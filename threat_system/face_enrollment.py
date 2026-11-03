"""Bounded browser uploads into the pipeline's existing face database."""

import base64
import binascii
from io import BytesIO
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import uuid

from constants import FACE_EMBEDDINGS_CACHE


MAX_PHOTOS = 5
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_ENROLLMENT_BYTES = 70 * 1024 * 1024


def enroll_photos(name, photos, database_dir):
    if not isinstance(name, str):
        raise ValueError("Enter the person's name.")
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}", name):
        raise ValueError("Use 1–64 letters, numbers, spaces, hyphens or underscores for the name.")
    if name.casefold() in {"unknown", "con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}:
        raise ValueError("Choose a different name; this name is reserved.")
    if not isinstance(photos, list) or not 1 <= len(photos) <= MAX_PHOTOS:
        raise ValueError("Upload 1–5 JPEG or PNG photos.")

    from PIL import Image, UnidentifiedImageError
    from enroll_faces import enroll_faces

    decoded = []
    for index, photo in enumerate(photos, 1):
        if not isinstance(photo, str) or not 0 < len(photo) <= 4 * ((MAX_PHOTO_BYTES + 2) // 3):
            raise ValueError(f"Photo {index} must be at most 10 MB.")
        try:
            raw = base64.b64decode(photo, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError(f"Photo {index} is not a valid image upload.") from error
        if not raw or len(raw) > MAX_PHOTO_BYTES:
            raise ValueError(f"Photo {index} must be at most 10 MB.")
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG"} or getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Photo {index} must be a still JPEG or PNG image.")
                if min(image.size) < 48 or image.width * image.height > 12_000_000:
                    raise ValueError(f"Photo {index} must be at least 48×48 pixels and at most 12 megapixels.")
                extension = ".jpg" if image.format == "JPEG" else ".png"
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"Photo {index} is damaged or is not a supported image.") from error
        decoded.append((raw, extension))

    root = Path(database_dir).resolve()
    target = root / name
    if target.is_symlink() or target.resolve().parent != root:
        raise ValueError("This identity folder cannot be used for uploads.")
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".face-upload-", dir=root.parent) as temporary:
        stage = Path(temporary)
        source = stage / "source" / name
        source.mkdir(parents=True)
        for raw, extension in decoded:
            (source / (uuid.uuid4().hex + extension)).write_bytes(raw)
        summary, _ = enroll_faces(source, stage / "enrolled", require_single_face=True)
        if not summary["faces_enrolled"] or not summary["embeddings_cache"]:
            raise ValueError("No usable faces were enrolled. Upload clear, well-lit photos showing exactly one face each.")
        target.mkdir(parents=True, exist_ok=True)
        # Running recognizers keep their templates; the next run rebuilds from persisted crops.
        (root / FACE_EMBEDDINGS_CACHE).unlink(missing_ok=True)
        saved = []
        try:
            for crop in (stage / "enrolled" / name).glob("*.jpg"):
                destination = target / crop.name
                os.replace(crop, destination)
                saved.append(destination)
        except OSError:
            for destination in saved:
                destination.unlink(missing_ok=True)
            raise
    return {"name": name, "saved": len(saved), "skipped": summary["images_skipped"],
            "restart_required": True}
