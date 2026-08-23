#!/usr/bin/env python3
"""
Download YuNet (face detection) and SFace (face recognition) ONNX models
from the official OpenCV model zoo.

Usage:
    python download_face_models.py

Models are placed in models/face_models/ which is the path expected by
constants.FACE_MODEL_DIR and constants.FACE_SFACE_MODEL.

No training, no external service — just two pre-trained ONNX files.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    {
        "name": "face_detection_yunet_2023mar.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "size_kb": 338,
        "description": "YuNet face detector (OpenCV DNN)",
    },
    {
        "name": "face_recognition_sface_2021dec.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "size_kb": 37_000,
        "description": "SFace face recognizer — 128-D deep embedding",
    },
]

OUTPUT_DIR = Path(__file__).parent / "models" / "face_models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


class _ProgressReporter:
    """Simple progress callback for urllib.request.urlretrieve."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if pct != self._last_pct and pct % 10 == 0:
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%  ({_human_size(downloaded)} / {_human_size(total_size)})",
                  end="", flush=True)
            self._last_pct = pct
        if pct == 100:
            print()  # newline after completion


def _download(model: dict, output_dir: Path) -> Path:
    dest = output_dir / model["name"]
    if dest.exists():
        size = dest.stat().st_size
        print(f"  [SKIP] {model['name']} already present ({_human_size(size)})")
        return dest

    print(f"  Downloading {model['name']} (~{model['size_kb']} KB) ...")
    print(f"  URL: {model['url']}")

    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(model["url"], str(tmp), _ProgressReporter(model["name"]))
        tmp.rename(dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download {model['name']}: {exc}\n"
            f"Manual download: {model['url']}\n"
            f"Place in: {output_dir}"
        ) from exc

    print(f"  [OK] {model['name']} saved ({_human_size(dest.stat().st_size)})")
    return dest


def _verify_loadable(model: dict, dest: Path) -> bool:
    """Try to load the ONNX model through OpenCV to confirm it is valid."""
    try:
        import cv2
        name = model["name"]
        if "yunet" in name:
            detector = cv2.FaceDetectorYN_create(str(dest), "", (320, 320))
            print(f"  [OK] {name} loadable via cv2.FaceDetectorYN_create")
        elif "sface" in name:
            recognizer = cv2.FaceRecognizerSF_create(str(dest), "")
            print(f"  [OK] {name} loadable via cv2.FaceRecognizerSF_create")
        return True
    except Exception as exc:
        print(f"  [WARN] Load verification failed for {dest.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("OpenCV Face Model Downloader")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        print(f"[INFO] OpenCV version: {cv2.__version__}")
        if not hasattr(cv2, "FaceDetectorYN_create"):
            print("[WARN] cv2.FaceDetectorYN_create not found — YuNet may not work at runtime.")
        if not hasattr(cv2, "FaceRecognizerSF_create"):
            print("[WARN] cv2.FaceRecognizerSF_create not found — SFace may not work at runtime.")
    except ImportError:
        print("[ERROR] OpenCV not installed. Run: pip install opencv-python>=4.8.0")
        return 1

    errors = []
    for model in MODELS:
        print(f"\n{'-' * 60}")
        print(f"  {model['description']}")
        try:
            dest = _download(model, OUTPUT_DIR)
            _verify_loadable(model, dest)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            errors.append(model["name"])

    print(f"\n{'=' * 60}")
    if errors:
        print(f"[FAIL] {len(errors)} model(s) failed to download:")
        for e in errors:
            print(f"  - {e}")
        print("\nManual download instructions:")
        for m in MODELS:
            if m["name"] in errors:
                print(f"  {m['url']}")
                print(f"  → Save to: {OUTPUT_DIR / m['name']}")
        return 1

    print(f"[OK] All {len(MODELS)} models downloaded and verified.")
    print(f"[OK] Path: {OUTPUT_DIR}")
    print("\nNext step:")
    print("  python enroll_faces.py --source collected_assets/family "
          "--output models/faces --overwrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
