#!/usr/bin/env python3
"""
Enroll family member face images into the local face database.

Expected layout for the source folder:
    source/
        Mom/
            img1.jpg
            img2.jpg
        Dad/
            img1.jpg

The script detects the largest face in each image, crops it, and writes the
normalized crop into models/faces/<identity>/ for use by the runtime identity
module.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from identity.recognizer import (
    estimate_face_quality,
    prepare_face_crop,
    build_identity_templates,
    save_template_cache,
)


MIN_FACE_BRIGHTNESS = 28.0
MIN_BLUR_VARIANCE = 18.0
MIN_EYE_COUNT = 2


def _get_face_cascade():
    cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f'Failed to load Haar cascade from {cascade_path}')
    return cascade


def _get_eye_cascade():
    cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_eye_tree_eyeglasses.xml'
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f'Failed to load eye cascade from {cascade_path}')
    return cascade


def _detect_largest_face(image, cascade, min_face_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )

    if len(faces) == 0:
        return None

    return max(faces, key=lambda rect: rect[2] * rect[3])


def _crop_face(image, face_rect):
    x, y, w, h = face_rect
    pad_x = int(w * 0.18)
    pad_y = int(h * 0.18)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    return image[y1:y2, x1:x2]


def _fallback_portrait_crop(image, min_face_size):
    height, width = image.shape[:2]

    crop_width = max(min_face_size * 3, int(width * 0.55))
    crop_height = max(min_face_size * 3, int(height * 0.60))

    center_x = width // 2
    x1 = max(0, center_x - crop_width // 2)
    x2 = min(width, x1 + crop_width)

    y1 = max(0, int(height * 0.08))
    y2 = min(height, y1 + crop_height)

    return image[y1:y2, x1:x2]


def _assess_face_quality(face_crop, eye_cascade, detected_face=True):
    if face_crop is None or face_crop.size == 0:
        return False, 'empty_crop'

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    brightness = float(gray.mean())
    if brightness < MIN_FACE_BRIGHTNESS:
        return False, 'too_dark'

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality_score = estimate_face_quality(face_crop)
    if blur_variance < MIN_BLUR_VARIANCE and quality_score < 0.22:
        return False, 'too_blurry'

    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(12, 12),
    )
    eye_count = len(eyes)
    if not detected_face and eye_count < MIN_EYE_COUNT and quality_score < 0.38:
        return False, 'not_frontal'

    aspect_ratio = face_crop.shape[1] / max(1, face_crop.shape[0])
    if aspect_ratio < 0.65 or aspect_ratio > 1.45:
        return False, 'bad_aspect_ratio'

    if quality_score < 0.18:
        return False, 'low_quality'

    return True, 'ok'


def _looks_like_cctv_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return brightness < MIN_FACE_BRIGHTNESS or contrast < 24.0


def enroll_faces(source_dir, output_dir, overwrite=False, min_face_size=48):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {source_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    cascade = _get_face_cascade()
    eye_cascade = _get_eye_cascade()

    summary = {
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'identities': {},
        'images_seen': 0,
        'faces_enrolled': 0,
        'images_skipped': 0,
        'skip_reasons': {},
    }

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    top_level_images = [p for p in sorted(source_dir.iterdir()) if p.is_file() and p.suffix.lower() in image_extensions]
    subfolder_identities = [p for p in sorted(source_dir.iterdir()) if p.is_dir()]

    enrollment_batches = []

    if top_level_images:
        enrollment_batches.append((source_dir.name, top_level_images))

    for identity_dir in subfolder_identities:
        identity_images = [p for p in sorted(identity_dir.iterdir()) if p.is_file() and p.suffix.lower() in image_extensions]
        if identity_images:
            enrollment_batches.append((identity_dir.name, identity_images))

    if not enrollment_batches:
        raise RuntimeError(
            f'No supported images found in {source_dir}. Provide either images directly in the folder '
            f'or identity subfolders containing images.'
        )

    for identity_name, image_paths in enrollment_batches:
        target_identity_dir = output_dir / identity_name
        target_identity_dir.mkdir(parents=True, exist_ok=True)

        enrolled_count = 0
        skipped_count = 0
        skip_reasons = {}

        def record_skip(reason):
            nonlocal skipped_count
            skipped_count += 1
            summary['images_skipped'] += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        for image_path in image_paths:
            summary['images_seen'] += 1
            image = cv2.imread(str(image_path))
            if image is None:
                record_skip('read_failed')
                continue

            face_rect = _detect_largest_face(image, cascade, min_face_size)
            if face_rect is None:
                face_crop = _fallback_portrait_crop(image, min_face_size)
                detected_face = False
            else:
                face_crop = _crop_face(image, face_rect)
                detected_face = True

            is_valid, reason = _assess_face_quality(face_crop, eye_cascade, detected_face=detected_face)
            if not is_valid:
                record_skip(reason)
                continue

            normalized = prepare_face_crop(face_crop)
            if normalized is None:
                record_skip('normalization_failed')
                continue

            resized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            destination = target_identity_dir / f'{image_path.stem}_face.jpg'

            if destination.exists() and not overwrite:
                record_skip('exists')
                continue

            cv2.imwrite(str(destination), resized)
            enrolled_count += 1
            summary['faces_enrolled'] += 1

        summary['identities'][identity_name] = {
            'images_enrolled': enrolled_count,
            'images_skipped': skipped_count,
            'skip_reasons': skip_reasons,
            'output_dir': str(target_identity_dir),
        }

        for reason, count in skip_reasons.items():
            summary['skip_reasons'][reason] = summary['skip_reasons'].get(reason, 0) + count

    templates = build_identity_templates(output_dir)
    cache_path = save_template_cache(output_dir, templates)
    summary['embeddings_cache'] = str(cache_path) if cache_path else None

    manifest_path = output_dir / 'enrollment_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    return summary, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description='Enroll family member faces into the local database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  python enroll_faces.py --source C:\\photos\\family --output models\\faces
        '''
    )
    parser.add_argument('--source', required=True, help='Folder containing identity subfolders')
    parser.add_argument('--output', default='models/faces', help='Target face database folder')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing enrolled face crops')
    parser.add_argument('--min-face-size', type=int, default=48, help='Minimum face size for detection')

    args = parser.parse_args()

    try:
        summary, manifest_path = enroll_faces(
            args.source,
            args.output,
            overwrite=args.overwrite,
            min_face_size=args.min_face_size,
        )
    except Exception as exc:
        print(f'[ERROR] Enrollment failed: {exc}')
        return 1

    print(f"[OK] Enrolled {summary['faces_enrolled']} face crops from {summary['images_seen']} images")
    print(f'[OK] Manifest written to: {manifest_path}')
    for identity, stats in summary['identities'].items():
        print(f"  - {identity}: {stats['images_enrolled']} enrolled, {stats['images_skipped']} skipped")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
