#!/usr/bin/env python3
"""
One-shot setup script: rebuilds the face embedding cache using the correct
YuNet → alignCrop → SFace feature path, then runs a self-verification test.

Run this after any model or code changes:
    python rebuild_face_db.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from identity.recognizer import (
    FaceIdentityRecognizer,
    build_identity_templates,
    save_template_cache,
)
from constants import FACE_DB_PATH

DB = Path(FACE_DB_PATH)

print("=" * 60)
print("Face DB Rebuild")
print(f"DB path: {DB.resolve()}")
print("=" * 60)

# Step 1: Load recognizer (prints backend info)
r = FaceIdentityRecognizer(database_dir=DB, verbose=True)

# Step 2: Rebuild templates from enrolled images using SFace
print("\nRebuilding identity templates...")
templates = build_identity_templates(DB, r.sface_recognizer)

if not templates:
    print("[WARN] No templates built — check that models/faces/ contains identity subfolders.")
    sys.exit(1)

cache_path = save_template_cache(DB, templates)

print(f"\nTemplates built: {list(templates.keys())}")
for name, vec in templates.items():
    print(f"  {name}: dim={vec.shape[0]}  norm={np.linalg.norm(vec):.6f}")
print(f"Cache saved: {cache_path}")

# Step 3: Self-verification — reload and test against enrolled images
print("\n" + "-" * 60)
print("Self-verification: testing enrolled images...")

r2 = FaceIdentityRecognizer(database_dir=DB, verbose=False)

pass_count = 0
fail_count = 0

for identity_dir in sorted(DB.iterdir()):
    if not identity_dir.is_dir():
        continue
    for img_path in sorted(identity_dir.iterdir()):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        r2.reset()
        # Run 3 frames to pass temporal confirmation window
        for _ in range(3):
            results = r2.update(img, {1: [0, 0, w, h]})
        res = results.get(1, {})
        name = res.get('identity_name', 'unknown')
        sim = res.get('identity_similarity', 0.0)
        fd = res.get('face_detected', False)
        status = "OK" if name == identity_dir.name else "MISS"
        if status == "OK":
            pass_count += 1
        else:
            fail_count += 1
        print(f"  [{status}] {identity_dir.name}/{img_path.name}  "
              f"-> '{name}'  sim={sim:.4f}  face_detected={fd}")

print(f"\nSelf-verification: {pass_count} passed, {fail_count} failed")
print("=" * 60)
