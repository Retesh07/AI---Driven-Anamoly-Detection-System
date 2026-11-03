"""Run in the inference environment: python -m unittest test_face_enrollment -v."""

import base64
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from constants import FACE_EMBEDDINGS_CACHE
from dashboard import Dashboard
from enroll_faces import _detect_largest_face
from face_enrollment import MAX_ENROLLMENT_BYTES, MAX_PHOTO_BYTES, enroll_photos
from identity.recognizer import FaceIdentityRecognizer


def photo(size=(160, 160), image_format="PNG"):
    image = Image.effect_noise(size, 40).convert("RGB")
    output = BytesIO()
    image.save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode()


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "faces"
        self.photo = photo()

    def test_rejects_invalid_names_payloads_and_images_before_saving(self):
        for name, photos in [(None, [self.photo]), ("../Alex", [self.photo]), ("C:\\Alex", [self.photo]),
                             ("CON", [self.photo]), ("unknown", [self.photo]), ("a" * 65, [self.photo]),
                             ("Alex", []), ("Alex", [self.photo] * 6), ("Alex", ["not base64!"]),
                             ("Alex", [base64.b64encode(b"not an image").decode()]),
                             ("Alex", ["A" * (4 * ((MAX_PHOTO_BYTES + 2) // 3) + 1)]),
                             ("Alex", [photo((40, 40))]), ("Alex", [photo(image_format="GIF")])]:
            with self.subTest(name=name, count=len(photos)), self.assertRaises(ValueError):
                enroll_photos(name, photos, self.root)
        self.assertFalse(self.root.exists())
        with self.assertRaisesRegex(ValueError, "at most 10 MB"):
            enroll_photos("Alex", [base64.b64encode(b"0" * (MAX_PHOTO_BYTES + 1)).decode()], self.root)
        output = BytesIO()
        Image.new("RGB", (4000, 3001)).save(output, format="PNG")
        with self.assertRaisesRegex(ValueError, "12 megapixels"):
            enroll_photos("Alex", [base64.b64encode(output.getvalue()).decode()], self.root)

    def test_persists_crops_and_new_recognizer_rebuilds_old_cache(self):
        self.root.mkdir()
        existing = self.root / "Existing"
        existing.mkdir()
        original = base64.b64decode(self.photo)
        (existing / "original.png").write_bytes(original)
        initial = FaceIdentityRecognizer(database_dir=self.root)
        self.assertEqual(set(initial.identity_templates), {"Existing"})
        self.assertTrue((self.root / FACE_EMBEDDINGS_CACHE).exists())
        cascade = Mock()
        cascade.detectMultiScale.return_value = np.array([[20, 15, 100, 110]])
        with patch("enroll_faces._get_face_cascade", return_value=cascade), patch("enroll_faces._assess_face_quality", return_value=(True, "ok")):
            result = enroll_photos(" Alex ", [self.photo, self.photo], self.root)
            self.assertEqual(result, {"name": "Alex", "saved": 2, "skipped": 0, "restart_required": True})
            self.assertFalse((self.root / FACE_EMBEDDINGS_CACHE).exists())
            restarted = FaceIdentityRecognizer(database_dir=self.root)
            self.assertEqual(set(restarted.identity_templates), {"Alex", "Existing"})
            self.assertEqual(set(initial.identity_templates), {"Existing"})
            enroll_photos("Alex", [self.photo], self.root)
        self.assertEqual(len(list((self.root / "Alex").glob("*.jpg"))), 3)
        self.assertEqual((existing / "original.png").read_bytes(), original)
        self.assertEqual(list(self.root.parent.glob(".face-upload-*")), [])

    def test_rejects_zero_or_multiple_faces_without_altering_database(self):
        for faces in [[], [[0, 0, 60, 60], [60, 60, 70, 70]]]:
            cascade = Mock()
            cascade.detectMultiScale.return_value = np.array(faces)
            with patch("enroll_faces._get_face_cascade", return_value=cascade), self.assertRaisesRegex(ValueError, "No usable faces"):
                enroll_photos("Alex", [self.photo], self.root)
            self.assertFalse((self.root / "Alex").exists())
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        self.assertEqual(tuple(_detect_largest_face(image, cascade, 48)), (60, 60, 70, 70))
        self.assertIsNone(_detect_largest_face(image, cascade, 48, require_single_face=True))

    def test_disk_failure_rolls_back_only_new_crops_and_rejects_symlinks(self):
        self.root.mkdir()
        outside = self.root.parent / "outside"
        outside.mkdir()
        if os.name != "nt":
            (self.root / "Alias").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                enroll_photos("Alias", [self.photo], self.root)
        target = self.root / "Alex"
        target.mkdir()
        (target / "keep.jpg").write_bytes(b"existing crop")
        cascade = Mock()
        cascade.detectMultiScale.return_value = np.array([[20, 15, 100, 110]])
        replace = os.replace
        moves = 0
        def fail_second(source, destination):
            nonlocal moves
            moves += 1
            if moves == 2:
                raise OSError("test disk full")
            return replace(source, destination)
        with patch("enroll_faces._get_face_cascade", return_value=cascade), patch("enroll_faces._assess_face_quality", return_value=(True, "ok")), patch("face_enrollment.os.replace", side_effect=fail_second), self.assertRaises(OSError):
            enroll_photos("Alex", [self.photo, self.photo], self.root)
        self.assertEqual([p.name for p in target.iterdir()], ["keep.jpg"])
        self.assertEqual((target / "keep.jpg").read_bytes(), b"existing crop")

    def test_http_enrollment_requires_auth_control_token_and_allowed_origin(self):
        origin = "https://sentinel-test.vercel.app"
        dashboard = Dashboard(face_db_path=self.root, allowed_origins=[origin], access_token="test-only-key")
        port = dashboard.start(port=0)
        self.addCleanup(dashboard.stop)
        self.assertTrue(dashboard.snapshot()["can_enroll"])
        headers = {"Content-Type": "application/json", "Origin": origin,
                   "Authorization": "Bearer test-only-key", "X-Sentinel-Token": dashboard.control_token}
        def post(payload, overrides=None):
            request = Request(f"http://127.0.0.1:{port}/api/faces/enroll", data=json.dumps(payload).encode(), headers={**headers, **(overrides or {})})
            return urlopen(request, timeout=10)
        for overrides, status in [({"Authorization": "Bearer wrong"}, 401), ({"X-Sentinel-Token": "wrong"}, 403),
                                  ({"Origin": "https://evil.example"}, 403), ({"Content-Length": str(MAX_ENROLLMENT_BYTES + 1)}, 400),
                                  ({"Content-Type": "text/plain"}, 400)]:
            with patch("dashboard.enroll_photos") as enroll, self.assertRaises(HTTPError) as error:
                post({}, overrides)
            self.assertEqual(error.exception.code, status)
            error.exception.close()
            enroll.assert_not_called()
        cascade = Mock()
        cascade.detectMultiScale.return_value = np.array([[20, 15, 100, 110]])
        raw = base64.b64decode(self.photo)
        largest_photo = base64.b64encode(raw + b"\0" * (MAX_PHOTO_BYTES - len(raw))).decode()
        with patch("enroll_faces._get_face_cascade", return_value=cascade), patch("enroll_faces._assess_face_quality", return_value=(True, "ok")):
            with post({"name": "Alex", "photos": [largest_photo]}) as response:
                self.assertEqual(response.status, 201)
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
                self.assertEqual(json.load(response)["saved"], 1)
        self.assertIn("Alex", FaceIdentityRecognizer(database_dir=self.root).identity_templates)
        for failure, status in [(ImportError("cv2"), 503), (OSError("private disk path"), 500)]:
            with patch("dashboard.enroll_photos", side_effect=failure), self.assertRaises(HTTPError) as error:
                post({"name": "Alex", "photos": [self.photo]})
            self.assertEqual(error.exception.code, status)
            self.assertNotIn(b"private disk path", error.exception.read())
            error.exception.close()
        self.assertFalse(Dashboard(demo=True).snapshot()["can_enroll"])
        with self.assertRaises(RuntimeError):
            Dashboard(demo=True).enroll_faces({})


if __name__ == "__main__":
    unittest.main()
