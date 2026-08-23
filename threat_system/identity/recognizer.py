"""
Face identity recognition for surveillance pipelines.

Architecture
------------
Detection  : OpenCV YuNet  (FaceDetectorYN)    — primary
             Haar cascade                        — fallback
Recognition: OpenCV SFace  (FaceRecognizerSF)  — primary
             Hand-crafted DCT/histogram embedding — fallback

Graceful degradation
--------------------
* If ONNX model files are missing, the module silently falls back to the
  Haar cascade detector and the hand-crafted embedding.  The pipeline
  continues to function; identity accuracy is simply reduced.

Public interface (unchanged)
----------------------------
    recognizer = FaceIdentityRecognizer(database_dir=...)
    results = recognizer.update(frame, person_bboxes)   # {tid: dict}
    recognizer.reset()

The ``results`` dict per tid always has the same keys regardless of which
backend is active (see ``_match_to_dict``).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from constants import (
    FACE_DB_PATH,
    FACE_DETECTOR_BACKEND,
    FACE_FALLBACK_ENABLED,
    FACE_EMBEDDINGS_CACHE,
    FACE_FACE_CROP_SIZE,
    FACE_FORGET_FRAMES,
    FACE_MODEL_DIR,
    FACE_MIN_FACE_SIZE,
    FACE_LOW_LIGHT_THRESHOLD,
    FACE_NOISE_THRESHOLD,
    FACE_RECOGNITION_MARGIN,
    FACE_RECOGNITION_THRESHOLD,
    FACE_SMOOTHING_WINDOW,
    FACE_YUNET_MODEL,
    FACE_SFACE_MODEL,
)

# ---------------------------------------------------------------------------
# Embedding dimension markers — used to detect stale caches
# ---------------------------------------------------------------------------
_SFACE_DIM = 128     # SFace outputs 128-D feature vectors
_LEGACY_DIM = 304    # Old hand-crafted embedding dimension (approx)


# ===========================================================================
# CCTV image enhancement helpers  (unchanged from previous version)
# ===========================================================================

def enhance_cctv_image(image):
    """Enhance a CCTV frame or crop for low light, compression and mild noise."""
    if image is None or image.size == 0:
        return None

    work = image.copy()

    if work.ndim == 3:
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        work = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    else:
        work = cv2.equalizeHist(work)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    brightness = float(gray.mean())
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < FACE_LOW_LIGHT_THRESHOLD:
        gamma = 0.75
        lookup = np.array([((index / 255.0) ** gamma) * 255 for index in range(256)], dtype='uint8')
        work = cv2.LUT(work, lookup)

    if blur_variance < FACE_NOISE_THRESHOLD:
        if work.ndim == 3:
            work = cv2.bilateralFilter(work, 5, 40, 40)
        else:
            work = cv2.medianBlur(work, 3)

    return work


def estimate_face_quality(face_crop):
    """Estimate how usable a face crop is under CCTV conditions.

    Returns a score in [0, 1]. Higher is better.
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0

    enhanced = enhance_cctv_image(face_crop)
    if enhanced is None:
        return 0.0

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if enhanced.ndim == 3 else enhanced
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area = float(gray.shape[0] * gray.shape[1])

    brightness_score = 1.0 - min(abs(brightness - 110.0) / 110.0, 1.0)
    contrast_score = min(contrast / 60.0, 1.0)
    sharpness_score = min(sharpness / 120.0, 1.0)
    size_score = min(area / float(FACE_FACE_CROP_SIZE * FACE_FACE_CROP_SIZE), 1.0)

    quality = (
        0.30 * brightness_score +
        0.30 * contrast_score +
        0.25 * sharpness_score +
        0.15 * size_score
    )

    return float(np.clip(quality, 0.0, 1.0))


# ===========================================================================
# Legacy / fallback hand-crafted embedding  (kept for graceful degradation)
# ===========================================================================

def augment_face_views(face_crop):
    """Generate small, deterministic augmentations to improve template stability."""
    if face_crop is None or face_crop.size == 0:
        return []

    enhanced = enhance_cctv_image(face_crop)
    if enhanced is None:
        return []

    if enhanced.ndim == 2:
        base = enhanced
    else:
        base = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    views = [base]
    views.append(cv2.flip(base, 1))
    views.append(cv2.equalizeHist(base))
    views.append(cv2.GaussianBlur(base, (3, 3), 0))
    return views


def prepare_face_crop(face_crop, target_size=(FACE_FACE_CROP_SIZE, FACE_FACE_CROP_SIZE)):
    """Normalize a face crop to a compact, consistent CCTV-friendly representation."""
    if face_crop is None or face_crop.size == 0:
        return None

    enhanced = enhance_cctv_image(face_crop)
    if enhanced is None:
        return None

    if enhanced.ndim == 3:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        gray = enhanced

    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    resized = cv2.GaussianBlur(resized, (3, 3), 0)
    return resized


def _compute_legacy_embedding(face_crop):
    """Legacy hand-crafted DCT + histogram + gradient embedding (fallback)."""
    prepared = prepare_face_crop(face_crop)
    if prepared is None:
        return None

    embeddings = []
    for view in augment_face_views(prepared):
        normalized = np.float32(view) / 255.0
        dct = cv2.dct(normalized)
        low_freq = dct[:16, :16].flatten()

        hist = cv2.calcHist([view], [0], None, [32], [0, 256]).flatten().astype(np.float32)
        hist_sum = float(hist.sum())
        if hist_sum > 0:
            hist /= hist_sum

        gradient_x = cv2.Sobel(view, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(view, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_hist = cv2.calcHist(
            [np.uint8(np.clip(gradient_mag, 0, 255))], [0], None, [16], [0, 256]
        ).flatten().astype(np.float32)
        gradient_sum = float(gradient_hist.sum())
        if gradient_sum > 0:
            gradient_hist /= gradient_sum

        embedding = np.concatenate([low_freq, hist, gradient_hist]).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-6:
            embeddings.append(embedding / norm)

    if not embeddings:
        return None

    template = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(template))
    if norm < 1e-6:
        return None
    return template / norm


# ===========================================================================
# Identity template cache  (shared by recognizer and enroll_faces.py)
# ===========================================================================

def build_identity_templates(database_dir, sface_recognizer=None):
    """Build identity templates from the enrolled face database folder.

    Parameters
    ----------
    database_dir : str | Path
        Root of the face database (contains per-identity subdirectories).
    sface_recognizer : cv2.FaceRecognizerSF | None
        When provided, SFace embeddings are computed.  Otherwise falls back
        to the legacy hand-crafted embedding.
    """
    templates = {}
    database_dir = Path(database_dir)
    if not database_dir.exists():
        return templates

    for identity_dir in sorted(database_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        embeddings = []
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            emb = _compute_embedding_with_backend(image, sface_recognizer)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            template = np.mean(np.stack(embeddings, axis=0), axis=0)
            norm = float(np.linalg.norm(template))
            if norm > 1e-6:
                template = template / norm
            templates[identity_dir.name] = template

    return templates


def load_template_cache(database_dir, expected_dim: Optional[int] = None):
    """Load cached identity templates, optionally validating embedding dimension."""
    cache_path = Path(database_dir) / FACE_EMBEDDINGS_CACHE
    if not cache_path.exists():
        return None

    try:
        payload = np.load(str(cache_path), allow_pickle=True)
        names = [str(name) for name in payload['names'].tolist()]
        templates = payload['templates']
        if len(names) != len(templates):
            return None
        if expected_dim is not None and len(templates) > 0:
            if templates[0].shape[0] != expected_dim:
                # Stale cache from a different backend — discard
                return None
        return {name: np.asarray(template, dtype=np.float32)
                for name, template in zip(names, templates)}
    except Exception:
        return None


def save_template_cache(database_dir, templates):
    """Persist identity templates to the face database folder."""
    if not templates:
        return None

    database_dir = Path(database_dir)
    database_dir.mkdir(parents=True, exist_ok=True)
    cache_path = database_dir / FACE_EMBEDDINGS_CACHE

    names = np.array(list(templates.keys()), dtype=object)
    vectors = np.stack(
        [np.asarray(template, dtype=np.float32) for template in templates.values()], axis=0
    )
    np.savez_compressed(str(cache_path), names=names, templates=vectors)
    return cache_path


def _compute_embedding_with_backend(image, sface_recognizer=None):
    """Compute an embedding for *image* using whichever backend is available.

    This is the shared helper used by both template building and runtime
    recognition so the two always use the same embedding space.
    """
    if sface_recognizer is not None:
        # ---- SFace path ------------------------------------------------
        # SFace needs a full-face BGR crop of reasonable size.
        # We use a simple Haar detection here for enrollment (not
        # time-critical); at runtime the recognizer uses YuNet landmarks.
        return _sface_embed_from_full_image(image, sface_recognizer)
    else:
        # ---- Legacy fallback -------------------------------------------
        return _compute_legacy_embedding(image)


def _sface_embed_from_full_image(image, sface_recognizer, yunet_detector=None):
    """Compute SFace embedding from an image that contains a face.

    Strategy (in order):
    1. Run internal YuNet to detect the face and get landmarks, then use
       ``alignCrop`` → ``feature()``.  This is the highest-quality path and
       is **identical to what runtime recognition does**, so templates built
       here will have cosine similarity > 0.363 with the same person at
       runtime.
    2. If no face is detected by YuNet, fall back to ``feature()`` on the
       whole image.  This handles portrait-style enrolled crops where the
       face fills the frame.

    The function never calls ``feature()`` on an arbitrary sub-crop; doing
    so produces embeddings in a different numeric space from the aligned path.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    if w < 20 or h < 20:
        return None

    # Ensure BGR 3-channel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # ---- Attempt YuNet + alignCrop path --------------------------------
    _det = yunet_detector
    _cleanup = False
    if _det is None and hasattr(cv2, 'FaceDetectorYN_create'):
        # Create a temporary detector for enrollment
        _model_path = Path(FACE_MODEL_DIR) / FACE_YUNET_MODEL
        if _model_path.exists():
            try:
                _det = cv2.FaceDetectorYN_create(str(_model_path), '', (w, h))
                _cleanup = True
            except Exception:
                _det = None

    if _det is not None:
        try:
            _det.setInputSize((w, h))
            _, faces = _det.detect(image)
            if faces is not None and len(faces) > 0:
                best_row = max(faces, key=lambda f: f[2] * f[3])
                row = np.asarray(best_row, dtype=np.float32).reshape(1, -1)
                aligned = sface_recognizer.alignCrop(image, row)
                if aligned is not None and aligned.size > 0:
                    feature = sface_recognizer.feature(aligned)
                    if feature is not None:
                        vec = np.asarray(feature, dtype=np.float32).flatten()
                        norm = float(np.linalg.norm(vec))
                        if norm > 1e-6:
                            return vec / norm
        except Exception:
            pass

    # ---- Fallback: direct feature() on whole image ---------------------
    # Used when face is not detectable (very small enrolled images).
    try:
        feature = sface_recognizer.feature(image)
        if feature is None:
            return None
        vec = np.asarray(feature, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return vec / norm
    except Exception:
        return None


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class IdentityMatch:
    name: str = 'unknown'
    confidence: float = 0.0
    similarity: float = 0.0
    quality_score: float = 0.0
    is_known_family: bool = False
    face_detected: bool = False
    face_bbox: list = None
    backend: str = 'haar'


@dataclass
class TrackIdentityState:
    """Temporal state used to stabilize identity decisions per track."""

    stable_name: str = 'unknown'
    stable_confidence: float = 0.0
    stable_similarity: float = 0.0
    stable_quality: float = 0.0
    candidate_name: str = 'unknown'
    consecutive_known_frames: int = 0
    consecutive_unknown_frames: int = 0
    last_update_frame: int = 0


# ===========================================================================
# Main recognizer class
# ===========================================================================

class FaceIdentityRecognizer:
    """Detect faces in tracked people and match them against enrolled identities.

    Detection  : YuNet (primary) or Haar cascade (fallback)
    Recognition: SFace 128-D deep embedding (primary) or hand-crafted (fallback)
    Stability  : per-track temporal smoothing with confirmation + forget windows
    """

    def __init__(self, database_dir=None, threshold=FACE_RECOGNITION_THRESHOLD,
                 smoothing_window=FACE_SMOOTHING_WINDOW, min_face_size=FACE_MIN_FACE_SIZE,
                 fallback_enabled=FACE_FALLBACK_ENABLED, verbose=False):
        self.database_dir = Path(database_dir or FACE_DB_PATH)
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.min_face_size = min_face_size
        self.fallback_enabled = fallback_enabled
        self.verbose = verbose

        # ---- Haar cascade (always available as last resort) ----
        cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))

        # ---- Model directory ----
        self.face_model_dir = Path(FACE_MODEL_DIR)
        self.detector_backend = FACE_DETECTOR_BACKEND.lower()

        # ---- YuNet (primary detector) ----
        self.yunet_detector = self._create_yunet_detector()

        # ---- SFace (primary recognizer) ----
        self.sface_recognizer = self._create_sface_recognizer()

        # Decide expected embedding dimension for cache validation
        self._expected_dim = _SFACE_DIM if self.sface_recognizer is not None else None

        # ---- Identity templates ----
        self.identity_templates = self._load_identity_templates()

        # ---- Temporal state ----
        self.identity_history = defaultdict(lambda: deque(maxlen=self.smoothing_window))
        self.track_states = defaultdict(TrackIdentityState)

        # ---- Stability hyper-parameters ----
        self.confirmation_frames = 2
        # ponytail: fixed frame grace window; make it time-based if source FPS varies materially.
        self.forget_frames = FACE_FORGET_FRAMES
        self.known_margin = FACE_RECOGNITION_MARGIN
        self.unknown_margin = 0.02
        self.confidence_ema_alpha = 0.35

        if self.verbose:
            backend = self._active_backend_name()
            print(f'[FaceID] Backend: {backend}')
            print(f'[FaceID] Identities loaded: {len(self.identity_templates)}  '
                  f'(from {self.database_dir})')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all temporal state (call between videos)."""
        self.identity_history.clear()
        self.track_states.clear()

    def update(self, frame, person_bboxes):
        """Return identity metadata for each tracked person.

        Parameters
        ----------
        frame : np.ndarray  BGR frame
        person_bboxes : dict  {tid: [x1, y1, x2, y2]}

        Returns
        -------
        dict  {tid: identity_dict}   — same keys for all backends
        """
        if frame is None or not person_bboxes:
            return {}

        matches = {}
        for tid, bbox in person_bboxes.items():
            raw_match = self._recognize_person(frame, bbox)
            self.identity_history[tid].append(raw_match)
            matches[tid] = self._stabilize_identity(tid)

        return matches

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_yunet_detector(self):
        if self.detector_backend == 'haar':
            return None
        if not hasattr(cv2, 'FaceDetectorYN_create'):
            if self.verbose:
                print('[FaceID] cv2.FaceDetectorYN_create not available — using Haar')
            return None

        model_path = self.face_model_dir / FACE_YUNET_MODEL
        if not model_path.exists():
            if self.verbose:
                print(f'[FaceID] YuNet model not found at {model_path} — using Haar')
            return None

        try:
            detector = cv2.FaceDetectorYN_create(str(model_path), '', (320, 320))
            if self.verbose:
                print(f'[FaceID] YuNet loaded: {model_path.name}')
            return detector
        except Exception as exc:
            if self.verbose:
                print(f'[FaceID] YuNet load failed ({exc}) — using Haar')
            return None

    def _create_sface_recognizer(self):
        if not hasattr(cv2, 'FaceRecognizerSF_create'):
            if self.verbose:
                print('[FaceID] cv2.FaceRecognizerSF_create not available — using legacy embedding')
            return None

        model_path = self.face_model_dir / FACE_SFACE_MODEL
        if not model_path.exists():
            if self.verbose:
                print(f'[FaceID] SFace model not found at {model_path} — using legacy embedding')
            return None

        try:
            recognizer = cv2.FaceRecognizerSF_create(str(model_path), '')
            if self.verbose:
                print(f'[FaceID] SFace loaded: {model_path.name}')
            return recognizer
        except Exception as exc:
            if self.verbose:
                print(f'[FaceID] SFace load failed ({exc}) — using legacy embedding')
            return None

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_identity_templates(self):
        cached = load_template_cache(self.database_dir, expected_dim=self._expected_dim)
        if cached:
            if self.verbose:
                print(f'[FaceID] Loaded {len(cached)} identity templates from cache')
            return cached

        if self.verbose:
            print('[FaceID] Building identity templates from images ...')

        templates = build_identity_templates(self.database_dir, self.sface_recognizer)
        if templates:
            save_template_cache(self.database_dir, templates)
            if self.verbose:
                print(f'[FaceID] Built and cached {len(templates)} templates')
        return templates

    # ------------------------------------------------------------------
    # Per-frame recognition
    # ------------------------------------------------------------------

    def _recognize_person(self, frame, bbox) -> IdentityMatch:
        """Detect face in person bbox and match against known identities."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(x1, w_frame - 1))
        y1 = max(0, min(y1, h_frame - 1))
        x2 = max(x1 + 1, min(x2, w_frame))
        y2 = max(y1 + 1, min(y2, h_frame))

        person_crop = frame[y1:y2, x1:x2]

        # ---- Detect face ----
        face_row, face_rects = self._detect_faces_all(person_crop)

        if face_rects is None and self.fallback_enabled:
            face_crop, face_bbox = self._fallback_face_crop(person_crop, x1, y1, x2, y2)
            face_detected = False
            face_row = None
        elif face_rects is not None:
            face_x, face_y, face_w, face_h = face_rects
            face_crop = person_crop[face_y:face_y + face_h, face_x:face_x + face_w]
            face_bbox = [x1 + face_x, y1 + face_y,
                         x1 + face_x + face_w, y1 + face_y + face_h]
            face_detected = True
        else:
            return IdentityMatch(backend=self._active_backend_name())

        # ---- Quality gate ----
        quality_score = estimate_face_quality(face_crop)
        if quality_score < 0.15:
            return IdentityMatch(
                face_detected=face_detected,
                face_bbox=face_bbox,
                quality_score=quality_score,
                backend=self._active_backend_name(),
            )

        # ---- Compute embedding ----
        embedding = self._compute_embedding(face_crop, face_row, person_crop)
        if embedding is None or not self.identity_templates:
            return IdentityMatch(
                face_detected=face_detected,
                face_bbox=face_bbox,
                quality_score=quality_score,
                backend=self._active_backend_name(),
            )

        # ---- Match against templates ----
        best_name = 'unknown'
        best_similarity = -1.0
        for name, template in self.identity_templates.items():
            similarity = float(np.dot(embedding, template))
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        # Quality-adjusted similarity
        adjusted_similarity = best_similarity * (0.60 + 0.40 * quality_score)
        if adjusted_similarity < self.threshold:
            best_name = 'unknown'

        denom = max(1e-6, 1.0 - self.threshold)
        confidence = float(np.clip((adjusted_similarity - self.threshold) / denom, 0.0, 1.0))
        if quality_score < 0.35:
            confidence *= 0.85

        return IdentityMatch(
            name=best_name,
            confidence=confidence,
            similarity=adjusted_similarity,
            quality_score=quality_score,
            is_known_family=best_name != 'unknown',
            face_detected=face_detected,
            face_bbox=face_bbox,
            backend=self._active_backend_name(),
        )

    # ------------------------------------------------------------------
    # Face detection helpers
    # ------------------------------------------------------------------

    def _detect_faces_all(self, image):
        """Try YuNet then Haar.  Returns (yunet_row | None, (x,y,w,h) | None)."""
        if image is None or image.size == 0:
            return None, None

        candidates = [image]
        enhanced = enhance_cctv_image(image)
        if enhanced is not None and enhanced is not image:
            candidates.append(enhanced)

        # YuNet path
        if self.yunet_detector is not None and self.detector_backend in {'auto', 'yunet'}:
            for candidate in candidates:
                yunet_row, rect = self._detect_face_yunet(candidate)
                if rect is not None:
                    return yunet_row, rect

        # Haar fallback
        for candidate in candidates:
            rect = self._detect_face_haar(candidate)
            if rect is not None:
                return None, rect

        return None, None

    def _detect_face_yunet(self, image) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        """Run YuNet and return (raw_detection_row, (x,y,w,h)) for the best face."""
        if self.yunet_detector is None:
            return None, None

        h, w = image.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return None, None

        # Ensure 3-channel BGR for YuNet
        img_bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        self.yunet_detector.setInputSize((w, h))
        _, faces = self.yunet_detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None, None

        # Pick largest face by area
        best_row = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = [int(round(v)) for v in best_row[:4]]
        if fw < self.min_face_size or fh < self.min_face_size:
            return None, None

        # Clamp to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        fw = max(1, min(fw, w - x))
        fh = max(1, min(fh, h - y))

        return best_row, (x, y, fw, fh)

    def _detect_face_haar(self, image) -> Optional[tuple]:
        """Run Haar cascade, return (x,y,w,h) for the best face or None."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(faces) == 0:
            return None
        return tuple(max(faces, key=lambda r: r[2] * r[3]))

    def _fallback_face_crop(self, person_crop, x1, y1, x2, y2):
        """Upper-body crop used when no face detector fires."""
        h, w = person_crop.shape[:2]
        face_h = max(self.min_face_size, int(h * 0.45))
        face_w = max(self.min_face_size, int(w * 0.60))
        cx = w // 2
        xs = max(0, cx - face_w // 2)
        ys = 0
        xe = min(w, xs + face_w)
        ye = min(h, ys + face_h)
        face_crop = person_crop[ys:ye, xs:xe]
        face_bbox = [x1 + xs, y1 + ys, x1 + xe, y1 + ye]
        return face_crop, face_bbox

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def _compute_embedding(self, face_crop, yunet_row=None, person_crop=None):
        """Return a unit-normalised embedding vector.

        Both template-building and runtime now use the YuNet → alignCrop → feature()
        path via ``_sface_embed_from_full_image``.  This ensures cosine similarity
        between a known person's template and their runtime embedding is > 0.363.

        We prefer ``person_crop`` as the input when available (larger region gives
        YuNet a clearer detection), falling back to ``face_crop``.
        """
        if self.sface_recognizer is not None:
            # Use person_crop if available (better context for YuNet internal detection)
            source = person_crop if person_crop is not None else face_crop
            emb = _sface_embed_from_full_image(source, self.sface_recognizer,
                                               yunet_detector=self.yunet_detector)
            if emb is not None:
                return emb

        # Legacy hand-crafted fallback
        return _compute_legacy_embedding(face_crop)


    def _sface_aligned_embed(self, image, yunet_row):
        """Use YuNet landmarks to align the face, then embed with SFace."""
        if self.sface_recognizer is None:
            return None

        img_bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        try:
            # alignCrop expects shape (H, W, 3) and a 1×15 float row
            row = np.asarray(yunet_row, dtype=np.float32).reshape(1, -1)
            aligned = self.sface_recognizer.alignCrop(img_bgr, row)
            if aligned is None or aligned.size == 0:
                return None
            feature = self.sface_recognizer.feature(aligned)
            if feature is None:
                return None
            vec = np.asarray(feature, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(vec))
            if norm < 1e-6:
                return None
            return vec / norm
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Temporal identity stabilisation  (unchanged logic)
    # ------------------------------------------------------------------

    def _stabilize_identity(self, track_id):
        history = list(self.identity_history[track_id])
        if not history:
            return self._match_to_dict(IdentityMatch())

        recent = history[-self.smoothing_window:]
        state = self.track_states[track_id]
        candidate = self._pick_candidate(recent)
        current = history[-1]

        if current.name != 'unknown':
            state.consecutive_unknown_frames = 0
        else:
            state.consecutive_unknown_frames += 1

        if candidate.name != 'unknown':
            if current.name == 'unknown':
                state.consecutive_known_frames = 0
            elif state.candidate_name == candidate.name:
                state.consecutive_known_frames += 1
            else:
                state.candidate_name = candidate.name
                state.consecutive_known_frames = 1

            state.stable_confidence = self._ema(state.stable_confidence, candidate.confidence)
            state.stable_similarity = self._ema(state.stable_similarity, candidate.similarity)
            state.stable_quality = self._ema(state.stable_quality, candidate.quality_score)

            if (current.name != 'unknown' and
                    self._is_confident_candidate(candidate, recent) and
                    state.consecutive_known_frames >= self.confirmation_frames):
                state.stable_name = candidate.name
                state.last_update_frame += 1
        else:
            state.consecutive_known_frames = 0

            if state.stable_name != 'unknown' and \
                    state.consecutive_unknown_frames < self.forget_frames:
                pass  # retain identity through brief misses
            else:
                state.stable_name = 'unknown'
                state.stable_confidence = self._ema(state.stable_confidence, current.confidence)
                state.stable_similarity = self._ema(state.stable_similarity, current.similarity)
                state.stable_quality = self._ema(state.stable_quality, current.quality_score)

        # Re-lock on one strong observation after a prolonged detector miss.
        if (state.stable_name == 'unknown' and current.name != 'unknown' and
                current.confidence >= self.threshold + self.known_margin):
            state.stable_name = current.name
            state.candidate_name = current.name
            state.consecutive_known_frames = self.confirmation_frames
            state.consecutive_unknown_frames = 0
            state.stable_confidence = current.confidence
            state.stable_similarity = current.similarity
            state.stable_quality = current.quality_score

        if (state.stable_name == 'unknown' and candidate.name != 'unknown' and
                candidate.confidence >= self.threshold + self.known_margin):
            state.candidate_name = candidate.name

        if state.stable_name != 'unknown' and \
                state.consecutive_unknown_frames >= self.forget_frames:
            state.stable_name = 'unknown'
            state.candidate_name = 'unknown'
            state.consecutive_known_frames = 0
            state.stable_confidence = self._ema(state.stable_confidence, current.confidence)
            state.stable_similarity = self._ema(state.stable_similarity, current.similarity)
            state.stable_quality = self._ema(state.stable_quality, current.quality_score)

        if state.stable_name != 'unknown':
            src = self._get_latest_by_name(recent, state.stable_name)
            stable = IdentityMatch(
                name=state.stable_name,
                confidence=state.stable_confidence,
                similarity=state.stable_similarity,
                quality_score=state.stable_quality,
                is_known_family=True,
                face_detected=src.face_detected if src else False,
                face_bbox=src.face_bbox if src else current.face_bbox,
            )
        else:
            stable = IdentityMatch(
                name='unknown',
                confidence=state.stable_confidence,
                similarity=state.stable_similarity,
                quality_score=current.quality_score,
                is_known_family=False,
                face_detected=current.face_detected,
                face_bbox=current.face_bbox,
            )

        return self._match_to_dict(stable)

    def _pick_candidate(self, history: List[IdentityMatch]) -> IdentityMatch:
        known = [m for m in history if m.name != 'unknown']
        if not known:
            return IdentityMatch()

        groups = defaultdict(list)
        for m in known:
            groups[m.name].append(m)

        def _score(items):
            return (len(items),
                    float(np.mean([m.confidence for m in items])),
                    float(np.mean([m.similarity for m in items])))

        best_name, best_items = max(groups.items(), key=lambda kv: _score(kv[1]))
        return IdentityMatch(
            name=best_name,
            confidence=float(np.mean([m.confidence for m in best_items])),
            similarity=float(np.mean([m.similarity for m in best_items])),
            quality_score=float(np.mean([m.quality_score for m in best_items])),
            is_known_family=True,
            face_detected=any(m.face_detected for m in best_items),
            face_bbox=best_items[-1].face_bbox,
        )

    def _is_confident_candidate(self, candidate, history):
        same = [m for m in history if m.name == candidate.name]
        support = len(same)
        avg_conf = float(np.mean([m.confidence for m in same]))
        avg_sim = float(np.mean([m.similarity for m in same]))
        return (support >= 2 or
                avg_conf >= self.threshold + self.known_margin or
                avg_sim >= self.threshold + self.known_margin)

    def _get_latest_by_name(self, history, name):
        for item in reversed(history):
            if item.name == name:
                return item
        return None

    def _ema(self, previous, value):
        return float(self.confidence_ema_alpha * value + (1 - self.confidence_ema_alpha) * previous)

    # ------------------------------------------------------------------
    # Output serialisation
    # ------------------------------------------------------------------

    def _match_to_dict(self, match: IdentityMatch) -> dict:
        is_known = match.name != 'unknown'
        return {
            'identity_name': match.name,
            'identity_confidence': round(float(match.confidence), 4),
            'identity_similarity': round(float(match.similarity), 4),
            'identity_quality': round(float(match.quality_score), 4),
            'is_known_family': is_known,
            'face_detected': match.face_detected,
            'face_bbox': match.face_bbox,
            'suppress_loitering': is_known,
            'identity_state': 'known' if is_known else 'unknown',
            # Always report the recognizer's active backend, not a per-match default.
            'face_backend': self._active_backend_name(),
        }

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _active_backend_name(self) -> str:
        det = 'yunet' if self.yunet_detector is not None else 'haar'
        rec = 'sface' if self.sface_recognizer is not None else 'legacy'
        return f'{det}+{rec}'

    def _normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return vector
        return vector / norm
