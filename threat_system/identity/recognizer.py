"""
Lightweight face identity recognition for surveillance pipelines.

This module intentionally avoids heavy new dependencies so it can run on a
Windows laptop now and move to Jetson Orin Nano later with minimal changes.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from constants import (
    FACE_DB_PATH,
    FACE_DETECTOR_BACKEND,
    FACE_FALLBACK_ENABLED,
    FACE_EMBEDDINGS_CACHE,
    FACE_FACE_CROP_SIZE,
    FACE_MODEL_DIR,
    FACE_MIN_FACE_SIZE,
    FACE_LOW_LIGHT_THRESHOLD,
    FACE_NOISE_THRESHOLD,
    FACE_RECOGNITION_MARGIN,
    FACE_RECOGNITION_THRESHOLD,
    FACE_SMOOTHING_WINDOW,
    FACE_YUNET_MODEL,
)


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

    Returns a score in [0, 1]. Higher is better. The score is intentionally
    forgiving for low-light footage and light compression noise.
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


def compute_face_embedding(face_crop):
    """Compute a lightweight embedding from a normalized face crop."""
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
        gradient_hist = cv2.calcHist([np.uint8(np.clip(gradient_mag, 0, 255))], [0], None, [16], [0, 256]).flatten().astype(np.float32)
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


def build_identity_templates(database_dir):
    """Build identity templates from the enrolled face database folder."""
    templates = {}
    database_dir = Path(database_dir)
    if not database_dir.exists():
        return templates

    for identity_dir in sorted(database_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        embeddings = []
        for image_path in identity_dir.iterdir():
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            embedding = compute_face_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            template = np.mean(np.stack(embeddings, axis=0), axis=0)
            norm = float(np.linalg.norm(template))
            if norm > 1e-6:
                template = template / norm
            templates[identity_dir.name] = template

    return templates


def load_template_cache(database_dir):
    """Load cached identity templates if they exist."""
    cache_path = Path(database_dir) / FACE_EMBEDDINGS_CACHE
    if not cache_path.exists():
        return None

    try:
        payload = np.load(str(cache_path), allow_pickle=False)
        names = [str(name) for name in payload['names'].tolist()]
        templates = payload['templates']
        if len(names) != len(templates):
            return None
        return {name: np.asarray(template, dtype=np.float32) for name, template in zip(names, templates)}
    except Exception:
        return None


def save_template_cache(database_dir, templates):
    """Persist identity templates to the face database folder."""
    if not templates:
        return None

    database_dir = Path(database_dir)
    database_dir.mkdir(parents=True, exist_ok=True)
    cache_path = database_dir / FACE_EMBEDDINGS_CACHE

    names = np.array(list(templates.keys()), dtype=str)
    vectors = np.stack([np.asarray(template, dtype=np.float32) for template in templates.values()], axis=0)
    np.savez_compressed(str(cache_path), names=names, templates=vectors)
    return cache_path


@dataclass
class IdentityMatch:
    name: str = 'unknown'
    confidence: float = 0.0
    similarity: float = 0.0
    quality_score: float = 0.0
    is_known_family: bool = False
    face_detected: bool = False
    face_bbox: list[int] | None = None
    backend: str = 'haar'


@dataclass
class TrackIdentityState:
    """Temporal state used to stabilize identity decisions per track."""

    stable_name: str = 'unknown'
    stable_confidence: float = 0.0
    stable_similarity: float = 0.0
    candidate_name: str = 'unknown'
    consecutive_known_frames: int = 0
    consecutive_unknown_frames: int = 0
    last_update_frame: int = 0


class FaceIdentityRecognizer:
    """Detect faces in tracked people and match them against enrolled identities."""

    def __init__(self, database_dir=None, threshold=FACE_RECOGNITION_THRESHOLD,
                 smoothing_window=FACE_SMOOTHING_WINDOW, min_face_size=FACE_MIN_FACE_SIZE,
                 fallback_enabled=FACE_FALLBACK_ENABLED, model_dir=None, verbose=False):
        self.database_dir = Path(database_dir or FACE_DB_PATH)
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.min_face_size = min_face_size
        self.fallback_enabled = fallback_enabled
        self.verbose = verbose

        cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self.face_model_dir = Path(model_dir or FACE_MODEL_DIR)
        self.detector_backend = FACE_DETECTOR_BACKEND.lower()
        self.yunet_detector = self._create_yunet_detector()

        self.identity_templates = self._load_identity_templates()
        self.identity_history = defaultdict(lambda: deque(maxlen=self.smoothing_window))
        self.track_states = defaultdict(TrackIdentityState)

        self.confirmation_frames = 2
        self.forget_frames = 3
        self.known_margin = FACE_RECOGNITION_MARGIN
        self.unknown_margin = 0.02
        self.confidence_ema_alpha = 0.35

        if self.verbose:
            print(f'[FaceID] Loaded {len(self.identity_templates)} identities from {self.database_dir}')

    def reset(self):
        self.identity_history.clear()
        self.track_states.clear()

    def update(self, frame, person_bboxes):
        """Return identity metadata for each tracked person."""
        if frame is None or not person_bboxes:
            return {}

        matches = {}
        for tid, bbox in person_bboxes.items():
            raw_match = self._recognize_person(frame, bbox)
            self.identity_history[tid].append(raw_match)
            matches[tid] = self._stabilize_identity(tid)

        return matches

    def _stabilize_identity(self, track_id):
        history = list(self.identity_history[track_id])
        if not history:
            return self._match_to_dict(IdentityMatch())

        recent_history = history[-self.smoothing_window:]
        state = self.track_states[track_id]

        candidate = self._pick_candidate(recent_history)
        current_frame_match = history[-1]

        if candidate.name != 'unknown':
            if state.candidate_name == candidate.name:
                state.consecutive_known_frames += 1
            else:
                state.candidate_name = candidate.name
                state.consecutive_known_frames = 1
                state.consecutive_unknown_frames = 0

            state.stable_confidence = self._ema(state.stable_confidence, candidate.confidence)
            state.stable_similarity = self._ema(state.stable_similarity, candidate.similarity)

            if self._is_confident_candidate(candidate, recent_history) and state.consecutive_known_frames >= self.confirmation_frames:
                state.stable_name = candidate.name
                state.last_update_frame += 1

        else:
            state.consecutive_unknown_frames += 1
            state.consecutive_known_frames = 0

            if state.stable_name != 'unknown' and state.consecutive_unknown_frames < self.forget_frames:
                # Keep the previously confirmed identity through brief misses.
                pass
            else:
                state.stable_name = 'unknown'
                state.stable_confidence = self._ema(state.stable_confidence, current_frame_match.confidence)
                state.stable_similarity = self._ema(state.stable_similarity, current_frame_match.similarity)

        if state.stable_name == 'unknown' and candidate.name != 'unknown' and candidate.confidence >= self.threshold + self.known_margin:
            # Start a confirmation window, but do not flip immediately.
            state.candidate_name = candidate.name

        if state.stable_name != 'unknown' and state.consecutive_unknown_frames >= self.forget_frames:
            state.stable_name = 'unknown'

        if state.stable_name != 'unknown':
            stable_source = self._get_latest_by_name(recent_history, state.stable_name)
            stable_match = IdentityMatch(
                name=state.stable_name,
                confidence=state.stable_confidence,
                similarity=state.stable_similarity,
                is_known_family=True,
                face_detected=stable_source.face_detected if stable_source else False,
                face_bbox=stable_source.face_bbox if stable_source else current_frame_match.face_bbox,
            )
        else:
            stable_match = IdentityMatch(
                name='unknown',
                confidence=state.stable_confidence,
                similarity=state.stable_similarity,
                is_known_family=False,
                face_detected=current_frame_match.face_detected,
                face_bbox=current_frame_match.face_bbox,
            )

        return self._match_to_dict(stable_match)

    def _pick_candidate(self, history):
        known_history = [item for item in history if item.name != 'unknown']
        if not known_history:
            return IdentityMatch()

        label_groups = defaultdict(list)
        for item in known_history:
            label_groups[item.name].append(item)

        def score(items):
            confidences = [m.confidence for m in items]
            similarities = [m.similarity for m in items]
            return (
                len(items),
                float(np.mean(confidences)),
                float(np.mean(similarities)),
            )

        best_name, best_items = max(label_groups.items(), key=lambda kv: score(kv[1]))
        confidence = float(np.mean([m.confidence for m in best_items]))
        similarity = float(np.mean([m.similarity for m in best_items]))

        return IdentityMatch(
            name=best_name,
            confidence=confidence,
            similarity=similarity,
            is_known_family=True,
            face_detected=any(m.face_detected for m in best_items),
            face_bbox=best_items[-1].face_bbox,
        )

    def _is_confident_candidate(self, candidate, history):
        support = sum(1 for item in history if item.name == candidate.name)
        average_conf = float(np.mean([item.confidence for item in history if item.name == candidate.name]))
        average_similarity = float(np.mean([item.similarity for item in history if item.name == candidate.name]))
        return support >= self.confirmation_frames and (
            average_conf >= self.threshold + self.known_margin or
            average_similarity >= self.threshold + self.known_margin
        )

    def _get_latest_by_name(self, history, name):
        for item in reversed(history):
            if item.name == name:
                return item
        return None

    def _ema(self, previous, value):
        return float(self.confidence_ema_alpha * value + (1 - self.confidence_ema_alpha) * previous)

    def _match_to_dict(self, match):
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
            'face_backend': match.backend,
        }

    def _load_identity_templates(self):
        cached_templates = load_template_cache(self.database_dir)
        if cached_templates:
            return cached_templates

        templates = build_identity_templates(self.database_dir)
        if templates:
            save_template_cache(self.database_dir, templates)
        return templates

    def _extract_reference_face(self, image):
        faces = self._detect_faces(image)
        if faces:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            return image[y:y + h, x:x + w]

        if not self.fallback_enabled:
            return None

        height, width = image.shape[:2]
        x1 = int(width * 0.20)
        x2 = int(width * 0.80)
        y1 = 0
        y2 = max(self.min_face_size, int(height * 0.55))
        return image[y1:y2, x1:x2]

    def _recognize_person(self, frame, bbox):
        x1, y1, x2, y2 = [int(value) for value in bbox]
        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        person_crop = frame[y1:y2, x1:x2]
        face_rects = self._detect_faces(person_crop)

        if not face_rects:
            enhanced_person = enhance_cctv_image(person_crop)
            if enhanced_person is not None:
                face_rects = self._detect_faces(enhanced_person)
                if face_rects:
                    person_crop = enhanced_person

        if face_rects:
            face_x, face_y, face_w, face_h = max(face_rects, key=lambda rect: rect[2] * rect[3])
            face_crop = person_crop[face_y:face_y + face_h, face_x:face_x + face_w]
            face_bbox = [x1 + face_x, y1 + face_y, x1 + face_x + face_w, y1 + face_y + face_h]
            face_detected = True
        elif self.fallback_enabled:
            face_crop, face_bbox = self._fallback_face_crop(person_crop, x1, y1, x2, y2)
            face_detected = False
        else:
            return IdentityMatch()

        quality_score = estimate_face_quality(face_crop)
        if quality_score < 0.15:
            return IdentityMatch(face_detected=face_detected, face_bbox=face_bbox, quality_score=quality_score, backend=self._active_backend_name())

        # Never suppress a security signal from a guessed upper-body crop.
        if not face_detected:
            return IdentityMatch(face_bbox=face_bbox, quality_score=quality_score, backend=self._active_backend_name())

        embedding = self._compute_embedding(face_crop)
        if embedding is None or not self.identity_templates:
            return IdentityMatch(face_detected=face_detected, face_bbox=face_bbox, quality_score=quality_score, backend=self._active_backend_name())

        best_name = 'unknown'
        best_similarity = -1.0
        for name, template in self.identity_templates.items():
            similarity = float(np.dot(embedding, template))
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        adjusted_similarity = best_similarity * (0.60 + 0.40 * quality_score)
        if adjusted_similarity < self.threshold:
            best_name = 'unknown'

        confidence = float(np.clip((adjusted_similarity - self.threshold) / max(1e-6, 1.0 - self.threshold), 0.0, 1.0))
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

    def _fallback_face_crop(self, person_crop, x1, y1, x2, y2):
        height, width = person_crop.shape[:2]
        face_height = max(self.min_face_size, int(height * 0.45))
        face_width = max(self.min_face_size, int(width * 0.60))

        center_x = width // 2
        x_start = max(0, center_x - face_width // 2)
        y_start = 0
        x_end = min(width, x_start + face_width)
        y_end = min(height, y_start + face_height)

        face_crop = person_crop[y_start:y_end, x_start:x_end]
        face_bbox = [x1 + x_start, y1 + y_start, x1 + x_end, y1 + y_end]
        return face_crop, face_bbox

    def _detect_faces(self, image):
        if image is None or image.size == 0:
            return []

        candidates = [image]
        enhanced = enhance_cctv_image(image)
        if enhanced is not None and enhanced is not image:
            candidates.append(enhanced)

        if self.yunet_detector is not None and self.detector_backend in {'auto', 'yunet'}:
            for candidate in candidates:
                faces = self._detect_faces_yunet(candidate)
                if faces:
                    return faces

        for candidate in candidates:
            gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) if candidate.ndim == 3 else candidate
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
            if len(faces) > 0:
                return list(faces)

        return []

    def _detect_faces_yunet(self, image):
        if self.yunet_detector is None:
            return []

        height, width = image.shape[:2]
        self.yunet_detector.setInputSize((width, height))
        _, faces = self.yunet_detector.detect(image)
        if faces is None or len(faces) == 0:
            return []

        detections = []
        for face in faces:
            x, y, w, h = [int(round(value)) for value in face[:4]]
            if w >= self.min_face_size and h >= self.min_face_size:
                detections.append((x, y, w, h))
        return detections

    def _create_yunet_detector(self):
        if self.detector_backend == 'haar':
            return None
        if not hasattr(cv2, 'FaceDetectorYN_create'):
            return None

        model_path = self.face_model_dir / FACE_YUNET_MODEL
        if not model_path.exists():
            return None

        try:
            detector = cv2.FaceDetectorYN_create(str(model_path), '', (320, 320))
            return detector
        except Exception:
            return None

    def _active_backend_name(self):
        if self.yunet_detector is not None:
            return 'yunet'
        return 'haar'

    def _compute_embedding(self, face_crop):
        return compute_face_embedding(face_crop)

    def _normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return vector
        return vector / norm
