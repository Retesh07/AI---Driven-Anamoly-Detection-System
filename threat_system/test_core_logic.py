import tempfile
import unittest
from collections import defaultdict, deque

import numpy as np

from constants import FACE_CONFIRMATION_FRAMES, FACE_FORGET_FRAMES
from fusion.temporal_fusion import TemporalFusion
from identity.recognizer import (
    FaceIdentityRecognizer,
    IdentityMatch,
    TrackIdentityState,
    load_template_cache,
    save_template_cache,
)


class TemporalFusionTests(unittest.TestCase):
    def interaction(self, first, second):
        fusion = TemporalFusion()
        signal = {'smooth_prob': 0.2}
        fusion.process_frame(signal, {}, {}, first, [1, 2])
        results, interactions = fusion.process_frame(signal, {}, {}, second, [1, 2])
        return results, interactions[(1, 2)]

    def test_approaching_people_have_positive_rate(self):
        _, interaction = self.interaction(
            {1: (0.0, 0.0), 2: (0.4, 0.0)},
            {1: (0.1, 0.0), 2: (0.3, 0.0)},
        )
        self.assertGreater(interaction.approach_rate, 0)

    def test_separating_people_have_negative_rate(self):
        _, interaction = self.interaction(
            {1: (0.1, 0.0), 2: (0.3, 0.0)},
            {1: (0.0, 0.0), 2: (0.4, 0.0)},
        )
        self.assertLess(interaction.approach_rate, 0)

    def test_interaction_adds_only_incremental_amplification(self):
        results, interaction = self.interaction(
            {1: (0.0, 0.0), 2: (0.1, 0.0)},
            {1: (0.02, 0.0), 2: (0.08, 0.0)},
        )
        self.assertGreater(interaction.interaction_intensity, 0.3)
        self.assertLess(results[1]['interaction_amplification'], 1.1)


class FaceIdentityTests(unittest.TestCase):
    def recognizer(self):
        recognizer = FaceIdentityRecognizer.__new__(FaceIdentityRecognizer)
        recognizer.threshold = 0.78
        recognizer.confirmation_frames = FACE_CONFIRMATION_FRAMES
        recognizer.forget_frames = FACE_FORGET_FRAMES
        recognizer.smoothing_window = 5
        recognizer.confidence_ema_alpha = 0.35
        recognizer.identity_history = defaultdict(lambda: deque(maxlen=5))
        recognizer.track_states = defaultdict(TrackIdentityState)
        return recognizer

    def stabilize(self, recognizer, match):
        recognizer.identity_history[1].append(match)
        return recognizer._stabilize_identity(1)['identity_name']

    def test_borderline_matches_do_not_confirm_identity(self):
        recognizer = self.recognizer()
        history = [
            IdentityMatch(name='Mom', confidence=0.05, similarity=0.79),
            IdentityMatch(name='Mom', confidence=0.05, similarity=0.79),
        ]
        self.assertFalse(recognizer._is_confident_candidate(history[-1], history))

    def test_identity_survives_brief_detection_misses(self):
        recognizer = self.recognizer()
        known = IdentityMatch(name='Mom', confidence=0.8, similarity=0.9, face_detected=True)
        unknown = IdentityMatch()

        self.assertEqual([self.stabilize(recognizer, known) for _ in range(FACE_CONFIRMATION_FRAMES)][-1], 'Mom')
        self.assertEqual(
            [self.stabilize(recognizer, unknown) for _ in range(FACE_FORGET_FRAMES - 1)],
            ['Mom'] * (FACE_FORGET_FRAMES - 1),
        )
        self.assertEqual(self.stabilize(recognizer, unknown), 'unknown')

    def test_single_wrong_label_does_not_replace_stable_identity(self):
        recognizer = self.recognizer()
        mom = IdentityMatch(name='Mom', confidence=0.8, similarity=0.9, face_detected=True)
        dad = IdentityMatch(name='Dad', confidence=0.8, similarity=0.9, face_detected=True)

        [self.stabilize(recognizer, mom) for _ in range(FACE_CONFIRMATION_FRAMES)]
        self.assertEqual(self.stabilize(recognizer, dad), 'Mom')
        self.assertEqual(self.stabilize(recognizer, mom), 'Mom')

    def test_fallback_crop_cannot_match_known_family(self):
        recognizer = FaceIdentityRecognizer.__new__(FaceIdentityRecognizer)
        recognizer.fallback_enabled = True
        recognizer.min_face_size = 16
        recognizer.identity_templates = {'Mom': np.ones(304, dtype=np.float32)}
        recognizer._detect_faces = lambda _: []
        recognizer._active_backend_name = lambda: 'haar'
        frame = np.full((120, 80, 3), 128, dtype=np.uint8)
        match = recognizer._recognize_person(frame, [0, 0, 80, 120])
        self.assertEqual(match.name, 'unknown')

    def test_template_cache_loads_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = {'Mom': np.array([1.0, 0.0], dtype=np.float32)}
            save_template_cache(directory, expected)
            loaded = load_template_cache(directory)
            np.testing.assert_array_equal(loaded['Mom'], expected['Mom'])


if __name__ == '__main__':
    unittest.main()
