from collections import defaultdict, deque

from identity.recognizer import FaceIdentityRecognizer, IdentityMatch, TrackIdentityState


def _recognizer():
    recognizer = object.__new__(FaceIdentityRecognizer)
    recognizer.smoothing_window = 5
    recognizer.threshold = 0.363
    recognizer.known_margin = 0.04
    recognizer.confirmation_frames = 2
    recognizer.forget_frames = 12
    recognizer.confidence_ema_alpha = 0.35
    recognizer.identity_history = defaultdict(lambda: deque(maxlen=recognizer.smoothing_window))
    recognizer.track_states = defaultdict(TrackIdentityState)
    recognizer._active_backend_name = lambda: 'test'
    return recognizer


def _feed(recognizer, match):
    recognizer.identity_history[1].append(match)
    return recognizer._stabilize_identity(1)


def demo():
    recognizer = _recognizer()
    known = IdentityMatch(
        name='Rishabh', confidence=0.9, similarity=0.9,
        quality_score=0.7, face_detected=True,
    )
    unknown = IdentityMatch()

    for _ in range(3):
        result = _feed(recognizer, known)
    assert result['identity_name'] == 'Rishabh'
    assert result['identity_quality'] > 0

    for _ in range(12):
        result = _feed(recognizer, unknown)
    assert result['identity_state'] == 'unknown'

    result = _feed(recognizer, known)
    assert result['identity_name'] == 'Rishabh'
    assert result['identity_quality'] == 0.7


if __name__ == '__main__':
    demo()
    print('identity stability check: OK')
