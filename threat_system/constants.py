"""
Global constants for threat detection system.
"""

# Feature dimensions
FEATURE_DIM = 126      # Total feature dimension
PERSON_DIM = 60        # Per-person feature dimension (keypoint + kinematic + motion)
INTERACTION_DIM = 6    # Interaction feature dimension
SEQ_LEN = 60           # Temporal sequence length (frames)

# Pose keypoint constants
NUM_KEYPOINTS = 17
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Pose annotation body parts
ARM_SEGS = [(5, 7), (7, 9), (6, 8), (8, 10)]      # Arm segments
LEG_SEGS = [(11, 13), (13, 15), (12, 14), (14, 16)]  # Leg segments
TORSO_SEG = [(5, 6), (5, 11), (6, 12), (11, 12)]   # Torso segments
HEAD_SEG = [(0, 1), (0, 2), (1, 3), (2, 4)]       # Head segments

# Model paths (relative to project root)
MODEL_PATHS = {
    'violence': 'models/best_model_v3.pth',
    'violence_mean': 'models/pose_features_v3/mean.npy',
    'violence_std': 'models/pose_features_v3/std.npy',
    'weapon': 'models/weapon_detector.pt',
}

# Face identity settings
FACE_DB_PATH = 'models/faces'
FACE_EMBEDDINGS_CACHE = 'face_embeddings.npz'
FACE_FACE_CROP_SIZE = 160
FACE_MODEL_DIR = 'models/face_models'
FACE_DETECTOR_BACKEND = 'auto'  # auto | yunet | haar
FACE_YUNET_MODEL = 'face_detection_yunet_2023mar.onnx'
FACE_RECOGNITION_THRESHOLD = 0.78
FACE_RECOGNITION_MARGIN = 0.06
FACE_SMOOTHING_WINDOW = 5
FACE_MIN_FACE_SIZE = 48
FACE_FALLBACK_ENABLED = True
FACE_LOW_LIGHT_THRESHOLD = 90.0
FACE_NOISE_THRESHOLD = 28.0
FACE_MIN_QUALITY_TO_ENROLL = 0.22
FACE_MIN_QUALITY_TO_MATCH = 0.18

# Detection parameters
POSE_CONFIDENCE_THRESHOLD = 0.25
PERSON_DETECTION_CLASS = 0  # YOLO class ID for person

# Tracking parameters
TRACK_INACTIVE_FRAMES = 30
NUM_TOP_PERSONS = 2  # Track top N persons by area

# Violence detection thresholds
# Will be set during calibration, but these are defaults
DEFAULT_VIOLENCE_THRESHOLD = 0.65
DEFAULT_WARNING_THRESHOLD = 0.45

# Inference smoothing
CONSECUTIVE_ALERT_FRAMES = 5  # Frames needed for sustained alert

# Per-module parameters
# Gun detection parameters
GUN_CONFIDENCE_THRESHOLD = 0.25  # Higher threshold - only strong detections
GUN_SIZE_MIN_WIDTH = 40  # Minimum gun width (wider than knife)
GUN_SIZE_MAX_WIDTH = 400  # Maximum gun width
GUN_SIZE_MIN_HEIGHT = 20  # Minimum gun height
GUN_SIZE_MAX_HEIGHT = 300  # Maximum gun height

# Knife detection parameters (separate tuning)
KNIFE_CONFIDENCE_THRESHOLD = 0.22  # Slightly lower for small blades
KNIFE_SIZE_MIN_WIDTH = 15  # Narrower than guns
KNIFE_SIZE_MAX_WIDTH = 150  # Smaller max (knife is small)
KNIFE_SIZE_MIN_HEIGHT = 10  # Shorter than guns
KNIFE_SIZE_MAX_HEIGHT = 250  # Smaller max

# Shared weapon parameters
WEAPON_CONFIDENCE_THRESHOLD = 0.25  # Fallback for unknown weapon types
WEAPON_TEMPORAL_BUFFER = 10  # Keep weapon detection active for N frames after detection
WEAPON_CONFIDENCE_BOOST = 0.35  # Boost confidence if weapon was recently detected
WEAPON_EMA_DECISION_THRESHOLD = 0.20  # Report weapon if smooth_score > this threshold
WEAPON_SPATIAL_THRESHOLD = 0.15  # Max normalized distance weapon can move between frames (0-1)
WEAPON_CLASS_CONSISTENCY_FRAMES = 3  # Require weapon class to be consistent over N frames
WEAPON_HISTORY_AGREEMENT_THRESHOLD = 0.55  # Require 55% class consistency (stricter)
EMA_ALPHA = 0.35  # EMA smoothing factor (used by violence and other modules)
LOITERING_TIME_THRESHOLD = 5.0  # seconds
LOITERING_MOVEMENT_THRESHOLD = 0.05  # normalized pixels
SKELETON_COLORS = [
    (80, 210, 255),    # Cyan
    (255, 170, 60),    # Orange
    (100, 255, 100),   # Green
    (255, 100, 200)    # Magenta
]

VIOLENCE_COLOR = (30, 30, 230)      # Red (BGR)
WARNING_COLOR = (30, 165, 240)       # Yellow (BGR)
NORMAL_COLOR = (60, 200, 60)         # Green (BGR)

# Frame processing
FRAME_SKIP = 2  # Process every Nth frame
