"""
Loitering analysis module - detects suspicious lingering behavior.

Improved beyond simple threshold-based approach:
  - Tracks time in region + repeated trajectories
  - Distinguishes idle vs purposeful motion
  - Uses per-person temporal history
"""

import numpy as np
from collections import defaultdict, deque
from constants import LOITERING_TIME_THRESHOLD, LOITERING_MOVEMENT_THRESHOLD


class LoiteringAnalyzer:
    """
    Context-aware loitering detection.
    
    Responsibilities:
      - Track position history per person
      - Measure time spent in small regions
      - Detect repeated trajectories
      - Distinguish purposeful vs idle motion
      - Output temporal loitering score
    
    Features analyzed:
      - Dwell time: How long person stays in region
      - Movement radius: How much area person covers
      - Velocity consistency: Idle vs moving
      - Trajectory repetition: Pacing in circles
    """
    
    def __init__(self, time_threshold=LOITERING_TIME_THRESHOLD,
                 movement_threshold=LOITERING_MOVEMENT_THRESHOLD,
                 fps=25, frame_skip=2):
        """
        Args:
            time_threshold: Seconds of inactivity to trigger loitering (default 5)
            movement_threshold: Normalized pixels threshold for movement (default 0.05)
            fps: Video frame rate
            frame_skip: Frame skip factor (actual fps = fps/frame_skip)
        """
        self.time_threshold = time_threshold
        self.movement_threshold = movement_threshold
        self.fps = fps / frame_skip  # Effective fps after skipping frames
        self.frame_skip = frame_skip
        
        # Per-person state
        self.position_history = defaultdict(lambda: deque(maxlen=300))  # Last 300 positions
        self.velocity_history = defaultdict(lambda: deque(maxlen=60))   # Last 60 velocities
        self.stationary_counter = defaultdict(int)  # Frames person stayed in same region
        self.loitering_ema = defaultdict(float)  # EMA loitering score
        self.centroid_trace = defaultdict(lambda: deque(maxlen=30))    # Last 30 centroids
    
    def update(self, person_bboxes, frame_shape=(480, 640), identity_info=None):
        """
        Update loitering state.
        
        Args:
            person_bboxes: Dict of {track_id: [x1, y1, x2, y2]} (pixel coordinates)
            frame_shape: (H, W)
            identity_info: Optional dict of {track_id: identity metadata}
        
        Returns:
            Dict of {track_id: loitering_info}
        """
        H, W = frame_shape
        identity_info = identity_info or {}
        
        person_results = {}
        
        for tid, bbox in person_bboxes.items():
            x1, y1, x2, y2 = bbox
            
            # Compute center (normalized)
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            
            # Store position
            self.position_history[tid].append((cx, cy))
            self.centroid_trace[tid].append((cx, cy))
            
            # ===== Feature 1: Dwell Time =====
            # How long has the person stayed in a small area?
            dwell_time = self._compute_dwell_time(tid)
            
            # ===== Feature 2: Movement Radius =====
            # How much area has the person covered recently?
            movement_radius = self._compute_movement_radius(tid)
            
            # ===== Feature 3: Velocity Pattern =====
            # Is the person idle or moving purposefully?
            velocity_consistency = self._compute_velocity_consistency(tid)
            
            # ===== Feature 4: Trajectory Repetition =====
            # Is the person pacing in circles?
            path_entropy = self._compute_path_entropy(tid)
            
            # ===== Combined Loitering Score =====
            # Combines all features
            loitering_score = self._compute_loitering_score(
                dwell_time, movement_radius, velocity_consistency, path_entropy
            )
            
            # EMA smoothing
            alpha = 0.2
            self.loitering_ema[tid] = (
                alpha * loitering_score + (1 - alpha) * self.loitering_ema[tid]
            )
            
            identity = identity_info.get(tid, {})
            is_known_family = bool(identity.get('is_known_family', False))

            person_results[tid] = {
                'dwell_time_s': dwell_time,
                'movement_radius': movement_radius,
                'velocity_consistency': velocity_consistency,
                'path_entropy': path_entropy,
                'instant_score': loitering_score,
                'smooth_score': 0.0 if is_known_family else self.loitering_ema[tid],
                'raw_smooth_score': self.loitering_ema[tid],
                'loitering_detected': False if is_known_family else self.loitering_ema[tid] > 0.6,
                'high_confidence': False if is_known_family else self.loitering_ema[tid] > 0.8,
                'loitering_suppressed': is_known_family,
                'identity_name': identity.get('identity_name', 'unknown'),
                'identity_confidence': identity.get('identity_confidence', 0.0),
                'known_family': is_known_family,
            }
        
        return person_results
    
    def _compute_dwell_time(self, track_id):
        """
        Compute how long person has stayed in a small region.
        
        Returns:
            float: Seconds spent in small area
        """
        history = self.position_history[track_id]
        if len(history) < 2:
            return 0.0
        
        recent = list(history)[-int(self.fps * 10):]  # Last 10 seconds
        if len(recent) < 2:
            return 0.0
        
        # Compute spatial variance
        pos_array = np.array(recent)
        spatial_variance = np.var(pos_array, axis=0).sum()
        
        # If variance is low, person is staying in place
        if spatial_variance < self.movement_threshold ** 2:
            return min(len(recent) / self.fps, self.time_threshold * 2)
        
        return 0.0
    
    def _compute_movement_radius(self, track_id):
        """
        Compute radius of movement area (normalized spatial extent).
        
        Returns:
            float: Movement radius in normalized coordinates [0, 1]
        """
        history = self.position_history[track_id]
        if len(history) < 5:
            return 1.0
        
        recent = list(history)[-int(self.fps * 30):]  # Last 30 seconds
        pos_array = np.array(recent)
        
        # Compute bounding box of movement
        x_range = pos_array[:, 0].max() - pos_array[:, 0].min()
        y_range = pos_array[:, 1].max() - pos_array[:, 1].min()
        
        # Average radius
        radius = np.sqrt(x_range ** 2 + y_range ** 2) / 2
        
        return radius
    
    def _compute_velocity_consistency(self, track_id):
        """
        Compute consistency of velocity (repeated motion pattern).
        
        Low variance in velocity = walking in circles
        High variance = varied movement = normal activity
        
        Returns:
            float: Consistency score [0, 1], 1 = very consistent (concerning)
        """
        history = self.position_history[track_id]
        if len(history) < 3:
            return 0.0
        
        recent = list(history)[-int(self.fps * 15):]  # Last 15 seconds
        if len(recent) < 3:
            return 0.0
        
        pos_array = np.array(recent)
        velocities = np.diff(pos_array, axis=0)
        
        # Speed at each step
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Consistency: low speed variance (always same speed = pacing)
        if speeds.mean() < 0.005:  # Moving very slowly
            return 1.0
        
        speed_cv = speeds.std() / (speeds.mean() + 1e-6)  # Coefficient of variation
        
        # Low CV = consistent speed (pacing)
        consistency = np.clip(1.0 - speed_cv / 2, 0, 1)
        
        return consistency
    
    def _compute_path_entropy(self, track_id):
        """
        Compute entropy of trajectory (randomness of path).
        
        Low entropy = predictable, repeated path (concerning)
        High entropy = random movement (normal)
        
        Returns:
            float: Entropy score [0, 1], 0 = predictable (concerning)
        """
        trace = self.centroid_trace[track_id]
        if len(trace) < 10:
            return 0.5  # Neutral
        
        recent = list(trace)[-20:]  # Last 20 frames
        pos_array = np.array(recent)
        
        # Quantize positions to grid
        grid_size = 10
        grid_positions = np.floor(pos_array * grid_size).astype(int)
        
        # Count unique positions
        unique_positions = len(set(tuple(p) for p in grid_positions))
        
        # Entropy: more unique = more randomness = less concerning
        max_unique = len(grid_positions)
        entropy = unique_positions / max_unique if max_unique > 0 else 0.5
        
        return entropy
    
    def _compute_loitering_score(self, dwell_time, movement_radius,
                                velocity_consistency, path_entropy):
        """
        Combine all features into single loitering score.
        
        Returns:
            float: Loitering score [0, 1]
        """
        # Dwell time component: longer dwell = higher score
        dwell_score = min(dwell_time / self.time_threshold, 1.0)
        
        # Small movement area = higher score
        movement_score = max(0, 1.0 - movement_radius / 0.3)
        
        # Consistent velocity (pacing) = higher score
        velocity_score = velocity_consistency
        
        # Low entropy (predictable) = higher score
        entropy_score = max(0, 1.0 - path_entropy / 0.5)
        
        # Weighted combination
        score = (
            dwell_score * 0.4 +
            movement_score * 0.3 +
            velocity_score * 0.2 +
            entropy_score * 0.1
        )
        
        return float(np.clip(score, 0.0, 1.0))
    
    def remove_person(self, track_id):
        """Clean up state for person who left frame."""
        self.position_history.pop(track_id, None)
        self.velocity_history.pop(track_id, None)
        self.stationary_counter.pop(track_id, None)
        self.loitering_ema.pop(track_id, None)
        self.centroid_trace.pop(track_id, None)
    
    def reset(self):
        """Clear all state for new video."""
        self.position_history.clear()
        self.velocity_history.clear()
        self.stationary_counter.clear()
        self.loitering_ema.clear()
        self.centroid_trace.clear()

