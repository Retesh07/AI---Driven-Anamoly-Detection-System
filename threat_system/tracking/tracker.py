"""
Shared tracking module - maintains person identities across frames.
"""

import numpy as np
from collections import defaultdict


class PersonTracker:
    """
    Manages temporal state (kinematics) for each person across frames.
    Separate from detection-based tracking - focuses on feature extraction state.
    """
    
    def __init__(self, max_inactive_frames=30):
        """
        Args:
            max_inactive_frames: Drop tracking state after N frames without detection
        """
        self.max_inactive_frames = max_inactive_frames
        self.prev_centers = {}   # tid -> (xc, yc)
        self.prev_vel = {}       # tid -> (vx, vy)
        self.prev_acc = {}       # tid -> (ax, ay)
        self.prev_kps = {}       # tid -> keypoints (17, 2)
        self.frame_count = {}    # tid -> last frame seen
        self.current_frame = 0
    
    def update(self, detection_ids):
        """
        Mark which persons were detected in this frame.
        Cleans up inactive persons.
        
        Args:
            detection_ids: List of track IDs detected in current frame
        """
        self.current_frame += 1
        
        # Mark detected persons as active
        for tid in detection_ids:
            self.frame_count[tid] = self.current_frame
        
        # Remove inactive persons
        to_remove = [tid for tid, frame in self.frame_count.items()
                    if self.current_frame - frame > self.max_inactive_frames]
        for tid in to_remove:
            self.remove(tid)
    
    def remove(self, tid):
        """Clean up state for a person."""
        self.prev_centers.pop(tid, None)
        self.prev_vel.pop(tid, None)
        self.prev_acc.pop(tid, None)
        self.prev_kps.pop(tid, None)
        self.frame_count.pop(tid, None)
    
    def set_kinematic_state(self, tid, xc, yc, dx, dy, ddx, ddy, kps):
        """
        Update kinematic state for a person.
        
        Args:
            tid: Track ID
            xc, yc: Normalized center (0-1)
            dx, dy: Velocity
            ddx, ddy: Acceleration
            kps: Keypoints array (17, 2)
        """
        self.prev_centers[tid] = (xc, yc)
        self.prev_vel[tid] = (dx, dy)
        self.prev_acc[tid] = (ddx, ddy)
        self.prev_kps[tid] = kps.copy()
    
    def get_prev_center(self, tid):
        """Get previous position of person."""
        return self.prev_centers.get(tid)
    
    def get_prev_vel(self, tid):
        """Get previous velocity of person."""
        return self.prev_vel.get(tid)
    
    def get_prev_acc(self, tid):
        """Get previous acceleration of person."""
        return self.prev_acc.get(tid)
    
    def get_prev_kps(self, tid):
        """Get previous keypoints of person."""
        return self.prev_kps.get(tid)
    
    def reset(self):
        """Clear all tracking state (e.g., between videos)."""
        self.prev_centers.clear()
        self.prev_vel.clear()
        self.prev_acc.clear()
        self.prev_kps.clear()
        self.frame_count.clear()
        self.current_frame = 0


class TemporalBuffer:
    """
    Maintains a sliding window of features for temporal modeling.
    """
    
    def __init__(self, max_len=60):
        """
        Args:
            max_len: Maximum buffer length (sequence length)
        """
        self.max_len = max_len
        self.buffer = []
    
    def append(self, feature):
        """Add feature to buffer."""
        self.buffer.append(feature)
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)
    
    def get_sequence(self):
        """Get current buffer as numpy array."""
        if len(self.buffer) == 0:
            return None
        return np.array(self.buffer)
    
    def is_full(self):
        """Check if buffer has reached max length."""
        return len(self.buffer) >= self.max_len
    
    def reset(self):
        """Clear buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class PersonTemporalState:
    """
    Maintains per-person temporal history for context-aware modeling.
    """
    
    def __init__(self, track_id, buffer_size=60):
        """
        Args:
            track_id: Person's tracking ID
            buffer_size: Temporal buffer size
        """
        self.track_id = track_id
        self.buffer = TemporalBuffer(buffer_size)
        self.last_updated = 0
        self.ema_threat_score = 0.0
        self.ema_loitering_score = 0.0
        self.ema_weapon_score = 0.0
        self.confidence_history = []
    
    def update(self, feature_vector, threat_score=None):
        """
        Update temporal state.
        
        Args:
            feature_vector: Current frame features
            threat_score: Current threat assessment (optional)
        """
        self.buffer.append(feature_vector)
        self.last_updated += 1
        if threat_score is not None:
            self.confidence_history.append(threat_score)
            if len(self.confidence_history) > 30:
                self.confidence_history.pop(0)
    
    def get_sequence(self):
        """Get temporal buffer as sequence."""
        return self.buffer.get_sequence()
    
    def is_ready(self):
        """Check if buffer has enough frames for inference."""
        return self.buffer.is_full()
    
    def reset(self):
        """Clear all temporal data."""
        self.buffer.reset()
        self.confidence_history.clear()
