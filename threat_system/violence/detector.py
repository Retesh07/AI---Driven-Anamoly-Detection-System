"""
Violence detection inference pipeline.

Maintains temporal buffers, applies temporal smoothing, and detects violence
with false-positive mitigation strategies.
"""

import numpy as np
import torch
from collections import defaultdict, deque
from constants import (
    SEQ_LEN, FEATURE_DIM, PERSON_DIM, INTERACTION_DIM,
    DEFAULT_VIOLENCE_THRESHOLD, DEFAULT_WARNING_THRESHOLD,
    EMA_ALPHA, CONSECUTIVE_ALERT_FRAMES
)
from violence.model import ViolenceDetectorV3


class ViolenceDetector:
    """
    Context-aware violence detection engine.
    
    Responsibilities:
      - Maintain temporal sequences per person pair
      - Apply Bi-LSTM + Attention model inference
      - Apply temporal smoothing (EMA)
      - Detect sustained alerts (consecutive frames)
      - Implement false-positive mitigation
    
    False positive mitigations:
      1. FP-conservative threshold from calibration (precision >= 0.82)
      2. Cold-start guard: skip prediction first 30 frames (buffer warming)
      3. Sustained alert: violence only if EMA >= threshold for 5 consecutive frames
      4. Single-person gate: raise threshold by +0.10 if < 2 people detected
    """
    
    def __init__(self, model_path, mean_path, std_path, device='cuda',
                 violence_threshold=None, warning_threshold=None,
                 ema_alpha=EMA_ALPHA, consecutive_frames=CONSECUTIVE_ALERT_FRAMES):
        """
        Args:
            model_path: Path to trained violence detector weights
            mean_path: Path to feature normalization mean
            std_path: Path to feature normalization std
            device: 'cuda' or 'cpu'
            violence_threshold: Threshold for violence detection (default uses calibrated)
            warning_threshold: Threshold for warning (default uses calibrated)
            ema_alpha: Smoothing factor for temporal filtering
            consecutive_frames: Consecutive frames needed for sustained alert
        """
        self.device = torch.device(device)
        
        # Load model
        self.model = ViolenceDetectorV3().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load normalization
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.load(std_path).astype(np.float32)
        
        # Settings
        self.violence_threshold = violence_threshold or DEFAULT_VIOLENCE_THRESHOLD
        self.warning_threshold = warning_threshold or DEFAULT_WARNING_THRESHOLD
        self.ema_alpha = ema_alpha
        self.consecutive_frames = consecutive_frames
        
        # State
        self.buffer = deque(maxlen=SEQ_LEN)
        self.ema_prob = 0.0
        self.raw_prob = 0.0
        self.last_attention_weights = None
        self.consecutive_alerts = 0
        self.confirmed_violence = False
        self.frame_count = 0
        self.person_ema_scores = defaultdict(float)  # Per-person threat scores
        self.last_attn_weights = None
    
    def reset(self):
        """Reset state for new video."""
        self.buffer.clear()
        self.ema_prob = 0.0
        self.raw_prob = 0.0
        self.last_attention_weights = None
        self.consecutive_alerts = 0
        self.confirmed_violence = False
        self.frame_count = 0
        self.person_ema_scores.clear()
        self.last_attn_weights = None
    
    def update(self, frame_features, num_persons=2):
        """
        Process frame features and update violence detection.
        
        Args:
            frame_features: Feature vector (FEATURE_DIM,) from current frame
            num_persons: Number of persons detected in frame
        
        Returns:
            Dict with detection results:
                'raw_prob': Raw network output
                'smooth_prob': EMA-smoothed probability
                'status': 'VIOLENCE', 'WARNING', or 'NORMAL'
                'confirmed': True if sustained violence alert
                'person_scores': List of (track_id, threat_score) tuples
        """
        self.frame_count += 1
        self.buffer.append(frame_features)
        
        # ===== FP Mitigation 1: Cold-start Guard =====
        # Wait for buffer to warm up before making predictions
        if len(self.buffer) < SEQ_LEN or self.frame_count <= 30:
            self.raw_prob = 0.0
            self.ema_prob = self.ema_alpha * self.raw_prob + (1 - self.ema_alpha) * self.ema_prob
        else:
            # ===== Run Inference =====
            seq = np.stack(list(self.buffer)).astype(np.float32)
            seq_norm = (seq - self.mean) / self.std
            
            with torch.no_grad():
                logit, attn_w = self.model(
                    torch.tensor(seq_norm).unsqueeze(0).to(self.device)
                )
            
            self.raw_prob = float(torch.sigmoid(logit).item())
            self.last_attn_weights = attn_w[0].cpu().numpy()
            
            # ===== Apply Temporal Smoothing =====
            self.ema_prob = self.ema_alpha * self.raw_prob + (1 - self.ema_alpha) * self.ema_prob
        
        # ===== FP Mitigation 4: Single-Person Gate =====
        # Raise threshold if only 1 person (motion alone shouldn't trigger violence)
        effective_threshold = self.violence_threshold
        if num_persons < 2:
            effective_threshold = self.violence_threshold + 0.10
        
        # ===== FP Mitigation 3: Sustained Alert Requirement =====
        if self.ema_prob >= effective_threshold:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = max(0, self.consecutive_alerts - 1)
        
        self.confirmed_violence = self.consecutive_alerts >= self.consecutive_frames
        
        # ===== Status Determination =====
        if self.confirmed_violence:
            status = 'VIOLENCE'
        elif self.ema_prob >= self.warning_threshold:
            status = 'WARNING'
        else:
            status = 'NORMAL'
        
        # ===== Per-Person Threat Scores =====
        # Extract interaction features for threat amplification
        inter_feat = frame_features[PERSON_DIM * 2:]
        approach_sig = max(0.0, -inter_feat[1])  # Negative = approaching
        iou_sig = inter_feat[2]  # Bounding box overlap
        pp_sig = min(1.0, self.ema_prob * (1 + approach_sig + iou_sig))
        
        person_scores = []
        
        return {
            'raw_prob': self.raw_prob,
            'smooth_prob': self.ema_prob,
            'status': status,
            'confirmed': self.confirmed_violence,
            'consecutive_alerts': self.consecutive_alerts,
            'effective_threshold': effective_threshold,
            'attention_weights': self.last_attn_weights,
            'inter_features': {
                'distance': float(inter_feat[0]),
                'approach_vel': float(inter_feat[1]),
                'bbox_iou': float(inter_feat[2]),
                'speed_diff': float(inter_feat[3]),
                'heading_align': float(inter_feat[4]),
                'motion_sync': float(inter_feat[5]),
            }
        }
    
    def get_person_threat(self, track_id, frame_feature):
        """
        Compute per-person threat score based on interaction features.
        
        Args:
            track_id: Person's tracking ID
            frame_feature: Full frame feature vector
        
        Returns:
            float: Threat score [0, 1]
        """
        inter_feat = frame_feature[PERSON_DIM * 2:]
        approach_sig = max(0.0, -inter_feat[1])
        iou_sig = inter_feat[2]
        pp_sig = min(1.0, self.ema_prob * (1 + approach_sig + iou_sig))
        
        # Apply EMA smoothing per person
        self.person_ema_scores[track_id] = (
            self.ema_alpha * pp_sig + (1 - self.ema_alpha) * self.person_ema_scores[track_id]
        )
        
        return self.person_ema_scores[track_id]
    
    def get_attention_heatmap(self, timestep_dim=SEQ_LEN):
        """
        Get attention weights over temporal sequence.
        
        Returns:
            np.ndarray: Attention heatmap for visualization
        """
        return self.last_attn_weights
    
    def set_thresholds(self, violence_threshold, warning_threshold):
        """Update detection thresholds (e.g., from calibration)."""
        self.violence_threshold = violence_threshold
        self.warning_threshold = warning_threshold
