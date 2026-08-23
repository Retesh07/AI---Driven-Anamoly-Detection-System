"""
Weapon detection module - context-aware integration with violence detection.

Philosophy:
  - Use YOLO for per-frame detection
  - Associate weapons with tracked persons
  - Apply temporal smoothing to reduce flickering
  - FOLLOW violence module (secondary analysis)
"""

import numpy as np
import torch
from collections import defaultdict, deque
from constants import (
    WEAPON_CONFIDENCE_THRESHOLD, WEAPON_TEMPORAL_BUFFER, 
    WEAPON_CONFIDENCE_BOOST, WEAPON_EMA_DECISION_THRESHOLD,
    GUN_CONFIDENCE_THRESHOLD, GUN_SIZE_MIN_WIDTH, GUN_SIZE_MAX_WIDTH, GUN_SIZE_MIN_HEIGHT, GUN_SIZE_MAX_HEIGHT,
    KNIFE_CONFIDENCE_THRESHOLD, KNIFE_SIZE_MIN_WIDTH, KNIFE_SIZE_MAX_WIDTH, KNIFE_SIZE_MIN_HEIGHT, KNIFE_SIZE_MAX_HEIGHT,
    WEAPON_SPATIAL_THRESHOLD, WEAPON_CLASS_CONSISTENCY_FRAMES, 
    WEAPON_HISTORY_AGREEMENT_THRESHOLD
)


class WeaponDetector:
    """
    YOLO-based weapon detector with temporal smoothing - Production Ready.
    
    Designed for real-world surveillance deployment (NOT overfitted to test videos).
    
    Features:
      - Weapon-specific thresholds (different for guns vs knives)
      - No temporal assumptions (weapons can appear at any frame)
      - Robust size filtering (rejects arms and background)
      - EMA smoothing for temporal stability
      - Pose-based hand filtering to reduce false positives during violence
      - Class consistency checks (gun stays gun, not flip-flopping)
    
    Responsibilities:
      - Run YOLO inference on each frame
      - Associate detections with tracked persons
      - Apply weapon-specific filtering
      - Output reliable threat assessment
    """
    
    def __init__(self, model_path, device='cuda', confidence_threshold=WEAPON_CONFIDENCE_THRESHOLD):
        """
        Args:
            model_path: Path to YOLO weapon detection model (.pt file)
            device: 'cuda' or 'cpu'
            confidence_threshold: Detection confidence threshold
        """
        from ultralytics import YOLO
        
        self.model = YOLO(model_path)
        self.device = 0 if device == 'cuda' else 'cpu'
        self.confidence_threshold = confidence_threshold
        
        # Temporal state
        self.weapon_history = defaultdict(lambda: deque(maxlen=5))  # Last 5 frames per person
        self.person_weapon_scores = defaultdict(float)  # EMA scores
        self.weapon_active_frames = defaultdict(int)  # Frames since last detection
        
        # WEAPON PERSISTENCE TRACKING
        # Once weapon is detected, track person until they leave
        self.weapon_detection_frame = defaultdict(int)  # Frame where weapon first detected per person
        self.weapon_persistence_count = defaultdict(int)  # How many frames weapon persisted
        self.has_ever_had_weapon = defaultdict(bool)  # Track if person ever had weapon
    
    def detect(self, frame):
        """
        Run weapon detection on frame.
        
        Args:
            frame: Input frame (H, W, 3)
        
        Returns:
            List of detections:
                [{
                    'bbox': [x1, y1, x2, y2],
                    'class': 'gun'|'knife'|'etc',
                    'confidence': float,
                    'xyxy': [x1, y1, x2, y2]
                }, ...]
        """
        results = self.model(frame, device=self.device, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = r.names[cls_id] if cls_id in r.names else f'class_{cls_id}'
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'xyxy': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': conf,
                    'class_id': cls_id
                })
        
        return detections
    
    def associate_with_persons(self, weapons, person_bboxes):
        """
        Associate weapon detections with tracked persons.
        
        Uses bounding box IoU to match weapons to nearest person.
        
        Args:
            weapons: List of weapon detections
            person_bboxes: Dict of {track_id: [x1, y1, x2, y2]}
        
        Returns:
            Dict of {track_id: [weapon_detection, ...]}
        """
        person_weapons = defaultdict(list)
        
        if not weapons or not person_bboxes:
            return person_weapons
        
        for weapon in weapons:
            wx1, wy1, wx2, wy2 = weapon['xyxy']
            best_tid = None
            best_iou = 0
            
            # Find person with highest IoU
            for tid, (px1, py1, px2, py2) in person_bboxes.items():
                inter_x = max(0, min(wx2, px2) - max(wx1, px1))
                inter_y = max(0, min(wy2, py2) - max(wy1, py1))
                inter = inter_x * inter_y
                
                p_area = (px2 - px1) * (py2 - py1)
                w_area = (wx2 - wx1) * (wy2 - wy1)
                union = p_area + w_area - inter
                
                if union > 0:
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid
            
            # Also match if weapon center is inside person bbox
            if best_tid is None:
                wx_center = (wx1 + wx2) / 2
                wy_center = (wy1 + wy2) / 2
                
                for tid, (px1, py1, px2, py2) in person_bboxes.items():
                    if px1 <= wx_center <= px2 and py1 <= wy_center <= py2:
                        best_tid = tid
                        break
            
            if best_tid is not None:
                person_weapons[best_tid].append(weapon)
        
        return person_weapons
    
    def update(self, frame, person_bboxes, ema_alpha=0.3, pose_result=None, det_info=None):
        """
        Update weapon detection state with temporal persistence.
        
        WEAPON PERSISTENCE: Once a weapon is detected on a person, we continue tracking
        that person and marking them as potentially armed until they leave the frame.
        
        Args:
            frame: Input frame
            person_bboxes: Dict of {track_id: [x1, y1, x2, y2]}
            ema_alpha: Smoothing factor
            pose_result: YOLO pose detection results (optional, for filtering hands)
            det_info: Detection info list with keypoints (optional, for filtering hands)            violence_prob: Violence probability (optional, stricter filtering during violence)        
        Returns:
            Dict of {track_id: weapon_info}
        """
        # Detect weapons
        weapons = self.detect(frame)
        
        # Filter out detections overlapping with hands/arms (common false positive source)
        if pose_result is not None and det_info is not None:
            weapons = self._filter_hand_overlaps(weapons, pose_result, det_info)
        
        # Associate with persons
        person_weapons = self.associate_with_persons(weapons, person_bboxes)
        
        # Update per-person scores
        person_results = {}
        
        for tid in person_bboxes.keys():
            detected_weapons = person_weapons.get(tid, [])
            
            # Apply spatial and class consistency filters
            filtered_weapons = self._filter_by_spatial_consistency(tid, detected_weapons)
            
            # Max confidence among filtered weapons
            max_conf = max([w['confidence'] for w in filtered_weapons]) if filtered_weapons else 0.0
            
            # Determine weapon type with consistency check FIRST (before persistence)
            weapon_type = 'unknown'
            if filtered_weapons:
                weapon_type = max(filtered_weapons, key=lambda w: w['confidence'])['class']
                
                # Verify class consistency in history
                if not self._check_class_consistency(tid, weapon_type):
                    weapon_type = 'unknown'  # Inconsistent class = likely false positive
            
            # ===== WEAPON PERSISTENCE TRACKING =====
            # CRITICAL FIX: Only persist for REAL weapons (gun/knife)
            # Unknown/unconfident detections should NOT trigger persistence
            # This prevents phones, bags, and other false positives from being tracked
            is_real_weapon = max_conf > 0 and weapon_type in ['gun', 'knife']
            
            if is_real_weapon:
                self.weapon_detection_frame[tid] = len(self.weapon_history[tid])  # Record frame
                self.has_ever_had_weapon[tid] = True  # Person is now armed
                self.weapon_persistence_count[tid] += 1
                self.weapon_active_frames[tid] = WEAPON_TEMPORAL_BUFFER
            else:
                # Decay buffer - ALWAYS decay for unknown or non-weapons
                self.weapon_active_frames[tid] = max(0, self.weapon_active_frames[tid] - 1)
            
            # Apply confidence boost ONLY for real weapons with active persistence
            if self.weapon_active_frames[tid] > 0 and (weapon_type in ['gun', 'knife']):
                max_conf = max(max_conf, WEAPON_CONFIDENCE_BOOST)
            
            # EMA smoothing with STRICTER filtering for unknown/false positives
            # Use lower alpha for unknown weapons to reduce noise
            actual_ema_alpha = ema_alpha if weapon_type in ['gun', 'knife'] else (ema_alpha * 0.5)
            self.person_weapon_scores[tid] = (
                actual_ema_alpha * max_conf + (1 - actual_ema_alpha) * self.person_weapon_scores[tid]
            )
            
            # History tracking
            self.weapon_history[tid].append({
                'detected': len(filtered_weapons) > 0,
                'confidence': max_conf,
                'weapons': filtered_weapons,
                'spatial_filtered': len(detected_weapons) > len(filtered_weapons)
            })
            
            # Check history agreement - require multiple frames to agree
            history_agreement = self._check_history_agreement(tid)
            
            # Decision: weapon present - STRICT to reject unknown/false positives
            # Only count as weapon_present if it's a REAL gun/knife (not unknown)
            # Unknown detections require HIGHER EMA threshold to be believed
            
            if weapon_type == 'gun':
                # Gun: trust if smooth score > threshold
                weapon_present = self.person_weapon_scores[tid] > WEAPON_EMA_DECISION_THRESHOLD
            elif weapon_type == 'knife':
                # Knife: trust if smooth score > threshold
                weapon_present = self.person_weapon_scores[tid] > WEAPON_EMA_DECISION_THRESHOLD
            else:
                # Unknown: NEVER report as weapon (strict filter)
                # Unknown detections are likely false positives (phones, hands, etc.)
                weapon_present = False
            
            # Apply persistence ONLY to real weapons
            if self.weapon_active_frames[tid] > 0:
                if weapon_type in ['gun', 'knife']:
                    weapon_present = True  # Keep reporting real weapons during persistence
            
            person_results[tid] = {
                'detected': len(detected_weapons) > 0,
                'frame_detections': detected_weapons,
                'smooth_score': self.person_weapon_scores[tid],
                'weapon_present': weapon_present,
                'weapon_type': weapon_type,  # 'gun' or 'knife'
                'history_agreement': self._check_history_agreement(tid),
                'has_ever_had_weapon': self.has_ever_had_weapon[tid],  # For tracking
                'weapon_detection_frame': self.weapon_detection_frame.get(tid, -1),  # When weapon first detected
                'weapon_persistence_frames': self.weapon_persistence_count[tid]  # How long armed
            }
        
        return person_results
    
    def _filter_hand_overlaps(self, detections, pose_result, det_info):
        """
        Filter detections that overlap with raised hand keypoints.
        Only filters when hands are ABOVE shoulder level (actual raising during violence).
        This prevents false positives from misclassifying raised hands as weapons.
        
        Args:
            detections: List of weapon detections
            pose_result: YOLO pose detection results
            det_info: Detection info with track IDs
        
        Returns:
            Filtered detections (excluding hand overlaps)
        """
        if not detections or pose_result is None or pose_result.keypoints is None:
            return detections
        
        filtered = []
        for det in detections:
            det_bbox = det['xyxy']  # [x1, y1, x2, y2]
            is_hand_overlap = False
            
            # Check overlap with all person poses
            for i, kps in enumerate(pose_result.keypoints.xy):
                if len(kps) < 10:  # Not enough keypoints
                    continue
                
                # Only check wrist keypoints (9=L_wrist, 10=R_wrist), not elbows
                wrist_kps = [kps[9], kps[10]]  # Left and right wrists
                
                # Get shoulder position to check if hands are "raised"
                # Shoulder keypoints: 5=L_shoulder, 6=R_shoulder
                shoulder_y = min([kps[5][1], kps[6][1]]) if len(kps) > 6 else 999999
                
                for kp in wrist_kps:
                    if kp[0] > 0 and kp[1] > 0 and kp[1] < shoulder_y:  # Valid AND raised above shoulder
                        # Create smaller bounding box around raised wrist
                        margin = 10
                        kp_x1, kp_y1 = int(kp[0] - margin), int(kp[1] - margin)
                        kp_x2, kp_y2 = int(kp[0] + margin), int(kp[1] + margin)
                        
                        # Check if detection overlaps with hand region
                        if self._bbox_overlap(det_bbox, [kp_x1, kp_y1, kp_x2, kp_y2]):
                            is_hand_overlap = True
                            break
                
                if is_hand_overlap:
                    break
            
            if not is_hand_overlap:
                filtered.append(det)
        
        return filtered
    
    def _bbox_overlap(self, bbox1, bbox2):
        """Check if two bboxes overlap. Args: bbox = [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _filter_by_spatial_consistency(self, track_id, detected_weapons):
        """
        Filter detections based on size constraints and confidence.
        Real-time robustness for production: weapon-specific thresholds.
        
        Uses different size/confidence constraints for guns vs knives:
        - Guns: larger, usually 40-400px wide
        - Knives: smaller, typically 15-150px wide
        
        Args:
            track_id: Person track ID
            detected_weapons: List of detected weapons in current frame
        
        Returns:
            Filtered list of realistic weapons (production-ready)
        """
        if not detected_weapons:
            return []
        
        filtered = []
        for w in detected_weapons:
            conf = w['confidence']
            weapon_class = w.get('class', 'unknown')
            
            # Size and confidence constraints vary by weapon type
            x1, y1, x2, y2 = w['xyxy']
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / (height + 1e-6)
            
            # ===== FALSE POSITIVE REJECTION FILTER =====
            # REJECT PHONE-LIKE OBJECTS: phones are tall with aspect < 0.8
            # Guns are wider (~1.5-3.0), knives are long/thin (0.2-1.0)
            # Low-confidence + tall = almost always phone/false positive
            if aspect_ratio < 0.6 and conf < 0.35:
                # Very likely a phone, reject immediately
                continue
            
            if weapon_class == 'gun':
                # Gun-specific constraints
                min_w, max_w = GUN_SIZE_MIN_WIDTH, GUN_SIZE_MAX_WIDTH
                min_h, max_h = GUN_SIZE_MIN_HEIGHT, GUN_SIZE_MAX_HEIGHT
                min_conf = GUN_CONFIDENCE_THRESHOLD
                
                # Guns should have aspect ratio 0.8-4.0 (wider-ish or square)
                if aspect_ratio < 0.8 or aspect_ratio > 4.0:
                    # Weird aspect ratio for gun
                    continue
                    
            elif weapon_class == 'knife':
                # Knife-specific constraints (smaller, thin/long)
                min_w, max_w = KNIFE_SIZE_MIN_WIDTH, KNIFE_SIZE_MAX_WIDTH
                min_h, max_h = KNIFE_SIZE_MIN_HEIGHT, KNIFE_SIZE_MAX_HEIGHT
                min_conf = KNIFE_CONFIDENCE_THRESHOLD
                
                # Knives: aspect ratio 0.15-1.5 (long/thin to balanced)
                if aspect_ratio > 1.5:
                    # Too wide for knife
                    continue
                    
            else:
                # Unknown weapon type - VERY STRICT for unknowns to prevent false positives
                # Unknown detections need high confidence AND reasonable proportions
                min_w, max_w = 30, 300  # Tighter range than all weapons
                min_h, max_h = 20, 250
                min_conf = 0.40  # Higher threshold for unknowns (was 0.25)
                
                # Unknown: Reject tall/thin objects (phones) or extreme aspect ratios
                if aspect_ratio < 0.5 or aspect_ratio > 3.5:
                    continue
            
            # Apply weapon-specific size and confidence filters
            if width >= min_w and width <= max_w and height >= min_h and height <= max_h and conf >= min_conf:
                filtered.append(w)
        
        return filtered
    
    def _check_class_consistency(self, track_id, current_class):
        """
        Check if weapon class is consistent across recent frames.
        Rapid switching between gun/knife indicates false positives.
        
        Args:
            track_id: Person track ID
            current_class: Current detected weapon class
        
        Returns:
            True if class is consistent, False if unreliable
        """
        history = self.weapon_history.get(track_id, [])
        if len(history) < WEAPON_CLASS_CONSISTENCY_FRAMES:
            return True  # Not enough history
        
        # Check recent frames
        recent = list(history)[-WEAPON_CLASS_CONSISTENCY_FRAMES:]
        classes = []
        
        for entry in recent:
            if entry.get('weapons'):
                class_name = entry['weapons'][0].get('class', 'unknown')
                classes.append(class_name)
        
        if not classes:
            return True
        
        # Check if current class matches majority
        matching = sum(1 for c in classes if c == current_class)
        consistency = matching / len(classes) if classes else 0
        
        return consistency >= WEAPON_HISTORY_AGREEMENT_THRESHOLD
    
    def _check_history_agreement(self, track_id):
        """Check if recent frames agree on weapon presence."""
        history = self.weapon_history.get(track_id, [])
        if len(history) < 3:
            return False
        
        detected_count = sum(1 for h in list(history)[-3:] if h['detected'])
        return detected_count >= 2  # At least 2 out of last 3 frames
    
    def reset(self):
        """Clear state for new video."""
        self.weapon_history.clear()
        self.person_weapon_scores.clear()
        self.weapon_active_frames.clear()
