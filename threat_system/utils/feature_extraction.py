"""
Core pose-based feature extraction for violence detection.

Extracts person-specific and interaction-based features from skeletal keypoints.
Designed to separate individual motion from interaction patterns.
"""

import numpy as np
from constants import (
    PERSON_DIM, INTERACTION_DIM, FEATURE_DIM, NUM_KEYPOINTS,
    ARM_SEGS, LEG_SEGS, TORSO_SEG, HEAD_SEG, NUM_TOP_PERSONS
)


def _angle_3pt(a, b, c):
    """
    Compute angle at point b (in radians, normalized to [-1, 1]).
    
    Args:
        a, b, c: 2D points as numpy arrays
    
    Returns:
        float: Angle normalized to [-1, 1] via arccos
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.clip(cos_angle, -1, 1))


def _limb_angle_2pt(a, b, W, H):
    """
    Compute limb orientation angle (normalized by frame dimensions).
    
    Args:
        a, b: 2D endpoints of limb
        W, H: Frame width, height
    
    Returns:
        float: Normalized limb angle in [-1, 1]
    """
    dy = (b[1] - a[1]) / (H + 1e-6)
    dx = (b[0] - a[0]) / (W + 1e-6)
    return float(np.arctan2(dy, dx) / np.pi)


def _segment_energy(kps_now, kps_prev, pairs):
    """
    Compute motion energy of a body segment (sum of endpoint displacements).
    
    Args:
        kps_now: Current keypoints (17, 2)
        kps_prev: Previous keypoints (17, 2) or None
        pairs: List of (joint_a, joint_b) indices
    
    Returns:
        float: Normalized motion energy
    """
    if kps_prev is None:
        return 0.0
    
    total_energy = 0.0
    for i, j in pairs:
        if i < len(kps_now) and j < len(kps_now):
            total_energy += (np.linalg.norm(kps_now[i] - kps_prev[i]) + 
                           np.linalg.norm(kps_now[j] - kps_prev[j]))
    
    return total_energy / (2 * len(pairs) + 1e-6)


def extract_person_features(kps, kps_prev, xc, yc, x1, y1, x2, y2,
                            dx, dy, ddx, ddy, dddx, dddy, kp_conf, W, H):
    """
    Extract comprehensive person-specific features from pose keypoints.
    
    Features include:
      - Normalized keypoint positions (34D)
      - Kinematics: velocity, acceleration, jerk (6D)
      - Limb angles: arm, arm-torso (3D)
      - Joint angles: elbows, knees, torso (5D)
      - Motion energy per segment (4D)
      - Bounding box area (1D)
      - Keypoint confidence (1D)
      - Pose speed (1D)
      - Arm raise indicators (1D)
      - Asymmetry features (3D)
      - Padding (1D)
    
    Total: 60D per person
    
    Args:
        kps: Current keypoints (17, 2)
        kps_prev: Previous keypoints (17, 2) or None
        xc, yc: Normalized center position (0-1)
        x1, y1, x2, y2: Bounding box
        dx, dy, ddx, ddy, dddx, dddy: Kinematics
        kp_conf: Keypoint confidence (17,)
        W, H: Frame dimensions
    
    Returns:
        np.ndarray: Feature vector (PERSON_DIM,) in float32
    """
    # 1. Normalized keypoints (34D)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    kp_norm = np.array([
        [(kps[i, 0] - x1) / bw, (kps[i, 1] - y1) / bh]
        for i in range(NUM_KEYPOINTS)
    ], dtype=np.float32).flatten()
    
    # 2. Kinematics: velocity, acceleration, jerk (6D)
    kinematics = np.array([dx, dy, ddx, ddy, dddx, dddy], dtype=np.float32)
    
    # 3. Limb orientations (3D)
    ra = _limb_angle_2pt(kps[6], kps[10], W, H)  # Right arm
    la = _limb_angle_2pt(kps[5], kps[9], W, H)   # Left arm
    hm = (kps[11] + kps[12]) / 2  # Hip midpoint
    sm = (kps[5] + kps[6]) / 2    # Shoulder midpoint
    limb_ang = np.array([
        ra,
        la,
        _limb_angle_2pt(hm, sm, W, H)  # Torso angle
    ], dtype=np.float32)
    
    # 4. Joint angles (5D)
    joint_ang = np.array([
        _angle_3pt(kps[6], kps[8], kps[10]),      # Right elbow
        _angle_3pt(kps[5], kps[7], kps[9]),       # Left elbow
        _angle_3pt(kps[12], kps[14], kps[16]),    # Right knee
        _angle_3pt(kps[11], kps[13], kps[15]),    # Left knee
        _angle_3pt(kps[5], kps[11], kps[13]),     # Torso-leg angle
    ], dtype=np.float32)
    
    # 5. Motion energy per segment (4D)
    motion_e = np.array([
        _segment_energy(kps, kps_prev, ARM_SEGS),
        _segment_energy(kps, kps_prev, LEG_SEGS),
        _segment_energy(kps, kps_prev, TORSO_SEG),
        _segment_energy(kps, kps_prev, HEAD_SEG),
    ], dtype=np.float32)
    
    # 6. Bounding box area (1D)
    bbox_area = np.array([(bw * bh) / (W * H)], dtype=np.float32)
    
    # 7. Keypoint confidence (1D)
    conf_mean = np.array([float(np.mean(kp_conf))], dtype=np.float32)
    
    # 8. Pose speed (1D)
    if kps_prev is not None:
        pose_speed = float(np.mean(np.linalg.norm(kps - kps_prev, axis=1))) / (W + 1e-6)
    else:
        pose_speed = 0.0
    pose_spd = np.array([pose_speed], dtype=np.float32)
    
    # 9. Arm raise (1D)
    r_raise = max(0.0, float((kps[6, 1] - kps[10, 1]) / bh))
    l_raise = max(0.0, float((kps[5, 1] - kps[9, 1]) / bh))
    arm_raise = np.array([max(r_raise, l_raise)], dtype=np.float32)
    
    # 10. Asymmetry features (3D)
    asym = np.array([
        abs(dx) - abs(dy),
        abs(ra - la),
        abs(r_raise - l_raise)
    ], dtype=np.float32)
    
    # 11. Padding (1D)
    pad = np.zeros(1, dtype=np.float32)
    
    # Concatenate all features
    person_features = np.concatenate([
        kp_norm,      # 34D
        kinematics,   # 6D
        limb_ang,     # 3D
        joint_ang,    # 5D
        motion_e,     # 4D
        bbox_area,    # 1D
        conf_mean,    # 1D
        pose_spd,     # 1D
        arm_raise,    # 1D
        asym,         # 3D
        pad           # 1D
    ])
    
    assert person_features.shape == (PERSON_DIM,), f"Expected {PERSON_DIM}D, got {person_features.shape}"
    return person_features


def compute_interaction(centers, velocities, bboxes, kps_list):
    """
    Compute interaction features between two people.
    
    Features:
      - Distance between centers
      - Approach velocity (negative = approaching)
      - Bounding box IoU
      - Speed difference
      - Heading alignment (negative = facing each other)
      - Motion synchronization
    
    Args:
        centers: List of 2 (xc, yc) tuples (normalized 0-1)
        velocities: List of 2 (vx, vy) tuples (normalized)
        bboxes: List of 2 bboxes [[x1,y1,x2,y2], ...]
        kps_list: List of 2 keypoint arrays (17, 2)
    
    Returns:
        np.ndarray: Interaction feature vector (6D,)
    """
    if len(centers) < 2 or len(velocities) < 2:
        return np.zeros(INTERACTION_DIM, dtype=np.float32)
    
    (x1c, y1c), (x2c, y2c) = centers[0], centers[1]
    
    # 1. Relative position and distance
    rel_pos = np.array([x1c - x2c, y1c - y2c])
    distance = float(np.linalg.norm(rel_pos))
    
    # 2. Approach velocity (dot product with relative position)
    rel_vel = np.array(velocities[0]) - np.array(velocities[1])
    approach_vel = float(np.dot(rel_pos, rel_vel) / (distance + 1e-6))
    
    # 3. Bounding box IoU
    b1, b2 = bboxes[0], bboxes[1]
    inter_x = max(0., min(b1[2], b2[2]) - max(b1[0], b2[0]))
    inter_y = max(0., min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = inter_x * inter_y
    
    union = ((b1[2] - b1[0]) * (b1[3] - b1[1]) + 
             (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter + 1e-6)
    iou = inter / union
    
    # 4. Speed difference
    s1 = np.linalg.norm(velocities[0])
    s2 = np.linalg.norm(velocities[1])
    speed_diff = abs(s1 - s2)
    
    # 5. Heading alignment (negative = facing each other, positive = moving same direction)
    v1_norm = np.array(velocities[0]) / (s1 + 1e-6)
    v2_norm = np.array(velocities[1]) / (s2 + 1e-6)
    heading_align = float(-np.dot(v1_norm, v2_norm))  # Negative to detect confrontation
    
    # 6. Motion synchronization
    motion_sync = 1.0 - (abs(s1 - s2) / (s1 + s2 + 1e-6))
    
    return np.array([
        distance,
        approach_vel,
        iou,
        speed_diff,
        heading_align,
        motion_sync
    ], dtype=np.float32)


def extract_frame_features(frame, result, dets, tracker, 
                          prev_centers, prev_vel, prev_acc, prev_kps):
    """
    Extract full frame features (both person and interaction features).
    
    Args:
        frame: Input frame (H, W, 3)
        result: YOLOv8 pose inference result
        dets: supervision.Detections object
        tracker: ByteTrack tracker
        prev_centers, prev_vel, prev_acc, prev_kps: Temporal tracking state
    
    Returns:
        Tuple of:
            - frame_feature: np.ndarray (FEATURE_DIM,) - composite features
            - det_info: List of dict with detection info (bbox, tid)
    """
    H, W = frame.shape[:2]
    
    # No detection fallback
    if len(dets) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32), []
    
    # Filter for person class (YOLO class 0)
    dets = dets[dets.class_id == 0]
    if len(dets) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32), []
    
    # Update tracker
    dets = tracker.update_with_detections(dets)
    
    # Use the two largest people for the fixed-size violence feature vector,
    # while returning every tracked person to the other detectors.
    areas = [(dets.xyxy[i, 2] - dets.xyxy[i, 0]) * (dets.xyxy[i, 3] - dets.xyxy[i, 1])
             for i in range(len(dets))]
    feature_indices = list(np.argsort(-np.array(areas))[:NUM_TOP_PERSONS])
    feature_index_set = set(feature_indices)
    indices = feature_indices + [idx for idx in range(len(dets)) if idx not in feature_index_set]
    
    person_feats = []
    centers = []
    velocities = []
    bboxes = []
    kps_list = []
    det_info = []
    
    for idx in indices:
        x1, y1, x2, y2 = dets.xyxy[idx]
        tid = int(dets.tracker_id[idx])
        
        # Normalized center
        xc = (x1 + x2) / 2 / W
        yc = (y1 + y2) / 2 / H
        det_info.append({'bbox': [int(x1), int(y1), int(x2), int(y2)], 'tid': tid})
        
        # Extract keypoints
        try:
            kps = result.keypoints.xy[idx].cpu().numpy().astype(np.float32)
            kp_conf = (result.keypoints.conf[idx].cpu().numpy() 
                      if result.keypoints.conf is not None 
                      else np.ones(NUM_KEYPOINTS, dtype=np.float32))
        except:
            kps = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            kp_conf = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        
        # Compute kinematics
        dx = dy = ddx = ddy = dddx = dddy = 0.0
        
        if tid in prev_centers:
            px, py = prev_centers[tid]
            dx, dy = xc - px, yc - py
        
        if tid in prev_vel:
            pdx, pdy = prev_vel[tid]
            ddx, ddy = dx - pdx, dy - pdy
        
        if tid in prev_acc:
            pddx, pddy = prev_acc[tid]
            dddx, dddy = ddx - pddx, ddy - pddy
        
        prev_centers[tid] = (xc, yc)
        prev_vel[tid] = (dx, dy)
        prev_acc[tid] = (ddx, ddy)
        
        # Extract person features
        pf = extract_person_features(
            kps, prev_kps.get(tid),
            xc, yc, x1, y1, x2, y2,
            dx, dy, ddx, ddy, dddx, dddy,
            kp_conf, W, H
        )
        
        prev_kps[tid] = kps.copy()
        if idx in feature_index_set:
            centers.append((xc, yc))
            bboxes.append([x1, y1, x2, y2])
            velocities.append((dx, dy))
            person_feats.append(pf)
            kps_list.append(kps)
    
    # Pad if less than 2 persons
    while len(person_feats) < 2:
        person_feats.append(np.zeros(PERSON_DIM, dtype=np.float32))
    while len(centers) < 2:
        centers.append((0.5, 0.5))
    while len(velocities) < 2:
        velocities.append((0.0, 0.0))
    while len(bboxes) < 2:
        bboxes.append([0.0, 0.0, 1.0, 1.0])
    while len(kps_list) < 2:
        kps_list.append(np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32))
    
    # Compute interaction
    inter = compute_interaction(centers, velocities, bboxes, kps_list)
    
    # Combine features
    frame_features = np.concatenate([
        person_feats[0],
        person_feats[1],
        inter
    ], dtype=np.float32)
    
    assert frame_features.shape == (FEATURE_DIM,), f"Expected {FEATURE_DIM}D, got {frame_features.shape}"
    
    return frame_features, det_info
