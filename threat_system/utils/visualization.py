"""
Visualization utilities for threat detection output.
"""

import cv2
import numpy as np
from constants import SKELETON_PAIRS, SKELETON_COLORS, VIOLENCE_COLOR, WARNING_COLOR, NORMAL_COLOR


def draw_skeleton(frame, kps, color=(255, 255, 255), thickness=2):
    """
    Draw pose skeleton on frame.
    
    Args:
        frame: Input frame (H, W, 3)
        kps: Keypoints array (17, 2)
        color: RGB color tuple (B, G, R in OpenCV)
        thickness: Line thickness
    """
    # Draw joint circles
    for i in range(len(kps)):
        x, y = int(kps[i, 0]), int(kps[i, 1])
        if kps[i, 0] > 0 and kps[i, 1] > 0:
            cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
    
    # Draw skeleton connections
    for a, b in SKELETON_PAIRS:
        if a < len(kps) and b < len(kps):
            if kps[a, 0] > 0 and kps[a, 1] > 0 and kps[b, 0] > 0 and kps[b, 1] > 0:
                pt1 = (int(kps[a, 0]), int(kps[a, 1]))
                pt2 = (int(kps[b, 0]), int(kps[b, 1]))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_attention_bar(frame, attn_weights, bar_height=18):
    """
    Draw attention weights as temporal bar at bottom of frame.
    
    Args:
        frame: Input frame (H, W, 3)
        attn_weights: Attention tensor (can be 1D, 2D, or 3D)
        bar_height: Height of attention bar
    """
    if attn_weights is None:
        return
    
    H, W = frame.shape[:2]
    aw = np.array(attn_weights)
    
    # Flatten attention weights
    if aw.ndim == 3:
        wts = aw.mean(axis=(0, 2))
    elif aw.ndim == 2:
        wts = aw.mean(axis=0)
    elif aw.ndim == 1:
        wts = aw
    else:
        return
    
    wts = wts.flatten()
    if len(wts) == 0:
        return
    
    # Normalize to [0, 1]
    wts_min, wts_max = wts.min(), wts.max()
    if wts_max - wts_min > 1e-6:
        wts = (wts - wts_min) / (wts_max - wts_min)
    else:
        wts = np.ones_like(wts) * 0.5
    
    # Draw bar
    step = max(1, W // len(wts))
    bar_y = H - bar_height - 2
    
    # Background
    cv2.rectangle(frame, (0, bar_y - 2), (W, H), (20, 20, 20), -1)
    
    # Attention bars
    for t, wt in enumerate(wts):
        x1 = t * step
        x2 = min(x1 + step, W)
        # Color gradient: blue (low) to orange/red (high)
        color_intensity = int(wt * 220)
        color = (int((1 - wt) * 200), 40, color_intensity)
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_height), color, -1)
    
    # Label
    cv2.putText(frame, 'Attn', (4, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def draw_hud(frame, raw_prob, ema_prob, status, person_scores, fps_val, frame_no):
    """
    Draw heads-up display with threat information.
    
    Args:
        frame: Input frame (H, W, 3)
        raw_prob: Raw violence probability [0, 1]
        ema_prob: EMA-smoothed probability [0, 1]
        status: 'VIOLENCE', 'WARNING', or 'NORMAL'
        person_scores: List of (track_id, threat_score) tuples
        fps_val: Frames per second
        frame_no: Current frame number
    """
    H, W = frame.shape[:2]
    
    # Semi-transparent overlay box
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (300, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    
    # Status text with color
    status_color = (VIOLENCE_COLOR if status == 'VIOLENCE' 
                   else WARNING_COLOR if status == 'WARNING' 
                   else NORMAL_COLOR)
    cv2.putText(frame, status, (16, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2, cv2.LINE_AA)
    
    # Probability bar
    bar_len = 240
    bar_x, bar_y = 16, 48
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 14), (60, 60, 60), -1)
    
    bar_color = (VIOLENCE_COLOR if ema_prob > 0.6 
                else WARNING_COLOR if ema_prob > 0.4 
                else NORMAL_COLOR)
    bar_width = int(ema_prob * bar_len)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 14), bar_color, -1)
    cv2.putText(frame, f'{ema_prob*100:.1f}%', (bar_x + bar_len + 6, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    
    # Info line
    cv2.putText(frame, f'Raw:{raw_prob*100:.1f}%  FPS:{fps_val:.1f}  F:{frame_no}',
                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1)
    
    # Per-person threat scores
    for pi, (tid, score) in enumerate(person_scores[:2]):
        py = 96 + pi * 16
        cv2.putText(frame, f'P{pi+1}#{tid}', (16, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1)
        
        plen = 140
        px = 60
        cv2.rectangle(frame, (px, py - 10), (px + plen, py - 2), (50, 50, 50), -1)
        
        p_color = (VIOLENCE_COLOR if score > 0.6 
                  else WARNING_COLOR if score > 0.35 
                  else NORMAL_COLOR)
        cv2.rectangle(frame, (px, py - 10), (px + int(score * plen), py - 2), p_color, -1)
        cv2.putText(frame, f'{score*100:.0f}%', (px + plen + 4, py - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)


def draw_detections(frame, det_info, result, status, frame_idx):
    """
    Draw bounding boxes and skeletons for detected persons.
    
    Args:
        frame: Input frame (H, W, 3)
        det_info: List of detection info dicts
        result: YOLOv8 result object with keypoints
        status: Current threat status
        frame_idx: Index in detection result
    """
    status_color = (VIOLENCE_COLOR if status == 'VIOLENCE' 
                   else WARNING_COLOR if status == 'WARNING' 
                   else NORMAL_COLOR)
    
    for pi, info in enumerate(det_info[:2]):
        x1, y1, x2, y2 = info['bbox']
        tid = info['tid']
        skel_color = SKELETON_COLORS[pi % len(SKELETON_COLORS)]
        
        # Bounding box color based on status
        box_color = status_color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        
        # Track ID label
        cv2.putText(frame, f'P{pi+1}#{tid}', (x1, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, skel_color, 1, cv2.LINE_AA)
        
        # Draw skeleton
        try:
            kps = result.keypoints.xy[pi].cpu().numpy()
            draw_skeleton(frame, kps, skel_color)
        except:
            pass


def create_timeline_visualization(frames, raw_probs, smooth_probs, timeline_data, 
                                 violence_threshold, warning_threshold,
                                 output_path):
    """
    Create timeline visualization showing violence probability over time.
    
    Args:
        frames: Frame numbers
        raw_probs: Raw probabilities
        smooth_probs: EMA-smoothed probabilities
        timeline_data: Full timeline data with status
        violence_threshold: Threshold for violence
        warning_threshold: Threshold for warning
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Confirmed violence region
    confirmed = [t['violence']['confirmed'] for t in timeline_data]
    ax.fill_between(frames, 0, 1, where=confirmed, alpha=0.20, color='red', label='VIOLENCE (confirmed)')
    
    # Warning region
    warning_mask = [warning_threshold <= s < violence_threshold for s in smooth_probs]
    ax.fill_between(frames, 0, 1, where=warning_mask, alpha=0.12, color='orange', label='WARNING')
    
    # Probability curves
    ax.plot(frames, raw_probs, color='#aaa', lw=0.7, alpha=0.6, label='raw')
    ax.plot(frames, smooth_probs, color='#d04020', lw=1.8, label='EMA smooth')
    
    # Threshold lines
    ax.axhline(violence_threshold, color='red', ls='--', lw=0.9, alpha=0.6)
    ax.axhline(warning_threshold, color='orange', ls='--', lw=0.9, alpha=0.6)
    
    ax.set_xlim(frames[0], frames[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Violence probability')
    ax.set_title('Temporal Violence Probability')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()


def get_threat_color(violence_score, weapon_present, loitering_detected):
    """
    Determine color based on threat composition.
    
    Args:
        violence_score: Violence probability [0, 1]
        weapon_present: Boolean
        loitering_detected: Boolean
    
    Returns:
        BGR color tuple
    """
    if violence_score > 0.6:
        if weapon_present:
            return (0, 0, 220)  # Bright red: Violence + Weapon (highest threat)
        else:
            return (0, 100, 220)  # Orange-red: Violence only
    elif weapon_present:
        return (0, 165, 240)  # Yellow: Weapon without violence
    elif loitering_detected:
        return (255, 0, 255)  # Magenta: Loitering
    else:
        return (0, 200, 0)  # Green: Safe


def draw_enhanced_detections(frame, det_info, result, fused_results, timeline_entry):
    """
    Draw enhanced detections with per-person threat visualization.
    
    Args:
        frame: Input frame (H, W, 3)
        det_info: List of detection info dicts with track IDs
        result: YOLOv8 result object with keypoints
        fused_results: Dict of {track_id: fused threat data}
        timeline_entry: Current frame's timeline data with violence scores
    """
    for pi, info in enumerate(det_info):
        x1, y1, x2, y2 = info['bbox']
        tid = info['tid']
        
        # Get fused threat data for this person
        if tid not in fused_results:
            continue
            
        threat_data = fused_results[tid]
        violence_score = threat_data['violence_score']
        weapon_present = threat_data['weapon_present']
        loitering_detected = threat_data['loitering_detected']
        threat_level = threat_data['threat_level']
        fused_score = threat_data['fused_score']
        identity_name = threat_data.get('identity_name', 'unknown')
        identity_confidence = threat_data.get('identity_confidence', 0.0)
        is_known_family = threat_data.get('is_known_family', False)
        
        # Determine color based on threat type
        box_color = get_threat_color(violence_score, weapon_present, loitering_detected)
        skel_color = SKELETON_COLORS[pi % len(SKELETON_COLORS)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3, cv2.LINE_AA)
        
        # Draw LARGER threat level label with better contrast
        identity_label = identity_name if is_known_family else 'Unknown'
        label_text = f"P#{tid} [{threat_data['threat_level']}] {identity_label}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
        label_y = max(y1 - 80, label_size[1] + 8)
        # Draw background with white border for contrast
        cv2.rectangle(frame, (x1 - 2, label_y - label_size[1] - 6), 
                     (x1 + label_size[0] + 6, label_y + 4), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1 - 2, label_y - label_size[1] - 6), 
                     (x1 + label_size[0] + 6, label_y + 4), box_color, 2)
        cv2.putText(frame, label_text, (x1 + 2, label_y - 3),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        identity_text = f"ID:{identity_name} {identity_confidence*100:.0f}%"
        identity_color = (0, 255, 0) if is_known_family else (180, 180, 180)
        cv2.putText(frame, identity_text, (x1 + 2, label_y + 18),
               cv2.FONT_HERSHEY_DUPLEX, 0.45, identity_color, 1, cv2.LINE_AA)
        
        # Draw all THREE anomalies EQUALLY prominent (balanced system - not violence-centric)
        indicator_x = x2 + 8
        indicator_y = y1 + 10
        font_size = 0.55
        
        # VIOLENCE Score - Full Size (EQUAL to others)
        violence_color = (0, 165, 240)  # Orange
        cv2.circle(frame, (indicator_x - 5, indicator_y), 6, violence_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"V:{violence_score:.2f}", (indicator_x + 8, indicator_y + 4),
                   cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
        
        # WEAPON Score - Full Size (EQUAL to others)
        indicator_y += 22
        weapon_color = (0, 0, 220) if threat_data.get('is_gun') else \
                       (0, 200, 255) if weapon_present else (100, 100, 100)
        circle_radius = 8 if weapon_present else 6
        cv2.circle(frame, (indicator_x - 5, indicator_y), circle_radius, weapon_color, -1, cv2.LINE_AA)
        weapon_label = f"WEAPON:{threat_data.get('weapon_score', 0):.2f}"
        if threat_data.get('is_gun'):
            weapon_label = f"🔫 GUN:{threat_data.get('weapon_score', 0):.2f}"
        cv2.putText(frame, weapon_label, (indicator_x + 8, indicator_y + 4),
                   cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
        
        # LOITERING Score - Full Size (EQUAL to others)
        indicator_y += 22
        loitering_color = (255, 0, 255) if loitering_detected else (100, 100, 100)
        circle_radius = 8 if loitering_detected else 6
        cv2.circle(frame, (indicator_x - 5, indicator_y), circle_radius, loitering_color, -1, cv2.LINE_AA)
        loitering_score = threat_data.get('loitering_score', 0)
        loitering_label = f"L:{loitering_score:.2f}"
        if threat_data.get('suppress_loitering', False):
            loitering_label += " (suppressed)"
        cv2.putText(frame, loitering_label, (indicator_x + 8, indicator_y + 4),
                   cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
        
        # OVERALL Threat Score
        indicator_y += 20
        fused_color = (0, 0, 220) if threat_data['threat_value'] == 4 else \
                      (0, 100, 220) if threat_data['threat_value'] == 3 else \
                      (0, 165, 240) if threat_data['threat_value'] == 2 else \
                      (100, 165, 240) if threat_data['threat_value'] == 1 else (0, 200, 0)
        cv2.circle(frame, (indicator_x - 5, indicator_y), 6, fused_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"THREAT:{threat_data['fused_score']:.2f}", (indicator_x + 8, indicator_y + 4),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw skeleton with threat color
        try:
            kps = result.keypoints.xy[pi].cpu().numpy()
            draw_skeleton(frame, kps, box_color)
        except:
            pass


def draw_enhanced_hud(frame, violence_data, overall_threat_level, person_threats, fps_val, frame_no):
    """
    Draw enhanced HUD showing violence, weapon, and loitering status.
    
    Args:
        frame: Input frame (H, W, 3)
        violence_data: Dict with 'raw', 'smooth', 'status' keys
        overall_threat_level: Overall threat level string
        person_threats: List of person threat dicts
        fps_val: Frames per second
        frame_no: Current frame number
    """
    H, W = frame.shape[:2]
    
    raw_prob = violence_data['raw']
    smooth_prob = violence_data['smooth']
    violence_status = violence_data['status']
    known_family_count = sum(1 for threat in person_threats if threat.get('is_known_family', False))
    unknown_count = max(0, len(person_threats) - known_family_count)
    
    # Main HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (420, 140), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Status line - Show OVERALL THREAT LEVEL (not just violence)
    threat_color_map = {
        'CRITICAL': VIOLENCE_COLOR,  # Magenta/Red
        'HIGH': WARNING_COLOR,       # Red
        'MEDIUM': (0, 165, 255),     # Orange
        'LOW': (0, 255, 255),        # Yellow
        'NORMAL': NORMAL_COLOR       # Green
    }
    status_color = threat_color_map.get(overall_threat_level, NORMAL_COLOR)
    cv2.putText(frame, f"STATUS: {overall_threat_level}", (16, 32),
               cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2, cv2.LINE_AA)
    
    # Probability bar
    bar_len = 240
    bar_x, bar_y = 16, 48
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 16), (50, 50, 50), -1)
    
    bar_color = (VIOLENCE_COLOR if smooth_prob > 0.65 
                else WARNING_COLOR if smooth_prob > 0.45 
                else NORMAL_COLOR)
    bar_width = int(smooth_prob * bar_len)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 16), bar_color, -1)
    cv2.putText(frame, f"Violence: {smooth_prob*100:.1f}%", (bar_x + bar_len + 6, bar_y + 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)
    
    # Threat summary
    threat_threat_count = sum(1 for p in person_threats if p['threat_level'] in ['CRITICAL', 'HIGH'])
    weapon_count = sum(1 for p in person_threats if p['weapon_present'])
    loitering_count = sum(1 for p in person_threats if p['loitering_detected'])
    
    info_y = 72
    cv2.putText(frame, f"People: {len(person_threats)} | Threats: {threat_threat_count} | Weapons: {weapon_count} | Loiter: {loitering_count}",
               (16, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    cv2.putText(frame, f"Known family: {known_family_count} | Unknown: {unknown_count}",
               (16, info_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    
    # FPS and frame info
    cv2.putText(frame, f"FPS: {fps_val:.1f} | Frame: {frame_no}",
               (16, info_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    
    # Overall threat level badge
    overall_threat_color = (VIOLENCE_COLOR if overall_threat_level == 'CRITICAL'
                           else WARNING_COLOR if overall_threat_level == 'HIGH'
                           else (100, 200, 100) if overall_threat_level == 'MEDIUM'
                           else NORMAL_COLOR)
    
    threat_label = f"THREAT: {overall_threat_level}"
    label_size = cv2.getTextSize(threat_label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
    cv2.rectangle(frame, (W - label_size[0] - 20, 10),
                 (W - 8, 38), overall_threat_color, -1)
    cv2.putText(frame, threat_label, (W - label_size[0] - 12, 30),
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw legend
    legend_y = H - 60
    cv2.putText(frame, "Legend:", (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    legends = [
        ((0, 0, 220), "Violence+Weapon"),
        ((0, 100, 220), "Violence"),
        ((0, 165, 240), "Weapon"),
        ((255, 0, 255), "Loitering"),
        ((0, 200, 0), "Safe")
    ]
    
    legend_x = 8
    for i, (color, label) in enumerate(legends):
        if legend_x + 100 > W - 10:
            legend_y += 18
            legend_x = 8
        cv2.circle(frame, (legend_x, legend_y), 4, color, -1)
        cv2.putText(frame, label, (legend_x + 10, legend_y + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        legend_x += 100
