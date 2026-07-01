"""
Enhanced visualization with per-module confidence indicators.

Shows breakdown of violence, weapon, and loitering scores separately
along with the fused threat assessment.
"""

import cv2
import numpy as np


class EnhancedVisualizer:
    """Visualize threat detection with detailed per-module breakdowns."""
    
    def __init__(self, frame_shape, font_scale=0.5):
        """
        Args:
            frame_shape: (height, width, channels) of frames
            font_scale: Text size scale
        """
        self.frame_shape = frame_shape
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Color assignments
        self.colors = {
            'CRITICAL': (128, 0, 255),    # Magenta
            'HIGH': (0, 0, 255),          # Red
            'MEDIUM': (0, 165, 255),      # Orange
            'LOW': (0, 255, 255),         # Yellow
            'NORMAL': (0, 255, 0),        # Green
        }
        
        self.module_colors = {
            'violence': (0, 0, 255),      # Red (primary)
            'weapon': (0, 255, 255),      # Yellow
            'loitering': (255, 0, 0),     # Blue
            'fused': (128, 0, 255),       # Magenta
        }
    
    def draw_frame(self, frame, detections, fusion_results, interactions=None, 
                  person_positions=None, video_metadata=None):
        """
        Draw comprehensive threat assessment visualization.
        
        Args:
            frame: Input image
            detections: Dict of {track_id: (x, y, w, h)}
            fusion_results: Dict from temporal_fusion.process_frame()
            interactions: Dict of interactions from temporal_fusion
            person_positions: Dict {track_id: (norm_x, norm_y)}
            video_metadata: Additional frame metadata
        
        Returns:
            Annotated frame
        """
        frame = frame.copy()
        
        # ===== Draw Person Detections with Detailed HUD =====
        for track_id, (x, y, w, h) in detections.items():
            if track_id not in fusion_results:
                continue
            
            threat = fusion_results[track_id]
            color = self.colors[threat['threat_level']]
            
            # Draw bounding box
            self._draw_bbox(frame, (x, y, w, h), color, thickness=2)
            
            # Draw ID and threat level
            self._draw_person_header(frame, track_id, threat, (x, y))
            
            # Draw module scores (vertical bars)
            self._draw_module_scores(frame, threat, (x, y + h + 5))
            
            # Draw risk factors
            self._draw_risk_factors(frame, threat, (x, y + h + 80))
            
            # Draw trend indicator
            self._draw_threat_trend(frame, threat, (x + w - 30, y + 5))
        
        # ===== Draw Interactions =====
        if interactions and person_positions:
            self._draw_interactions(frame, interactions, person_positions, detections)
        
        # ===== Draw Global HUD =====
        self._draw_global_hud(frame, fusion_results, video_metadata)
        
        return frame
    
    def _draw_bbox(self, frame, bbox, color, thickness=2):
        """Draw bounding box."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    def _draw_person_header(self, frame, track_id, threat, top_left):
        """Draw person ID and threat level header."""
        x, y = top_left
        threat_level = threat['threat_level']
        threat_value = threat['threat_value']
        
        color = self.colors[threat_level]
        
        # ID and threat level
        label = f"ID:{track_id} {threat_level} ({threat['fused_score']:.2f})"
        cv2.putText(frame, label, (x, y - 5), self.font, 
                   self.font_scale + 0.1, color, 2)
    
    def _draw_module_scores(self, frame, threat, top_left):
        """
        Draw horizontal score bars for each module.
        
        Format:
        V: [========>      ] 0.78
        W: [====           ] 0.40
        L: [==             ] 0.18
        """
        x, y = top_left
        bar_width = 100
        bar_height = 12
        spacing = 16
        
        modules = [
            ('V', 'violence_score', 'violence'),
            ('W', 'weapon_score', 'weapon'),
            ('L', 'loitering_score', 'loitering'),
        ]
        
        for i, (label, key, mod_type) in enumerate(modules):
            score = threat.get(key, 0.0)
            filled = int(bar_width * score)
            
            # Label
            cv2.putText(frame, f"{label}:", (x, y + i*spacing), self.font,
                       self.font_scale, self.module_colors[mod_type], 1)
            
            # Background bar
            x_bar = x + 20
            cv2.rectangle(frame, (x_bar, y + i*spacing - 10),
                         (x_bar + bar_width, y + i*spacing - 10 + bar_height),
                         (50, 50, 50), 1)
            
            # Filled bar
            if filled > 0:
                cv2.rectangle(frame, (x_bar, y + i*spacing - 10),
                             (x_bar + filled, y + i*spacing - 10 + bar_height),
                             self.module_colors[mod_type], -1)
            
            # Score text
            score_text = f"{score:.2f}"
            cv2.putText(frame, score_text, (x_bar + bar_width + 5, y + i*spacing),
                       self.font, self.font_scale, (255, 255, 255), 1)
    
    def _draw_risk_factors(self, frame, threat, top_left):
        """Draw detected risk factors."""
        x, y = top_left
        
        factors = threat.get('risk_factors', [])
        if not factors:
            return
        
        factors_text = "Risks: " + ", ".join(factors[:3])
        if len(factors) > 3:
            factors_text += f" +{len(factors)-3}"
        
        # Background for text
        text_size = cv2.getTextSize(factors_text, self.font, self.font_scale, 1)[0]
        cv2.rectangle(frame, (x - 2, y - text_size[1] - 2),
                     (x + text_size[0] + 2, y + 2),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, factors_text, (x, y), self.font,
                   self.font_scale, (0, 165, 255), 1)
    
    def _draw_threat_trend(self, frame, threat, top_right):
        """Draw threat trend indicator (rising/falling/stable)."""
        x, y = top_right
        trend = threat.get('threat_trend', 0.0)
        
        # Arrow direction
        if trend > 0.1:
            arrow = "↑"
            color = (0, 0, 255)  # Red for increasing
        elif trend < -0.1:
            arrow = "↓"
            color = (0, 255, 0)  # Green for decreasing
        else:
            arrow = "→"
            color = (0, 165, 255)  # Orange for stable
        
        cv2.putText(frame, arrow, (x, y + 20), self.font,
                   0.8, color, 2)
    
    def _draw_interactions(self, frame, interactions, person_positions, detections):
        """Draw person-to-person interactions."""
        for (tid1, tid2), inter in interactions.items():
            if inter.interaction_intensity < 0.3:
                continue
            
            if tid1 not in person_positions or tid2 not in person_positions:
                continue
            
            if tid1 not in detections or tid2 not in detections:
                continue
            
            # Get centers
            x1, y1, w1, h1 = detections[tid1]
            x2, y2, w2, h2 = detections[tid2]
            center1 = (x1 + w1 // 2, y1 + h1 // 2)
            center2 = (x2 + w2 // 2, y2 + h2 // 2)
            
            # Line thickness based on intensity
            thickness = max(1, int(3 * inter.interaction_intensity))
            
            # Color based on threat amplification
            amp = inter.threat_amplification
            if amp > 1.5:
                color = (0, 0, 255)  # Red for high amplification
            else:
                color = (0, 165, 255)  # Orange for medium
            
            # Draw line
            cv2.line(frame, center1, center2, color, thickness)
            
            # Draw distance label
            mid_x = (center1[0] + center2[0]) // 2
            mid_y = (center1[1] + center2[1]) // 2
            dist_text = f"D:{inter.distance:.2f}"
            
            cv2.putText(frame, dist_text, (mid_x - 20, mid_y - 5),
                       self.font, self.font_scale - 0.1, color, 1)
    
    def _draw_global_hud(self, frame, fusion_results, video_metadata=None):
        """Draw global HUD with summary and alerts."""
        height = self.frame_shape[0]
        
        # ===== Global Statistics =====
        threat_counts = {'NORMAL': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for threat in fusion_results.values():
            level = threat['threat_level']
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        # Status bar
        status_y = 30
        status_text = f"Persons: {len(fusion_results)} | "
        status_text += f"Critical:{threat_counts['CRITICAL']} | "
        status_text += f"High:{threat_counts['HIGH']} | "
        status_text += f"Medium:{threat_counts['MEDIUM']}"
        
        # Background
        text_size = cv2.getTextSize(status_text, self.font, 0.6, 1)[0]
        cv2.rectangle(frame, (5, status_y - 25), (15 + text_size[0], status_y),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, status_text, (10, status_y - 5), self.font, 0.6,
                   (255, 255, 255), 1)
        
        # ===== Alerts =====
        alerts = self._collect_all_alerts(fusion_results)
        
        alert_y = status_y + 30
        for alert in alerts[:5]:
            # Parse alert format
            if alert.startswith('⚠️'):
                color = (0, 0, 255)
            elif alert.startswith('🔴'):
                color = (0, 0, 255)
            elif alert.startswith('🟠'):
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            
            # Background
            text_size = cv2.getTextSize(alert, self.font, 0.5, 1)[0]
            cv2.rectangle(frame, (5, alert_y - 18),
                         (15 + text_size[0], alert_y + 2),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, alert, (10, alert_y - 3), self.font, 0.5, color, 1)
            alert_y += 20
        
        # ===== Video Metadata =====
        if video_metadata:
            meta_text = f"Frame: {video_metadata.get('frame_count', 0)} | "
            meta_text += f"FPS: {video_metadata.get('fps', 0):.1f}"
            
            cv2.putText(frame, meta_text, (10, height - 10), self.font, 0.5,
                       (200, 200, 200), 1)
    
    def _collect_all_alerts(self, fusion_results):
        """Collect unique alerts from all persons."""
        all_alerts = []
        seen = set()
        
        for threat in fusion_results.values():
            for alert in threat.get('alerts', []):
                if alert not in seen:
                    all_alerts.append(alert)
                    seen.add(alert)
        
        # Prioritize by severity
        priority_order = ['⚠️', '🔴', '🟠', '🟡']
        all_alerts.sort(key=lambda a: next((i for i, p in enumerate(priority_order) if a.startswith(p)), 999))
        
        return all_alerts
    
    def draw_detections_only(self, frame, detections, fusion_results):
        """Simple visualization - just bboxes and labels."""
        frame = frame.copy()
        
        for track_id, (x, y, w, h) in detections.items():
            if track_id not in fusion_results:
                continue
            
            threat = fusion_results[track_id]
            color = self.colors[threat['threat_level']]
            
            # Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"{track_id} {threat['threat_level']}"
            cv2.putText(frame, label, (x, y - 5), self.font, 0.6, color, 2)
        
        return frame
    
    def draw_confidence_matrix(self, frame, fusion_results):
        """
        Draw a small matrix showing all module scores for all persons.
        Useful for at-a-glance monitoring.
        """
        if not fusion_results:
            return frame
        
        frame = frame.copy()
        
        # Matrix properties
        cell_width = 60
        cell_height = 30
        start_x = self.frame_shape[1] - (len(fusion_results) + 1) * cell_width
        start_y = 10
        
        # Header
        cv2.putText(frame, "ID", (start_x + 5, start_y + 20), self.font, 0.4,
                   (255, 255, 255), 1)
        cv2.putText(frame, "F", (start_x + cell_width + 5, start_y + 20), self.font,
                   0.4, (255, 255, 255), 1)
        cv2.putText(frame, "V", (start_x + cell_width * 2 + 5, start_y + 20), self.font,
                   0.4, (255, 255, 255), 1)
        cv2.putText(frame, "W", (start_x + cell_width * 3 + 5, start_y + 20), self.font,
                   0.4, (255, 255, 255), 1)
        cv2.putText(frame, "L", (start_x + cell_width * 4 + 5, start_y + 20), self.font,
                   0.4, (255, 255, 255), 1)
        
        # Data rows
        row_y = start_y + 30
        for i, (tid, threat) in enumerate(fusion_results.items()):
            # ID
            cv2.putText(frame, str(tid), (start_x + 15, row_y + 12), self.font, 0.4,
                       (255, 255, 255), 1)
            
            # Fused
            f_val = threat['fused_score']
            f_color = self.colors[threat['threat_level']]
            cv2.putText(frame, f"{f_val:.2f}", (start_x + cell_width + 5, row_y + 12),
                       self.font, 0.35, f_color, 1)
            
            # Violence
            v_val = threat['violence_score']
            cv2.putText(frame, f"{v_val:.2f}", (start_x + cell_width * 2 + 5, row_y + 12),
                       self.font, 0.35, self.module_colors['violence'], 1)
            
            # Weapon
            w_val = threat['weapon_score']
            cv2.putText(frame, f"{w_val:.2f}", (start_x + cell_width * 3 + 5, row_y + 12),
                       self.font, 0.35, self.module_colors['weapon'], 1)
            
            # Loitering
            l_val = threat['loitering_score']
            cv2.putText(frame, f"{l_val:.2f}", (start_x + cell_width * 4 + 5, row_y + 12),
                       self.font, 0.35, self.module_colors['loitering'], 1)
            
            row_y += cell_height
        
        return frame
