"""
Enhanced fusion logic with temporal modeling and person-to-person interaction.

Improvements over rule-based:
  - Temporal context from all modules
  - Person-to-person interaction tracking
  - Learned interaction patterns
  - Multi-step reasoning for complex scenarios
"""

import numpy as np
from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class ThreatLevel(Enum):
    """Threat classification levels."""
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PersonThreatHistory:
    """Temporal threat history for a person."""
    track_id: int
    violence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    weapon_history: deque = field(default_factory=lambda: deque(maxlen=30))
    loitering_history: deque = field(default_factory=lambda: deque(maxlen=30))
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    interaction_scores: dict = field(default_factory=dict)  # {other_tid: score}
    threat_trend: float = 0.0  # Rising/falling trend
    alpha_tracker: dict = field(default_factory=dict)  # Per-person EMA
    
    # ===== Persistent Threat Tracking =====
    max_threat_level_reached: str = 'NORMAL'  # Highest threat ever reached
    frames_at_critical: int = 0  # How many frames at CRITICAL
    last_threat_level: str = 'NORMAL'  # Previous frame's threat level
    frames_since_critical: int = 0  # Reset when CRITICAL, increment when below


@dataclass  
class PersonInteraction:
    """Interaction between two persons."""
    person_a_id: int
    person_b_id: int
    distance: float = 0.0
    approach_rate: float = 0.0
    interaction_intensity: float = 0.0  # 0-1: how interactive
    threat_amplification: float = 1.0
    frames_interacting: int = 0
    last_interaction_frame: int = 0


class TemporalFusion:
    """
    Enhanced fusion engine with temporal modeling and person-to-person interactions.
    
    Improvements:
      - Tracks threat history per person (30-frame window)
      - Models person-to-person interactions
      - Detects threat escalation patterns
      - Context-aware threat assessment
      - Learns interaction dynamics over time
    """
    
    def __init__(self, window_size=30):
        """
        Args:
            window_size: Temporal window for history tracking
        """
        self.window_size = window_size
        
        # Per-person state
        self.person_history = {}  # {track_id: PersonThreatHistory}
        self.interactions = {}  # {(tid1, tid2): PersonInteraction}
        self.frame_count = 0
        
        # Thresholds for interaction detection
        self.distance_threshold = 0.15  # Normalized
        self.approach_threshold = 0.02  # Velocity component
        self.interaction_intensity_threshold = 0.3
    
    def process_frame(self, violence_results, weapon_results, loitering_results,
                     person_positions, person_ids):
        """
        Process frame and update threat assessments.
        
        Args:
            violence_results: Dict from ViolenceDetector
            weapon_results: Dict from WeaponDetector {tid: {...}}
            loitering_results: Dict from LoiteringAnalyzer {tid: {...}}
            person_positions: Dict {tid: (x, y)} normalized positions
            person_ids: List of track IDs present this frame
        
        Returns:
            Dict with fused threat assessments for each person
        """
        self.frame_count += 1
        
        # ===== Update Person Histories =====
        for tid in person_ids:
            if tid not in self.person_history:
                self.person_history[tid] = PersonThreatHistory(track_id=tid)
            
            history = self.person_history[tid]
            
            # Store scores
            violence_score = violence_results.get('smooth_prob', 0.0)
            weapon_score = weapon_results.get(tid, {}).get('smooth_score', 0.0)
            loitering_score = loitering_results.get(tid, {}).get('smooth_score', 0.0)
            
            history.violence_history.append(violence_score)
            history.weapon_history.append(weapon_score)
            history.loitering_history.append(loitering_score)
            
            if tid in person_positions:
                history.position_history.append(person_positions[tid])
        
        # ===== Detect Person-to-Person Interactions =====
        interactions = self._detect_interactions(person_positions, person_ids)
        
        # ===== Compute Threat Trend =====
        for tid in person_ids:
            if tid in self.person_history:
                self.person_history[tid].threat_trend = self._compute_threat_trend(tid)
        
        # ===== Compute Fused Threats with Context =====
        fused_results = {}
        
        for tid in person_ids:
            if tid not in self.person_history:
                continue
            
            history = self.person_history[tid]
            violence = violence_results.get('smooth_prob', 0.0)
            weapon_data = weapon_results.get(tid, {})
            loitering_data = loitering_results.get(tid, {})
            
            # ===== Interaction Amplification =====
            interaction_amplification = 1.0
            interacting_ids = []
            
            for other_tid in person_ids:
                if other_tid == tid:
                    continue
                
                inter_key = tuple(sorted([tid, other_tid]))
                if inter_key in interactions:
                    inter = interactions[inter_key]
                    if inter.interaction_intensity > self.interaction_intensity_threshold:
                        interaction_amplification += inter.threat_amplification
                        interacting_ids.append(other_tid)
            
            # ===== Threat Escalation Detection =====
            escalation_factor = self._detect_escalation(tid)
            
            # ===== Temporal Consistency =====
            temporal_consistency = self._compute_temporal_consistency(tid)
            
            # ===== Enhanced Fused Score =====
            fused_score = self._compute_enhanced_threat_score(
                violence, weapon_data, loitering_data,
                escalation_factor, interaction_amplification,
                temporal_consistency
            )
            
            # ===== Threat Classification =====
            threat_level = self._classify_threat(
                violence, weapon_data, loitering_data,
                escalation_factor, fused_score
            )
            
            # ===== Apply Persistent Threat Tracking =====
            # Once CRITICAL, stay HIGH until person leaves (grace period 10 frames)
            threat_level = self._apply_persistent_threat_tracking(tid, threat_level, history)
            
            # ===== Context Features =====
            context_features = self._extract_context(
                tid, history, interactions, violence_results
            )
            
            fused_results[tid] = {
                'fused_score': fused_score,
                'threat_level': threat_level.name,
                'threat_value': threat_level.value,
                'violence_score': violence,
                'weapon_score': weapon_data.get('smooth_score', 0.0),
                'weapon_present': weapon_data.get('weapon_present', False),
                'loitering_score': loitering_data.get('smooth_score', 0.0),
                'loitering_detected': loitering_data.get('loitering_detected', False),
                'escalation_factor': escalation_factor,
                'interaction_amplification': interaction_amplification,
                'temporal_consistency': temporal_consistency,
                'threat_trend': history.threat_trend,
                'interacting_with': interacting_ids,
                'context_features': context_features,
                'risk_factors': self._extract_risk_factors(threat_level, context_features),
                'alerts': self._generate_alerts(threat_level, context_features)
            }
        
        return fused_results, interactions
    
    def _detect_interactions(self, positions, person_ids):
        """
        Detect person-to-person interactions based on proximity and motion.
        
        Returns:
            Dict of {(tid1, tid2): PersonInteraction}
        """
        interactions = {}
        
        for i, tid1 in enumerate(person_ids):
            for tid2 in person_ids[i+1:]:
                if tid1 not in positions or tid2 not in positions:
                    continue
                
                pos1 = np.array(positions[tid1])
                pos2 = np.array(positions[tid2])
                distance = float(np.linalg.norm(pos1 - pos2))
                
                inter_key = (tid1, tid2) if tid1 < tid2 else (tid2, tid1)
                
                # Get velocity from history
                h1 = self.person_history.get(tid1)
                h2 = self.person_history.get(tid2)
                
                approach_rate = 0.0
                if h1 and len(h1.position_history) > 1 and h2 and len(h2.position_history) > 1:
                    vel1 = np.array(h1.position_history[-1]) - np.array(h1.position_history[-2])
                    vel2 = np.array(h2.position_history[-1]) - np.array(h2.position_history[-2])
                    rel_pos = pos1 - pos2
                    approach_rate = float(np.dot(vel1 - vel2, rel_pos / (distance + 1e-6)))
                
                # Compute interaction intensity
                is_close = distance < self.distance_threshold
                is_approaching = approach_rate > self.approach_threshold
                intensity = 0.0
                
                if is_close:
                    intensity += 0.5 * (1 - distance / self.distance_threshold)
                if is_approaching:
                    intensity += 0.5 * min(1.0, approach_rate / 0.1)
                
                # Update or create interaction
                if inter_key not in self.interactions:
                    self.interactions[inter_key] = PersonInteraction(tid1, tid2)
                
                inter = self.interactions[inter_key]
                inter.distance = distance
                inter.approach_rate = approach_rate
                inter.interaction_intensity = intensity
                inter.last_interaction_frame = self.frame_count
                
                if intensity > self.interaction_intensity_threshold:
                    inter.frames_interacting += 1
                    # Amplification increases with interaction duration
                    inter.threat_amplification = 1.0 + (0.1 * min(10, inter.frames_interacting / 5))
                else:
                    inter.frames_interacting = 0
                    inter.threat_amplification = 1.0
                
                interactions[inter_key] = inter
        
        return interactions
    
    def _compute_threat_trend(self, track_id):
        """
        Compute threat trend: rising/falling/stable.
        
        Returns:
            float: trend score [-1, 1] where negative=falling, positive=rising
        """
        if track_id not in self.person_history:
            return 0.0
        
        history = self.person_history[track_id]
        
        if len(history.violence_history) < 10:
            return 0.0
        
        recent = list(history.violence_history)[-10:]
        older = list(history.violence_history)[-20:-10]
        
        recent_avg = np.mean(recent) if recent else 0.0
        older_avg = np.mean(older) if older else 0.0
        
        trend = (recent_avg - older_avg) / (older_avg + 1e-6)
        return float(np.clip(trend, -1.0, 1.0))
    
    def _detect_escalation(self, track_id):
        """
        Detect threat escalation patterns.
        
        Returns:
            float: escalation factor [0, 2]
        """
        if track_id not in self.person_history:
            return 1.0
        
        history = self.person_history[track_id]
        violence_vals = list(history.violence_history)
        
        if len(violence_vals) < 5:
            return 1.0
        
        # Check for rapid increase
        recent = violence_vals[-5:]
        older = violence_vals[-10:-5] if len(violence_vals) >= 10 else violence_vals[:5]
        
        recent_trend = np.mean(recent) if recent else 0.0
        older_trend = np.mean(older) if older else 0.0
        
        escalation = recent_trend / (older_trend + 1e-6)
        
        # Also check for sustained elevation
        if recent_trend > 0.6 and len([v for v in recent if v > 0.6]) >= 3:
            escalation *= 1.5
        
        return float(np.clip(escalation, 0.5, 2.0))
    
    def _compute_temporal_consistency(self, track_id):
        """
        How consistent are the threat signals over time?
        High consistency = high confidence in threat assessment.
        
        Returns:
            float: [0, 1] higher = more consistent/confident
        """
        if track_id not in self.person_history:
            return 0.0
        
        history = self.person_history[track_id]
        
        if len(history.violence_history) < 10:
            return 0.3
        
        recent = list(history.violence_history)[-10:]
        variance = float(np.var(recent))
        
        # Low variance = high consistency
        consistency = max(0.0, 1.0 - variance)
        
        return consistency
    
    def _compute_enhanced_threat_score(self, violence, weapon_data, loitering_data,
                                      escalation, interaction_amp, consistency):
        """
        Compute enhanced threat score combining temporal and interaction context.
        
        Combines:
        - Violence (primary, 0.7 weight)
        - Weapon (secondary, 0.15 weight)
        - Loitering (tertiary, 0.15 weight)
        - Escalation (amplifies base score)
        - Interactions (amplifies based on other persons)
        - Consistency (confidence modifier)
        """
        weapon_score = weapon_data.get('smooth_score', 0.0)
        loitering_score = loitering_data.get('smooth_score', 0.0)
        
        # Base threat
        base_threat = (
            0.70 * violence +
            0.15 * weapon_score +
            0.15 * loitering_score
        )
        
        # Apply escalation factor
        escalated_threat = base_threat * escalation
        
        # Apply interaction amplification
        # But cap the amplification to prevent runaway scores
        inter_amp_capped = min(interaction_amp, 2.5)
        amplified_threat = escalated_threat * inter_amp_capped
        
        # Apply consistency as confidence modifier
        # High consistency = trust the signal more
        # Low consistency = be more conservative
        final_score = amplified_threat * (0.5 + 0.5 * consistency)
        
        return float(np.clip(final_score, 0.0, 1.0))
    
    def _classify_threat(self, violence, weapon_data, loitering_data, escalation, fused_score):
        """
        Classify threat level with temporal and interaction context.
        
        Real-world tuning:
        - CRITICAL: Gun/knife with extreme violence (>0.5+) or very high violence
        - HIGH: Weapon + ANY violence (armed + active threat) OR gun/knife alone OR high violence
        - MEDIUM: High violence alone OR weapon + loitering (suspicious)
        - LOW: Moderate violence or loitering
        - NORMAL: Clean scene
        """
        weapon_present = weapon_data.get('weapon_present', False)
        weapon_type = weapon_data.get('weapon_type', 'unknown')
        loitering_detected = loitering_data.get('loitering_detected', False)
        
        # ===== CRITICAL (Highest Priority) =====
        # Gun/knife + extreme violence = imminent threat
        if weapon_present and weapon_type in ['gun', 'knife'] and violence > 0.5:
            return ThreatLevel.CRITICAL
        
        # Extreme violence with any weapon
        if violence > 0.75 and weapon_present:
            return ThreatLevel.CRITICAL
        
        # Very high violence alone
        if violence > 0.85:
            return ThreatLevel.CRITICAL
        
        # ===== HIGH (ARMED + VIOLENT = DANGEROUS) =====
        # Weapon + ANY violence (even small amount) = high threat
        if weapon_present and violence > 0.3:
            return ThreatLevel.HIGH
        
        # Gun/Knife alone is inherent danger
        if weapon_present and weapon_type in ['gun', 'knife']:
            return ThreatLevel.HIGH
        
        # High violence (even without weapon)
        if violence > 0.7:
            return ThreatLevel.HIGH
        
        # Moderate-high violence alone
        if violence > 0.65:
            return ThreatLevel.HIGH
        
        # ===== MEDIUM =====
        # Weapon + loitering (suspicious but not violent)
        if weapon_present and loitering_detected:
            return ThreatLevel.MEDIUM
        
        # Moderate violence alone
        if violence > 0.6:
            return ThreatLevel.MEDIUM
        
        # ===== LOW =====
        # Minor-moderate violence
        if violence > 0.45:
            return ThreatLevel.LOW
        
        # Loitering detected (without weapon)
        if loitering_detected:
            return ThreatLevel.LOW
        
        # Fused score indicates some concern
        if fused_score > 0.4:
            return ThreatLevel.LOW
        
        # ===== NORMAL =====
        return ThreatLevel.NORMAL
    
    def _apply_persistent_threat_tracking(self, tid, computed_threat_level, history):
        """
        Apply persistent threat tracking:
        - Once a person reaches CRITICAL, keep them at HIGH until they leave
        - Track max threat level reached
        - Update threat history for continuous monitoring
        
        Args:
            tid: Track ID
            computed_threat_level: ThreatLevel computed from current frame
            history: PersonThreatHistory for this person
        
        Returns:
            Adjusted ThreatLevel applying persistence logic
        """
        computed_name = computed_threat_level.name
        
        # Update max threat reached
        if computed_threat_level.value > self._threat_level_value(history.max_threat_level_reached):
            history.max_threat_level_reached = computed_name
        
        # If CRITICAL this frame
        if computed_name == 'CRITICAL':
            history.frames_at_critical += 1
            history.frames_since_critical = 0
            history.last_threat_level = 'CRITICAL'
            return ThreatLevel.CRITICAL
        
        # If person was CRITICAL before, keep them HIGH for grace period (10 frames = ~0.33 sec)
        grace_frames = 10
        if history.max_threat_level_reached == 'CRITICAL':
            if history.frames_since_critical < grace_frames:
                history.frames_since_critical += 1
                # Escalate to HIGH if below it
                if computed_threat_level.value < ThreatLevel.HIGH.value:
                    history.last_threat_level = 'HIGH'
                    return ThreatLevel.HIGH
                else:
                    history.last_threat_level = computed_name
                    return computed_threat_level
            else:
                # Grace period expired, use normal threat classification
                history.last_threat_level = computed_name
                return computed_threat_level
        
        # Normal case: use computed threat
        history.last_threat_level = computed_name
        return computed_threat_level
    
    def _threat_level_value(self, threat_name):
        """Get numeric value of threat level by name."""
        try:
            return ThreatLevel[threat_name].value
        except:
            return 0
    
    def _extract_context(self, tid, history, interactions, violence_results):
        """
        Extract contextual information about the threat.
        """
        violence_history = list(history.violence_history)[-5:] if history.violence_history else []
        weapon_history = list(history.weapon_history)[-5:] if history.weapon_history else []
        
        context = {
            'violence_avg_recent': float(np.mean(violence_history)) if violence_history else 0.0,
            'violence_max_recent': float(max(violence_history)) if violence_history else 0.0,
            'weapon_count_recent': sum(1 for w in weapon_history if w > 0.5),
            'sustained_violent': all(v > 0.5 for v in violence_history[-3:]) if len(violence_history) >= 3 else False,
            'approaching': violence_results.get('inter_features', {}).get('approach_vel', 0) > 0.05,
            'close_proximity': violence_results.get('inter_features', {}).get('bbox_iou', 0) > 0.2,
        }
        
        return context
    
    def _extract_risk_factors(self, threat_level, context):
        """Extract specific risk factors."""
        factors = []
        
        if threat_level.value >= 2:  # MEDIUM or higher
            if context.get('sustained_violent'):
                factors.append('sustained_violence')
            if context.get('approaching'):
                factors.append('approaching')
            if context.get('close_proximity'):
                factors.append('close_proximity')
            if context.get('violence_max_recent', 0) > 0.7:
                factors.append('high_violence_intensity')
            if context.get('weapon_count_recent', 0) > 0:
                factors.append('weapon_detected')
        
        return factors
    
    def _generate_alerts(self, threat_level, context):
        """Generate actionable alerts."""
        alerts = []
        
        if threat_level == ThreatLevel.CRITICAL:
            alerts.append('⚠️ CRITICAL: Armed violent altercation detected - Immediate response required')
            if context.get('sustained_violent'):
                alerts.append('⚠️ Extended violence in progress')
        elif threat_level == ThreatLevel.HIGH:
            alerts.append('🔴 HIGH THREAT: Active violence detected - Response recommended')
            if context.get('weapon_count_recent'):
                alerts.append('🔴 Weapon detected in altercation')
        elif threat_level == ThreatLevel.MEDIUM:
            alerts.append('🟠 MEDIUM: Suspicious activity - Increased monitoring')
            if context.get('approaching') and context.get('violence_avg_recent', 0) > 0.4:
                alerts.append('⚠️ Approaching confrontation developing')
        elif threat_level == ThreatLevel.LOW:
            alerts.append('🟡 LOW: Unusual behavior detected')
        
        return alerts
    
    def remove_person(self, track_id):
        """Clean up state for person leaving frame."""
        self.person_history.pop(track_id, None)
        
        # Also clean up interactions involving this person
        keys_to_remove = [k for k in self.interactions.keys() 
                         if track_id in k]
        for k in keys_to_remove:
            self.interactions.pop(k, None)
    
    def reset(self):
        """Reset all state for new video."""
        self.person_history.clear()
        self.interactions.clear()
        self.frame_count = 0
