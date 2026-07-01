"""
Anomaly Detection Fusion - Context-aware fusion of violence, weapon, and loitering.

Philosophy:
  - Violence, Weapon, and Loitering are THREE EQUAL ANOMALIES
  - Gun presence always triggers HIGH threat (weapon of choice)
  - Weapon + Violence = CRITICAL threat
  - Context-aware: Hand raised + weapon = active usage intent
  - Loitering adds suspicion independently
  - Kept modular so can be replaced with learned model later
"""

import numpy as np
from enum import Enum


class AnomalyType(Enum):
    """Anomaly classifications."""
    NORMAL = 0
    VIOLENCE = 1
    WEAPON = 2
    LOITERING = 3
    VIOLENCE_WEAPON = 4  # Combined threat


class ThreatLevel(Enum):
    """Threat classification levels."""
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ThreatFusion:
    """
    Multi-modal anomaly detection combining:
      - Violence detection (equal anomaly)
      - Weapon detection (equal anomaly, GUN = automatic HIGH)
      - Loitering analysis (equal anomaly)
    
    Context-aware features:
      - Hand raised + weapon = active usage
      - Gun detected = automatic HIGH threat
      - Multiple anomalies = escalating threat
    
    Outputs per-person threat assessment with anomaly labels.
    """
    
    def __init__(self):
        """Initialize fusion engine."""
        # Equal weighting for all three anomalies
        self.weight_violence = 0.33      # Violence anomaly
        self.weight_weapon = 0.33        # Weapon anomaly  
        self.weight_loitering = 0.34     # Loitering anomaly
        
        # Gun escalation
        self.gun_threat_multiplier = 2.0  # Gun doubles threat
    
    def fuse(self, violence_result, weapon_results, loitering_results, person_ids):
        """
        Fuse all three anomalies into per-person threat assessment.
        Treats violence, weapon, and loitering as EQUAL anomalies.
        
        Args:
            violence_result: Dict from ViolenceDetector.update()
            weapon_results: Dict from WeaponDetector.update() for all persons
            loitering_results: Dict from LoiteringAnalyzer.update() for all persons
            person_ids: List of (track_id, bbox) for current detected persons
        
        Returns:
            Dict of {track_id: threat_assessment}
        """
        fused_results = {}
        
        violence_status = violence_result['status']
        violence_prob = violence_result['smooth_prob']
        violence_confirmed = violence_result['confirmed']
        
        for track_id, bbox in person_ids:
            # ===== Get Individual Anomaly Scores =====
            violence_score = violence_prob  # Shared across frame
            
            weapon_info = weapon_results.get(track_id, {})
            weapon_present = weapon_info.get('weapon_present', False)
            weapon_score = weapon_info.get('smooth_score', 0.0)
            weapon_type = weapon_info.get('weapon_type', 'unknown')  # 'gun' or 'knife'
            
            loitering_info = loitering_results.get(track_id, {})
            loitering_detected = loitering_info.get('loitering_detected', False)
            loitering_score = loitering_info.get('smooth_score', 0.0)
            
            # ===== Context-Aware Weapon Assessment =====
            is_gun = weapon_type == 'gun'
            weapon_usage_intent = False
            
            if weapon_present and violence_score > 0.4:
                # Hand raised + weapon = active usage intent
                weapon_usage_intent = True
            
            # ===== Anomaly Detection Flags =====
            has_violence = violence_status in ['VIOLENCE', 'WARNING']
            has_weapon = weapon_present
            has_loitering = loitering_detected
            
            # Count active anomalies
            active_anomalies = sum([has_violence, has_weapon, has_loitering])
            
            # ===== Calculate Threat Score (Equal Weight) =====
            # All three treated equally
            fused_score = (
                self.weight_violence * violence_score +
                self.weight_weapon * weapon_score +
                self.weight_loitering * loitering_score
            )
            
            # ===== Gun Escalation =====
            # Gun presence automatically escalates threat
            if is_gun and weapon_present:
                fused_score = min(1.0, fused_score * self.gun_threat_multiplier)
            
            # ===== Multi-Anomaly Escalation =====
            # Multiple anomalies together = higher threat
            if active_anomalies >= 2:
                fused_score = min(1.0, fused_score * (1.0 + 0.3 * (active_anomalies - 1)))
            
            fused_score = np.clip(fused_score, 0.0, 1.0)
            
            # ===== Threat Classification =====
            threat_level = self._classify_threat(
                violence_status, violence_confirmed, violence_prob,
                weapon_present, is_gun, weapon_usage_intent,
                loitering_detected, active_anomalies, fused_score
            )
            
            # ===== Anomaly Labels =====
            anomalies = self._get_anomaly_labels(
                has_violence, has_weapon, has_loitering, is_gun, weapon_usage_intent
            )
            
            # ===== Risk Indicators =====
            risk_factors = self._extract_risk_factors(
                violence_result, weapon_info, loitering_info, is_gun, weapon_usage_intent
            )
            
            fused_results[track_id] = {
                'fused_score': fused_score,
                'threat_level': threat_level.name,
                'threat_value': threat_level.value,
                'anomaly_type': anomalies,
                'anomaly_count': active_anomalies,
                'violence_score': violence_score,
                'weapon_score': weapon_score,
                'weapon_present': weapon_present,
                'weapon_type': weapon_type,
                'is_gun': is_gun,
                'weapon_usage_intent': weapon_usage_intent,
                'loitering_score': loitering_score,
                'loitering_detected': loitering_detected,
                'risk_factors': risk_factors,
                'alerts': self._generate_alerts(
                    threat_level, anomalies, is_gun, weapon_usage_intent, risk_factors
                )
            }
        
        return fused_results
    
    def _classify_threat(self, violence_status, violence_confirmed, violence_prob,
                         weapon_present, is_gun, weapon_usage_intent,
                         loitering_detected, anomaly_count, fused_score):
        """
        Classify threat level based on anomaly combinations.
        Treats all three anomalies equally with escalation rules.
        
        Returns:
            ThreatLevel enum
        """
        # ===== CRITICAL THREATS =====
        # Gun + Weapon Usage Intent = ALWAYS CRITICAL
        if is_gun and weapon_usage_intent:
            return ThreatLevel.CRITICAL
        
        # Violence Confirmed + Weapon = CRITICAL
        if violence_confirmed and weapon_present:
            return ThreatLevel.CRITICAL
        
        # Violence Confirmed + Gun = CRITICAL (gun is never safe)
        if violence_confirmed and is_gun:
            return ThreatLevel.CRITICAL
        
        # ===== HIGH THREATS =====
        # Gun presence alone = HIGH (guns are dangerous weapons)
        if is_gun:
            return ThreatLevel.HIGH
        
        # Violence Status + Weapon Present = HIGH
        if violence_status == 'VIOLENCE' and weapon_present:
            return ThreatLevel.HIGH
        
        # Weapon with Usage Intent = HIGH (threat intent detected)
        if weapon_present and weapon_usage_intent:
            return ThreatLevel.HIGH
        
        # Violence Confirmed = HIGH
        if violence_confirmed:
            return ThreatLevel.HIGH
        
        # Weapon Present with high score = HIGH (not LOW)
        if weapon_present and fused_score > 0.5:
            return ThreatLevel.HIGH
        
        # ===== MEDIUM THREATS =====
        # Violence Status (WARNING) = MEDIUM
        if violence_status == 'VIOLENCE':
            return ThreatLevel.MEDIUM
        
        # Multiple anomalies (2+) = MEDIUM
        if anomaly_count >= 2:
            return ThreatLevel.MEDIUM
        
        # Weapon Present (moderate confidence) = MEDIUM
        if weapon_present and fused_score > 0.3:
            return ThreatLevel.MEDIUM
        
        # High fused score = MEDIUM
        if fused_score > 0.6:
            return ThreatLevel.MEDIUM
        
        # ===== LOW THREATS =====
        # Warning Status = LOW
        if violence_status == 'WARNING':
            return ThreatLevel.LOW
        
        # Weapon Present (low confidence) = LOW
        if weapon_present:
            return ThreatLevel.LOW
        
        # Loitering alone = LOW
        if loitering_detected:
            return ThreatLevel.LOW
        
        # Moderate fused score = LOW
        if fused_score > 0.3:
            return ThreatLevel.LOW
        
        # ===== NORMAL =====
        return ThreatLevel.NORMAL
    
    def _get_anomaly_labels(self, has_violence, has_weapon, has_loitering, 
                            is_gun, weapon_usage_intent):
        """
        Get human-readable anomaly labels.
        
        Returns:
            List of anomaly type strings
        """
        labels = []
        
        if has_violence:
            labels.append('VIOLENCE')
        
        if has_weapon:
            if is_gun:
                labels.append('GUN')
            else:
                labels.append('WEAPON')
        
        if weapon_usage_intent:
            labels.append('ACTIVE_INTENT')
        
        if has_loitering:
            labels.append('LOITERING')
        
        if not labels:
            labels.append('NORMAL')
        
        return labels
    
    def _extract_risk_factors(self, violence_result, weapon_info, loitering_info,
                             is_gun, weapon_usage_intent):
        """
        Extract specific risk factors from all anomalies.
        
        Returns:
            List of risk factor strings
        """
        factors = []
        
        # Violence factors
        if violence_result['status'] == 'VIOLENCE':
            factors.append('ACTIVE_VIOLENCE')
        elif violence_result['status'] == 'WARNING':
            factors.append('POTENTIAL_VIOLENCE')
        
        if violence_result['confirmed']:
            factors.append('SUSTAINED_VIOLENCE')
        
        inter_feat = violence_result.get('inter_features', {})
        if inter_feat.get('approach_vel', 0) > 0.1:
            factors.append('RAPID_APPROACH')
        if inter_feat.get('bbox_iou', 0) > 0.3:
            factors.append('CLOSE_PROXIMITY')
        
        # Weapon factors
        if weapon_info.get('weapon_present', False):
            if is_gun:
                factors.append('GUN_DETECTED')
                if weapon_usage_intent:
                    factors.append('GUN_USAGE_INTENT')
            else:
                factors.append('WEAPON_DETECTED')
                if weapon_usage_intent:
                    factors.append('WEAPON_USAGE_INTENT')
        
        # Loitering factors
        if loitering_info.get('loitering_detected', False):
            dwell = loitering_info.get('dwell_time_s', 0)
            if dwell > 30:
                factors.append('EXTENDED_LOITERING')
            else:
                factors.append('LOITERING')
        
        return factors
    
    def _generate_alerts(self, threat_level, anomalies, is_gun, weapon_usage_intent, 
                        risk_factors):
        """
        Generate context-aware alerts based on detected anomalies.
        
        Returns:
            List of alert strings (empty if NORMAL)
        """
        alerts = []
        
        if threat_level == ThreatLevel.CRITICAL:
            if is_gun and weapon_usage_intent:
                alerts.append('🚨 CRITICAL: ARMED AND ACTIVELY USING GUN')
            elif 'ACTIVE_VIOLENCE' in anomalies and is_gun:
                alerts.append('🚨 CRITICAL: VIOLENT ACTIVITY WITH GUN')
            elif 'ACTIVE_VIOLENCE' in anomalies and 'WEAPON' in anomalies:
                alerts.append('🚨 CRITICAL: VIOLENT ACTIVITY WITH WEAPON')
            else:
                alerts.append('🚨 CRITICAL THREAT DETECTED')
        
        elif threat_level == ThreatLevel.HIGH:
            if is_gun:
                alerts.append('⚠️  HIGH: GUN DETECTED - INHERENT DANGER')
                if weapon_usage_intent:
                    alerts.append('⚠️  Weapon usage intent detected')
            elif 'ACTIVE_VIOLENCE' in anomalies:
                alerts.append('⚠️  HIGH: VIOLENCE IN PROGRESS')
                if 'WEAPON' in anomalies:
                    alerts.append('⚠️  Armed individual involved')
            elif 'WEAPON_USAGE_INTENT' in risk_factors:
                alerts.append('⚠️  HIGH: WEAPON USAGE INTENT DETECTED')
            else:
                alerts.append('⚠️  HIGH THREAT')
        
        elif threat_level == ThreatLevel.MEDIUM:
            anomaly_list = [a for a in anomalies if a != 'NORMAL']
            alert_msg = f"⚡ MEDIUM: {' + '.join(anomaly_list)} detected"
            alerts.append(alert_msg)
            
            if 'POTENTIAL_VIOLENCE' in anomalies:
                alerts.append('Situation escalating - monitor closely')
            if 'WEAPON' in anomalies and 'VIOLENCE' not in anomalies:
                alerts.append('Weapon present without violence - still elevated risk')
        
        elif threat_level == ThreatLevel.LOW:
            if 'LOITERING' in anomalies:
                alerts.append('✓ LOW: Suspicious loitering detected')
            elif 'POTENTIAL_VIOLENCE' in anomalies:
                alerts.append('✓ LOW: Potential violence warning')
            else:
                alerts.append('✓ Monitoring recommended')
        
        return alerts
    
    def set_weights(self, violence_weight, weapon_weight, loitering_weight):
        """
        Update fusion weights (advanced tuning).
        
        Should sum to 1.0 for normalized output.
        """
        total = violence_weight + weapon_weight + loitering_weight
        if total > 0:
            self.weight_violence = violence_weight / total
            self.weight_weapon = weapon_weight / total
            self.weight_loitering = loitering_weight / total

