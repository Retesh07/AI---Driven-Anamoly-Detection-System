"""
Unified inference pipeline - orchestrates all modules.

Pipeline flow:
    Video → Frame extraction → Tracking → Violence → Weapon → Loitering
    → Face Identity → Fusion → Visualization → Output
"""

import cv2
import json
import subprocess
import time
import shutil
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import constants
from utils.feature_extraction import extract_frame_features
from utils.visualization import (
    draw_detections, draw_attention_bar, draw_hud, create_timeline_visualization,
    draw_enhanced_detections, draw_enhanced_hud
)
from utils.stats_visualization import (
    generate_weapon_statistics_graph,
    generate_loitering_statistics_graph,
    generate_combined_threat_heatmap,
    generate_threat_level_distribution,
    generate_per_person_weapon_timeline
)
from tracking.tracker import PersonTracker
from violence.model import ViolenceDetectorV3
from violence.detector import ViolenceDetector
from weapon.detector import WeaponDetector
from loitering.analyzer import LoiteringAnalyzer
from identity.recognizer import FaceIdentityRecognizer
from fusion.temporal_fusion import TemporalFusion

try:
    from ultralytics import YOLO
    import supervision as sv
except ImportError:
    raise ImportError("Please install: pip install ultralytics supervision")


class ThreatDetectionPipeline:
    """
    Complete threat detection system combining all modules.
    """
    
    def __init__(self, project_root, device='cuda', verbose=True, face_db_path=None):
        """
        Initialize pipeline with all sub-modules.
        
        Args:
            project_root: Root directory containing models/
            device: 'cuda' or 'cpu'
            verbose: Print detailed info
        """
        self.project_root = Path(project_root)
        self.device = device
        self.verbose = verbose
        self.face_db_path = Path(face_db_path) if face_db_path else self.project_root / constants.FACE_DB_PATH
        
        if self.verbose:
            print('[Pipeline] Loading models...')

        required_models = [
            self.project_root / constants.MODEL_PATHS[key]
            for key in ('violence', 'violence_mean', 'violence_std', 'weapon')
        ]
        missing_models = [str(path) for path in required_models if not path.is_file()]
        if missing_models:
            raise FileNotFoundError('Missing required model files:\n  ' + '\n  '.join(missing_models))
        
        # ===== Load YOLO Pose (shared) =====
        # Try local model first, then allow YOLO to download if needed
        pose_model_path = self.project_root / 'models' / 'yolov8s-pose.pt'
        if pose_model_path.exists():
            self.pose_model = YOLO(str(pose_model_path))
        else:
            # YOLO will download from ultralytics hub if local file not found
            try:
                self.pose_model = YOLO('yolov8s-pose.pt')
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load pose model. Please download yolov8s-pose.pt manually:\n"
                    f"  1. Download from: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-pose.pt\n"
                    f"  2. Place in: {pose_model_path}\n"
                    f"Error: {e}"
                )
        
        self.tracker = sv.ByteTrack()
        
        # ===== Load Violence Detector =====
        model_path = self.project_root / constants.MODEL_PATHS['violence']
        mean_path = self.project_root / constants.MODEL_PATHS['violence_mean']
        std_path = self.project_root / constants.MODEL_PATHS['violence_std']
        
        self.violence_detector = ViolenceDetector(
            str(model_path), str(mean_path), str(std_path),
            device=device
        )
        
        # ===== Load Weapon Detector =====
        weapon_path = self.project_root / constants.MODEL_PATHS['weapon']
        self.weapon_detector = WeaponDetector(str(weapon_path), device=device)
        
        # ===== Load Loitering Analyzer =====
        self.loitering_analyzer = LoiteringAnalyzer()

        # ===== Load Face Identity Recognizer =====
        self.face_recognizer = FaceIdentityRecognizer(
            database_dir=self.face_db_path,
            model_dir=self.project_root / constants.FACE_MODEL_DIR,
            verbose=self.verbose
        )
        
        # ===== Load Fusion Engine =====
        # Using TemporalFusion (v3.0) for superior threat assessment
        # Features: 70% violence, 15% weapon, 15% loitering weighting
        # + temporal history tracking, person interaction modeling
        self.fusion = TemporalFusion(window_size=30)
        
        # ===== Initialize Trackers =====
        self.person_tracker = PersonTracker()
        
        if self.verbose:
            print('[Pipeline] All models loaded successfully.')
    
    def process_video(self, video_path, output_dir=None, 
                     violence_threshold=None, warning_threshold=None,
                     ema_alpha=constants.EMA_ALPHA,
                     export_json=True):
        """
        Process video and generate threat detection output.
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output (default: video dir)
            violence_threshold: Override violence threshold
            warning_threshold: Override warning threshold
            ema_alpha: Temporal smoothing factor
            export_json: Save JSON timeline
        
        Returns:
            Dict with processing results and timeline
        """
        
        is_stream = isinstance(video_path, int) or str(video_path).lower().startswith(('rtsp://', 'http://', 'https://'))
        video_source = video_path if isinstance(video_path, int) else str(video_path)
        if not is_stream and not Path(video_source).exists():
            raise FileNotFoundError(f"Video not found: {video_source}")
        
        if output_dir is None:
            output_dir = Path(video_source).parent if not is_stream else Path('./results')
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ===== Open Video =====
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_source}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.verbose:
            print(f'[Pipeline] Input: {W}x{H} @ {fps:.1f}fps, {total_frames} frames')
        
        # ===== Set Thresholds =====
        if violence_threshold is not None or warning_threshold is not None:
            self.violence_detector.set_thresholds(
                self.violence_detector.violence_threshold if violence_threshold is None else violence_threshold,
                self.violence_detector.warning_threshold if warning_threshold is None else warning_threshold,
            )
        
        # ===== Initialize State =====
        self.violence_detector.reset()
        self.weapon_detector.reset()
        self.loitering_analyzer.reset()
        self.face_recognizer.reset()
        self.fusion.reset()
        self.tracker.reset()
        self.person_tracker.reset()
        
        # ===== Output Writers =====
        output_video = output_dir / 'output.mp4'
        temp_video = output_dir / '.tmp_output.mp4'
        
        # Adjust FPS for frame skipping
        output_fps = fps / constants.FRAME_SKIP
        self.loitering_analyzer.fps = max(output_fps, 1e-6)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(temp_video), fourcc, output_fps, (W, H))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'Cannot create output video: {temp_video}')
        
        # ===== Processing Loop =====
        frame_idx = 0
        processed = 0
        timeline = []
        frame_times = []
        t_last = time.time()
        fps_disp = 0.0
        
        with tqdm(total=total_frames // constants.FRAME_SKIP, 
                 desc='Processing', unit='frame') as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # ===== Skip Frames =====
                if frame_idx % constants.FRAME_SKIP != 0:
                    continue
                
                processed += 1
                t_start_frame = time.time()
                
                # ===== Run Pose Detection =====
                pose_result = self.pose_model(
                    frame, device=0 if self.device == 'cuda' else 'cpu',
                    conf=constants.POSE_CONFIDENCE_THRESHOLD, verbose=False
                )[0]
                
                dets = sv.Detections.from_ultralytics(pose_result)
                
                # ===== Extract Features & Update Tracking =====
                frame_features, det_info = extract_frame_features(
                    frame, pose_result, dets, self.tracker,
                    self.person_tracker.prev_centers,
                    self.person_tracker.prev_vel,
                    self.person_tracker.prev_acc,
                    self.person_tracker.prev_kps
                )
                
                detection_ids = [info['tid'] for info in det_info]
                self.person_tracker.update(detection_ids)
                
                # ===== Get Person Bboxes =====
                person_bboxes = {info['tid']: info['bbox'] for info in det_info}

                # ===== Face Identity Recognition =====
                identity_results = self.face_recognizer.update(frame, person_bboxes)
                
                # ===== Violence Detection =====
                num_persons = len(det_info)
                violence_result = self.violence_detector.update(frame_features, num_persons)
                
                # ===== Weapon Detection =====
                # Real-time ready: works for weapons appearing at any frame
                weapon_results = self.weapon_detector.update(
                    frame, person_bboxes, 
                    pose_result=pose_result, det_info=det_info
                )
                
                # ===== Loitering Analysis =====
                loitering_results = self.loitering_analyzer.update(
                    person_bboxes, frame_shape=(H, W), identity_info=identity_results
                )
                
                # ===== Fusion =====
                # Compute normalized person positions for TemporalFusion
                person_positions = {}
                for tid, bbox in person_bboxes.items():
                    x1, y1, x2, y2 = bbox
                    # Normalize center position to 0-1 range
                    cx_norm = ((x1 + x2) / 2) / W
                    cy_norm = ((y1 + y2) / 2) / H
                    person_positions[tid] = (cx_norm, cy_norm)
                
                # Call TemporalFusion with proper parameters
                fused_results, interactions = self.fusion.process_frame(
                    violence_result, weapon_results, loitering_results,
                    person_positions, list(person_bboxes.keys()), identity_results=identity_results
                )
                
                # ===== Visualization =====
                # Determine overall threat level
                overall_threat_level = 'LOW'
                if any(p['threat_level'] == 'CRITICAL' for p in fused_results.values()):
                    overall_threat_level = 'CRITICAL'
                elif any(p['threat_level'] == 'HIGH' for p in fused_results.values()):
                    overall_threat_level = 'HIGH'
                elif any(p['threat_level'] == 'MEDIUM' for p in fused_results.values()):
                    overall_threat_level = 'MEDIUM'
                
                draw_enhanced_detections(frame, det_info, pose_result, fused_results, identity_results)
                draw_enhanced_hud(frame, {
                    'raw': violence_result['raw_prob'],
                    'smooth': violence_result['smooth_prob'],
                    'status': violence_result['status']
                }, overall_threat_level, list(fused_results.values()), fps_disp, processed)
                draw_attention_bar(frame, violence_result['attention_weights'], bar_height=18)
                
                # ===== Write Frame =====
                writer.write(frame)
                
                # ===== Timeline Entry =====
                timeline_entry = {
                    'frame': processed,
                    'timestamp_s': round(processed / output_fps, 3),
                    'violence': {
                        'raw': round(violence_result['raw_prob'], 4),
                        'smooth': round(violence_result['smooth_prob'], 4),
                        'status': violence_result['status'],
                        'confirmed': violence_result['confirmed']
                    },
                    'persons': []
                }
                
                for tid, results in fused_results.items():
                    # Extract weapon type from weapon_results if available
                    weapon_type = weapon_results.get(tid, {}).get('weapon_type', 'unknown')
                    
                    # Status = threat_level (same fused result)
                    threat_level = results['threat_level']
                    
                    timeline_entry['persons'].append({
                        'track_id': tid,
                        'threat_level': threat_level,
                        'threat_status': threat_level,  # Same as threat_level (fused)
                        'threat_value': results['threat_value'],
                        'fused_score': round(results['fused_score'], 4),
                        'violence_score': round(results['violence_score'], 4),
                        'weapon_score': round(results['weapon_score'], 4),
                        'loitering_score': round(results['loitering_score'], 4),
                        'weapon_present': results['weapon_present'],
                        'weapon_type': weapon_type,
                        'loitering_detected': results['loitering_detected'],
                        'loitering_suppressed': results.get('suppress_loitering', False),
                        'identity_name': results.get('identity_name', 'unknown'),
                        'identity_confidence': results.get('identity_confidence', 0.0),
                        'is_known_family': results.get('is_known_family', False),
                        'face_detected': results.get('face_detected', False),
                        'escalation_factor': round(results.get('escalation_factor', 1.0), 4),
                        'temporal_consistency': round(results.get('temporal_consistency', 0.0), 4),
                        'threat_trend': round(results.get('threat_trend', 0.0), 4),
                        'interacting_with': results.get('interacting_with', []),
                        'risk_factors': results.get('risk_factors', []),
                        'alerts': results.get('alerts', [])
                    })
                
                timeline.append(timeline_entry)
                # ===== FPS Calculation =====
                t_frame = time.time() - t_start_frame
                frame_times.append(t_frame)
                fps_disp = 1.0 / max(t_frame, 1e-6)
                
                pbar.update(1)
        
        cap.release()
        writer.release()
        
        # ===== Finalize Video Output =====
        # Rename temp video to output (skip ffmpeg re-encoding if not available)
        if temp_video.exists():
            try:
                # Try ffmpeg re-encoding for better compression if available
                if self.verbose:
                    print('[Pipeline] Finalizing video...')
                subprocess.run([
                    'ffmpeg', '-y', '-loglevel', 'error', '-i', str(temp_video),
                    '-vcodec', 'libx264', '-crf', '22', '-preset', 'fast', str(output_video),
                ], check=True)
                temp_video.unlink()
            except (FileNotFoundError, subprocess.CalledProcessError):
                # Fallback: just rename temp file
                if output_video.exists():
                    output_video.unlink()
                shutil.move(str(temp_video), str(output_video))
        
        if self.verbose:
            print(f'[Pipeline] Video saved: {output_video}')
        
        # ===== Save JSON Timeline =====
        if export_json:
            json_path = output_dir / 'output.json'
            with open(json_path, 'w') as f:
                json.dump(timeline, f, indent=2)
            if self.verbose:
                print(f'[Pipeline] Timeline saved: {json_path}')
        
        # ===== Generate Visualization =====
        timeline_viz_path = output_dir / 'output.png'
        frames_data = [t['frame'] for t in timeline]
        raw_probs = [t['violence']['raw'] for t in timeline]
        smooth_probs = [t['violence']['smooth'] for t in timeline]
        
        create_timeline_visualization(
            frames_data, raw_probs, smooth_probs, timeline,
            self.violence_detector.violence_threshold,
            self.violence_detector.warning_threshold,
            str(timeline_viz_path)
        )
        
        if self.verbose:
            print(f'[Pipeline] Visualization saved: {timeline_viz_path}')
        
        # ===== Generate Statistical Graphs =====
        # Weapon detection graphs
        weapon_stats_path = output_dir / 'weapon_statistics.png'
        generate_weapon_statistics_graph(timeline, str(weapon_stats_path), 
                                        title="Weapon Detection Statistics")
        if self.verbose:
            print(f'[Pipeline] Weapon statistics saved: {weapon_stats_path}')
        
        # Loitering detection graphs
        loitering_stats_path = output_dir / 'loitering_statistics.png'
        generate_loitering_statistics_graph(timeline, str(loitering_stats_path),
                                          title="Loitering Detection Statistics")
        if self.verbose:
            print(f'[Pipeline] Loitering statistics saved: {loitering_stats_path}')
        
        # Combined threat heatmap
        threat_heatmap_path = output_dir / 'threat_composition.png'
        generate_combined_threat_heatmap(timeline, str(threat_heatmap_path),
                                        title="Threat Composition - Violence vs Weapon vs Loitering")
        if self.verbose:
            print(f'[Pipeline] Threat composition heatmap saved: {threat_heatmap_path}')
        
        # Threat level distribution
        threat_dist_path = output_dir / 'threat_level_distribution.png'
        generate_threat_level_distribution(timeline, str(threat_dist_path),
                                          title="Threat Level Distribution Across All Frames")
        if self.verbose:
            print(f'[Pipeline] Threat level distribution saved: {threat_dist_path}')
        
        # Per-person weapon timeline
        person_weapon_path = output_dir / 'person_weapon_timeline.png'
        generate_per_person_weapon_timeline(timeline, str(person_weapon_path),
                                           title="Per-Person Weapon Detection Timeline")
        if self.verbose:
            print(f'[Pipeline] Per-person weapon timeline saved: {person_weapon_path}')
        
        # ===== Summary Statistics =====
        violence_frames = sum(1 for t in timeline if t['violence']['confirmed'])
        warning_frames = sum(1 for t in timeline 
                           if t['violence']['status'] == 'WARNING')
        alert_frames = sum(1 for t in timeline 
                          if any(p['alerts'] for p in t['persons']))
        known_family_frames = sum(1 for t in timeline
                      if any(p.get('is_known_family', False) for p in t['persons']))
        
        results = {
            'input_video': str(video_source),
            'output_video': str(output_video),
            'timeline_json': str(json_path if export_json else None),
            'timeline_plot': str(timeline_viz_path),
            'graphs': {
                'weapon_statistics': str(weapon_stats_path),
                'loitering_statistics': str(loitering_stats_path),
                'threat_composition': str(threat_heatmap_path),
                'threat_distribution': str(threat_dist_path),
                'person_weapon_timeline': str(person_weapon_path)
            },
            'processing_stats': {
                'total_frames': processed,
                'violence_frames': violence_frames,
                'warning_frames': warning_frames,
                'alert_frames': alert_frames,
                'known_family_frames': known_family_frames,
                'violence_percentage': round(violence_frames / processed * 100, 2) if processed else 0.0,
                'avg_fps': round(len(frame_times) / sum(frame_times) if frame_times else 0, 1),
                'total_time_s': round(sum(frame_times), 2)
            },
            'timeline': timeline
        }
        
        if self.verbose:
            print(f'[Pipeline] Done!')
            print(f'  Frames: {processed}')
            print(f'  Violence: {violence_frames} ({results["processing_stats"]["violence_percentage"]}%)')
            print(f'  Warnings: {warning_frames}')
            print(f'  Output: {output_video}')
        
        return results
