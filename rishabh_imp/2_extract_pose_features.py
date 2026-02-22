#!/usr/bin/env python3
"""
Script to extract pose features from RWF-2000 dataset videos.

This script:
1. Reads the master_splits.csv file
2. Processes each video to extract pose features using YOLOv8 pose estimation
3. Implements advanced feature engineering for action recognition
4. Saves features as .npy files in a structured directory

Enhancements:
- Uses yolov8m-pose for better small object detection
- Implements dense frame sampling for transient actions
- Uses motion-based actor selection for crowded scenes
- Extracts features for top 2 most active people
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from functools import partial
import json
import time
import shutil
from pathlib import Path
from scipy.signal import savgol_filter

# Constants
NUM_JOINTS = 17  # COCO keypoints
NUM_COORDS = 3    # (x, y, confidence)
TARGET_FRAMES = 30  # Number of frames to sample per video
FEATURES_PER_PERSON = 102  # Number of features per person
NUM_PEOPLE = 2     # Number of people to track
# Force CPU usage by default
DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

# Configuration
class Config:
    # Model settings
    MODEL_SIZE = 'm'  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.7
    
    # Smoothing parameters
    SMOOTH_WINDOW_LENGTH = 5  # Window length for Savitzky-Golay filter (must be odd)
    SMOOTH_POLYORDER = 2     # Polynomial order for Savitzky-Golay filter
    
    # Feature extraction
    MIN_KEYPOINT_CONFIDENCE = 0.1
    MIN_PERSON_CONFIDENCE = 0.3
    
    # Directories
    OUTPUT_ROOT = 'data/pose_features_new'
    DEBUG_ROOT = 'debug_videos'  # Directory for debug outputs
    
    # Processing
    NUM_WORKERS = max(1, mp.cpu_count() - 2)  # Leave 2 cores free
    CHUNK_SIZE = 10  # Number of videos to process before updating progress bar
    
    # Debug settings
    DEBUG = False  # Set to True to enable debug mode
    DEBUG_MAX_VIDEOS = 2  # Max number of videos to process in debug mode per class

def load_model():
    """Load the YOLOv8 pose estimation model."""
    model_name = f'yolov8{Config.MODEL_SIZE}-pose.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    model.to(DEVICE)
    return model

def get_video_properties(video_path: str) -> Tuple[int, int, int, int]:
    """Get video properties (width, height, fps, total_frames)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return width, height, fps, total_frames

def sample_frames(video_path: str, num_frames: int = TARGET_FRAMES) -> List[int]:
    """
    Sample a dense block of consecutive frames from the video.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample
        
    Returns:
        List of frame indices to extract
    """
    _, _, _, total_frames = get_video_properties(video_path)
    
    # If video is shorter than target frames, pad with last frame
    if total_frames <= num_frames:
        return list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    
    # Randomly select a starting point for the dense block
    start_frame = np.random.randint(0, total_frames - num_frames + 1)
    return list(range(start_frame, start_frame + num_frames))

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score (0-1) 
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Avoid division by zero
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_motion_energy(current_keypoints: np.ndarray, 
                          previous_keypoints: np.ndarray,
                          current_bbox: np.ndarray,
                          previous_bbox: np.ndarray) -> float:
    """
    Calculate motion energy between current and previous detections.
    
    Args:
        current_keypoints: Current frame keypoints [17, 3]
        previous_keypoints: Previous frame keypoints [17, 3]
        current_bbox: Current bounding box [x1, y1, x2, y2]
        previous_bbox: Previous bounding box [x1, y1, x2, y2]
        
    Returns:
        Motion energy score (higher means more motion)
    """
    # Calculate keypoint displacement
    displacement = current_keypoints[:, :2] - previous_keypoints[:, :2]
    squared_displacement = np.sum(displacement ** 2, axis=1)
    
    # Weight by keypoint confidence
    confidence = current_keypoints[:, 2:3]
    keypoint_motion = np.sum(squared_displacement * confidence)
    
    # Calculate bbox center displacement
    prev_center = np.array([
        (previous_bbox[0] + previous_bbox[2]) / 2,
        (previous_bbox[1] + previous_bbox[3]) / 2
    ])
    curr_center = np.array([
        (current_bbox[0] + current_bbox[2]) / 2,
        (current_bbox[1] + current_bbox[3]) / 2
    ])
    bbox_motion = np.sum((curr_center - prev_center) ** 2)
    
    # Combine scores (weighted sum)
    return float(0.7 * keypoint_motion + 0.3 * bbox_motion)

class PersonTracker:
    """Tracks persons across frames using IoU and motion features."""
    
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        """
        Initialize the person tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching detections
            max_missed: Maximum frames to keep a track without updates
        """
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}  # {track_id: {
                          #     'keypoints': [17, 3],
                          #     'bbox': [4,],
                          #     'confidence': float,
                          #     'missed': int,
                          #     'motion_history': [float, ...],
                          #     'last_seen': int
                          # }}
        self.next_id = 0
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of {
                'keypoints': [17, 3],
                'bbox': [4,],
                'confidence': float
            }
            
        Returns:
            Dictionary of {track_id: detection} with updated motion information
        """
        # Update existing tracks or create new ones
        if not self.tracks:
            # First frame, create new tracks for all detections
            updated_tracks = {}
            for det in detections:
                updated_tracks[self.next_id] = {
                    **det,
                    'missed': 0,
                    'motion_history': [0.0],  # Initialize with zero motion
                    'last_seen': 0
                }
                self.next_id += 1
            self.tracks = updated_tracks
            return updated_tracks
        
        # Calculate IoU between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        det_boxes = np.array([d['bbox'] for d in detections])
        
        # Create cost matrix for Hungarian algorithm
        cost_matrix = np.ones((len(track_ids), len(detections))) * 1000  # Large initial cost
        
        for i, track_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou = calculate_iou(self.tracks[track_id]['bbox'], det['bbox'])
                if iou > self.iou_threshold:
                    # Calculate motion score (lower is better)
                    motion_energy = calculate_motion_energy(
                        det['keypoints'],
                        self.tracks[track_id]['keypoints'],
                        det['bbox'],
                        self.tracks[track_id]['bbox']
                    )
                    cost_matrix[i, j] = 1.0 - iou + motion_energy  # Combine IoU and motion
        
        # Solve assignment problem
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update matched tracks
        updated_tracks = {}
        matched_det_indices = set()
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1.0:  # Valid match
                track_id = track_ids[i]
                det = detections[j]
                
                # Calculate motion energy
                motion_energy = calculate_motion_energy(
                    det['keypoints'],
                    self.tracks[track_id]['keypoints'],
                    det['bbox'],
                    self.tracks[track_id]['bbox']
                )
                
                # Update track
                updated_tracks[track_id] = {
                    **det,
                    'missed': 0,
                    'motion_history': self.tracks[track_id]['motion_history'][-9:] + [motion_energy],
                    'last_seen': 0,
                    'motion_energy': np.mean(self.tracks[track_id]['motion_history'][-5:] + [motion_energy])
                }
                matched_det_indices.add(j)
        
        # Handle unmatched tracks (increment missed counter or remove)
        for track_id in self.tracks:
            if track_id not in updated_tracks:
                if self.tracks[track_id]['missed'] < self.max_missed:
                    updated_tracks[track_id] = {
                        **self.tracks[track_id],
                        'missed': self.tracks[track_id]['missed'] + 1,
                        'last_seen': self.tracks[track_id]['last_seen'] + 1,
                        'motion_energy': 0.0  # No motion if not detected
                    }
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                updated_tracks[self.next_id] = {
                    **det,
                    'missed': 0,
                    'motion_history': [0.0],
                    'last_seen': 0,
                    'motion_energy': 0.0
                }
                self.next_id += 1
        
        self.tracks = updated_tracks
        return updated_tracks

def process_frame(frame: np.ndarray, model, frame_idx: int, 
                 tracker: PersonTracker) -> Dict[int, Dict]:
    """
    Process a single frame to detect poses and track persons.
    
    Args:
        frame: Input frame (BGR)
        model: YOLO pose estimation model
        frame_idx: Frame index
        tracker: PersonTracker instance
        
    Returns:
        Dictionary of {
            track_id: {
                'keypoints': [17, 3] array of keypoints,
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'motion_energy': float
            }
        }
    """
    # Run inference
    results = model(frame, verbose=False, conf=Config.CONFIDENCE_THRESHOLD, 
                   iou=Config.IOU_THRESHOLD, device=DEVICE)
    
    if len(results) == 0 or results[0].keypoints is None:
        return {}
    
    # Get detections
    detections = []
    keypoints = results[0].keypoints.xy.cpu().numpy()  # [N, 17, 2]
    confs = results[0].keypoints.conf.cpu().numpy()    # [N, 17]
    boxes = results[0].boxes.xyxy.cpu().numpy()        # [N, 4]
    scores = results[0].boxes.conf.cpu().numpy()       # [N,]
    
    # Prepare detections
    for i in range(len(scores)):
        if scores[i] < Config.MIN_PERSON_CONFIDENCE:
            continue
            
        detections.append({
            'keypoints': np.concatenate([keypoints[i], confs[i][..., np.newaxis]], axis=-1),
            'bbox': boxes[i],
            'confidence': float(scores[i])
        })
    
    # Update tracks
    current_poses = tracker.update(detections)
    
    # Filter out low confidence and stale tracks, ensure consistent structure
    filtered_poses = {}
    for track_id, track in current_poses.items():
        if track['confidence'] >= Config.MIN_PERSON_CONFIDENCE and track.get('missed', 0) == 0:
            filtered_poses[track_id] = {
                'keypoints': track['keypoints'],
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'motion_energy': track.get('motion_energy', 0.0)  # Default to 0 if missing
            }
    current_poses = filtered_poses
    
    return current_poses

def smooth_keypoints(keypoints_sequence: np.ndarray, window_length: int = 5, 
                     polyorder: int = 2) -> np.ndarray:
    """
    Smooth keypoint trajectories using Savitzky-Golay filter.
    
    Args:
        keypoints_sequence: [T, 17, 3] array of keypoints (x, y, confidence)
        window_length: Length of the filter window (must be odd)
        polyorder: Order of the polynomial used to fit the samples
        
    Returns:
        Smoothed keypoints with same shape as input
    """
    if len(keypoints_sequence) < window_length:
        return keypoints_sequence
    
    # Make window length odd if it's even
    if window_length % 2 == 0:
        window_length += 1
        window_length = min(window_length, len(keypoints_sequence))
    
    smoothed = keypoints_sequence.copy()
    
    # Apply smoothing to x and y coordinates (not confidence)
    for k in range(keypoints_sequence.shape[1]):  # For each keypoint
        for c in range(2):  # Only smooth x (0) and y (1) coordinates
            try:
                smoothed[:, k, c] = savgol_filter(
                    keypoints_sequence[:, k, c],
                    window_length=window_length,
                    polyorder=min(polyorder, window_length - 1),  # polyorder must be < window_length
                    mode='nearest'
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                # Fallback to original values if smoothing fails
                print(f"Smoothing failed for keypoint {k}, coord {c}: {e}")
                smoothed[:, k, c] = keypoints_sequence[:, k, c]
    
    return smoothed

def extract_features_for_person(keypoints: np.ndarray, bbox: np.ndarray, 
                              frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Extract engineered features for a single person.
    
    Args:
        keypoints: [17, 3] array of keypoints (x, y, confidence)
        bbox: [x1, y1, x2, y2] bounding box
        frame_size: (width, height) of the frame
        
    Returns:
        Feature vector of length FEATURES_PER_PERSON
    """
    features = []
    
    # 1. Raw keypoint coordinates (normalized by frame size)
    kp_coords = keypoints[:, :2] / np.array([frame_size[0], frame_size[1]])
    features.extend(kp_coords.flatten())
    
    # 2. Keypoint confidences
    features.extend(keypoints[:, 2])
    
    # 3. Bounding box features (normalized)
    bbox_features = [
        bbox[0] / frame_size[0],  # x1
        bbox[1] / frame_size[1],  # y1
        (bbox[2] - bbox[0]) / frame_size[0],  # width
        (bbox[3] - bbox[1]) / frame_size[1],  # height
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (frame_size[0] * frame_size[1])  # area ratio
    ]
    features.extend(bbox_features)
    
    # 4. Keypoint velocities (if previous frame available)
    # This will be calculated in the main processing loop
    
    # 5. Pairwise distances between keypoints (normalized by frame diagonal)
    frame_diag = np.sqrt(frame_size[0]**2 + frame_size[1]**2)
    for i in range(NUM_JOINTS):
        for j in range(i+1, NUM_JOINTS):
            dist = np.linalg.norm(keypoints[i, :2] - keypoints[j, :2]) / frame_diag
            features.append(dist)
    
    # Ensure we have exactly FEATURES_PER_PERSON features
    if len(features) > FEATURES_PER_PERSON:
        features = features[:FEATURES_PER_PERSON]
    elif len(features) < FEATURES_PER_PERSON:
        features = features + [0.0] * (FEATURES_PER_PERSON - len(features))
    
    return np.array(features, dtype=np.float32)

def draw_skeleton(image, keypoints, bbox, color, track_id=None, motion_energy=None):
    """Draw skeleton and bounding box on the image."""
    # Draw bounding box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw track ID and motion energy
    if track_id is not None:
        label = f"ID: {track_id}"
        if motion_energy is not None:
            label += f" | Motion: {motion_energy:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > Config.MIN_KEYPOINT_CONFIDENCE:
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
    
    # Draw skeleton connections (simplified COCO keypoint connections)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (6, 8),  # Shoulders and arms
        (7, 9), (8, 10),  # Elbows to wrists
        (11, 12), (11, 13), (12, 14),  # Hips and legs
        (13, 15), (14, 16)  # Knees to ankles
    ]
    
    for i, j in connections:
        if (i < len(keypoints) and j < len(keypoints) and 
            keypoints[i, 2] > Config.MIN_KEYPOINT_CONFIDENCE and 
            keypoints[j, 2] > Config.MIN_KEYPOINT_CONFIDENCE):
            pt1 = tuple(map(int, keypoints[i, :2]))
            pt2 = tuple(map(int, keypoints[j, :2]))
            cv2.line(image, pt1, pt2, color, 2)
    
    return image

def process_video(video_path: str, output_dir: str, model, debug: bool = False) -> bool:
    """
    Process a single video to extract pose features.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the features
        model: YOLO pose estimation model
        debug: If True, save debug visualization
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{Path(video_path).stem}.npy")
    
    # Create debug directory if needed
    if debug:
        debug_dir = os.path.join(Config.DEBUG_ROOT, os.path.relpath(output_dir, Config.OUTPUT_ROOT))
        os.makedirs(debug_dir, exist_ok=True)
        debug_video_path = os.path.join(debug_dir, f"{Path(video_path).stem}_debug.mp4")
    else:
        debug_video_path = None
    
    # Skip if already processed and not in debug mode
    if os.path.exists(output_path) and not debug:
        return True
    
    try:
        # Get video properties
        width, height, fps, total_frames = get_video_properties(video_path)
        
        # Sample frames to process
        frame_indices = sample_frames(video_path)
        
        # Initialize variables
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer for debug
        if debug and debug_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                debug_video_path, 
                fourcc, 
                min(fps, 30),  # Cap at 30 FPS for reasonable file size
                (width, height)
            )
        
        tracker = PersonTracker(iou_threshold=0.3, max_missed=3)
        all_features = []
        
        # Initialize storage for all frames' keypoints for smoothing
        all_keypoints = []
        all_poses = []
        
        # First pass: collect all keypoints for smoothing
        for frame_idx in frame_indices:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Process frame with tracking
            current_poses = process_frame(frame, model, frame_idx, tracker)
            all_poses.append(current_poses)
            
            # Store keypoints for smoothing
            frame_keypoints = {}
            for track_id, person in current_poses.items():
                frame_keypoints[track_id] = person['keypoints']
            all_keypoints.append(frame_keypoints)
        
        # Apply smoothing to keypoints
        smoothed_keypoints = []
        for i in range(len(all_keypoints)):
            smoothed_frame = {}
            for track_id in all_keypoints[i].keys():
                # Get keypoint sequence for this track
                track_sequence = []
                for j in range(max(0, i - Config.SMOOTH_WINDOW_LENGTH // 2), 
                             min(len(all_keypoints), i + Config.SMOOTH_WINDOW_LENGTH // 2 + 1)):
                    if track_id in all_keypoints[j]:
                        track_sequence.append(all_keypoints[j][track_id])
                
                if track_sequence:
                    # Stack keypoints across frames [T, 17, 3]
                    track_sequence = np.stack(track_sequence)
                    # Apply smoothing
                    smoothed = smooth_keypoints(
                        track_sequence,
                        window_length=min(Config.SMOOTH_WINDOW_LENGTH, len(track_sequence)),
                        polyorder=Config.SMOOTH_POLYORDER
                    )
                    # Take the center frame (current frame)
                    center_idx = len(track_sequence) // 2
                    smoothed_frame[track_id] = smoothed[center_idx]
            
            smoothed_keypoints.append(smoothed_frame)
        
        # Second pass: extract features with smoothed keypoints
        for frame_idx, (current_poses, smoothed_frame) in enumerate(zip(all_poses, smoothed_keypoints)):
            # Update keypoints with smoothed versions
            for track_id in current_poses:
                if track_id in smoothed_frame:
                    current_poses[track_id]['keypoints'] = smoothed_frame[track_id]
            
            # Create a copy of the frame for visualization
            if debug:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[frame_idx])
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                vis_frame = frame.copy()
            
            # Select top 2 people by motion energy (or confidence if first frame or motion_energy not available)
            if frame_idx == 0 or not all('motion_energy' in person for person in current_poses.values()):
                # Sort by confidence if first frame or if any person is missing motion_energy
                selected = sorted(
                    current_poses.items(), 
                    key=lambda x: x[1].get('confidence', 0), 
                    reverse=True
                )[:NUM_PEOPLE]
                sort_method = 'confidence'
            else:  # Subsequent frames, use motion energy
                selected = sorted(
                    current_poses.items(), 
                    key=lambda x: x[1].get('motion_energy', 0), 
                    reverse=True
                )[:NUM_PEOPLE]
                sort_method = 'motion_energy'
            
            # Draw all detections (with different colors for selected vs not selected)
            if debug:
                # Draw non-selected detections in gray
                non_selected = [p for p in current_poses.items() if p not in selected]
                for track_id, person in non_selected:
                    vis_frame = draw_skeleton(
                        vis_frame, 
                        person['keypoints'], 
                        person['bbox'], 
                        (128, 128, 128),  # Gray for non-selected
                        track_id,
                        person.get('motion_energy')
                    )
                
                # Draw selected detections in green (primary) and blue (secondary)
                for i, (track_id, person) in enumerate(selected):
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for primary, blue for secondary
                    vis_frame = draw_skeleton(
                        vis_frame, 
                        person['keypoints'], 
                        person['bbox'], 
                        color,
                        track_id,
                        person.get('motion_energy')
                    )
                
                # Add frame info
                cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(vis_frame, f"Sort by: {sort_method}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Write frame to debug video
                out.write(vis_frame)
            
            # Extract features for each selected person
            frame_features = []
            for _, person in selected:
                features = extract_features_for_person(
                    person['keypoints'],
                    person['bbox'],
                    (width, height)
                )
                frame_features.append(features)
            
            # Pad with zeros if fewer than NUM_PEOPLE people detected
            while len(frame_features) < NUM_PEOPLE:
                frame_features.append(np.zeros(FEATURES_PER_PERSON, dtype=np.float32))
            
            # Concatenate features for all people
            all_features.append(np.concatenate(frame_features))
        
        # Save features if not in debug mode
        if not debug:
            features_array = np.stack(all_features, axis=0)  # [T, NUM_PEOPLE * FEATURES_PER_PERSON]
            np.save(output_path, features_array)
        
        # Release debug video writer
        if debug and 'out' in locals():
            out.release()
            print(f"Debug video saved to: {debug_video_path}")
            
        return True
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False
    finally:
        if 'cap' in locals():
            cap.release()

# Global model variable for multiprocessing
_global_model = None

def init_worker():
    """Initialize worker with global model."""
    global _global_model
    if _global_model is None:
        _global_model = load_model()

def process_video_wrapper(args):
    """Wrapper function for multiprocessing."""
    video_path, output_dir, debug = args
    return process_video(video_path, output_dir, _global_model, debug)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract pose features from RWF-2000 dataset')
    parser.add_argument('--input_csv', type=str, default='master_splits.csv',
                       help='Path to the master_splits.csv file')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_ROOT,
                       help='Root directory to save the extracted features')
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS,
                       help='Number of worker processes')
    parser.add_argument('--model_size', type=str, default=Config.MODEL_SIZE,
                       choices=['n', 's', 'm', 'l'],
                       help='YOLOv8 model size (nano, small, medium, large)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to visualize detections')
    parser.add_argument('--debug_max_videos', type=int, default=Config.DEBUG_MAX_VIDEOS,
                       help='Max number of videos to process in debug mode per class')
    
    args = parser.parse_args()
    
    # Update config
    Config.MODEL_SIZE = args.model_size
    Config.OUTPUT_ROOT = args.output_dir
    Config.NUM_WORKERS = args.num_workers
    
    # Load the dataset
    print(f"Loading dataset from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # In debug mode, limit the number of videos per class
    if args.debug:
        debug_dfs = []
        for label in [0, 1]:
            label_df = df[df['label'] == label].head(args.debug_max_videos)
            debug_dfs.append(label_df)
        df = pd.concat(debug_dfs, axis=0)
        print(f"Debug mode: Processing {len(df)} videos ({args.debug_max_videos} per class)")
        
        # Create debug directory
        os.makedirs(Config.DEBUG_ROOT, exist_ok=True)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for label in [0, 1]:
            label_name = 'Fight' if label == 1 else 'NonFight'
            os.makedirs(os.path.join(Config.OUTPUT_ROOT, split, label_name), exist_ok=True)
    
    # Load the model (will be loaded in workers)
    model = load_model()
    
    # Process videos in parallel
    print(f"Processing {len(df)} videos with {Config.NUM_WORKERS} workers...")
    
    # Prepare arguments for multiprocessing
    tasks = []
    for _, row in df.iterrows():
        video_path = row['path']
        label = 'Fight' if row['label'] == 1 else 'NonFight'
        output_dir = os.path.join(Config.OUTPUT_ROOT, row['split'], label)
        tasks.append((video_path, output_dir, args.debug))
    
    # Set global model for main process
    global _global_model
    _global_model = model
    
    # Process videos in parallel or sequentially for debugging
    success_count = 0
    
    if args.debug:
        # In debug mode, process sequentially for better error reporting
        print("Debug mode: Processing videos sequentially")
        init_worker()  # Initialize model
        results = []
        for task in tqdm(tasks, desc="Debug processing"):
            results.append(process_video_wrapper(task))
        success_count = sum(results)
    else:
        # Normal parallel processing
        with mp.Pool(
            processes=Config.NUM_WORKERS,
            initializer=init_worker,
            maxtasksperchild=10  # Helps with memory management
        ) as pool:
            results = list(tqdm(
                pool.imap(process_video_wrapper, tasks, chunksize=1),
                total=len(tasks),
                desc="Extracting features"
            ))
            success_count = sum(results)
    
    # Clean up
    _global_model = None
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(df)} videos")
    print(f"Features saved to: {os.path.abspath(Config.OUTPUT_ROOT)}")

if __name__ == '__main__':
    main()
