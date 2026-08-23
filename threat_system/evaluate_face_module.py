#!/usr/bin/env python3
"""
Standalone evaluation script for the face recognition module.

Runs the FaceIdentityRecognizer independently from the full pipeline so that
detection rate, recognition accuracy, and temporal identity stability can be
assessed on sample CCTV videos before integration.

Usage
-----
    # Evaluate on a video (no known identities enrolled — all should be Unknown)
    python evaluate_face_module.py --video collected_assets/n1.mp4

    # Evaluate with the enrolled face database
    python evaluate_face_module.py --video collected_assets/f1.mp4 --face-db models/faces

    # Evaluate with YuNet-only bounding boxes (no pose tracker needed)
    python evaluate_face_module.py --video collected_assets/loitering.mp4 --face-db models/faces

Output
------
* Console report: detection rate, recognition breakdown, backend info
* Per-frame CSV log: face_eval_<video_name>_<timestamp>.csv
* Summary JSON: face_eval_<video_name>_<timestamp>.json

Design note
-----------
This script does NOT load the YOLO pose model or violence detector.
It uses its own simple person detector (YOLOv8n, lightweight) to get bounding
boxes, then passes them to FaceIdentityRecognizer.update() exactly as the
pipeline would.  This keeps the evaluation self-contained.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from identity.recognizer import FaceIdentityRecognizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_SKIP = 2           # Evaluate every Nth frame (same as pipeline default)
DISPLAY_SCALE = 0.6      # Scale factor for display window
BBOX_COLOR_KNOWN = (50, 200, 50)      # Green
BBOX_COLOR_UNKNOWN = (50, 50, 220)    # Red-ish
BBOX_COLOR_FACE = (230, 160, 30)      # Orange — face box


# ===========================================================================
# Simple person detector using YOLOv8n (no pose required)
# ===========================================================================

class SimplePeopleDetector:
    """Wraps YOLO nano to get person bounding boxes without pose estimation.

    Falls back to a full-frame single-person bbox if YOLO is not available.
    """

    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu',
                 conf: float = 0.35):
        self._yolo = None
        self._device = device
        self._conf = conf
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(model_path)
            self._yolo.to(device)
        except Exception as exc:
            print(f'[Eval] YOLO load failed ({exc}) — full-frame fallback active')

    def detect(self, frame) -> dict:
        """Return {fake_tid: [x1,y1,x2,y2]} for each person detected."""
        if self._yolo is None:
            h, w = frame.shape[:2]
            return {0: [0, 0, w, h]}

        results = self._yolo(
            frame, classes=[0], conf=self._conf,
            device=self._device, verbose=False
        )[0]

        bboxes = {}
        if results.boxes is None:
            return bboxes

        boxes = results.boxes.xyxy.cpu().numpy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            bboxes[idx] = [x1, y1, x2, y2]

        return bboxes


# ===========================================================================
# Metrics accumulator
# ===========================================================================

class EvalMetrics:
    def __init__(self):
        self.total_persons = 0          # sum of detections across frames
        self.face_detected_count = 0    # person bbox where face was found
        self.known_count = 0            # recognised as known identity
        self.unknown_count = 0          # unknown
        self.backend_used = 'unknown'
        self.similarities: list[float] = []
        self.confidences: list[float] = []
        self.qualities: list[float] = []

        # Temporal stability per track
        self._track_labels: dict[int, list[str]] = {}   # tid -> [name, name, ...]

    def record(self, tid: int, result: dict):
        self.total_persons += 1
        if result.get('face_detected'):
            self.face_detected_count += 1
        if result.get('identity_state') == 'known':
            self.known_count += 1
        else:
            self.unknown_count += 1

        sim = result.get('identity_similarity', 0.0)
        conf = result.get('identity_confidence', 0.0)
        qual = result.get('identity_quality', 0.0)
        if sim > 0:
            self.similarities.append(sim)
        if conf > 0:
            self.confidences.append(conf)
        if qual > 0:
            self.qualities.append(qual)

        self.backend_used = result.get('face_backend', 'unknown')

        if tid not in self._track_labels:
            self._track_labels[tid] = []
        self._track_labels[tid].append(result.get('identity_name', 'unknown'))

    def detection_rate(self) -> float:
        if self.total_persons == 0:
            return 0.0
        return self.face_detected_count / self.total_persons

    def recognition_rate(self) -> float:
        if self.face_detected_count == 0:
            return 0.0
        return self.known_count / self.total_persons

    def stability_per_track(self) -> dict:
        """Fraction of frames where a track's label did not flip."""
        results = {}
        for tid, labels in self._track_labels.items():
            if len(labels) < 2:
                results[tid] = {'frames': len(labels), 'stability': 1.0, 'dominant': labels[0]}
                continue
            dominant = max(set(labels), key=labels.count)
            same = sum(1 for l in labels if l == dominant)
            results[tid] = {
                'frames': len(labels),
                'stability': round(same / len(labels), 4),
                'dominant': dominant,
            }
        return results

    def summary(self) -> dict:
        return {
            'total_person_detections': self.total_persons,
            'face_detected': self.face_detected_count,
            'detection_rate': round(self.detection_rate(), 4),
            'known_count': self.known_count,
            'unknown_count': self.unknown_count,
            'recognition_rate': round(self.recognition_rate(), 4),
            'avg_similarity': round(float(np.mean(self.similarities)), 4) if self.similarities else 0.0,
            'avg_confidence': round(float(np.mean(self.confidences)), 4) if self.confidences else 0.0,
            'avg_quality': round(float(np.mean(self.qualities)), 4) if self.qualities else 0.0,
            'backend': self.backend_used,
            'track_stability': self.stability_per_track(),
        }


# ===========================================================================
# Visualization helpers
# ===========================================================================

def _draw_result(frame, bbox, result):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    is_known = result.get('identity_state') == 'known'
    color = BBOX_COLOR_KNOWN if is_known else BBOX_COLOR_UNKNOWN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    name = result.get('identity_name', 'unknown')
    conf = result.get('identity_confidence', 0.0)
    sim = result.get('identity_similarity', 0.0)
    qual = result.get('identity_quality', 0.0)
    fd = result.get('face_detected', False)
    backend = result.get('face_backend', '?')

    label1 = f"{name}  [{conf:.2f}]"
    label2 = f"sim:{sim:.2f}  q:{qual:.2f}  fd:{'Y' if fd else 'N'}  [{backend}]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    th = 1
    (tw1, lh1), _ = cv2.getTextSize(label1, font, scale, th)
    (tw2, lh2), _ = cv2.getTextSize(label2, font, scale, th)
    bg_x2 = max(x1 + tw1, x1 + tw2) + 4

    cv2.rectangle(frame, (x1, y1 - lh1 - lh2 - 8), (bg_x2, y1), (30, 30, 30), -1)
    cv2.putText(frame, label1, (x1 + 2, y1 - lh2 - 6), font, scale, color, th, cv2.LINE_AA)
    cv2.putText(frame, label2, (x1 + 2, y1 - 3), font, scale, (200, 200, 200), th, cv2.LINE_AA)

    # Draw face bbox if available
    fb = result.get('face_bbox')
    if fb is not None:
        fx1, fy1, fx2, fy2 = [int(v) for v in fb]
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), BBOX_COLOR_FACE, 1)


def _draw_hud(frame, frame_idx, fps, metrics: EvalMetrics):
    h, w = frame.shape[:2]
    dr = metrics.detection_rate()
    rr = metrics.recognition_rate()
    lines = [
        f"Frame: {frame_idx}   FPS: {fps:.1f}",
        f"Backend: {metrics.backend_used}",
        f"DetRate: {dr:.1%}   RecRate: {rr:.1%}",
        f"Known: {metrics.known_count}   Unknown: {metrics.unknown_count}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    th = 1
    y = 18
    for line in lines:
        cv2.putText(frame, line, (6, y), font, scale, (220, 220, 220), th, cv2.LINE_AA)
        y += 18


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def evaluate(video_path: Path, face_db: Path, output_dir: Path,
             show: bool = False, no_yolo: bool = False) -> dict:

    print(f"[Eval] Video  : {video_path}")
    print(f"[Eval] FaceDB : {face_db}")
    print(f"[Eval] Output : {output_dir}")

    # ---- Open video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Eval] {W}x{H} @ {fps_src:.1f}fps  total={total}")

    # ---- Initialise modules ----
    recognizer = FaceIdentityRecognizer(database_dir=face_db, verbose=True)
    detector = SimplePeopleDetector(device='cpu') if not no_yolo else None

    metrics = EvalMetrics()
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"face_eval_{video_path.stem}_{ts}.csv"
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'frame', 'tid', 'identity_name', 'identity_confidence', 'identity_similarity',
        'identity_quality', 'face_detected', 'identity_state', 'face_backend',
    ])
    writer.writeheader()

    frame_idx = 0
    processed = 0
    t_prev = time.time()
    disp_fps = 0.0

    # ---- Annotated video writer ----
    vid_path = output_dir / f"face_eval_{video_path.stem}_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(str(vid_path), fourcc, fps_src / FRAME_SKIP, (W, H))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        processed += 1
        t_start = time.time()

        # ---- Person detection ----
        if detector is not None:
            person_bboxes = detector.detect(frame)
        else:
            # Full-frame single-person fallback
            person_bboxes = {0: [0, 0, W, H]}

        # ---- Face recognition ----
        results = recognizer.update(frame, person_bboxes)

        # ---- Accumulate metrics ----
        for tid, result in results.items():
            metrics.record(tid, result)
            writer.writerow({
                'frame': processed,
                'tid': tid,
                'identity_name': result.get('identity_name', 'unknown'),
                'identity_confidence': result.get('identity_confidence', 0.0),
                'identity_similarity': result.get('identity_similarity', 0.0),
                'identity_quality': result.get('identity_quality', 0.0),
                'face_detected': result.get('face_detected', False),
                'identity_state': result.get('identity_state', 'unknown'),
                'face_backend': result.get('face_backend', 'unknown'),
            })

        # ---- Annotate frame ----
        for tid, result in results.items():
            if tid in person_bboxes:
                _draw_result(frame, person_bboxes[tid], result)

        disp_fps = 0.7 * disp_fps + 0.3 * (1.0 / max(time.time() - t_start, 1e-6))
        _draw_hud(frame, processed, disp_fps, metrics)
        vid_writer.write(frame)

        if show:
            small = cv2.resize(frame, (int(W * DISPLAY_SCALE), int(H * DISPLAY_SCALE)))
            cv2.imshow('Face Eval', small)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if processed % 50 == 0:
            print(f"  [Eval] frame={processed}  "
                  f"DetRate={metrics.detection_rate():.1%}  "
                  f"Known={metrics.known_count}  Unknown={metrics.unknown_count}")

    cap.release()
    vid_writer.release()
    csv_file.close()
    if show:
        cv2.destroyAllWindows()

    # ---- Summary ----
    summary = metrics.summary()
    summary['video'] = str(video_path)
    summary['frames_processed'] = processed
    summary['fps_effective'] = round(fps_src / FRAME_SKIP, 2)
    summary['csv_log'] = str(csv_path)
    summary['annotated_video'] = str(vid_path)

    json_path = output_dir / f"face_eval_{video_path.stem}_{ts}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FACE MODULE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Backend          : {summary['backend']}")
    print(f"  Frames processed : {summary['frames_processed']}")
    print(f"  Person detections: {summary['total_person_detections']}")
    print(f"  Face detected    : {summary['face_detected']}  "
          f"({summary['detection_rate']:.1%})")
    print(f"  Known identity   : {summary['known_count']}  "
          f"(of all detections)")
    print(f"  Unknown          : {summary['unknown_count']}")
    print(f"  Avg similarity   : {summary['avg_similarity']:.4f}")
    print(f"  Avg confidence   : {summary['avg_confidence']:.4f}")
    print(f"  Avg quality      : {summary['avg_quality']:.4f}")
    print(f"\n  Track stability:")
    for tid, ts_info in summary['track_stability'].items():
        print(f"    Track {tid}: {ts_info['frames']} frames  "
              f"stability={ts_info['stability']:.1%}  "
              f"dominant={ts_info['dominant']}")
    print(f"\n  CSV log     : {csv_path}")
    print(f"  JSON summary: {json_path}")
    print(f"  Video       : {vid_path}")
    print("=" * 60)

    return summary


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Standalone evaluation of the face recognition module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run on a video (unknown persons expected)
  python evaluate_face_module.py --video collected_assets/n1.mp4

  # Run with enrolled database
  python evaluate_face_module.py --video collected_assets/f1.mp4 --face-db models/faces

  # Show live display window
  python evaluate_face_module.py --video collected_assets/loitering.mp4 --show

  # Skip YOLO detection (full-frame single-person mode)
  python evaluate_face_module.py --video collected_assets/n1.mp4 --no-yolo
        '''
    )
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--face-db', default='models/faces',
                        help='Face database dir (default: models/faces)')
    parser.add_argument('--output', default='results/face_eval',
                        help='Output directory (default: results/face_eval)')
    parser.add_argument('--show', action='store_true',
                        help='Show live annotated display')
    parser.add_argument('--no-yolo', action='store_true',
                        help='Skip YOLO person detection (full-frame mode)')

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f'[ERROR] Video not found: {video_path}')
        return 1

    face_db = Path(args.face_db)
    output_dir = Path(args.output)

    try:
        evaluate(video_path, face_db, output_dir, show=args.show, no_yolo=args.no_yolo)
    except KeyboardInterrupt:
        print('\n[Interrupted]')
    except Exception as exc:
        print(f'[ERROR] {exc}')
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
