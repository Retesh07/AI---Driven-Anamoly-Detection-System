#  Realtime AI Driven Context-Aware Anomaly-Detection in Surveillance Camera

An **edge-deployable surveillance intelligence system** that detects aggressive behavior, weapon presence, and suspicious loitering in real time from video, and recognizes known individuals to reduce false alerts. The system fuses four specialized AI branches per tracked person into a single, explainable threat level.

---

##  System Overview

The pipeline processes each frame through four parallel branches, tracks every person over time, fuses their scores, and raises alerts.

Input Frame (camera / RTSP stream)

├── YOLOv8-Pose + ByteTrack → per-person boxes, keypoints, track IDs

↓

├── Violence Branch → Bi-LSTM + Multi-Head Attention → violence score

├── Weapon Branch → YOLO gun/knife detector → weapon score

├── Loitering Branch → dwell/movement/path analysis → loitering score

└── Identity Branch → YuNet + SFace → known vs unknown

↓

Temporal Fusion Engine (weighted score + rule engine)

↓

Threat Level: Normal / Low / Medium / High / Critical

↓

Visualization, Alerts, JSON Logging

---

##  Core Modules

### 1. Capture and Tracking
- Threaded frame capture from webcam or RTSP (always serves the newest frame, never a stale backlog)
- YOLOv8-Pose extracts skeleton keypoints for every person in frame
- ByteTrack assigns a persistent ID to each person across frames

---

### 2. Violence Branch
- Builds a 126-D feature vector (60 pose/kinematic features per person for the two largest tracked people, plus 6 interaction features) over a 60-frame window
- Bi-LSTM (2 layers, 64 hidden units, bidirectional) + 4-head self-attention classify the window
- EMA-smoothed and confirmed only after 5 consecutive threshold crossings, to suppress single-frame noise

---

### 3. Weapon Branch
- YOLO-based detector for guns and knives
- Class-specific confidence and size filters, associated with the nearest tracked person
- 3-frame label agreement + 10-frame persistence to reduce flicker and false positives

---

### 4. Loitering Branch
- Tracks dwell time, movement radius, velocity consistency, and path coverage per person
- Raises a loitering flag after sustained low-movement presence
- Automatically suppressed for recognized/known individuals

---

### 5. Identity Branch
- YuNet face detector (Haar cascade fallback) + SFace embeddings
- Matches against an enrolled face database by cosine similarity
- Classifies each tracked person as:
  - **Known**
  - **Unknown**
- Identity is not used to raise the threat score directly — only to suppress loitering alerts for enrolled individuals

---

##  Temporal Fusion Engine

Combines all branch outputs per tracked person into one fused score:

`fused = 0.70 × violence + 0.15 × weapon + 0.15 × loitering`

A rule engine then escalates this into a final level, with a confirmed weapon or a violence-plus-weapon combination escalating directly to **High** or **Critical**:

- **Normal**
- **Low**
- **Medium**
- **High**
- **Critical**

---

##  Alerting and Output

- Real-time on-screen overlay with per-person threat level and HUD
- JSON timeline logging of detections and threat levels per session
- Live recording and snapshot capture during a session
- Post-session analytics graphs

---

##  Key Features

- Real-time multi-branch inference (violence, weapon, loitering, identity)
- Persistent per-person identity tracking across frames
- Explainable, rule-based threat fusion rather than a single opaque score
- Identity-aware suppression to cut false loitering alerts for known people
- Works on recorded video or a live camera/RTSP stream
- JSON timeline + live recording for after-the-fact review

---

##  Hardware Plan (Phase 3)

A Raspberry Pi 4 (4 GB) cannot run the full AI pipeline in real time, so the system is deployed as a two-tier architecture:

- **Raspberry Pi 4 (4 GB) + Pi camera** — captures video and streams it as RTSP/H.264 over Ethernet
- **Inference host** — runs the full `threat_system` pipeline against that stream, loads all model weights and the face database, and drives display/recording/alerts
- **Future upgrade path:** Raspberry Pi 5 with a Hailo AI accelerator, for on-device inference

---

##  Future Improvements

- On-device inference via Pi 5 + Hailo accelerator (INT8/FP16 quantization)
- Multi-camera cross tracking
- Audio-based threat detection
- Crowd-scale anomaly detection (beyond the current two-person violence limit)
- Calibrated, independently validated alert thresholds

---

##  Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8 (pose estimation + weapon detection)
- supervision (ByteTrack)
- OpenCV (YuNet + SFace, ONNX runtime)
- NumPy, scikit-learn
- Edge gateway: Raspberry Pi 4 (4 GB)

---

##  Repository Structure

```
threat_system/
├── main.py                 # CLI entry point (--video or --webcam)
├── pipeline.py              # ThreatDetectionPipeline orchestration
├── constants.py              # Thresholds and configuration values
├── tracking/                 # Person tracking (ByteTrack wrapper)
├── violence/                 # Violence model + detector
├── weapon/                   # Weapon detector
├── loitering/                 # Loitering analyzer
├── identity/                  # Face recognition (YuNet + SFace)
├── fusion/                    # Temporal fusion engine
├── utils/                     # Feature extraction, visualization
├── enroll_faces.py             # Enroll known faces into the database
└── requirements.txt
member-1/                    # Early person detection/tracking prototypes
VOILENCE.ipynb                # Violence model training notebook
```

---

##  Getting Started

```bash
# install dependencies
pip install -r threat_system/requirements.txt

# enroll known faces (optional, enables identity-aware suppression)
python threat_system/enroll_faces.py

# run on a recorded video
python threat_system/main.py --video path/to/video.mp4

# run live on a webcam or RTSP stream
python threat_system/main.py --webcam
python threat_system/main.py --webcam --camera-source rtsp://<pi-ip>:<port>/<stream>
```

Live-session controls: `q` quit · `space` pause · `s` snapshot · `r` record · `f` fullscreen.

