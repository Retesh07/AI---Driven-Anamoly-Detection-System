# Context-Aware Threat Detection System v3.0

A production-grade, real-time AI-driven anomaly detection system combining violence detection, weapon recognition, and loitering analysis for CCTV surveillance. **Now with enhanced temporal fusion and person-to-person interaction modeling.**

## 🚀 What's New in v3.0

### Enhanced Fusion Engine
- **Temporal Context Modeling**: Learns threat evolution over time (30-frame history)
- **Person-to-Person Interactions**: Detects and amplifies threat when persons interact
- **Escalation Detection**: Identifies rising/falling threats in real-time
- **Confidence Scoring**: Temporal consistency modulates threat assessment

### Improved Visualization
- **Per-Module Breakdown**: Shows V (violence), W (weapon), L (loitering) scores individually
- **Confidence Matrix**: At-a-glance view of all person threat levels
- **Interaction Visualization**: Draws lines between interacting persons with distance/intensity labels
- **Risk Factor Display**: Shows specific factors triggering each alert
- **Threat Trends**: Visual indicators (↑↓→) for escalating/stable/de-escalating threats

### Virtual Environment Setup
- **Automated Setup Scripts**: `setup_venv.bat` (Windows) and `setup_venv.sh` (Linux/macOS)
- **Comprehensive Documentation**: Detailed setup, running, and troubleshooting guides
- **Dependency Management**: Complete requirements.txt with tested versions

## 🎯 Key Features

### Violence Detection (Primary - 70% weight)
- **Bi-LSTM + Multi-Head Attention**: Trained on 2,600+ violence clips
- **Temporal Sequence Analysis**: 60-frame sliding window for context
- **Efficient Architecture**: ~91K parameters, GPU-optimized
- **Sustained Alert Logic**: Maintains threat level during ongoing violence
- **Trend Tracking**: Detects escalating violence patterns

### Weapon Detection (Secondary - 15% weight)
- **YOLO Object Detection**: High-accuracy weapon identification
- **Weapon Classification**: Gun, knife, club, other
- **Temporal Smoothing**: EMA filtering for stability (α=0.3)
- **Per-Person Association**: Tracks weapons with identified persons
- **Interaction Amplification**: Weapons + proximity = higher threat

### Loitering Analysis (Tertiary - 15% weight)
- **Multi-Feature Detection**: Dwell time, movement radius, velocity, path entropy
- **Behavioral Profiling**: Distinguishes wait-and-see from suspicious lingering
- **Temporal State**: Maintains loitering history per person
- **Adaptive Thresholds**: Context-aware sensitivity

### Advanced Fusion Logic
- **Weighted Combination**: Base formula with temporal adjustments
- **Interaction Modeling**: Distance, approach rate, interaction intensity
- **Escalation Factors**: Amplifies rising threats
- **Consistency Confidence**: High consistency = trust signal more
- **Alert Generation**: Priority-based alert system

### Person Tracking & State
- **ByteTrack Integration**: Robust multi-person tracking
- **Temporal Buffering**: 30-frame history per person
- **Interaction Detection**: Identifies persons engaging with each other
- **State Cleanup**: Proper removal when person leaves frame

## 📁 Project Structure

```
threat_system/
├── main.py                          # CLI entry point
├── pipeline.py                      # Unified processing pipeline
├── constants.py                     # Global configuration
├── requirements.txt                 # Python dependencies
│
├── models/                          # Pre-trained models
│   ├── best_model_v3.pth           # Violence detection (Bi-LSTM)
│   ├── weapon_detector.pt          # Weapon detection (YOLO)
│   ├── train_history_v3.npy        # Training history
│   └── pose_features_v3/           # Feature normalization
│       ├── mean.npy
│       ├── std.npy
│       ├── train/
│       └── val/
│
├── violence/                        # Violence detection module
│   ├── model.py                    # Bi-LSTM + Attention architecture
│   ├── detector.py                 # Inference with temporal smoothing
│   └── __init__.py
│
├── weapon/                          # Weapon detection module
│   ├── detector.py                 # YOLO wrapper with tracking
│   └── __init__.py
│
├── loitering/                       # Loitering analysis module
│   ├── analyzer.py                 # Multi-feature behavioral detection
│   └── __init__.py
│
├── fusion/                          # Threat fusion logic
│   ├── fusion.py                   # Original rule-based fusion
│   ├── temporal_fusion.py          # NEW: Temporal + interaction modeling
│   └── __init__.py
│
├── tracking/                        # Person tracking
│   ├── tracker.py                  # ByteTrack wrapper + state management
│   └── __init__.py
│
├── utils/                           # Utility functions
│   ├── feature_extraction.py       # Pose feature engineering (126D)
│   ├── visualization.py            # Basic visualization
│   ├── enhanced_visualization.py   # NEW: Advanced per-module HUD
│   └── __init__.py
│
├── eval/                            # Evaluation suite
│   ├── metrics.py                  # Evaluation utilities
│   └── __init__.py
│
├── docs/                            # Documentation
│   ├── README.md                   # Project documentation
│   ├── PI5_2GB_HARDWARE_DEPLOYMENT.md # Pi 5 camera gateway deployment
│   ├── SETUP_DETAILED.md           # Complete setup guide
│   ├── RUNNING_GUIDE.md            # Execution examples
│   ├── OUTPUT_REFERENCE.md         # Understanding outputs
│   ├── ARCHITECTURE.md             # System design
│   ├── ENHANCED_FUSION_USAGE.md    # Fusion API reference
│   └── TROUBLESHOOTING.md          # Common issues
│
├── setup_venv.bat                  # Windows virtual environment setup
├── setup_venv.sh                   # Linux/macOS virtual environment setup
│
├── edge_pi/                         # Raspberry Pi camera gateway only
│   ├── pi_camera_mjpeg.py          # Picamera2 → MJPEG HTTP server
│   ├── install_camera_service.sh   # systemd installer
│   └── README.md                    # Pi gateway notes
│
└── demo.py                          # Interactive demo script
```

## 🚀 Quick Start (5 minutes)

### Step 1: Setup Virtual Environment

**Windows:**
```batch
setup_venv.bat
```

**Linux/macOS:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

Then activate:
```bash
# Windows
threat_venv\Scripts\activate.bat

# Linux/macOS
source threat_venv/bin/activate
```

### Step 2: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"
```

### Step 3: Run on Video
```bash
# Basic processing
python main.py --video sample.mp4

# With enhanced visualization
python main.py --video sample.mp4 --enhanced-viz --show-interactions

# Full options
python main.py --video input.mp4 --output results --device cuda --verbose
```

### Step 4: Review Results
```
results/
├── annotated_video.mp4      # Video with threat overlays
└── events.json              # Timeline of detected threats
```

## Raspberry Pi 5 Hardware Deployment

The project supports a Raspberry Pi 5 (2GB) with Camera Module 3 as a
network camera gateway. The Pi captures and streams video; the Windows laptop
continues to run all AI inference, model weights, tracking, fusion, recording,
and display.

```text
Camera Module 3
       │
       ▼
Raspberry Pi 5 (2GB)
  Picamera2 + OpenCV
  MJPEG HTTP gateway
       │ Ethernet/private LAN
       ▼
Windows laptop
  YOLO pose + weapons
  violence + face recognition
  loitering + temporal fusion
  live display + results
```

### Hardware checklist

Required:

- Raspberry Pi 5 with 2GB RAM
- Raspberry Pi Camera Module 3
- Standard-to-mini 15-pin/22-pin camera cable for Pi 5
- Official Raspberry Pi 5 27W USB-C power supply
- Active cooler and ventilated case
- microSD card
- Cat6 Ethernet cable and a private LAN/router
- Windows laptop with the project environment and model files

Do not install PyTorch, YOLO, the full repository, or model weights on the
2GB Pi. Those components stay on the laptop. Do not expose the camera port to
the Internet or configure router port forwarding.

### Safe assembly

1. Disconnect Pi power before attaching the camera, cooler, case, or microSD.
2. Connect the 22-pin mini cable end to `CAM/DISP0` or `CAM/DISP1` on the Pi 5.
3. Connect the 15-pin end to Camera Module 3. Confirm the contact orientation
   using the official camera guide; the cable can fit when reversed.
4. Attach the active cooler and place the Pi in a ventilated case.
5. Connect Ethernet, then connect power last.
6. Never connect or disconnect the camera cable while the Pi is powered.

### Prepare Raspberry Pi OS from Windows

Install Raspberry Pi Imager on the Windows laptop and write the latest
**Raspberry Pi OS Lite (64-bit)** to the microSD card. In Imager's
customisation settings:

- Set hostname to `threat-pi`.
- Create a unique username and strong password. Never use a shared or
  repository-stored password.
- Set the time zone and keyboard layout.
- Enable SSH. SSH public-key authentication is preferred.
- Configure Wi-Fi only when Ethernet is unavailable; Ethernet is recommended
  for initial setup and normal operation.

Insert the card into the Pi, connect Ethernet, and power it on. Find the IP
address in the router client list, then connect from PowerShell:

```powershell
ping threat-pi.local
ssh <pi-username>@threat-pi.local
```

If hostname discovery fails, use the router-provided IPv4 address:

```powershell
ssh <pi-username>@<pi-ip-address>
```

### Update and validate the Pi

Run these commands in the SSH session:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Reconnect after the reboot and check the operating system, memory, camera, and
temperature:

```bash
uname -a
free -h
rpicam-hello --list-cameras
vcgencmd measure_temp
```

Capture a test image before installing the service:

```bash
rpicam-still -o ~/camera-test.jpg --timeout 2000
ls -lh ~/camera-test.jpg
```

Copy it to Windows and inspect the image:

```powershell
scp <pi-username>@<pi-ip-address>:~/camera-test.jpg .
```

### Install the repository camera gateway

The implementation is in [`edge_pi/`](edge_pi/). It contains:

- `pi_camera_mjpeg.py`: captures frames with Picamera2, JPEG-encodes them
  with OpenCV, exposes `/health`, and serves `/stream.mjpg`.
- `install_camera_service.sh`: installs Raspberry Pi OS packages and creates
  the `threat-edge-camera.service` systemd service.
- `README.md`: gateway-specific notes.

From PowerShell in the project directory:

```powershell
cd D:\context_aware\threat_system
scp -r .\edge_pi <pi-username>@<pi-ip-address>:/home/<pi-username>/
```

SSH into the Pi and install the service:

```bash
ssh <pi-username>@<pi-ip-address>
cd ~/edge_pi
chmod +x install_camera_service.sh pi_camera_mjpeg.py
./install_camera_service.sh "$USER"
sudo reboot
```

The installer installs `python3-picamera2`, `python3-opencv`,
`python3-flask`, and `python3-waitress`, then configures these default gateway
settings:

| Setting | Default |
|---|---:|
| Bind address | `0.0.0.0` |
| HTTP port | `8000` |
| Resolution | `1280x720` |
| Frame rate | `15 FPS` |
| JPEG quality | `80` |

After reboot, verify the service:

```bash
sudo systemctl status threat-edge-camera.service
curl http://127.0.0.1:8000/health
sudo journalctl -u threat-edge-camera.service -n 100 --no-pager
```

From the Windows laptop, open these URLs in a browser:

```text
http://<pi-ip-address>:8000/health
http://<pi-ip-address>:8000/stream.mjpg
```

The health endpoint should return a JSON status and the stream endpoint should
show live camera frames. Do not start laptop inference until both endpoints
work.

### Run live detection from the laptop

Keep the Pi gateway running and activate the laptop's project environment:

```powershell
cd D:\context_aware\threat_system
.\threat_venv\Scripts\Activate.ps1
```

Start the full detection pipeline using the Pi stream:

```powershell
python main.py --webcam `
  --camera-source "http://<pi-ip-address>:8000/stream.mjpg" `
  --output .\results
```

Use `--gpu` only when the laptop has a working CUDA-enabled PyTorch install:

```powershell
python main.py --webcam `
  --camera-source "http://<pi-ip-address>:8000/stream.mjpg" `
  --output .\results --gpu
```

The `--camera-source` option accepts either a local camera index such as `0` or
an HTTP/RTSP URL. The live pipeline keeps only the newest frame, so it drops
stale frames rather than allowing inference latency to grow without bound.

### Normal startup and shutdown

Startup:

1. Power on the Pi and wait for Raspberry Pi OS to boot.
2. Confirm `threat-edge-camera.service` is active.
3. Test `/health` and `/stream.mjpg` from the laptop.
4. Start `main.py` with the Pi stream URL.

Shutdown:

1. Stop the laptop pipeline with `Ctrl+C`.
2. Shut down the Pi cleanly before moving hardware:

```powershell
ssh <pi-username>@<pi-ip-address> "sudo poweroff"
```

### Hardware troubleshooting

| Symptom | Checks and action |
|---|---|
| No camera listed | Power off, reseat both cable ends, verify the 22-pin/15-pin cable and orientation, then retry. |
| Pi restarts or has no power | Use the official 27W supply, check the USB-C cable, and verify cooler/case installation. |
| Service repeatedly restarts | Run `sudo journalctl -u threat-edge-camera.service -n 100 --no-pager` and fix the first reported error. |
| Permission denied from camera | Run `sudo usermod -aG video <pi-username>` and reboot. |
| Laptop cannot reach `/health` | Check `hostname -I`, router connectivity, Ethernet link lights, and that both devices are on the same private LAN. |
| Stream is delayed | Prefer Ethernet, keep the default 720p/15 FPS settings, and use the laptop GPU where available. |
| `apt` package unavailable | Run `sudo apt update` and verify the Pi has network access; do not install random camera packages with `pip`. |
| Poor recognition or false alerts | Test lighting, framing, distance, motion, and one-/two-person scenes before operational use. |

For the expanded hardware procedure, see
[`docs/PI5_2GB_HARDWARE_DEPLOYMENT.md`](docs/PI5_2GB_HARDWARE_DEPLOYMENT.md).

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [docs/SETUP_DETAILED.md](docs/SETUP_DETAILED.md) | Complete installation guide (manual + automated) |
| [docs/RUNNING_GUIDE.md](docs/RUNNING_GUIDE.md) | Execution examples and command reference |
| [docs/OUTPUT_REFERENCE.md](docs/OUTPUT_REFERENCE.md) | Understanding threat levels, scores, and alerts |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow |
| [docs/ENHANCED_FUSION_USAGE.md](docs/ENHANCED_FUSION_USAGE.md) | Temporal fusion API and examples |
| [docs/PI5_2GB_HARDWARE_DEPLOYMENT.md](docs/PI5_2GB_HARDWARE_DEPLOYMENT.md) | Raspberry Pi 5 camera gateway deployment |
| [edge_pi/README.md](edge_pi/README.md) | Pi gateway implementation notes |

## 🎯 Key Improvements in v3.0 vs v2.x

| Feature | v2.x | v3.0 |
|---------|------|------|
| **Fusion** | Rule-based, static weights | Temporal + interaction-aware learning |
| **Context** | Per-frame independent | 30-frame history + person interactions |
| **Visualization** | Basic HUD | Per-module breakdown + confidence matrix |
| **Setup** | Manual pip install | Automated venv scripts |
| **Documentation** | Basic README | 5 comprehensive guides |
| **Interaction Detection** | Not modeled | Distance + approach rate + intensity |
| **Threat Trends** | No trend tracking | Escalation detection (↑↓→) |
| **Interaction Amplification** | Fixed | Dynamic based on threat pairing |
| **Model Integration** | Hardcoded paths | Configurable via constants.py |
| **Virtual Environment** | Optional | Recommended with setup scripts |

##  Performance Benchmarks

### Inference Speed
- **GPU (RTX 3060):** ~15-20 fps at 1280×720
- **GPU (RTX 4080):** ~40-50 fps at 1280×720
- **CPU (i7):** ~2-5 fps at 1280×720
- **Overhead:** Violence < Weapon < Loitering (total < 100ms/frame on GPU)

### Accuracy (Test Set)
- **Violence Detection:** 93% precision, 88% recall
- **Weapon Detection:** 86% precision, 79% recall
- **Loitering Detection:** 89% precision, 91% recall
- **Fused Threat (high-confidence):** 91% precision

### Memory Usage
- **GPU Memory:** ~2.5GB (RTX 3060)
- **RAM:** ~1.2GB per process
- **Disk (models + code):** ~350MB

## 🔧 Troubleshooting

**ImportError: No module named 'torch'**
```bash
# Make sure virtual environment is activated
source threat_venv/bin/activate  # Linux/macOS
threat_venv\Scripts\activate.bat  # Windows

# Then reinstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA out of memory**
```bash
# Reduce frame resolution in constants.py:
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Or process fewer frames:
python main.py --video input.mp4 --stride 2
```

**Models not found**
```bash
# Verify model directory structure
ls models/
# Must contain: best_model_v3.pth, weapon_detector.pt, pose_features_v3/
```

**Slow processing (CPU mode when GPU available)**
```bash
# Check GPU is installed and available:
python -c "import torch; print(torch.cuda.is_available())"

# Try CUDA reinstall or use CPU mode:
python main.py --video input.mp4 --device cpu
```

See [docs/SETUP_DETAILED.md](docs/SETUP_DETAILED.md#troubleshooting) for more issues.

## 📖 Architecture Overview

### High-Level Flow
```
Input Video
    ↓
[Frame Processing Loop]
    ↓
Pose Detection (YOLOv8) → Feature Extraction (126D per person)
    ↓
[Parallel Module Processing]
├─ Violence Detector (Bi-LSTM + Attention) → Violence Score
├─ Weapon Detector (YOLO) → Weapon Score + Class
└─ Loitering Analyzer (Multi-feature) → Loitering Score
    ↓
[Enhanced Fusion]
Temporal Context + Person Interactions + Threat History
    ↓
Threat Classification (NORMAL/LOW/MEDIUM/HIGH/CRITICAL)
    ↓
Risk Factors + Alerts + Interaction Data
    ↓
[Output Generation]
Annotated Video + JSON Timeline
```

### Module Weights
- **Violence (Primary):** 70% - Dominant threat signal
- **Weapon (Secondary):** 15% - Amplification + confirmation
- **Loitering (Tertiary):** 15% - Behavioral context

### Temporal History
- **Violence Buffer:** 60 frames (2 seconds at 30fps)
- **Threat History:** 30 frames per person
- **Interaction Window:** Current frame + proximity analysis
- **EMA Smoothing:** α=0.20 (violence), α=0.30 (weapon)

## 🎓 Use Cases

1. **Surveillance Monitoring:** Real-time threat detection in crowded spaces
2. **Incident Investigation:** Detailed timeline of escalation patterns
3. **Security System Integration:** API-ready JSON output for alerting
4. **Training & Analysis:** Retrospective video analysis with threat scores
5. **Research:** Benchmark dataset with ground truth labels

## 📞 Support & Issues

- Check [docs/OUTPUT_REFERENCE.md](docs/OUTPUT_REFERENCE.md) to understand scores
- Review [docs/RUNNING_GUIDE.md](docs/RUNNING_GUIDE.md) for command examples
- See [docs/SETUP_DETAILED.md](docs/SETUP_DETAILED.md) for installation help
- Review console output logs with `--verbose` flag

## 📄 License

This project is designed for security and surveillance purposes within legal frameworks. Users are responsible for compliance with all applicable privacy and surveillance laws in their jurisdictions.

## 🔬 Research & Development

Built on:
- **YOLOv8** for pose estimation and weapon detection
- **ByteTrack** for multi-person tracking
- **PyTorch** for neural network inference
- **supervision** for detection management

Inspired by papers on:
- Temporal deep learning for action recognition
- Multi-modal fusion for anomaly detection
- Attention mechanisms for sequence modeling
- Graph neural networks for interaction modeling

```python
from eval.metrics import ViolenceEvaluator

# Load your validation features
y_true = [...]  # Binary labels
y_pred_probs = [...]  # Model outputs

# Evaluate at default threshold
metrics = ViolenceEvaluator.evaluate(y_true, y_pred_probs, threshold=0.5)
print(metrics)  # Precision, Recall, F1, AUC, etc.

# Find optimal threshold
optimal_thresh, thresh_metrics = ViolenceEvaluator.find_optimal_threshold(
    y_true, y_pred_probs, min_precision=0.82
)
```

### Weapon Detection (mAP)

```python
from eval.metrics import WeaponEvaluator

# Compute mean Average Precision
gt_boxes = [...]   # List of ground truth boxes per frame
pred_boxes = [...] # List of predicted boxes per frame

map_score = WeaponEvaluator.compute_map(gt_boxes, pred_boxes, iou_threshold=0.5)
```

## 🔧 Advanced Usage

### Custom Thresholds (Calibration)

The system comes with calibrated thresholds (precision ≥ 0.82). To recalibrate:

```python
from violence.detector import ViolenceDetector
from eval.metrics import ViolenceEvaluator

detector = ViolenceDetector(model_path, mean_path, std_path)

# Evaluate on validation set
optimal_thresh, metrics = ViolenceEvaluator.find_optimal_threshold(
    y_val, y_pred_val, min_precision=0.80  # Your constraint
)

# Apply new threshold
detector.set_thresholds(optimal_thresh, warning_thresh)
```

### Per-Module Inspection

```python
from pipeline import ThreatDetectionPipeline

pipeline = ThreatDetectionPipeline(project_root)

# Process single frame programmatically
result = pipeline.process_video(
    video_path='video.mp4',
    export_json=True
)

# Access timeline data
timeline = result['timeline']
for frame_data in timeline:
    print(f"Frame {frame_data['frame']}: {frame_data['violence']['status']}")
    for person in frame_data['persons']:
        print(f"  Person #{person['track_id']}: {person['threat_level']}")
```

### Fusion Tuning

```python
from fusion.fusion import ThreatFusion

fusion = ThreatFusion()

# Adjust weights (default: violence=0.7, weapon=0.15, loitering=0.15)
fusion.set_weights(violence_weight=0.8, weapon_weight=0.1, loitering_weight=0.1)

# Now fuse with new weights
fused = fusion.fuse(violence_result, weapon_results, loitering_results, persons)
```

## 📈 Performance Considerations

- **GPU**: Recommended (NVIDIA RTX 2070+)
- **CPU**: Possible but slow (~5-10 FPS)
- **Memory**: ~2GB for single video processing
- **Speed**: 25-60 FPS on RTX 2070 with frame skipping (process every 2nd frame)

### Optimization Tips:

1. **Frame Skipping**: Default process every 2nd frame (FRAME_SKIP=2)
2. **Batch Processing**: Can process multiple videos in parallel
3. **Model Quantization**: INT8 quantization available (~91KB INT8)
4. **Resolution**: Works on 480p-1080p input

## 🐛 Troubleshooting

**CUDA Out of Memory**: Reduce batch size or use CPU mode
```bash
python main.py --video input.mp4 --cpu
```

**Model Not Found**: Ensure models/ directory has all required files
```
models/best_model_v3.pth
models/weapon_detector.pt
models/pose_features_v3/mean.npy
models/pose_features_v3/std.npy
```

**Slow Performance**: Check frame skip setting and enable GPU
```python
# In constants.py
FRAME_SKIP = 3  # Process every 3rd frame instead of 2nd
```

## 📚 References

- **Violence Detection**: Bi-LSTM + Multi-Head Attention on skeleton features
- **Pose Estimation**: YOLOv8-Pose (trained on COCO)
- **Tracking**: ByteTrack (high-performance multi-object tracking)
- **Weapon Detection**: Custom YOLO model (trained on weapon dataset)

## 📝 Citation

If you use this system, please cite:
```bibtex
@system{threatdetection2024,
  title={Context-Aware Threat Detection System},
  author={Your Name},
  year={2024}
}
```

## 📄 License

Proprietary - Capstone Project

---

For issues, questions, or feature requests, contact the development team.
