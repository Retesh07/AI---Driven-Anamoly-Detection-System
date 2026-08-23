# Running Guide

## Setup

Run commands from `threat_system/`:

```bash
python -m venv threat_venv
source threat_venv/bin/activate  # Windows: threat_venv\Scripts\activate
python -m pip install -r requirements.txt
```

The trained models are not stored in Git. Add these files before running:

```text
threat_system/models/
  best_model_v3.pth
  weapon_detector.pt
  pose_features_v3/
    mean.npy
    std.npy
```

`yolov8s-pose.pt` is downloaded automatically when it is not already under
`threat_system/models/`.

## Enroll known faces

```bash
python enroll_faces.py --source /path/to/family_photos --output models/faces
```

Use one subfolder per identity, for example `family_photos/Mom/*.jpg`.

## Run

```bash
python main.py --video sample.mp4 --output results --verbose
python main.py --video 'rtsp://camera/stream' --output results
python main.py --webcam --output results
```

Supported options:

```text
--video FILE_OR_STREAM | --webcam
--output DIRECTORY
--violence-threshold VALUE
--warning-threshold VALUE
--face-db DIRECTORY
--gpu | --cpu
--verbose
```

Without `--gpu` or `--cpu`, CUDA is used when available and CPU otherwise.

Outputs are written as `output.mp4`, `output.json`, `output.png`, and the
statistics PNG files under the selected output directory.
