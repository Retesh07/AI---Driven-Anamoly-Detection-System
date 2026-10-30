# Raspberry Pi 5 (2GB) camera gateway

The Pi runs only camera capture and streaming. The laptop keeps the entire
threat-detection pipeline and every current model weight. This avoids trying to
run two YOLO models, face recognition, and temporal inference on a 2GB Pi.

The gateway supplies a private-LAN MJPEG URL:

```text
http://<pi-ip-address>:8000/stream.mjpg
```

Use that URL as the laptop's `--camera-source` value.

The installer installs the required Raspberry Pi OS packages and registers
`pi_camera_mjpeg.py` as `threat-edge-camera.service`; no project Python virtual
environment or AI model files belong on the Pi.

See [the full deployment guide](../docs/PI5_2GB_HARDWARE_DEPLOYMENT.md) before
powering or connecting the hardware.
