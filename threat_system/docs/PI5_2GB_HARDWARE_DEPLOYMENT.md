# Safe Raspberry Pi 5 (2GB) deployment guide

## What this setup does

Your Raspberry Pi 5 with 2GB RAM is enough for a reliable **camera gateway**.
It captures the Camera Module 3 feed and sends it over your private network to
the laptop. The laptop runs pose, tracking, face recognition, violence, weapon,
loitering, temporal fusion, recording, and the live display.

Do not install the full AI project or its PyTorch models on the Pi. That would
not provide dependable real-time performance on 2GB RAM and would make the
working laptop pipeline harder to maintain.

```text
Camera Module 3 -> Pi 5 (capture + stream) -> Ethernet/router -> laptop (all AI)
```

## Before opening or powering anything

1. Keep the Pi unplugged while attaching the camera, cooler, or microSD card.
2. Work on a dry, non-conductive table. Touch grounded metal before handling
   the camera board or Pi. Do not touch the camera lens or exposed contacts.
3. Use only the supplied official 27W USB-C Pi 5 power supply. Do not use a
   phone charger or a random USB-C laptop charger.
4. Confirm that the microSD card contains Raspberry Pi OS before inserting it.
5. Use Ethernet for the first setup and for normal deployment. Configure Wi-Fi
   only as a fallback.

## Phase 1: Hardware assembly

### 1. Fit the camera cable

1. Leave power disconnected.
2. Find either connector labelled `CAM/DISP0` or `CAM/DISP1` on the Pi 5.
3. Gently lift the connector latch. Do   not force it or pull it completely off.
4. Insert the **22-pin mini end** of the Standard-Mini camera cable evenly.
   Check the official orientation diagram before closing the latch; the contacts
   must face the correct direction. A cable fits physically even when reversed.
5. Connect the 15-pin end to Camera Module 3 in the same careful way and close
   its latch.
6. Remove the protective film from the camera lens, if present.

Raspberry Pi 5 uses a 22-pin mini camera connector. Camera Module 3 uses a
15-pin standard connector, so the Standard-Mini cable is required.

### 2. Fit cooling and case

1. Attach the active cooler only after the camera cable is seated. The cooler
   can block access to the camera connector.
2. Follow the cooler's supplied mounting guide. It should sit flat; never press
   hard on the Pi board.
3. Put the Pi in its ventilated case only after checking the cable is not bent
   sharply or trapped.
4. Insert the microSD card into the underside slot until it clicks.

### 3. Connect the network and power

1. Plug the Cat6 cable from the Pi into your router or switch.
2. Connect power **last**. The Pi boots as soon as power is connected.
3. Do not connect or disconnect the camera cable while power is present.

## Phase 2: Create the Pi operating system

Use a Windows laptop to install the latest **Raspberry Pi OS Lite (64-bit)**
with Raspberry Pi Imager. Lite reduces RAM use and has no unnecessary desktop.

In Imager settings before writing the microSD card:

1. Set a hostname such as `threat-pi`.
2. Create a unique username and a strong password. Do not use `pi` as a
   password and do not share the password in code or chat.
3. Set your time zone and keyboard layout.
4. Enable SSH. Prefer SSH public-key authentication if you already use it;
   password authentication is acceptable only for the first private-LAN setup.
5. Enter Wi-Fi credentials only if Ethernet will not be available.
6. Write the image, safely eject the card, put it into the Pi, and connect power.

Find the Pi IP address in your router's client list or try:

```powershell
ping threat-pi.local
ssh <your-pi-username>@threat-pi.local
```

If the hostname does not resolve on Windows, use the IPv4 address shown by the
router and run `ssh <your-pi-username>@<pi-ip-address>`.

## Phase 3: Update and validate the Pi

After logging in through SSH, run these commands one at a time:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Reconnect after the reboot, then confirm the OS, memory, camera, and heat:

```bash
uname -a
free -h
rpicam-hello --list-cameras
vcgencmd measure_temp
```

If the camera appears, capture one test image:

```bash
rpicam-still -o ~/camera-test.jpg --timeout 2000
ls -lh ~/camera-test.jpg
```

Copy it to the laptop and inspect it before mounting the camera permanently:

```powershell
scp <your-pi-username>@<pi-ip-address>:~/camera-test.jpg .
```

## Phase 4: Install the camera gateway

Copy the `edge_pi` folder from this repository to the Pi. From PowerShell in
the project directory:

```powershell
scp -r edge_pi <your-pi-username>@<pi-ip-address>:/home/<your-pi-username>/
```

Then SSH back into the Pi and run:

```bash
cd ~/edge_pi
chmod +x install_camera_service.sh pi_camera_mjpeg.py
./install_camera_service.sh "$USER"
sudo reboot
```

After reboot, inspect the service:

```bash
sudo systemctl status threat-edge-camera.service
curl http://127.0.0.1:8000/health
sudo journalctl -u threat-edge-camera.service -n 100 --no-pager
```

From a browser on the laptop, open:

```text
http://<pi-ip-address>:8000/health
http://<pi-ip-address>:8000/stream.mjpg
```

The stream must be reachable before starting the threat-detection application.
This service deliberately listens only for local-network use. Do **not** expose
port 8000 to the Internet or configure router port forwarding.

## Phase 5: Connect the existing laptop pipeline

On the laptop, activate the project's normal environment and start live mode:

```powershell
python main.py --webcam --camera-source "http://<pi-ip-address>:8000/stream.mjpg" --output .\results --gpu
```

If OpenCV cannot open the stream, first keep the stream page open in a browser
to prove the Pi side works. Then use the troubleshooting table below.

## Error handling and common mistakes

| Symptom | Likely cause | Safe action |
|---|---|---|
| Pi has no power LED or repeatedly restarts | Incorrect/weak power supply or loose connector | Disconnect power, check the official 27W supply and cable, then retry. |
| `rpicam-hello --list-cameras` shows no camera | Cable reversed, not fully seated, wrong connector, or camera connected while powered | Power off first. Reseat the cable evenly and verify connector orientation from the official guide. |
| Camera works once then fails | Thermal or power issue | Check `vcgencmd measure_temp`, cooler seating, airflow, and power supply. |
| `Permission denied` from camera service | Service user lacks the `video` group | Run `sudo usermod -aG video <username>` and reboot. |
| `threat-edge-camera.service` restarts | Python dependency or camera error | Run `sudo journalctl -u threat-edge-camera.service -n 100 --no-pager`; fix the first error, do not repeatedly power-cycle. |
| Laptop cannot open stream | Wrong Pi IP, different network, firewall, or service stopped | Test `/health` in a laptop browser, check Ethernet/router link lights, and check the systemd status. |
| Stream is slow | Wi-Fi congestion or overly high settings | Use Ethernet. Keep the supplied default 1280x720 at 15 FPS. |
| Threat display is delayed | Laptop inference is slower than incoming frames | The project drops stale frames by design. Keep full model inference on the laptop and close other GPU applications. |
| Face recognition is poor | Backlighting, subject too far away, or motion blur | Reposition the camera away from windows, add even light, and validate at the actual operating distance. |
| `apt` says a package is unavailable | OS repositories were not updated or network is unavailable | Run `sudo apt update`; verify `ping -c 3 raspberrypi.com`; do not install camera libraries using random `pip` instructions. |

## Safe operating checklist

- Keep the Pi in a ventilated case and out of direct sunlight.
- Keep the camera cable strain-free. Never fold it sharply.
- Shut down from SSH with `sudo poweroff` before moving wiring or removing the microSD card.
- Change the Pi password after first login and use SSH keys before permanent deployment.
- Keep the Pi and laptop on a private trusted network.
- Treat HIGH and CRITICAL alerts as prompts for human review, not autonomous decisions.
- Test camera framing, low light, one-person scenes, and two-person scenes before relying on alerts.

## Official resources

- [Raspberry Pi getting started and Imager](https://www.raspberrypi.com/documentation/computers/getting-started.html)
- [Raspberry Pi remote access and SSH](https://www.raspberrypi.com/documentation/computers/remote-access.html)
- [Camera Module connection and Pi 5 cable guidance](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [Raspberry Pi camera software and Picamera2](https://www.raspberrypi.com/documentation/computers/camera_software.html)
- [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
