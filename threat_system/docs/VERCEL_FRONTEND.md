# Test Sentinel from Vercel

Live frontend: [sentinel-edge-monitor.vercel.app](https://sentinel-edge-monitor.vercel.app).
The current production version was deployed directly from the committed source.
Automatic deployments require granting Vercel's GitHub integration access to this
repository and linking the Vercel project with Root Directory `frontend`.

The frontend can be deployed as static files on Vercel. Detection continues on a
host on the Pi's network. The Pi gateway only exposes `/health` and `/stream.mjpg`;
it does not run the threat models or expose the dashboard API.

```text
Browser on Vercel → HTTPS monitoring host → Pi's private-LAN camera gateway
                        │
                        └─ Existing threat pipeline → alerts + annotated frames
```

## 1. Deploy the frontend

In the Vercel project, set:

- **Root Directory:** `frontend`
- **Framework Preset:** Other
- **Build Command:** empty
- **Output Directory:** `.`

The included `frontend/vercel.json` configures the static output and browser
headers. There is no package installation or server-side Vercel function. Only the
frontend directory is published. Do not deploy the repository root as static files.

## 2. Prepare the colleague's monitoring host

On the laptop/host that can reach the Pi, use the project's existing inference
Python environment and model files. From `threat_system`, set a strong secret
environment variable named `DASHBOARD_ACCESS_TOKEN` privately, then run:

```sh
python3 dashboard.py --allow-origin https://sentinel-edge-monitor.vercel.app
```

Use a different frontend origin if needed, without a trailing slash. Each
preview URL is a different origin; add another `--allow-origin` for each exact
origin that should be allowed. Wildcards are not supported. Remote browser access
requires the access key; requests without it are rejected. Keys are not returned
by the API, saved in the frontend source, or stored in browser storage.

Expose this host's `http://127.0.0.1:8080` through an **HTTPS tunnel or authenticated
HTTPS reverse proxy** that forwards the API routes and Authorization header.
Do not disable certificate verification. The frontend needs the resulting HTTPS
monitoring URL, not the URL of the Pi's camera service. If a tunnel presents an
interstitial page, configure it to pass API traffic without HTML interstitials.

The host only grants CORS access to the configured frontend origins. CORS is not
authentication; the bearer access key protects state, frames and control routes.
Anybody with this key can view that host's camera/alerts and connect or disconnect
its camera, so share it only with the intended tester.

## 3. Connect from the frontend

In **Connect Raspberry Pi**:

1. Set **Monitoring host URL** to the HTTPS tunnel/reverse-proxy origin, such as
   `https://monitor.example.com`.
2. Enter **Host access key**, matching the host's `DASHBOARD_ACCESS_TOKEN`, and
   select **Use monitoring host**. The key stays in page memory; re-enter it after
   a reload. The non-secret URL can be remembered locally.
3. Enter **Camera gateway address**, such as `raspberrypi.local:8000`,
   `192.168.1.50`, or `http://192.168.1.50:8000/stream.mjpg`.
4. Select **Connect & monitor**. The host checks the gateway's exact health-service
   identity and reads a complete MJPEG frame before starting inference.

The Pi address is resolved from the **monitoring host**, so it can stay on the
private network. The Vercel server never proxies the Pi's private IP. A Vercel HTTPS
page cannot use a plain HTTP monitoring URL; the form explains this before sending
requests. No CORS changes are needed on the Pi camera service.

If models/packages are missing, live raw frames remain available as **CAMERA ONLY**
and the interface explicitly says **Detection offline**. No alerts are fabricated.
Restart the host in the configured inference environment, then reconnect to enable
real monitoring. **Disconnect** cancels capture/inference without stopping the
dashboard server. A new connection starts a fresh in-memory alert session.

## Local testing

The **Add a person** form uploads photos to the selected monitoring host's face
database. Deploying a new frontend is not enough: pull the backend update and
restart it to enable `/api/faces/enroll`. Photos persist on the host across runs;
restart the pipeline (or disconnect/reconnect its camera) after enrollment to
load them. See [face enrollment](ALERT_DASHBOARD.md#add-faces-from-the-frontend).

Run `python3 dashboard.py` and open `http://127.0.0.1:8080`. Leave the monitoring
URL empty and use the Pi field directly. For simulated alerts, add `--demo`;
connecting a real gateway exits demo mode.

The current implementation has been checked against a local mock gateway and
cross-origin HTTP test host. Testing with a physical Pi and a deployed HTTPS
frontend still requires that hardware, host environment and deployment.

## Checks

```sh
python3 -m unittest test_dashboard test_camera_gateway -v
node ../frontend/test_app.cjs
```
