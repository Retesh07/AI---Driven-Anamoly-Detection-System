# Sentinel alert dashboard

The Raspberry Pi service remains the camera gateway. It serves
`http://<pi-ip>:8000/stream.mjpg`; the **host** runs the existing inference pipeline
and serves this browser interface. No new Python or JavaScript packages are needed
for the dashboard itself.

## Start with the Pi camera

To connect from the browser, run `python3 dashboard.py`, open
[localhost:8080](http://127.0.0.1:8080), and enter the Pi address in **Connect
Raspberry Pi**. The dashboard verifies the camera gateway and starts the existing
host pipeline. If inference dependencies or assets are missing, it displays raw
camera footage with a clear **Detection offline** status. Disconnect is available
without stopping the dashboard server. Connecting begins a fresh alert session.

For a Vercel frontend and a colleague's HTTPS monitoring host, follow
[the hosted frontend setup](VERCEL_FRONTEND.md).

With the existing inference dependencies and model assets installed on the host:

```sh
cd threat_system
python3 main.py --webcam --camera-source http://<pi-ip>:8000/stream.mjpg --dashboard --headless
```

Replace `<pi-ip>` with the Pi's LAN address. Open
[the local dashboard](http://127.0.0.1:8080) in a browser on the **host**.
Use `--cpu` if needed. Omit `--headless` to also keep the existing OpenCV window
and its recording/snapshot controls. Stop the host with Ctrl+C.

The browser receives JSON from `/api/state` approximately once per second and
displays the latest annotated JPEG from `/api/frame.jpg`. JPEG previews are capped
at 960 pixels wide and one encode per second. All inference still uses the existing
pipeline frame size. The dashboard flags inference as stale after ten seconds
without an update and retries failed requests automatically. A camera failure
propagates through the existing pipeline error handling; restart the pipeline
after restoring the camera connection.

The dashboard can also display recorded-video detections with
`python3 main.py --video input.mp4 --dashboard`. Its server closes when processing
and exports finish; use live mode for continuous monitoring.

## Preview without hardware or model weights

```sh
cd threat_system
python3 dashboard.py --demo
```

Open [localhost:8080](http://127.0.0.1:8080). The interface explicitly labels demo
mode and generates simulated normal, loitering and violence states in a repeating
48-second cycle (a state changes every 12 seconds). Demo mode has no camera image.
`dashboard.py` without `--demo` lets an operator connect a real Pi from the browser.
Connecting a Pi exits demo mode immediately.

## Alerts and review

- Uses the existing fusion categories and messages; does not change thresholds.
- Creates an event when a non-normal track appears or its severity/messages change.
  Continuous identical detections are deduplicated. A track returning after it
  disappears or becomes normal creates a new event.
- Retains the latest **200 events in host memory**. Restarting the host clears them.
  This is a monitoring view, not a permanent incident audit log.
- Search matches detection labels, tracks, identity estimates, risk factors and
  messages. Severity and review filters combine.
- Acknowledgements are local to the browser, saved in local storage when available.
  They do not silence detection, clear model state, or synchronize across operators.
- Desktop notifications require opting in on this page, browser permission, and
  HTTPS or `localhost`. New events notify while the page is open; alerts already
  present on initial connection are not replayed as desktop notifications. Demo
  notifications are labelled. Closed-page, SMS and email delivery are not included.

## Another device on the LAN

Add `--dashboard-host 0.0.0.0` and visit `http://<host-lan-ip>:8080`.
Use `--dashboard-port 8081` to change the port. The default binds to localhost.
The server has no built-in TLS. By default it grants no cross-origin API access;
browser camera controls require a same-origin control token. Hosted frontends need
an explicit `--allow-origin` and `DASHBOARD_ACCESS_TOKEN` as documented above.
Only expose it on a trusted private network; remote deployment requires an
authenticated HTTPS reverse proxy or private VPN. LAN HTTP supports in-page alerts,
but browsers generally require HTTPS for desktop notifications.

## Checks

```sh
cd threat_system
python3 -m unittest test_dashboard test_camera_gateway -v
node ../frontend/test_app.cjs
python3 main.py --help
```

These checks cover alert transitions, deduplication, bounded retention, HTTP
routes and headers, concurrency, filtering and acknowledgement selection. Real
camera/inference validation requires a reachable Pi and the existing model files.
