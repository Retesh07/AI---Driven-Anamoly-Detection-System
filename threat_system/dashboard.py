#!/usr/bin/env python3
"""Local threat dashboard with Raspberry Pi gateway connection controls."""

import argparse
import hmac
from collections import deque
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

from camera_gateway import check_gateway, gateway_origin, read_gateway_frame


LEVELS = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
ASSETS = Path(__file__).resolve().parent.parent / "frontend"


class Dashboard:
    def __init__(self, source="0", demo=False, allow_connect=False, allowed_origins=(), access_token=""):
        self.session_id = uuid.uuid4().hex
        self.source = str(source)
        self.demo = demo
        self.allow_connect = allow_connect
        self.allowed_origins = set(allowed_origins)
        self.access_token = access_token
        if self.allowed_origins and not access_token:
            raise ValueError("Set DASHBOARD_ACCESS_TOKEN before allowing a hosted frontend.")
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or not parsed.netloc or origin != f"{parsed.scheme}://{parsed.netloc}" or parsed.username is not None:
                raise ValueError("--allow-origin must be an exact frontend origin, without a path or wildcard.")
        self.control_token = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._worker = None
        self._cancel = threading.Event()
        self._connection = {"status": "demo" if demo else "disconnected" if allow_connect else "external",
                            "gateway": None, "error": None, "camera_updated_at": None}
        self._preview_mode = "none"
        # shortcut: keep the latest 200 alerts in memory; add durable storage for audit retention.
        self._events = deque(maxlen=200)
        self._signatures = {}
        self._persons = []
        self._updated_at = None
        self._fps = 0.0
        self._frames = 0
        self._jpeg = None
        self._frame_at = 0.0
        self._counter = 0
        self._server = None
        self._thread = None

    def publish(self, results, frame=None, fps=0, frames=0, demo=False):
        """Receive the pipeline's fused results without exposing model internals."""
        now = time.time()
        persons = []
        for track_id, result in results.items():
            level = str(result.get("threat_level", "NORMAL"))
            if level not in LEVELS:
                level = "NORMAL"
            persons.append({
                "track_id": str(track_id),
                "level": level,
                "score": float(result.get("fused_score", 0)),
                "identity": str(result.get("identity_name", "unknown")),
                "messages": [str(item) for item in result.get("alerts", [])],
                "factors": [str(item) for item in result.get("risk_factors", [])],
                "violence": float(result.get("violence_score", 0)),
                "weapon": float(result.get("weapon_score", 0)),
                "loitering": float(result.get("loitering_score", 0)),
            })
        jpeg = None
        if frame is not None and now - self._frame_at >= 1:
            import cv2
            height, width = frame.shape[:2]
            preview = cv2.resize(frame, (960, max(1, round(height * 960 / width)))) if width > 960 else frame
            ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                jpeg = encoded.tobytes()
                self._frame_at = now
        with self._lock:
            if (demo and not self.demo) or (not demo and self.allow_connect and self._cancel.is_set()):
                return
            signatures = {}
            for person in persons:
                if person["level"] == "NORMAL":
                    continue
                signature = (person["level"], tuple(person["messages"]))
                signatures[person["track_id"]] = signature
                if self._signatures.get(person["track_id"]) != signature:
                    self._counter += 1
                    self._events.appendleft({**person, "id": f"{self.session_id}-{self._counter}", "timestamp": now})
            self._signatures = signatures
            self._persons = persons
            self._updated_at = now
            self._fps = float(fps)
            self._frames = int(frames)
            if jpeg is not None:
                self._jpeg = jpeg
                self._preview_mode = "annotated"
            if self.allow_connect and not self.demo and self._connection["status"] != "stopping":
                self._connection["status"] = "running"

    def connect_gateway(self, address):
        origin = gateway_origin(address)
        with self._control_lock:
            if not self.allow_connect:
                raise RuntimeError("This dashboard is controlled by main.py. Use dashboard.py for browser connection controls.")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("Disconnect the current camera and wait for it to stop before connecting again.")
            with self._lock:
                self.demo = False
                self.session_id = uuid.uuid4().hex
                self.source = origin + "/stream.mjpg"
                self._events.clear()
                self._signatures.clear()
                self._persons = []
                self._updated_at = None
                self._jpeg = None
                self._preview_mode = "none"
                self._frames = 0
                self._fps = 0.0
                self._connection = {"status": "checking", "gateway": origin, "error": None, "camera_updated_at": None}
            self._cancel = threading.Event()
            self._worker = threading.Thread(target=self._run_gateway, args=(origin, self._cancel), daemon=True)
            self._worker.start()

    def disconnect_gateway(self):
        with self._control_lock:
            if not self.allow_connect:
                raise RuntimeError("This dashboard is controlled by main.py.")
            self._cancel.set()
            with self._lock:
                self.demo = False
                self._connection["status"] = "stopping" if self._worker is not None and self._worker.is_alive() else "disconnected"
                self._connection["error"] = None
                self._updated_at = None
                self._jpeg = None
                self._preview_mode = "none"
                self._persons = []

    def _load_pipeline(self):
        from pipeline import ThreatDetectionPipeline
        import constants
        import torch
        root = Path(__file__).parent
        required = list(constants.MODEL_PATHS.values()) + ["models/yolov8s-pose.pt"]
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise RuntimeError("Missing model assets: " + ", ".join(missing))
        return ThreatDetectionPipeline(project_root=root, device="cuda" if torch.cuda.is_available() else "cpu")

    def _run_gateway(self, origin, cancel):
        try:
            origin, jpeg = check_gateway(origin)
            if cancel.is_set():
                return
            with self._lock:
                self.source = origin + "/stream.mjpg"
                self._jpeg = jpeg
                self._preview_mode = "camera"
                self._connection.update(status="starting", gateway=origin, camera_updated_at=time.time())
            try:
                pipeline = self._load_pipeline()
                if cancel.is_set():
                    return
                pipeline.process_realtime(camera_source=self.source, dashboard=self, headless=True, stop_event=cancel)
                if not cancel.is_set():
                    raise RuntimeError("Inference stopped. Reconnect the camera to restart monitoring.")
            except Exception as error:
                if cancel.is_set():
                    return
                with self._lock:
                    self._connection.update(status="camera_only", error=f"Detection unavailable: {error}. Use the host's inference Python environment and model assets.")
                    self._updated_at = None
                    self._persons = []
                    self._preview_mode = "camera"
                # shortcut: camera-only preview opens one stream per second; use a persistent reader for higher preview rates.
                while not cancel.wait(1):
                    jpeg = read_gateway_frame(origin)
                    if cancel.is_set():
                        return
                    with self._lock:
                        self._jpeg = jpeg
                        self._connection["camera_updated_at"] = time.time()
        except Exception as error:
            if not cancel.is_set():
                with self._lock:
                    self._connection.update(status="error", error=f"Could not connect to the Pi gateway: {error}")
                    self._updated_at = None
                    self._jpeg = None
                    self._preview_mode = "none"
        finally:
            if cancel.is_set():
                with self._lock:
                    self._connection["status"] = "disconnected"
                    self._jpeg = None
                    self._preview_mode = "none"
                    self._updated_at = None

    def snapshot(self):
        with self._lock:
            return {
                "session_id": self.session_id,
                "demo": self.demo,
                "source": self.source,
                "updated_at": self._updated_at,
                "fps": round(self._fps, 1),
                "frames": self._frames,
                "level": max((p["level"] for p in self._persons), key=LEVELS.get, default="NORMAL"),
                "persons": list(self._persons),
                "events": list(self._events),
                "has_frame": self._jpeg is not None,
                "preview_mode": self._preview_mode,
                "connection": dict(self._connection),
                "can_connect": self.allow_connect,
                "control_token": self.control_token,
            }

    def start(self, host="127.0.0.1", port=8080):
        self._server = ThreadingHTTPServer((host, port), partial(DashboardHandler, dashboard=self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._server.server_address[1]

    def stop(self):
        if self.allow_connect:
            self.disconnect_gateway()
            if self._worker is not None:
                self._worker.join(timeout=6)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=2)


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, dashboard, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, **kwargs)

    def _origin_allowed(self):
        origin = self.headers.get("Origin")
        return not origin or origin in self.dashboard.allowed_origins or origin in (
            "http://" + self.headers.get("Host", ""), "https://" + self.headers.get("Host", ""))

    def _authorize(self):
        if not self._origin_allowed():
            self.send_error(403, "Frontend origin is not allowed")
            return False
        expected = "Bearer " + self.dashboard.access_token
        if self.dashboard.access_token and not hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected.encode()):
            self.send_error(401, "Monitoring host access key required")
            return False
        return True

    def end_headers(self):
        origin = self.headers.get("Origin")
        if origin in self.dashboard.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self):
        if not self.path.startswith("/api/") or self.headers.get("Origin") not in self.dashboard.allowed_origins:
            self.send_error(403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Sentinel-Token")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/") and not self._authorize():
            return
        if path == "/api/state":
            body = json.dumps(self.dashboard.snapshot(), allow_nan=False).encode()
            content_type = "application/json; charset=utf-8"
        elif path == "/api/frame.jpg":
            with self.dashboard._lock:
                body = self.dashboard._jpeg
            if body is None:
                self.send_error(404, "Waiting for a camera frame")
                return
            content_type = "image/jpeg"
        else:
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/styles.css": ("styles.css", "text/css; charset=utf-8")}
            if path not in assets:
                self.send_error(404)
                return
            name, content_type = assets[path]
            body = (ASSETS / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self' https: http:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if self.path not in ("/api/gateway/connect", "/api/gateway/disconnect"):
            self.send_error(404)
            return
        if not self._authorize():
            return
        token = self.headers.get("X-Sentinel-Token", "")
        if not hmac.compare_digest(token.encode(), self.dashboard.control_token.encode()):
            self.send_error(403, "Use connection controls from this dashboard")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON request of at most 4096 bytes.")
            self.connection.settimeout(5)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            if self.path.endswith("/connect"):
                self.dashboard.connect_gateway(payload.get("address"))
            else:
                self.dashboard.disconnect_gateway()
            body = json.dumps({"status": "accepted"}).encode()
            status = 202
        except (ValueError, UnicodeError) as error:
            body, status = json.dumps({"error": str(error)}).encode(), 400
        except RuntimeError as error:
            body, status = json.dumps({"error": str(error)}).encode(), 409
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--demo", action="store_true", help="Generate clearly labelled simulated alerts; no models required")
    parser.add_argument("--allow-origin", action="append", default=[], help="Exact hosted frontend origin; requires DASHBOARD_ACCESS_TOKEN")
    args = parser.parse_args()
    try:
        dashboard = Dashboard(source="Simulated Raspberry Pi camera" if args.demo else "No Pi connected", demo=args.demo,
                              allow_connect=True, allowed_origins=args.allow_origin, access_token=os.environ.get("DASHBOARD_ACCESS_TOKEN", ""))
    except ValueError as error:
        parser.error(str(error))
    port = dashboard.start(args.host, args.port)
    print(f"Dashboard: http://{args.host}:{port}", flush=True)
    print("Connect the Pi from the browser. Press Ctrl+C to stop.", flush=True)
    tick = 0
    try:
        while True:
            if dashboard.demo:
                # Replay a predictable cycle so connection, deduplication and escalation are reviewable.
                phase = (tick // 12) % 4
                level = ("NORMAL", "MEDIUM", "HIGH", "NORMAL")[phase]
                result = {"threat_level": level, "fused_score": (0.08, 0.48, 0.81, 0.12)[phase],
                          "identity_name": "unknown", "violence_score": 0.79 if phase == 2 else 0.06,
                          "weapon_score": 0.03, "loitering_score": 0.65 if phase == 1 else 0.12,
                          "alerts": [] if level == "NORMAL" else ["Suspicious loitering — increased monitoring" if phase == 1 else "Active violence detected — response recommended"],
                          "risk_factors": ["Prolonged presence"] if phase == 1 else ["Sustained aggression"] if phase == 2 else []}
                dashboard.publish({7: result}, fps=14.8, frames=tick * 15, demo=True)
                tick += 1
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()


if __name__ == "__main__":
    main()
