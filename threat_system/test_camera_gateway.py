"""Gateway contract and connection lifecycle checks without Pi/model dependencies."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from camera_gateway import check_gateway, gateway_origin, resolve_gateway
from dashboard import Dashboard


JPEG = b"\xff\xd8camera-test-frame\xff\xd9"


@contextmanager
def fake_gateway(health=None, frame=JPEG, redirect=False):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", "http://example.com/health")
                    self.end_headers()
                    return
                body = json.dumps(health if health is not None else {"status": "ok", "service": "threat-edge-camera"}).encode()
                mime = "application/json"
            elif self.path == "/stream.mjpg":
                body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                mime = "multipart/x-mixed-replace; boundary=frame"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class GatewayTests(unittest.TestCase):
    def test_normalizes_gateway_addresses_and_rejects_unrelated_urls(self):
        for value, expected in [("192.168.1.50", "http://192.168.1.50:8000"),
                                ("raspberrypi.local:8001", "http://raspberrypi.local:8001"),
                                ("http://pi.local:80/stream.mjpg", "http://pi.local:80")]:
            self.assertEqual(gateway_origin(value), expected)
        for value in (None, "", "ftp://pi.local", "http://user:pass@pi.local", "http://pi.local/admin", "http://pi.local/?token=secret", "http://pi.local:0", "http://pi.local:99999", "http://pi.local/#secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gateway_origin(value)

    def test_pins_lan_dns_and_rejects_public_and_link_local_addresses(self):
        def answer(ip):
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 8000))]
        with patch("camera_gateway.socket.getaddrinfo", return_value=answer("192.168.1.50")):
            self.assertEqual(resolve_gateway("http://pi.local:8000"), "http://192.168.1.50:8000")
        for ip in ("8.8.8.8", "169.254.169.254", "0.0.0.0", "224.0.0.1"):
            with patch("camera_gateway.socket.getaddrinfo", return_value=answer(ip)), self.assertRaises(ValueError):
                resolve_gateway("http://pi.local:8000")

    def test_checks_actual_gateway_health_and_mjpeg_contract(self):
        with fake_gateway() as origin:
            self.assertEqual(check_gateway(origin), (origin, JPEG))
        with fake_gateway(health={"status": "ok", "service": "unrelated"}) as origin:
            with self.assertRaisesRegex(ValueError, "not the project's"):
                check_gateway(origin)
        with fake_gateway(frame=b"incomplete") as origin:
            with self.assertRaisesRegex(ValueError, "no complete camera frame"):
                check_gateway(origin)
        with fake_gateway(redirect=True) as origin:
            with self.assertRaises(HTTPError) as error:
                check_gateway(origin)
            self.assertEqual(error.exception.code, 302)
            error.exception.close()

    def test_connect_starts_existing_inference_and_disconnect_cancels_it(self):
        dashboard = Dashboard(demo=True, allow_connect=True)
        started = threading.Event()
        def run(**kwargs):
            self.assertEqual(kwargs["camera_source"], "http://192.168.1.50:8000/stream.mjpg")
            self.assertTrue(kwargs["headless"])
            kwargs["dashboard"].publish({7: {"threat_level": "HIGH", "alerts": ["Test alert"]}})
            started.set()
            kwargs["stop_event"].wait(2)
        pipeline = Mock()
        pipeline.process_realtime.side_effect = run
        with patch("dashboard.check_gateway", return_value=("http://192.168.1.50:8000", JPEG)), patch.object(dashboard, "_load_pipeline", return_value=pipeline):
            dashboard.connect_gateway("192.168.1.50")
            self.assertTrue(started.wait(2))
            self.assertFalse(dashboard.snapshot()["demo"])
            self.assertEqual(dashboard.snapshot()["connection"]["status"], "running")
            self.assertEqual(dashboard.snapshot()["events"][0]["level"], "HIGH")
            with self.assertRaises(RuntimeError):
                dashboard.connect_gateway("192.168.1.51")
            dashboard.disconnect_gateway()
            dashboard._worker.join(timeout=2)
            self.assertFalse(dashboard._worker.is_alive())
            self.assertEqual(dashboard.snapshot()["connection"]["status"], "disconnected")
            self.assertIsNone(dashboard.snapshot()["updated_at"])

    def test_missing_models_keeps_raw_preview_without_claiming_detection(self):
        dashboard = Dashboard(allow_connect=True)
        with patch("dashboard.check_gateway", return_value=("http://192.168.1.50:8000", JPEG)), patch.object(dashboard, "_load_pipeline", side_effect=RuntimeError("Missing model assets")), patch("dashboard.read_gateway_frame", return_value=JPEG):
            dashboard.connect_gateway("192.168.1.50")
            # Acquire the lock after the worker has reached its camera-only loop.
            for _ in range(100):
                if dashboard.snapshot()["connection"]["status"] == "camera_only":
                    break
                threading.Event().wait(0.01)
            state = dashboard.snapshot()
            self.assertEqual(state["connection"]["status"], "camera_only")
            self.assertIn("Missing model assets", state["connection"]["error"])
            self.assertTrue(state["has_frame"])
            self.assertEqual(state["preview_mode"], "camera")
            self.assertIsNone(state["updated_at"])
            self.assertEqual(state["events"], [])
            dashboard.stop()

    def test_failed_gateway_exits_demo_and_reports_failure(self):
        dashboard = Dashboard(demo=True, allow_connect=True)
        with patch("dashboard.check_gateway", side_effect=TimeoutError("Gateway timed out")):
            dashboard.connect_gateway("192.168.1.50")
            dashboard._worker.join(timeout=2)
        state = dashboard.snapshot()
        self.assertEqual(state["connection"]["status"], "error")
        self.assertIn("timed out", state["connection"]["error"])
        dashboard.publish({7: {"threat_level": "HIGH"}}, demo=True)
        self.assertEqual(dashboard.snapshot()["events"], [])
        self.assertFalse(state["demo"])

    def test_control_api_requires_token_origin_and_valid_json(self):
        dashboard = Dashboard(allow_connect=True)
        port = dashboard.start(port=0)
        base = f"http://127.0.0.1:{port}"
        def post(path, data=b"{}", token=None, origin=None):
            headers = {"Content-Type": "application/json"}
            if token is not None:
                headers["X-Sentinel-Token"] = token
            if origin is not None:
                headers["Origin"] = origin
            return urlopen(Request(base + path, data=data, headers=headers), timeout=2)
        try:
            for token, origin, data, code in [(None, base, b"{}", 403),
                    (dashboard.control_token, "http://evil.example", b"{}", 403),
                    (dashboard.control_token, base, b"[]", 400),
                    (dashboard.control_token, base, b"{}", 400)]:
                with self.assertRaises(HTTPError) as error:
                    post("/api/gateway/connect", data, token, origin)
                self.assertEqual(error.exception.code, code)
                error.exception.close()
            with patch.object(dashboard, "connect_gateway") as connect:
                with post("/api/gateway/connect", b'{"address":"192.168.1.50"}', dashboard.control_token, base) as response:
                    self.assertEqual(response.status, 202)
                connect.assert_called_once_with("192.168.1.50")
            with post("/api/gateway/disconnect", token=dashboard.control_token, origin=base) as response:
                self.assertEqual(response.status, 202)
        finally:
            dashboard.stop()

    def test_hosted_frontend_requires_access_key_and_exact_origin(self):
        frontend = "https://sentinel-test.vercel.app"
        with self.assertRaises(ValueError):
            Dashboard(allowed_origins=[frontend])
        dashboard = Dashboard(allow_connect=True, allowed_origins=[frontend], access_token="test-only-key")
        port = dashboard.start(port=0)
        base = f"http://127.0.0.1:{port}"
        try:
            request = Request(base + "/api/state", headers={"Origin": frontend})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 401)
            self.assertEqual(error.exception.headers["Access-Control-Allow-Origin"], frontend)
            error.exception.close()
            request = Request(base + "/api/state", headers={"Origin": frontend, "Authorization": "Bearer test-only-key"})
            with urlopen(request) as response:
                body = response.read()
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], frontend)
                self.assertNotIn(b"test-only-key", body)
            request = Request(base + "/api/state", headers={"Origin": "https://untrusted.vercel.app", "Authorization": "Bearer test-only-key"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
            self.assertNotIn("Access-Control-Allow-Origin", error.exception.headers)
            error.exception.close()
            request = Request(base + "/api/gateway/connect", method="OPTIONS", headers={"Origin": frontend, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type,x-sentinel-token"})
            with urlopen(request) as response:
                self.assertEqual(response.status, 204)
                self.assertIn("Authorization", response.headers["Access-Control-Allow-Headers"])
            request = Request(base + "/api/gateway/disconnect", data=b"{}", headers={"Origin": frontend, "Authorization": "Bearer test-only-key", "Content-Type": "application/json", "X-Sentinel-Token": dashboard.control_token})
            with urlopen(request) as response:
                self.assertEqual(response.status, 202)
        finally:
            dashboard.stop()


if __name__ == "__main__":
    unittest.main()
