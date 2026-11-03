"""Run with: python3 -m unittest test_dashboard -v (no model dependencies)."""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

from dashboard import Dashboard


def detection(level="MEDIUM", message="Suspicious activity"):
    return {"threat_level": level, "alerts": [message], "fused_score": 0.6,
            "risk_factors": ["Sustained aggression"], "identity_name": "unknown"}


class DashboardTests(unittest.TestCase):
    def test_records_transitions_without_per_frame_duplicates(self):
        dashboard = Dashboard()
        self.assertIsNone(dashboard.snapshot()["updated_at"])
        dashboard.publish({7: detection()})
        dashboard.publish({7: detection()})
        self.assertEqual(len(dashboard.snapshot()["events"]), 1)
        dashboard.publish({7: detection("HIGH")}, fps=14.5, frames=300)
        state = dashboard.snapshot()
        self.assertEqual(len(state["events"]), 2)
        self.assertEqual(state["events"][0]["level"], "HIGH")
        self.assertEqual(state["events"][0]["track_id"], "7")
        self.assertEqual(state["level"], "HIGH")
        self.assertEqual(state["fps"], 14.5)
        self.assertEqual(state["frames"], 300)
        dashboard.publish({})
        self.assertEqual(dashboard.snapshot()["level"], "NORMAL")
        dashboard.publish({7: detection("HIGH")})
        self.assertEqual(len(dashboard.snapshot()["events"]), 3)

    def test_same_level_new_signal_generates_an_alert(self):
        dashboard = Dashboard()
        dashboard.publish({7: detection()})
        dashboard.publish({7: detection(message="Weapon detected")})
        self.assertEqual(len(dashboard.snapshot()["events"]), 2)

    def test_history_is_bounded_and_ids_survive_reconnect_without_collisions(self):
        dashboard = Dashboard()
        for i in range(250):
            dashboard.publish({i: detection()})
        events = dashboard.snapshot()["events"]
        self.assertEqual(len(events), 200)
        self.assertEqual(len({event["id"] for event in events}), 200)
        self.assertNotEqual(dashboard.session_id, Dashboard().session_id)

    def test_normal_and_invalid_levels_do_not_create_alerts(self):
        dashboard = Dashboard()
        dashboard.publish({1: detection("NORMAL"), 2: detection("<script>")})
        self.assertEqual(dashboard.snapshot()["events"], [])
        json.dumps(dashboard.snapshot(), allow_nan=False)

    def test_current_identities_update_without_alerts_and_clear_when_people_leave(self):
        dashboard = Dashboard()
        dashboard.publish({7: {**detection("NORMAL"), "identity_name": "Alex"}, 8: detection("NORMAL")})
        state = dashboard.snapshot()
        self.assertEqual([(p["track_id"], p["identity"]) for p in state["persons"]], [("7", "Alex"), ("8", "unknown")])
        self.assertEqual(state["events"], [])
        dashboard.publish({7: {**detection("NORMAL"), "identity_name": "Jordan"}})
        self.assertEqual([p["identity"] for p in dashboard.snapshot()["persons"]], ["Jordan"])
        dashboard.publish({})
        self.assertEqual(dashboard.snapshot()["persons"], [])

    def test_http_serves_ui_state_and_frame_and_rejects_other_paths(self):
        dashboard = Dashboard(source="http://pi:8000/stream.mjpg")
        port = dashboard.start(port=0)
        base = f"http://127.0.0.1:{port}"
        try:
            for path, content_type in [("/", "text/html"), ("/app.js", "text/javascript"), ("/styles.css", "text/css")]:
                with urlopen(base + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(content_type, response.headers["Content-Type"])
                    self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/api/frame.jpg")
            self.assertEqual(error.exception.code, 404)
            error.exception.close()
            with dashboard._lock:
                dashboard._jpeg = b"jpeg-bytes"
            dashboard.publish({7: detection(message="<img src=x onerror=alert(1)>")})
            with urlopen(base + "/api/state") as response:
                state = json.load(response)
                self.assertEqual(state["events"][0]["messages"], ["<img src=x onerror=alert(1)>"])
                self.assertTrue(state["has_frame"])
                self.assertNotIn("Access-Control-Allow-Origin", response.headers)
            with urlopen(base + "/api/frame.jpg?cache=123") as response:
                self.assertEqual(response.read(), b"jpeg-bytes")
            for path in ("/../dashboard.py", "/api/ack", "/index.html"):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + path)
                self.assertEqual(error.exception.code, 404)
                error.exception.close()
        finally:
            dashboard.stop()

    def test_snapshots_remain_consistent_during_publishing(self):
        dashboard = Dashboard()
        writer = threading.Thread(target=lambda: [dashboard.publish({i: detection()}, frames=i) for i in range(100)])
        writer.start()
        while writer.is_alive():
            state = dashboard.snapshot()
            json.dumps(state, allow_nan=False)
            self.assertLessEqual(len(state["events"]), 200)
        writer.join()
        self.assertEqual(dashboard.snapshot()["frames"], 99)

    def test_cli_connects_live_pipeline_and_closes_server_on_failure(self):
        import main
        pipeline = Mock()
        pipeline.process_realtime.side_effect = RuntimeError("camera disconnected")
        dashboard = Mock()
        dashboard.start.return_value = 8080
        factory = Mock(return_value=dashboard)
        argv = ["main.py", "--webcam", "--cpu", "--camera-source", "http://pi:8000/stream.mjpg", "--dashboard", "--headless"]
        with patch("sys.argv", argv), patch.dict("sys.modules", {"pipeline": SimpleNamespace(ThreatDetectionPipeline=Mock(return_value=pipeline))}), patch("dashboard.Dashboard", factory), patch("traceback.print_exc"), patch("builtins.print"):
            self.assertEqual(main.main(), 1)
        factory.assert_called_once_with(source="http://pi:8000/stream.mjpg")
        pipeline.process_realtime.assert_called_once_with(camera_source="http://pi:8000/stream.mjpg", output_dir="./results", violence_threshold=None, warning_threshold=None, dashboard=dashboard, headless=True)
        dashboard.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
