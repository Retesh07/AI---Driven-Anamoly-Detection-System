#!/usr/bin/env python3
"""Serve Raspberry Pi Camera Module frames as a private-LAN MJPEG stream."""

import argparse
import threading

import cv2
from flask import Flask, Response, jsonify
from picamera2 import Picamera2
from waitress import serve


class CameraFrames:
    def __init__(self, width, height, fps, jpeg_quality):
        self.jpeg_quality = jpeg_quality
        self._condition = threading.Condition()
        self._jpeg = None
        self._stopped = False
        self._camera = Picamera2()
        configuration = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": fps},
        )
        self._camera.configure(configuration)
        self._camera.start()

    def capture(self):
        try:
            while not self._stopped:
                frame = self._camera.capture_array("main")
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    continue
                with self._condition:
                    self._jpeg = encoded.tobytes()
                    self._condition.notify_all()
        finally:
            self._camera.stop()
            self._camera.close()

    def wait_for_frame(self):
        with self._condition:
            self._condition.wait_for(lambda: self._jpeg is not None or self._stopped)
            return self._jpeg

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


def create_app(camera):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "threat-edge-camera"})

    @app.get("/stream.mjpg")
    def stream():
        def frames():
            while True:
                jpeg = camera.wait_for_frame()
                if jpeg is None:
                    return
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    camera = CameraFrames(args.width, args.height, args.fps, args.jpeg_quality)
    capture_thread = threading.Thread(target=camera.capture, daemon=True)
    capture_thread.start()
    app = create_app(camera)
    try:
        serve(app, host=args.host, port=args.port)
    finally:
        camera.stop()
        capture_thread.join(timeout=2)


if __name__ == "__main__":
    main()
