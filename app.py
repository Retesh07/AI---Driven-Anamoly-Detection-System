from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, flash
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime
import cv2
import json
import os
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'family-loitering-alert-system'

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = BASE_DIR / 'static' / 'uploads' / 'family'
ALERT_FILE = DATA_DIR / 'alerts.json'
FAMILY_FILE = DATA_DIR / 'family.json'

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def load_alerts():
    if not ALERT_FILE.exists():
        return []
    try:
        return json.loads(ALERT_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []


def save_alerts(alerts):
    ALERT_FILE.write_text(json.dumps(alerts, indent=2), encoding='utf-8')


def load_family_members():
    if not FAMILY_FILE.exists():
        return []
    try:
        return json.loads(FAMILY_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []


def save_family_members(members):
    FAMILY_FILE.write_text(json.dumps(members, indent=2), encoding='utf-8')


def add_alert(title, message, severity='high', person='Unknown visitor'):
    alerts = load_alerts()
    alerts.insert(0, {
        'id': int(time.time() * 1000),
        'title': title,
        'message': message,
        'severity': severity,
        'person': person,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    save_alerts(alerts[:25])


class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.tracks = {}
        self.next_track_id = 1
        self.alerted_tracks = set()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            processed = self._process_frame(frame)
            with self.lock:
                self.frame = processed

            time.sleep(0.03)

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        rects = [(x, y, w, h) for (x, y, w, h) in rects if w * h > 1800]
        active = {}

        for x, y, w, h in rects:
            cx = x + w / 2
            cy = y + h / 2
            best_id = None
            best_dist = 999999

            for track_id, track in self.tracks.items():
                dx = track['cx'] - cx
                dy = track['cy'] - cy
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            track_id = best_id if best_id is not None and best_dist < 180 else self.next_track_id
            if best_id is None:
                self.next_track_id += 1

            track = self.tracks.get(track_id, {'dwell': 0.0, 'last_seen': time.time(), 'cx': cx, 'cy': cy})
            if best_id is not None:
                track['dwell'] += 0.1
            else:
                track['dwell'] = 0.1

            track['cx'] = cx
            track['cy'] = cy
            track['bbox'] = (x, y, w, h)
            track['last_seen'] = time.time()
            active[track_id] = track

        now = time.time()
        for track_id in list(self.tracks.keys()):
            if track_id not in active and (now - self.tracks[track_id]['last_seen']) > 1.0:
                self.alerted_tracks.discard(track_id)

        self.tracks = active

        for track_id, track in self.tracks.items():
            x, y, w, h = track['bbox']
            dwell = track['dwell']
            loitering = dwell > 7.0

            if loitering and track_id not in self.alerted_tracks:
                add_alert(
                    title='Loitering Alert',
                    message=f'Unknown person remained in the monitored area for {dwell:.1f}s.',
                    severity='high',
                    person=f'Unknown visitor #{track_id}'
                )
                self.alerted_tracks.add(track_id)

            color = (0, 255, 0)
            if loitering:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'Person {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if loitering:
                cv2.putText(frame, f'Loitering {dwell:.1f}s', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


camera = VideoCamera()


def gen_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', family_members=load_family_members(), alerts=load_alerts()[:5])


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', family_members=load_family_members(), alerts=load_alerts()[:10])


@app.route('/upload_family', methods=['POST'])
def upload_family():
    name = request.form.get('name', '').strip()
    image = request.files.get('image')

    if not name:
        flash('Please enter a family member name.', 'error')
        return redirect(url_for('index'))

    if not image or image.filename == '':
        flash('Please choose an image.', 'error')
        return redirect(url_for('index'))

    safe_name = secure_filename(name).replace(' ', '_')
    file_ext = os.path.splitext(image.filename)[1].lower() or '.jpg'
    saved_name = f'{safe_name}_{int(time.time())}{file_ext}'
    file_path = UPLOAD_DIR / saved_name
    image.save(file_path)

    families = load_family_members()
    families.append({
        'id': int(time.time() * 1000),
        'name': name,
        'image': saved_name,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    save_family_members(families)

    flash(f'{name} was added to the family watch list.', 'success')
    return redirect(url_for('index'))


@app.route('/api/family')
def api_family():
    return jsonify(load_family_members())


@app.route('/api/alerts')
def api_alerts():
    return jsonify(load_alerts())


@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    save_alerts([])
    return jsonify({'status': 'success'})


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'camera': 'live', 'family_members': len(load_family_members())})


if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        camera.stop()
