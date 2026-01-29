from ultralytics import YOLO
import cv2
import time

# -----------------------------
# Load YOLOv8 model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Track memory
# -----------------------------
track_memory = {}
TRACK_TIMEOUT = 2.0  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for stability
    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()

    # -----------------------------
    # YOLO + ByteTrack
    # -----------------------------
    results = model.track(
        frame,
        persist=True,
        classes=[0],      # person only
        conf=0.5,
        iou=0.5,
        tracker="bytetrack.yaml",
        imgsz=640,
        max_det=50
    )

    tracked_objects = {}

    for r in results:
        if r.boxes.id is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])
            conf = float(box.conf[0])

            # Update track memory
            track_memory[track_id] = current_time

            # Store structured output
            tracked_objects[track_id] = {
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "last_seen": current_time
            }

            # Draw bounding box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # -----------------------------
    # Remove stale tracks
    # -----------------------------
    expired_ids = [
        tid for tid, t in track_memory.items()
        if current_time - t > TRACK_TIMEOUT
    ]

    for tid in expired_ids:
        track_memory.pop(tid, None)
        tracked_objects.pop(tid, None)

    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow("Member 1 - Multi Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
