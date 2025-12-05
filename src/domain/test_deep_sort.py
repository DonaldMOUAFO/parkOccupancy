import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Parameters
# -----------------------------
RTSP_URL = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
DETECT_EVERY_N = 5      # run YOLO every N frames
QUEUE_SIZE = 5          # async buffer
VIDEO_FPS = 60          # expected camera fps

# -----------------------------
# Load models
# -----------------------------
model = YOLO("yolov8s.pt")  # or TensorRT model later
tracker = DeepSort(max_age=30, n_init=2, max_iou_distance=0.7)

# -----------------------------
# Async Queue
# -----------------------------
frame_queue = queue.Queue(maxsize=QUEUE_SIZE)

# -----------------------------
# Parking Slot Homography
# -----------------------------
H = np.array([
    [1.12, 0.02, -200],
    [0.01, 1.08, -120],
    [0.00001, 0.00002, 1]
])  # Example homography – replace with yours

def warp_point(x, y):
    p = np.array([x, y, 1.0])
    p2 = H @ p
    return p2[0]/p2[2], p2[1]/p2[2]

# Example parking slots
parking_slots = [
    # Each: (ID, x, y)
    ("A1", 850, 320),
    ("A2", 940, 320),
    ("A3", 1030, 320)
]

# -----------------------------
# Capture Thread
# -----------------------------
def capture_worker():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_queue.full():
            frame_queue.get_nowait()

        frame_queue.put(frame)

capture_thread = threading.Thread(target=capture_worker, daemon=True)
capture_thread.start()

# -----------------------------
# Processing Loop
# -----------------------------
frame_id = 0
while True:
    frame = frame_queue.get()
    frame_id += 1

    # ----------------------------------
    # 1. YOLO DETECT EVERY N FRAMES
    # ----------------------------------
    if frame_id % DETECT_EVERY_N == 0:
        results = model(frame)[0]
        results = results.cpu().numpy()

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls != 2 and cls != 3 and cls != 5 and cls != 7:
                continue
            x1, y1, x2, y2 = box.xyxy[0] 
            print(type(box))
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
        
        print(f"Frame {frame_id}: Detected {len(detections)} vehicles.", type(detections))
        tracks = tracker.update_tracks(detections, frame=frame)

    # ----------------------------------
    # 2. TRACK between detections
    # ----------------------------------
    else:
        tracks = tracker.update_tracks([], frame=frame)

    # ----------------------------------
    # 3. Draw tracks
    # ----------------------------------
    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id

        # tracking box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ----------------------------------
        # 4. Parking slot projection (homography)
        # ----------------------------------
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        wx, wy = warp_point(cx, cy)

        for pid, sx, sy in parking_slots:
            dist = np.sqrt((wx - sx)**2 + (wy - sy)**2)
            if dist < 50:   # inside slot
                cv2.putText(frame, f"{pid} OCCUPIED", (sx-20, sy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.circle(frame, (sx, sy), 8, (0,0,255), -1)
            else:
                cv2.putText(frame, f"{pid}", (sx-20, sy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.circle(frame, (sx, sy), 8, (255,255,255), -1)

    # ----------------------------------
    # 5. Show
    # ----------------------------------
    cv2.imshow("Parking AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
