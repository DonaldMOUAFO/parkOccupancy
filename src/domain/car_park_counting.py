"""
Advanced parking occupancy tracker with:
 - Kalman filter per track (constant velocity)
 - Temporary occlusion tolerance (disappear grace frames)
 - State machine: MOVING -> PARKING_CANDIDATE -> PARKED
 - Producer (frame reader) + single worker (inference + tracking)
 - Periodic export of parked vehicles to CSV/JSON

Requirements:
  pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import time
import threading
import queue
import csv
import json
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Any
from ultralytics import YOLO

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "yolo11m.pt"
VIDEO_PATH = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"

VEHICLE_CLASSES = [2, 3, 5, 7] # car, truck, bus, motorbike

MOTION_THRESHOLD = 6.0 # pixels/frame considered "no motion"
FRAMES_TO_PARK = 30 # frames of stillness required to become PARKED
DISAPPEAR_GRACE = 30    # frames to tolerate a disappearance before deleting track

FRAME_QUEUE_MAX = 8     # frame queue size for producer-consumer
EXPORT_INTERVAL = 60.0  # seconds between exports

OUTPUT_CSV = "parked_vehicles.csv"
OUTPUT_JSON = "parked_vehicles.json"

SHOW_WINDOW = True      # Set False to disable display (headless)

# --------------------------
# Utilities / Kalman filter
# --------------------------
@dataclass
class Kalman2D:
    """
    Simple Kalman filter for 2D position with constant velocity model.
    State vector: [x, y, vx, vy]
    """
    x: np.ndarray = field(default_factory=lambda: np.zeros((4, 1))) # state
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 500.) # covariance
    F: np.ndarray = field(default_factory=lambda: np.eye(4))    # transition
    H: np.ndarray = field(default_factory=lambda: np.zeros((2, 4))) # measurement matrix
    Q: np.ndarray = field(default_factory=lambda: np.eye(4) * 0.01)     # process noise
    R: np.ndarray = field(default_factory=lambda: np.eye(2) * 5.0)      # measurement noise

    def __post_init__(self):
        # dt will be updated per predict call
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

    def init_state(self, cx: float, cy: float):
        self.x = np.array([[cx], [cy], [0.0], [0.0]])
        self.P = np.eye(4) * 10.0

    def predict(self, dt: float = 1.0):
        # update transition matrix for dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, cx: float, cy: float):
        z = np.array([[cx], [cy]])
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def current_pos(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

# --------------------------
# Track data structure
# --------------------------
@dataclass
class Track:
    track_id: int
    kalman: Kalman2D
    last_seen: float
    last_frame_idx: int
    disappear_count: int = 0
    still_count: int = 0
    state: str = "MOVING"   # MOVING, PARKING_CANDIDATE, PARKED
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    cls: int = -1
    
    def mark_seen(self, cx: float, cy: float, frame_idx: int, timestamp: float, bbox: Tuple[int,int,int,int], cls:int):
        dt = max(1.0, frame_idx - self.last_frame_idx) # simple dt in frames
        self.kalman.predict(dt) # self.kalman.predict(dt=1.0) using dt=1 frame step since frame_idx increments by 1
        self.kalman.update(cx, cy)
        self.last_seen = timestamp
        self.last_frame_idx = frame_idx
        self.disappear_count = 0
        self.bbox = bbox
        self.cls = cls
        
    def predict_only(self):
        self.kalman.predict(dt=1.0)
        # do not update last_seen; used when not observed this frame
        # 
    def get_position(self):
        return self.kalman.current_pos()

# --------------------------
# Tracker manager
# --------------------------
class ParkingTracker:
    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.parked_ids = set()
        self.last_export = time.time()
    
    def process_detections(self, detections, frame_idx: int, timestamp: float):
        """
        detections: list of dicts with keys:
          'id' (int track id), 'cls' (int), 'bbox' (x1,y1,x2,y2)
        """
        current_ids = set()
        # first, mark predicted for existing tracks (so Kalman evolves)
        for t in self.tracks.values():
            t.predict_only()

        for det in detections:
            track_id = det['id']
            cls = det['cls']
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_ids.add(track_id)
            
            if track_id not in self.tracks:
                kf = Kalman2D()
                kf.init_state(cx, cy)
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    kalman=kf,
                    last_seen=timestamp,
                    last_frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    cls=cls
                )
            else:
                self.tracks[track_id].mark_seen(cx, cy, frame_idx, timestamp, (x1,y1,x2,y2), cls)

            # compute motion distance using kalman estimated position and last position
            t = self.tracks[track_id]
            # est_x, est_y = t.get_position()
            # use bbox center to compute instantaneous movement (could also use velocity)
            # distance is relative to previous predicted state (we used predict then update)
            # We'll approximate movement with magnitude of velocity from state
            vx = float(t.kalman.x[2,0])
            vy = float(t.kalman.x[3,0])
            speed = np.hypot(vx, vy)

            # update still / parking logic
            if speed < MOTION_THRESHOLD:
                t.still_count += 1
            else:
                t.still_count = 0
                if t.state != "MOVING":
                    t.state = "MOVING"
                    
            if t.still_count >= FRAMES_TO_PARK and t.state != "PARKED":
                t.state = "PARKED"
                self.parked_ids.add(track_id)

        # Handle disappeared tracks
        to_delete = []
        for tid, t in list(self.tracks.items()):
            if tid not in current_ids:
                t.disappear_count += 1
                if t.disappear_count > DISAPPEAR_GRACE:
                    # If it was parked, keep record in exported list but remove from active
                    if tid in self.parked_ids:
                        # keep parked id in export but remove from active tracking
                        pass
                    to_delete.append(tid)

        for tid in to_delete:
            self.tracks.pop(tid, None)
            # do not remove from parked_ids so the export remembers historical parked vehicles

        # Periodic export
        if time.time() - self.last_export >= EXPORT_INTERVAL:
            self.export_parked()
            self.last_export = time.time()

    def export_parked(self):
        # Build records from parked_ids using last known info
        rows = []
        for pid in sorted(self.parked_ids):
            t = self.tracks.get(pid, None)
            if t is not None:
                bbox = t.bbox
                cls = t.cls
                rows.append({
                    "id": pid,
                    "class": int(cls),
                    "bbox": list(map(int, bbox)),
                    "state": t.state,
                    "last_seen": t.last_seen
                })
            else:
                # If track not active anymore, still export id only
                rows.append({"id": pid, "class": None, "bbox": [], "state": "PARKED", "last_seen": None})

        # CSV (append without duplicates handling here)
        with open(OUTPUT_CSV, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "class", "bbox", "state", "last_seen"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        # JSON
        with open(OUTPUT_JSON, mode='w') as f:
            json.dump(rows, f, indent=2, default=str)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Exported {len(rows)} parked records -> {OUTPUT_CSV}, {OUTPUT_JSON}")

# --------------------------
# Producer: read frames into a queue
# --------------------------
def frame_producer(cap: cv2.VideoCapture, frame_queue: queue.Queue):
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None) # sentinel to signal end
            break
        frame_queue.put((idx, frame))
        idx += 1
        # throttle if queue full (blocking put would already block; small sleep optional)
    print("Producer finished reading frames.")
    
# --------------------------
# Worker: inference + tracking + display
# --------------------------
def worker(model: YOLO, frame_queue: queue.Queue, tracker: ParkingTracker):
    window_name = "Advanced Parking Tracker"
    frame_count = 0
    start_time = time.time()
    while True:
        item = frame_queue.get()
        if item is None:
            break
        frame_idx, frame = item
        timestamp = time.time()
        
        # Run tracking/inference
        # Ultralytics 'track' may return multiple results per frame; we iterate boxes
        results = model.track(frame, persist=True, verbose=False)
        
        # Collect detection dicts expected by our tracker
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
    
                if cls not in VEHICLE_CLASSES:
                    continue
                # track id may be provided
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"id": track_id, "cls": cls, "bbox": (x1, y1, x2, y2)})

        # pass detections to tracker
        tracker.process_detections(detections, frame_idx, timestamp)
        
        # Visualization
        for tid, tr in tracker.tracks.items():
            x1, y1, x2, y2 = tr.bbox
            color = (0, 0, 255) if tr.state == "PARKED" else (0, 255, 255)
            label = f"ID:{tid} {tr.state}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay counts and perf info
        parked_count = len(tracker.parked_ids)
        fps = (frame_count / (time.time() - start_time + 1e-6)) if frame_count > 0 else 0.0
        cv2.putText(frame, f"Parked: {parked_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        frame_count += 1
        
        if SHOW_WINDOW:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Export once more at end
    tracker.export_parked()
    print("Worker finished.")

# --------------------------
# Main
# --------------------------
def main():
     # Load model
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open video:", VIDEO_PATH)
        return
    
    frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
    tracker = ParkingTracker()
    
    # Start producer thread
    prod_thread = threading.Thread(target=frame_producer, args=(cap, frame_q), daemon=True)
    prod_thread.start()
    
    # Start worker in main thread (so we can display cv2 window safely on many platforms)
    try:
        worker(model, frame_q, tracker)
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
