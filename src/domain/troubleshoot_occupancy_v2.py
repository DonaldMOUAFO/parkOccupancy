import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
vehicle_classes = [2, 3, 5, 7] #["car", "truck", "bus", "motorbike"]

# -------------------------------------------------------------------
# Parking logic
# -------------------------------------------------------------------
last_positions = {}         # {track_id: (x,y)}
no_motion_frames = {}       # {track_id: n}
PARKED = set()              # parked track IDs
object_names = {}           # {track_id: class name}

MOTION_THRESHOLD = 3
FRAME_TO_PARK = 30
TOTAL_SPOTS = 25

# -------------------------------------------------------------------
# Parking polygons
# -------------------------------------------------------------------
PARKING_POLYGON_LEFT = np.array([
    [0, 420],
    [0,528],
    [725, 252],
    [725, 165],
], dtype=np.int32)

PARKING_POLYGON_RIGHT = np.array([
    [540, 1080],
    [1500,0],
    [1919, 0],
    [1919, 1080],
], dtype=np.int32)

PARKING_PENTAGON_RIGHT = np.array([
    [540, 1076],
    [1372,150],
    [1015, 150],
    [0, 580],
    [0, 970]
], dtype=np.int32)

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)
    current_ids = set()
    # ----------------------- DETECTION LOOP -------------------------
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])

            # Only consider vehicle classes
            if cls not in vehicle_classes:
                continue

            track_id = int(box.id[0]) if box.id is not None else -1
            current_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            center = (cx, cy)

            # -------------------------------------------------------
            # Motion detection (Compute BEFORE updating last_positions)
            # -------------------------------------------------------
            if track_id in last_positions:
                px, py = last_positions[track_id]
                dist = np.hypot(cx - px, cy - py)
            else:
                dist = 9999

            # Update stillness counter
            if dist < MOTION_THRESHOLD:
                no_motion_frames[track_id] = no_motion_frames.get(track_id, 0) + 1
            else:
                no_motion_frames[track_id] = 0

            # Update last position AFTER computing dist
            last_positions[track_id] = center

            # Mark as parked
            if no_motion_frames.get(track_id, 0) >= FRAME_TO_PARK:
                PARKED.add(track_id)
                object_names[track_id] = model.names[cls]
    
    # ---------------- REMOVE LOST TRACKS ----------------------------
    for tid in list(PARKED):
        if tid not in current_ids:
            PARKED.remove(tid)

    # ---------------- CLASSIFY PARKED / WRONG -----------------------
    parked_cars = 0
    wrongly_parked_cars = 0

    for tid in PARKED:
        if tid not in last_positions:
            continue

        cx, cy = last_positions[tid]

        in_left  = cv2.pointPolygonTest(PARKING_POLYGON_LEFT, (cx, cy), False) >= 0
        in_right = cv2.pointPolygonTest(PARKING_PENTAGON_RIGHT, (cx, cy), False) >= 0
        in_wrong = cv2.pointPolygonTest(PARKING_POLYGON_RIGHT, (cx, cy), False) >= 0

        # We need bounding box for drawing : get box ONLY FOR THIS tid
        for r in results:
            for box in r.boxes:
                if box.id is None: 
                    continue
                if int(box.id[0]) == tid:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    break
            else:
                continue
            break

        if in_left or in_right:
            parked_cars += 1
            color = (0,255,0)
        elif in_wrong:
            wrongly_parked_cars += 1
            color = (0,0,255)
        else:
            color = (255,255,0)

        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, object_names.get(tid,"Vehicle"), (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
    
    # ---------------- DISPLAY COUNTS -------------------------------
    free_spots = TOTAL_SPOTS - parked_cars

    cv2.putText(
        frame, 
        f"Parked cars: {parked_cars}", (1500,800),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
    )

    cv2.putText(frame, 
        f"Free spots: {free_spots}", (1500,850),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2
    )

    cv2.putText(frame, 
        f"Wrong parking: {wrongly_parked_cars}", (1500,900),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2
    )

    cv2.putText(frame, 
        f"Total Parked (motion-based): {len(PARKED)}",
        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2
    )

    print("PARKED:", PARKED)

    cv2.imshow("Parking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
