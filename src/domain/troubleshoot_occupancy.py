from collections import defaultdict
import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # you can also try yolo11s.pt or larger models

video_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"

vehicle_classes = [2, 3, 5, 7] #["car", "truck", "bus", "motorbike"]

# ---------------------------------------------
# Variables for tracking parked vehicles
#---------------------------------------------

last_positions = {}  # vehicle_id -> (x, y)
no_motion_frames = {}  # vehicle_id -> number of consecutive frames with little motion 
PARKED = set()  # track IDs of vehicules that are considered parked 

MOTION_THRESHOLD = 3  # MAX PIXELS PER FRAME TO COUNT AS "not moving"
FRAME_TO_PARK = 30    # How long the car must be still to count as parked

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

TOTAL_SPOTS = 25              # <<< SET THIS to your parking capacity

cap = cv2.VideoCapture(video_path)

object_names = {}

static_vehicles = set()
parked_cars = 0
wrongly_parked_cars = 0
parked_vehicles_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)

    current_ids = set()

    for r in results:
        
        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0]) # box.cls[0]

            # Only consider vehicle classes
            if cls not in vehicle_classes:
                continue

            # track_id = int(box.id[0]) if box.id is not None else None
            track_id = int(box.id) if box.id is not None else -1
            current_ids.add( track_id )

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int( (x1 + x2) / 2 ) 
            cy = int( (y1 + y2) / 2 )
            center = (cx, cy)
            last_positions[track_id] = center

            # ---------------------------------------------------------------
            # Compute motion 
            #---------------------------------------------------------------
            if track_id in last_positions:
                prev_x, prev_y = last_positions[track_id]
                dist = np.hypot(cx - prev_x, cy - prev_y)
            else:
                dist = 9999  # first frame for this vehicle

            # ---------------------------------------------------------------
            # Update "stillness counter
            #---------------------------------------------------------------
            if dist < MOTION_THRESHOLD:
                no_motion_frames[track_id] = no_motion_frames.get(track_id, 0) + 1
            else:
                no_motion_frames[track_id] = 0 # reset counter if vehicle moved

            # ----------------------------------------------------
            # Mark as parked after enough still frames
            #---------------------------------------------------
            if no_motion_frames.get(track_id, 0) >= FRAME_TO_PARK:
                PARKED.add(track_id)
                object_names[track_id] = model.names[cls]

            # ---------------------------------------------------------------
            # Draw bounding box and status
            #---------------------------------------------------------------
            # cv2.rectangle(
            #     frame, (x1, y1), (x2, y2), 
            #     (0, 255, 255) if track_id not in PARKED else (0, 0, 255), 2
            # )
            # state = "PARKED" if track_id in PARKED else "MOVING"
            # cv2.putText(
            #     frame, f"ID:{track_id} {state}", (x1, y1 - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #     (0, 0, 255) if track_id in PARKED else (0, 255, 255), 2)

            # ---------------------------------------------------------------
            # Check if inside parking area
            #---------------------------------------------------------------

    #---------------------------------------------------------------
    # Remove track IDs that disappeared (no longer tracked)
    # ---------------------------------------------------------------
    for tid in list(PARKED):
        if tid not in current_ids:
            PARKED.remove(tid)

    # ---------------------------------------------------------------
    # Count parked vehicles and wrongly parked vehicles
    # ---------------------------------------------------------------  
    parked_cars = 0
    wrongly_parked_cars = 0
    for tid in PARKED:
        if tid not in last_positions:
            continue
        cx, cy = last_positions[tid]

        in_location_left  = cv2.pointPolygonTest(PARKING_POLYGON_LEFT, (cx, cy), False)
        in_location_right = cv2.pointPolygonTest(PARKING_PENTAGON_RIGHT, (cx, cy), False)
        in_out_location = cv2.pointPolygonTest(PARKING_POLYGON_RIGHT, (cx, cy), False)

        if (in_location_right >= 0) or (in_location_left >= 0):
            parked_cars += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, object_names.get(track_id, "None"), (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        elif in_out_location >= 0:
            wrongly_parked_cars += 1  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, object_names.get(track_id, "None"), (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  

    # ---------------------------------------------------------------    # Display parked and free spots
    # ---------------------------------------------------------------
    # cls_name = model.names[cls]
    # in_location_left  = cv2.pointPolygonTest(PARKING_POLYGON_LEFT, last_positions[track_id], False)
    # in_location_right = cv2.pointPolygonTest(PARKING_PENTAGON_RIGHT, last_positions[track_id], False)
    # in_out_location   = cv2.pointPolygonTest(PARKING_POLYGON_RIGHT, last_positions[track_id], False)
    
    # if (in_location_right >= 0) or (in_location_left >= 0):
    #     parked_cars += 1
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, cls_name, (x1, y1 - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # elif in_out_location >= 0:
    #     wrongly_parked_cars += 1
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     cv2.putText(frame, cls_name, (x1, y1 - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    free_spots = TOTAL_SPOTS - parked_cars
    cv2.putText(
        frame, 
        f"Parked car: {parked_cars}", 
        (1500, 800), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 0, 255), 2
    )

    cv2.putText(
        frame, 
        f"Free spots: {wrongly_parked_cars}", 
        (1500, 850), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, (0, 0, 255), 2
    )

    # ---------------------------------------------------------------
    # Display occupancy count
    # ---------------------------------------------------------------  
    cv2.putText(
        frame, 
        f"Parked Vehicles: {len(PARKED)}", 
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)     

    print(f"Parked Vehicles: {len(PARKED)}, {PARKED}")      

    cv2.imshow("Parking Occupancy Without Slot Locations", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()