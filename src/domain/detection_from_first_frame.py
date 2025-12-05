import cv2
import torch
import numpy as np
from ultralytics import YOLO

# -----------------------------
# PARAMETERS
# -----------------------------
VIDEO_PATH = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"  # your video file
TOTAL_SPOTS = 25              # <<< SET THIS to your parking capacity

VEHICLE_CLASSES = [2, 3, 5, 7] # COCO classes: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck

# -----------------------------
# LOAD FIRST FRAME
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Failed to read the first frame from the video.")

# >>>>> Replace these points with your parking zone polygon <<<<<
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

PARKING_PENTAGON_RIGHT = PARKING_PENTAGON_RIGHT.reshape((-1, 1, 2))

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
# YOLOv8 pretrained on COCO (class 2 = 'car')
model = YOLO("yolo11l.pt")   # you can use yolov8s.pt or yolov8m.pt for higher accuracy yolo11m.pt

# -----------------------------
# RUN DETECTION ON FIRST FRAME
# -----------------------------
results = model(frame)[0]

print(f"Total detections in first frame: {len(results.boxes)}", type(results.boxes))

#cv2.polylines(frame, [PARKING_PENTAGON_RIGHT], isClosed=True, color=(255, 0, 0), thickness=2)
#cv2.polylines(frame, [PARKING_POLYGON_LEFT], isClosed=True, color=(0, 255, 0), thickness=2)
#cv2.polylines(frame, [PARKING_POLYGON_RIGHT], isClosed=True, color=(255, 0, 255), thickness=2)

detected_cars = 0
parked_cars = 0
wrongly_parked_cars = 0

for box in results.boxes:

    cls = int(box.cls[0])
    if cls not in VEHICLE_CLASSES:
        continue
      
    # COCO classes:
    # 2 = car, 5 = bus, 7 = truck
    detected_cars += 1
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    in_location_left  = cv2.pointPolygonTest(PARKING_POLYGON_LEFT, (cx, cy), False)
    in_location_right = cv2.pointPolygonTest(PARKING_PENTAGON_RIGHT, (cx, cy), False)
    in_out_location = cv2.pointPolygonTest(PARKING_POLYGON_RIGHT, (cx, cy), False)

    if (in_location_right >= 0) or (in_location_left >= 0):
        parked_cars += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "right Car", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    elif in_out_location >= 0:
        wrongly_parked_cars += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Car wrongly", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# -----------------------------
# COUNT AVAILABLE SPACES
# -----------------------------
free_spots = TOTAL_SPOTS - parked_cars 
detected_cars
if free_spots < 0:
    free_spots = 0  # sanitize

# -----------------------------
# VISUALIZATION
# -----------------------------
#annotated = results.plot()

# cv2.putText(
#     annotated, 
#     f"Cars detected: {detected_cars}", 
#     (20, 40), 
#     cv2.FONT_HERSHEY_SIMPLEX, 
#     1, (0, 255, 0), 2
# )

cv2.putText(
    frame, 
    f"Parked car: {parked_cars}", 
    (1500, 800), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1, (0, 0, 255), 2
)

cv2.putText(
    frame, 
    f"Free spots: {free_spots}", 
    (1500, 880), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1, (0, 0, 255), 2
)

cv2.imshow("Parking Detection", frame) #annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Detected cars: {detected_cars}")
print(f"Available parking spots: {free_spots}")
