from ultralytics import YOLO
import numpy as np
import cv2

# 1. Load YOLOv11 model (pretrained on COCO)
model = YOLO("yolo11m.pt")  # you can also try yolo11s.pt or larger models
#model = YOLO("yolov8m.pt")

# 2. Define the vehicle classes we care about (COCO names)
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

# 3. Load image
image_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/frame_focale_0_1.jpg"# "focale_1.jpg"   # change this to your image  vlc-record-2025-09-12-11h39m43s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4
#image_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
image = cv2.imread(image_path)

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

# take part of the image
image_croped = image[:, :1400]  # Adjust these values as needed

# 4. Run inference
results = model(image, verbose=False)

# 5. Initialize counter
vehicle_count = {cls: 0 for cls in VEHICLE_CLASSES}

parked_cars = 0
wrongly_parked_cars = 0
parked_vehicles_count = 0

# 6. Process detections
for r in results:
    # print results to see available attributes
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        if cls_name not in VEHICLE_CLASSES:
            continue

        vehicle_count[cls_name] += 1

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(image, cls_name, (x1, y1 - 5),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        in_location_left  = cv2.pointPolygonTest(PARKING_POLYGON_LEFT, (cx, cy), False)
        in_location_right = cv2.pointPolygonTest(PARKING_PENTAGON_RIGHT, (cx, cy), False)
        in_out_location = cv2.pointPolygonTest(PARKING_POLYGON_RIGHT, (cx, cy), False)
        
        if (in_location_right >= 0) or (in_location_left >= 0):
            parked_cars += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        elif in_out_location >= 0:
            wrongly_parked_cars += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, cls_name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 7. Display results on image
summary = " | ".join([f"{cls}: {cnt}" for cls, cnt in vehicle_count.items()])

cv2.putText(image, f"{summary}", (600, 60),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

text_good_wrongly_parked = f"Parked | {parked_cars} wrongly | {wrongly_parked_cars}"

cv2.putText(image, f"{text_good_wrongly_parked}", (600, 100),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

# cv2.putText(image,
#             summary,
#             (1200, 60),
#             fontScale=cv2.FONT_HERSHEY_SIMPLEX,
#             color=(255, 0, 0),
#             thickness=2,
#             lineType=cv2.LINE_AA,
#         ) #   fontScale=self.sf, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

cv2.imshow("YOLOv11 Vehicle Counting", image)
cv2.imwrite("vehicle_counting_focal_1_output_complete_part2.jpg", image) # Save output image
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print counts in terminal too
print("Detected vehicles:", vehicle_count)