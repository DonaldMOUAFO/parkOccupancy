import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.application.utils import tensor_to_numpy

# set the device
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

# Load YOLO model (fast + accurate)
model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=5)

cap = cv2.VideoCapture("/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4")

current_vehicles = set()    # vehicles currently in parking
entered_count = 0
exited_count = 0

# List of vehicle classes to detect
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0] #, stream=True)

    detections = []
    for r in results:

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # Only detect vehicles in parking
            if label not in ["car", "truck", "bus", "motorbike"]:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = tensor_to_numpy(x1) 
            y1 = tensor_to_numpy(y1)
            x2 = tensor_to_numpy(x2)
            y2 = tensor_to_numpy(y2)

            #print(x1) #, y1, x2, y2)

            #print(f"Detected {label} with confidence {conf} at [{x1}, {y1}, {x2}, {y2}]")

            detections.append( ([x1, y1, x2-x1, y2-y1], conf, cls) )  # DeepSORT expects [x, y, w, h]

            #print(f"Detected {label} with confidence {conf} at {detections}")

    # Update tracker
    #print(f"""===== Detections: {detections} ===== """)
    # bbs expected to be a list of tuples (bbox, confidence, class)
    # tracks = tracker.update_tracks(detections, frame=frame)

    # tracked_ids = set()

    # for track in tracks:
    #   if not track.is_confirmed():
    #     continue

    #   tid = track.track_id
    #   x1, y1, x2, y2 = track.to_ltrb()

    #   tracked_ids.add(tid)

    # #   # Draw tracked box
    # #   cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    # #   cv2.putText(frame, f"ID {tid}", (int(x1), int(y1)-10),
    # #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
      
    # # Compare previous set with new set
    # # Vehicles that appear = entered
    # new_entries = tracked_ids - current_vehicles
    # entered_count += len(new_entries)

    # # Vehicles that disappear = exited
    # new_exits = current_vehicles - tracked_ids
    # exited_count += len(new_exits)

    # current_vehicles = tracked_ids

    # Display
    cv2.putText(frame, f"Parked: {len(current_vehicles)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    cv2.putText(frame, f"Entered: {entered_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.putText(frame, f"Exited: {exited_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Parking Lot Counting", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

 