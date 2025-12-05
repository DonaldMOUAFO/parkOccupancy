import cv2
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from src.application.utils import get_parking_status, over_lay, draw_filled_box, draw_status_panel, draw_roundrect, draw_info_panel
from src.config.config import prepare_flux_url, CAMERA_LOGIN_PATH, image_annotation, PARKING_INFORM_JSON_FILE_PATH, PARKING_INFORM_CSV_FILE_PATH
from .save_predictions import save_prediction

# -------------------------------------------------------
# Load YOLO with GPU optimizations
# -------------------------------------------------------
model = YOLO("yolo11n.pt")
#model.to("cuda")           # Force GPU
model.fuse()               # Faster inference
#model.half()               # FP16 speed boost

flux_url = prepare_flux_url(
    CAMERA_LOGIN_PATH,
    focal=0
)

#cap = cv2.VideoCapture(flux_url)

video_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
#video_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-09-24-11h56m08s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
#video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-19-10h02m09s-rtsp___10.210.19.177_554_2_profile2_media.smp-.mp4"
#video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-05-14h18m43s-rtsp___10.210.19.177_554_2_profile2_media.smp-.mp4"
#video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-12-11h53m42s-rtsp___10.210.19.177_554_1_profile2_media.smp-.mp4"
#video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-05-14h16m20s-rtsp___10.210.19.177_554_1_profile2_media.smp-.mp4"
#video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-05-14h03m19s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
#video_path = "/home/besttic-rd/Vidéos/vlc-record-2025-09-12-11h39m43s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"

def run_place_counting(video_path=video_path, model=model, focal=0):
    """Run the parking detection on the video feed."""

    cap = cv2.VideoCapture(video_path)
    vehicle_classes = [2, 3, 5, 7]

    # -------------------------------------------------------
    # PARKING STATE
    # -------------------------------------------------------
    last_positions = {}
    no_motion_frames = {}
    PARKED = set()
    object_names = {}

    MOTION_THRESHOLD = 3
    FRAME_TO_PARK = 25

    if int(focal) == 0:
        TOTAL_SPOTS = 25
    elif int(focal) == 1:
        TOTAL_SPOTS = 15
    elif int(focal) == 2:
        TOTAL_SPOTS = 10

    # -------------------------------------------------------
    # SPEED BOOST SETTINGS
    # -------------------------------------------------------
    DETECTION_INTERVAL = 5     # Detect every 5 frames, track in-between
    frame_count = 0
    active_tracks = {}         # Store track/id → bbox

    # -------------------------------------------------------
    # TIMED SAVING SETTINGS
    # -------------------------------------------------------
    last_save_time = time.time()
    SAVE_INTERVAL = 10   # 1 minute but for testing, you can set it to a lower value

    while True:
        ret, frame = cap.read()
        frame = frame[:, :1500]  # Crop to speed up processing
        if not ret:
            break

        frame_count += 1
        height, width = frame.shape[:2]

        # ---------------------------------------------------
        # DETECT every N frames (heavy)
        # TRACK between detections (light)
        # ---------------------------------------------------
        if frame_count % DETECTION_INTERVAL == 0:
            results = model(frame, conf=0.1) #.track(frame,task="detect", persist=False, verbose=False)

        else:
            results = model.track(
                frame, 
                persist=True, 
                verbose=False, 
                task="track",
                conf=0.25,
                tracker="bytetrack.yaml"
            )

        current_ids = set()
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in vehicle_classes:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1
                current_ids.add(track_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                center = (cx, cy)

                # ------------------- Motion distance -------------------
                if track_id in last_positions:
                    px, py = last_positions[track_id]
                    dist = np.hypot(cx - px, cy - py)
                else:
                    dist = 10000

                # Stillness counter
                if dist < MOTION_THRESHOLD:
                    no_motion_frames[track_id] = no_motion_frames.get(track_id, 0) + 1
                else:
                    no_motion_frames[track_id] = 0

                last_positions[track_id] = center
                # Parked logic
                if no_motion_frames.get(track_id, 0) >= FRAME_TO_PARK:
                    PARKED.add(track_id)
                    object_names[track_id] = model.names[cls]

                # Store bbox for drawing
                active_tracks[track_id] = (x1, y1, x2, y2)

        # Remove disappeared tracks
        for tid in list(PARKED):
            if tid not in current_ids:
                PARKED.remove(tid)

        # -------------------------------------------------------
        # COUNT PARKED + WRONG
        # -------------------------------------------------------
        parked_cars = 0
        wrong_cars = 0

        for tid in PARKED:
            if tid not in last_positions:
                continue

            cx, cy = last_positions[tid]
            x1, y1, x2, y2 = active_tracks.get( tid, (0,0,0,0) )

            parked_cars, wrong_cars, color = get_parking_status(
                video_path, cx, cy, parked_cars, wrong_cars
            )

            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, object_names.get(tid,"vehicle"), (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        # -------------------------------------------------------
        # Visual Overlay of Parking Zones
        # -------------------------------------------------------
        overlay = frame.copy()
        overlay = over_lay(overlay, video_path)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # -------------------------------------------------------
        image_annotation(frame, video_path)

        # -------------------------------------------------------
        # UI Text
        # -------------------------------------------------------
        free_spots = TOTAL_SPOTS - parked_cars

        # show detected vehicles count
        total_vehicles = len(results[0].boxes) if results and results[0].boxes is not None else 0

        overlay = frame.copy()
        draw_status_panel(overlay, parked_cars, free_spots, wrong_cars, total_vehicles)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # -------------------------------------------------------
        # TIMED SAVING (every 60 seconds)
        # -------------------------------------------------------
        now = time.time()
        if now - last_save_time >= SAVE_INTERVAL:
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            save_prediction(
                timestamp=timestamp,
                parked=parked_cars,
                free=free_spots,
                wrong=wrong_cars,
                detected=total_vehicles,
                muliple_files=False,
                focal=focal,
                file_path=PARKING_INFORM_CSV_FILE_PATH, #PARKING_INFORM_JSON_FILE_PATH
                frame=frame  # Save image with overlay also
            )

            print(f"[SAVED] Predictions at {timestamp}")
            last_save_time = now

        # -------------------------------------------------------
        # Show
        # -------------------------------------------------------
        cv2.imshow("Optimized Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"FRAME SHAPE: {frame.shape} ")
            break

    cap.release()
    cv2.destroyAllWindows()