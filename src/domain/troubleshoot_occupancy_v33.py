import time
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import prepare_flux_url, CAMERA_LOGIN_PATH

# ---------------------------
# Config / Params
# ---------------------------
MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"
CROP_WIDTH = 1500           # crop width to speed up processing (set None to disable)
VEHICLE_CLASSES = [2, 3, 5, 7]

DETECTION_INTERVAL = 7      # run heavy detection every N frames
MOTION_THRESHOLD = 3
FRAME_TO_PARK = 25
TOTAL_SPOTS = 25

# Remove parked/active tracks if they haven't been seen for this many detection cycles
GRACE_DETECTION_CYCLES = 3

# ---------------------------
# Load model (GPU if available)
# ---------------------------
model = YOLO(MODEL_PATH)
# attempt to enable GPU + fp16 (if available)
try:
    #model.to("cuda")
    # model.fuse()  # optional; already called internally in some versions
    #model.half()
    print("Model moved to CUDA (FP16).")
except Exception:
    # fallback to CPU if CUDA not present
    print("CUDA not available; running on CPU.")

# ---------------------------
# Video capture
# ---------------------------
#flux_url = prepare_flux_url(CAMERA_LOGIN_PATH, focal=0)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

# ---------------------------
# Zones / Polygons
# ---------------------------
PARKING_POLYGON_LEFT = np.array([[0, 420], [0,528], [725, 252], [725, 165]], dtype=np.int32)
PARKING_POLYGON_RIGHT = np.array([[540, 1076], [1372,150], [1015,150], [0,580], [0,970]], dtype=np.int32)
ILLEGAL_POLYGON = np.array([[540,1080], [1500,0], [1919,0], [1919,1080]], dtype=np.int32)

# ---------------------------
# State
# ---------------------------
last_positions = {}       # track_id -> (x,y)
no_motion_frames = {}     # track_id -> frames still
PARKED = set()            # set of parked track ids
parked_last_seen = {}     # track_id -> detection_cycle index when last seen
object_names = {}         # track_id -> class name
active_tracks = {}        # track_id -> bbox (x1,y1,x2,y2)

frame_count = 0
detection_cycle = 0       # increments each time we run a detection frame

# ---------------------------
# Helper
# ---------------------------
def safe_crop(frame, w):
    if w is None:
        return frame
    h, width = frame.shape[:2]
    if width <= w:
        return frame
    return frame[:, :w]

def point_in_poly(poly, pt):
    # return True if point inside polygon (closed)
    return cv2.pointPolygonTest(poly, (int(pt[0]), int(pt[1])), False) >= 0

# ---------------------------
# Main loop
# ---------------------------
try:
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            # end of file or camera error
            print("Frame not read; exiting.")
            break

        # crop early (safe)
        frame = safe_crop(frame, CROP_WIDTH)

        frame_count += 1
        height, width = frame.shape[:2]

        # Decide detect vs track-only frames
        is_detection_frame = (frame_count % DETECTION_INTERVAL == 0)

        if is_detection_frame:
            detection_cycle += 1
            # Run detection+tracking to update tracker and IDs
            # This is the heavy call (YOLO inference)
            #results = model.track(frame, task="detect", mode="predict", persist=True, verbose=False)
            results = model(frame, task="detect", mode="predict")
            # update last-seen cycles for active tracks
        else:
            # TRACK-only frame: tell the Ultralytics tracker to not run the detector
            # `detector=False` prevents running the network and only propagates tracks.
            # NOTE: this requires an Ultralytics version that supports detector=False.
            results = model.track(frame, persist=True, verbose=False, task="track", tracker="botsort.yaml") #"bytetrack.yaml")
           # try:
           #     results = model.track(frame, persist=True, verbose=False, task="track", tracker="bytetrack.yaml")  #mode="predict",
           # except TypeError:
           #    # Some ultralytics versions might not expose detector=False.
           #     # Fallback: call model.track(frame, persist=True) (still works but slower).
           #  results = model.track(frame, persist=True, verbose=False)
        
        # convert to CPU for processing
        #results = results.cpu().numpy()
        current_ids = set()

        # Parse results: Ultralytics returns a sequence of results (one per image)
        # We expect a single-image call, so iterate over results (usually length 1)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in VEHICLE_CLASSES:
                    continue

                # box.id is usually an array-like with [id], convert safely
                try:
                    track_id = int(box.id[0]) if box.id is not None else -1
                except Exception:
                    # Some versions expose scalar id
                    track_id = int(box.id) if box.id is not None else -1

                # bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                # motion: compute before updating last_positions
                if track_id in last_positions:
                    px, py = last_positions[track_id]
                    dist = np.hypot(cx - px, cy - py)
                else:
                    dist = float("inf")

                if dist < MOTION_THRESHOLD:
                    no_motion_frames[track_id] = no_motion_frames.get(track_id, 0) + 1
                else:
                    no_motion_frames[track_id] = 0

                last_positions[track_id] = (cx, cy)

                if no_motion_frames.get(track_id, 0) >= FRAME_TO_PARK:
                    PARKED.add(track_id)
                    object_names[track_id] = model.names[cls]

                # store bbox and mark last seen in detection cycles
                active_tracks[track_id] = (x1, y1, x2, y2)
                parked_last_seen[track_id] = detection_cycle
                current_ids.add(track_id)

        # Clean up active_tracks and PARKED entries that disappeared for several detection cycles
        # This prevents unlimited growth and removes stale entries
        to_remove = []
        for tid, last_seen_cycle in list(parked_last_seen.items()):
            if (detection_cycle - last_seen_cycle) > GRACE_DETECTION_CYCLES:
                to_remove.append(tid)
        for tid in to_remove:
            parked_last_seen.pop(tid, None)
            active_tracks.pop(tid, None)
            last_positions.pop(tid, None)
            no_motion_frames.pop(tid, None)
            PARKED.discard(tid)
            object_names.pop(tid, None)

        # Also remove PARKED IDs that are no longer in current_ids (immediate)
        for tid in list(PARKED):
            if tid not in current_ids:
                # allow a small grace window handled above, but also immediate removal if not present
                PARKED.discard(tid)

        # -----------------------
        # Count parked and wrong
        # -----------------------
        parked_cars = 0
        wrong_cars = 0
        # For drawing we need bbox — ensure we fetch it
        for tid in list(PARKED):
            if tid not in last_positions:
                continue
            cx, cy = last_positions[tid]
            bbox = active_tracks.get(tid, None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox

            in_left = point_in_poly(PARKING_POLYGON_LEFT, (cx, cy))
            in_right = point_in_poly(PARKING_POLYGON_RIGHT, (cx, cy))
            in_illegal = point_in_poly(ILLEGAL_POLYGON, (cx, cy))

            if in_left or in_right:
                parked_cars += 1
                color = (0,255,0)
            elif in_illegal:
                wrong_cars += 1
                color = (0,0,255)
            else:
                color = (255,255,0)

            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, object_names.get(tid,"vehicle"), (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # -----------------------
        # Visual overlay
        # -----------------------
        overlay = frame.copy()
        cv2.fillPoly(overlay, [PARKING_POLYGON_LEFT], (0,255,0))
        cv2.fillPoly(overlay, [PARKING_POLYGON_RIGHT], (0,200,0))
        cv2.fillPoly(overlay, [ILLEGAL_POLYGON], (0,0,255))
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        cv2.putText(frame, "LEFT ZONE", (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, "RIGHT ZONE", (900,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, "ILLEGAL", (1200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        free_spots = TOTAL_SPOTS - parked_cars
        cv2.putText(frame, f"Parked: {parked_cars}", (1200,800), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Free: {free_spots}", (1200,850), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, f"Wrong: {wrong_cars}", (1200,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Show
        cv2.imshow("Optimized Parking Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped, cleaned up.")
