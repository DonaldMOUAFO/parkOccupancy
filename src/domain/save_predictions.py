import time
import json
import csv
import os
import cv2
from src.config.config import PARKING_INFORM_JSON_FILE_PATH

last_save_time = time.time()
SAVE_INTERVAL = 60   # 1 minute

def save_prediction(
    timestamp, parked, free, wrong, detected, 
    muliple_files=False,
    focal=None,
    file_path=PARKING_INFORM_JSON_FILE_PATH, 
    frame=None
    ):
    """Save prediction results in CSV, JSON, and optionally image form."""

    file_exists = os.path.isfile(file_path)

    if muliple_files:
        # Create a new file for each timestamp
        file_path = file_path.replace(
            '.csv', f"_{timestamp.replace(':','-')}.csv") if ".csv" in file_path \
                else file_path.replace('.json', f"_{timestamp.replace(':','-')}.json" 
        )
        
        if ".json" in file_path:
            # Load existing JSON
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}

            # Append new entry
            data[timestamp] = {
                "timestamp": timestamp,
                "focal": focal,
                "parked": parked,
                "free": free,
                "wrong": wrong,
                "detected_total": detected,
            }

            # Save back to disk
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        
        if ".csv" in file_path:
            # ---------------------- CSV ----------------------
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "focal", "parked", "free", "wrong", "detected"])
                writer.writerow([timestamp, focal, parked, free, wrong, detected])

        # ---------------------- IMAGE SAVE ----------------------
        if frame is not None:
            img_path = file_path.replace(
                '.csv', ".jpg" ) if ".csv" in file_path else file_path.replace('.json', ".jpg" 
            )
            cv2.imwrite(img_path, frame)

    else:  
        if  ".json" in file_path:
            # ---------------------- JSON ----------------------
           # Load existing JSON
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}

            # Append new entry
            data[timestamp] = {
                "timestamp": timestamp,
                "focal": focal,
                "parked": parked,
                "free": free,
                "wrong": wrong,
                "detected_total": detected,
            }

            # Save back to disk
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)   

        if ".csv" in file_path:
            # ---------------------- CSV ----------------------
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "focal", "parked", "free", "wrong", "detected"])
                writer.writerow([timestamp, focal, parked, free, wrong, detected])
                
        # ---------------------- IMAGE SAVE ----------------------
        if frame is not None:
            img_path = file_path.replace(
                '.csv', f"_{timestamp.replace(':','-')}.jpg" ) if ".csv" in file_path \
                else file_path.replace('.json', f"_{timestamp.replace(':','-')}.jpg" 
            )  
            cv2.imwrite(img_path, frame)

# -------------------------------------------------------
# TIMED SAVING (every 60 seconds)
# -------------------------------------------------------
# now = time.time()
# if now - last_save_time >= SAVE_INTERVAL:
    
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     save_prediction(
#         timestamp=timestamp,
#         parked=parked_cars,
#         free=free_spots,
#         wrong=wrong_cars,
#         detected=total_vehicles,
#         frame=frame  # Save image with overlay also
#     )

#     print(f"[SAVED] Predictions at {timestamp}")
#     last_save_time = now