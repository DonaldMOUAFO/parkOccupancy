import os
import cv2
import json
import time
import threading
import queue
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Nums GPUs Available :", len(tf.config.list_physical_devices("GPU")))
print("GPU is", "available" if tf.test.is_gpu_available() else "Not available")
# ------------------------------
# =========== CONFIG ===========
# ------------------------------
path_file = "cam_access_info.json"
with open(os.path.join(path_file), "r+") as f:
    cam_access_info = json.load(f)

#VIDEO_URL =f"rtsp://{cam_access_info['0']['login']}:{cam_access_info['0']['pasword']}@{cam_access_info['0']['ip']}/profile2/media.smp"  # or local file
VIDEO_URL = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/vlc-record-2025-11-14-14h12m58s-rtsp___10.210.19.177_554_0_profile2_media.smp-.mp4"

ENGINE_PATH = "yolo11s_fp16.engine"   # or .pt / .onnx / engine
USE_TENSORRT_ENGINE = True            # True if ENGINE_PATH is a TensorRT engine
DETECTION_INTERVAL = 4                # detect every N frames (1 = every frame)
NUM_WORKERS = 2                       # number of worker threads (inference)
CAPTURE_QUEUE_MAX = 8                 # bounded queue sizes
DISPLAY_QUEUE_MAX = 8
MOTION_THRESHOLD = 3
FRAME_TO_PARK = 25
TOTAL_SLOTS = 25

# Homography / slots
# If you already know the 4 corners (in image coords) of the parking ground (clockwise),
# set MANUAL_HOMOGRAPHY = True and provide PARK_SRC_POINTS. Then set GRID_ROWS x GRID_COLS.
MANUAL_HOMOGRAPHY = True # if  PARK_SRC_POINTS not None, if False PARK_SRC_POINTS not None 
# Example: PARK_SRC_POINTS = np.array([[755,844],[1218,325],[608,323],[0,575]], dtype=np.float32)
#PARK_SRC_POINTS = None
PARK_SRC_POINTS = np.array([[755,844],[1218,325],[608,323],[0,575]], dtype=np.float32)
GRID_ROWS = 5
GRID_COLS = 5

# Automatic detection parameters (top-down)
AUTO_HOUGH_RHO      = 1
AUTO_HOUGH_THETA    = np.pi / 180
AUTO_HOUGH_THRESH   = 100
AUTO_SLOT_MIN_WIDTH = 30     # px in top-down coordinates (tune per scene)
# ------------------------------
# ======== END CONFIG ==========
# ------------------------------

# ------------------------------
# ========== Globals ===========
# ------------------------------
capture_queue = queue.Queue(maxsize=CAPTURE_QUEUE_MAX)
display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_MAX)
stop_event = threading.Event()

# Shared tracking state (thread-safe access via simple locks)
state_lock = threading.Lock()
last_positions   = {}
no_motion_frames = {}
PARKED = set()
object_names  = {}
active_tracks = {}   # track_id -> bbox (x1,y1,x2,y2)
frame_counter = 0

# set the device
device_cv2 = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
print(f"device_cv2 : {device_cv2}, device_torch : {device}")

#model = YOLO("yolo11s.pt")
#model.export(format="engine", device=0) #, imgsz=640)   # produces yolo11s.engine


# Load model (Ultralytics)
print("Loading model...")
model = YOLO(ENGINE_PATH)  # loads engine/onnx/pt transparently
# If using PT and want GPU: model.to('cuda')
# If engine is TensorRT, it's GPU-native.

# Initialize external DeepSort (we use an external tracker for better robustness)
deep_sort = DeepSort(max_age=30)

cap = cv2.VideoCapture(VIDEO_URL)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------------------------------
    # Show
    # -------------------------------------------------------
    cv2.imshow("Optimized Parking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# tensorflow                   2.13.0
# tensorrt                     8.5.1.7
# torch                        2.8.0
# onnx                         1.14.1
# cudnn                        8.9.7
# cuda_11.5.r11.5