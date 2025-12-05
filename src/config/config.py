import json
import time
import cv2
import numpy as np
from os import path


PRESENT_DIR = path.dirname(path.realpath(__file__)) 

camera_frame_path = path.join(
    PRESENT_DIR, "../../data/camera_frame.png"
)

camera_video_path = path.join(
    PRESENT_DIR, 
    "../../data/2020-03-19-08-00-00_scene_0013_BB_write.avi"
)

coordinate_file = path.join(
    PRESENT_DIR, "../../data/parking_place_coordinates.json"
)

coordinate_file2 = path.join(
    PRESENT_DIR, "../../data/parking_place_coordinates2.json"
)

points_correspondance_file = path.join(
    PRESENT_DIR, "../../data/parking_points_correspondance.json"  # points_correspondance_file
)

PARKING_INFORM_JSON_FILE_PATH= path.join(
    PRESENT_DIR, "../../data/data_file.json"
)

PARKING_INFORM_CSV_FILE_PATH = path.join(
    PRESENT_DIR, "../../data/data_file.csv"
)

camera_image_view_path = path.join(
    PRESENT_DIR, "../../data/scale/homography/frame_focale_0.jpg" #"../../data/camera_image_view.png" IMG_20250711_163748.jpg data/scale/homography/frame_focale_0_bird_view.png
)

bird_view_path = path.join(
    PRESENT_DIR, "../../data/scale/homography/frame_focale_0_bird_view.png" #, ../../data/field_to_view.png" IMG_20250711_163552.jpg
)

HOMAGRAPHY_MATRIX_PATH = path.join(
    PRESENT_DIR, "../../data/homography_matrix.npy"
)

PARKING_FILE_PATH = path.join(
    PRESENT_DIR, "../../data/scale/parking_place_coordinates.json" 
) 

CAMERA_LOGIN_PATH = path.join(
    PRESENT_DIR, "../../data/scale/cam_access_info.json"
)

def prepare_flux_url(camera_login_path, focal: int) -> str:
    """Returns the video frame path based on the focal length."""

    with open(camera_login_path, "r+") as f:
        cam_access_info = json.load(f)

    if int(focal)==0:
        cam_access_info = cam_access_info['0']
        return f"rtsp://{cam_access_info['login']}:{cam_access_info['pasword']}@{cam_access_info['ip']}/profile2/media.smp"
        
    elif int(focal)==1:
        cam_access_info = cam_access_info['1']
        return f"rtsp://{cam_access_info['login']}:{cam_access_info['pasword']}@{cam_access_info['ip']}/profile2/media.smp"
    
    elif int(focal)==2:    
        cam_access_info = cam_access_info['2']
        return f"rtsp://{cam_access_info['login']}:{cam_access_info['pasword']}@{cam_access_info['ip']}/profile2/media.smp"
    
    elif int(focal)==3:    
        cam_access_info = cam_access_info['3']
        return f"rtsp://{cam_access_info['login']}:{cam_access_info['pasword']}@{cam_access_info['ip']}/profile2/media.smp"
    
def draw_filled_box(img, pts1, label, fill_color=(0,255,0), alpha=0.4, border_color=(0,255,0), border_th=2):
    """
    Draw a filled rectangle with optional border and transparency.
    img : frame
    x1,y1,x2,y2 : box coordinates
    fill_color : BGR fill color
    alpha : transparency (0 = transparent, 1 = opaque)
    border_color : BGR border color
    border_th : border thickness
    """

    fill_color = (0, 255, 0) if label == "LEGAL" else (0, 0, 255) # Green for LEGAL, Red for ILLEGAL
    border_color = (0, 225, 0) if label == "LEGAL" else (0, 0, 225)

    #draw_filled_box(image, x1, y1, x2, y2, fill_color=(0,255,0), alpha=0.3, border_color=(0,255,0),border_th=2)
    fs = 0.7
    th = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (tw, th_text), _ = cv2.getTextSize(label, font, fs, th)
    p1 = (pts1[0], pts1[1] - th_text - 10)
    p2 = (pts1[0] + tw + 10, pts1[1])

    # Create overlay
    overlay = img.copy()

    # Draw filled box (only on overlay)
    cv2.rectangle(overlay, p1, p2, fill_color, -1)

    # Blend overlay and image
    img[:] = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    # Optional border (drawn on top)
    if border_th > 0:
        cv2.rectangle(img, p1, p2, border_color, border_th)

    # Label text
    cv2.putText(img, label, (pts1[0]+5, pts1[1]-5), font, fs, (0,0,0), 2)

    return img

def image_annotation(frame, img_path):
    """
    Annotates the image with the given text at the specified position.
    """
    if "_0_" in img_path:
        draw_filled_box(
            frame,
            (540, 160),
            fill_color=(0,255,0), alpha=0.3,
            border_color=(0,255,0), border_th=1,
            label="LEGAL"
        )

        draw_filled_box(
            frame,
            (1190, 130),
            fill_color=(0,255,0), alpha=0.3,
            border_color=(0,255,0), border_th=1,
            label="LEGAL"
        )

        draw_filled_box(
            frame,
            (1430, 135),
            fill_color=(0,255,0), alpha=0.3,
            border_color=(0,255,0), border_th=1,
            label="ILLEGAL"
        )
    elif "_1_" in img_path:
        draw_filled_box(
            frame,
            (1425, 108),
            fill_color=(0,255,0), alpha=0.3,
            border_color=(0,255,0), border_th=1,
            label="ILLEGAL"
        )
    elif "_2_" in img_path:
        draw_filled_box(
            frame,
            (10, 160),
            fill_color=(0,255,0), alpha=0.3,
            border_color=(0,255,0), border_th=1,
            label="ILLEGAL"
        )
# -------------------------------------------------------
# PARKING ZONES
# -------------------------------------------------------
PARKING_POLYGON_F0_LEFT = np.array([
    [0, 420],
    [0,528],
    [725, 252],
    [725, 165],
], dtype=np.int32)

PARKING_POLYGON_F0_RIGHT = np.array([
    [540, 1076],
    [1372,150],
    [1015,150],
    [0,580],
    [0,970]
], dtype=np.int32)

ILLEGAL_POLYGON_F0 = np.array([
    [540,1080],
    [1500,0],
    [1919,0],
    [1919,1080]
], dtype=np.int32)

PARKING_POLYGON_F1_1 = np.array([
    [1265, 422],
    [1920,483],
    [915, 846],
    [1720, 1072],
    [780, 1070],
    [957, 882],
    [964, 554]
], dtype=np.int32)

PARKING_POLYGON_F1_2 = np.array([
    [1425, 108],
    [1605,103],
    [1719, 465],
    [1919, 468],
    [1919, 1080],
    [765, 1080],
    [908, 886],
    [962, 558],
    [1269, 430],
    [1480, 442],
    [1520, 150],
    [1380, 140]
], dtype=np.int32)

PARKING_POLYGON_F2_1 = np.array([
    [0, 166],
    [0,349],
    [0, 1080],
    [1038, 1078]
], dtype=np.int32)

PARKING_POLYGON_F2_2 = np.array([
    [0, 166],
    [1237,20],
    [1305, 218],
    [324, 447]
], dtype=np.int32)

