import numpy as np
from os import path
import cv2
from src.config import config

def load_image(camera_frame_path) :
    """
    Load image that with help to locale parking places
    """

    if path.exists(camera_frame_path) :
        print(f"==== Load the file : {camera_frame_path} ====")
        frame = cv2.imread(camera_frame_path)
    else :
        print(f"==== Could not read the file {camera_frame_path} ====")
        frame = None
    
    return frame

def first_camera_frame(camera_video_path, camera_frame_path):
    """
    Load the video and take the first frame and use it to locale parking places
    """

    cap = cv2.VideoCapture(camera_video_path)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(camera_frame_path, frame)

    else :
        print(f"==== Could not read the video {camera_video_path} ========")
        print(f"======= Check the requirement and process again ============")
    
    cap.release()
    
    return frame

def draw_filled_box(img, pts1, label, fill_color=(0,255,0), alpha=0.4, border_color=(0,255,0), border_th=1):
    """
    Draw a filled rectangle with optional border and transparency.
    img : frame
    x1,y1,x2,y2 : box coordinates
    fill_color : BGR fill color
    alpha : transparency (0 = transparent, 1 = opaque)
    border_color : BGR border color
    border_th : border thickness
    """
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

def draw_contours(image,
    coordinates, label, font_color,
    border_color=(255, 0, 0),
    line_thickness=1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5):

    cv2.drawContours(
        image, [coordinates], 
        contourIdx=-1, color=border_color, 
        thickness=2, ineType=cv2.LINE_8
    )
    
    moments = cv2.moments(coordinates)

    center = (
        int(moments["m10"] / moments["m00"]) - 3,
        int(moments["m01"] / moments["m00"]) + 3
    )

    cv2.putText(
        image, label, center, font, font_scale, 
        font_color, line_thickness, cv2.LINE_AA
    )

def load_homography_matrix(homography_matrix_path):
    """
    Load the homography matrix from a file
    """
    if path.exists(homography_matrix_path):
        print(f"==== Load the homography matrix from {homography_matrix_path} ====")
        H = np.load(homography_matrix_path)
        if H.shape != (3, 3):
            raise ValueError("Homography matrix must be of shape (3, 3)")
        print(f"==== Homography matrix loaded successfully ====")
        print(f"==== Shape of the homography matrix: {H.shape} ====")
        return H
    else:
        print(f"==== Could not read the homography matrix file {homography_matrix_path} ====")
        return None
    
def show_image(image, window_name="Image"):
    """
    Show image in a window
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy() #tensor.cpu().numpy(), # 
    else:
        return tensor.numpy()


def get_parking_status(video_path, cx, cy, parked_cars, wrong_cars):

    if "_0_" in video_path:

        in_left    = cv2.pointPolygonTest(config.PARKING_POLYGON_F0_LEFT, (cx, cy), False) >= 0
        in_right   = cv2.pointPolygonTest(config.PARKING_POLYGON_F0_RIGHT, (cx, cy), False) >= 0
        in_illegal = cv2.pointPolygonTest(config.ILLEGAL_POLYGON_F0, (cx, cy), False) >= 0

        if in_left or in_right:
            parked_cars += 1
            color = (0,255,0)
        elif in_illegal:
            wrong_cars += 1
            color = (0,0,255)
        else:
            color = (255,255,0)
    
    elif "_1_" in video_path:
        in_f2 = cv2.pointPolygonTest(config.PARKING_POLYGON_F1_1, (cx, cy), False) >= 0
        in_f1 = cv2.pointPolygonTest(config.PARKING_POLYGON_F1_2, (cx, cy), False) >= 0

        if in_f2 or in_f1:
            wrong_cars += 1
            color = (0,0,255)
        else:
            parked_cars += 1
            color = (0,255,0)
    
    elif "_2_" in video_path:
        in_f2 = cv2.pointPolygonTest(config.PARKING_POLYGON_F2_1, (cx, cy), False) >= 0
        in_f1 = cv2.pointPolygonTest(config.PARKING_POLYGON_F2_2, (cx, cy), False) >= 0

        if in_f2 or in_f1:
            wrong_cars += 1
            color = (0,0,255)
        else:
            parked_cars += 1
            color = (0,255,0)

    return parked_cars, wrong_cars, color

# -------------------------------------------------------
def over_lay(overlay, video_path):

    if "_0_" in video_path:
        cv2.fillPoly(overlay, [config.PARKING_POLYGON_F0_LEFT], (0,255,0))
        cv2.fillPoly(overlay, [config.PARKING_POLYGON_F0_RIGHT], (0,200,0))
        cv2.fillPoly(overlay, [config.ILLEGAL_POLYGON_F0], (0,0,255))

    elif "_1_" in video_path:
        #cv2.fillPoly(overlay, [config.PARKING_POLYGON_F1_1], (0,0,255))
        cv2.fillPoly(overlay, [config.PARKING_POLYGON_F1_2], (0,0,255))
    
    elif "_2_" in video_path:
        cv2.fillPoly(overlay, [config.PARKING_POLYGON_F2_1], (0,0,255))
        cv2.fillPoly(overlay, [config.PARKING_POLYGON_F2_2], (0,0,255))

    return overlay


def draw_roundrect(img, pt1, pt2, radius=20, color=(50, 50, 50), thickness=-1):
    """Draw a rounded rectangle using cv2."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1

    # main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    # 4 circles for corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_status_panel(frame, parked, free, wrong, detected):
    # Panel position (right side) (1080, 1920, 3)
    panel_x1, panel_y1 = frame.shape[1]-220, frame.shape[0]-330 #1700, 750
    panel_x2, panel_y2 = panel_x1+210, panel_y1+230 #1900, 980

    # Draw rounded panel background
    draw_roundrect(
        frame, (panel_x1, panel_y1), (panel_x2, panel_y2),
        radius=25, color=(40, 40, 40), thickness=-1
    )

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    # Text coords
    x = panel_x1 + 30
    y = panel_y1 + 50
    dy = 45   # line spacing

    cv2.putText(frame, f"Parked:  {parked}", (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Free:    {free}",   (x, y + dy), font, scale, (255, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Wrong:   {wrong}",  (x, y + 2*dy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Detected:{detected}", (x, y + 3*dy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_info_panel(img, x, y, text, color, bg_color=(20,20,20), alpha=0.5):
    """
    Draws a clean info panel with background + text.
    img     : frame
    x, y    : top-left corner
    text    : text content
    color   : text color (BGR)
    bg_color: background color
    alpha   : transparency of background
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    # Measure text
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 10

    # Background panel coordinates
    x2 = x + tw + pad * 2
    y2 = y + th + pad * 2

    # Draw transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw the text
    cv2.putText(img, text, (x + pad, y + th + pad - 2),
                font, font_scale, color, thickness, cv2.LINE_AA)

