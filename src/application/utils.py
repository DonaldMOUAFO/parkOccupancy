import numpy as np
from os import path
import cv2

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
