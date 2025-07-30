import cv2 
import json
from os import path
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from src.config.config import camera_image_view_path, bird_view_path, points_correspondance_file

class DetectFeature :
    """Class to detect features in images using SIFT or ORB and compute homography."""
    
    def __init__(self, method="sift", from_pts=False):
        self.FROM_PTS = from_pts

        if self.FROM_PTS:
            if not path.exists(points_correspondance_file):
                raise FileNotFoundError(
                    f"""Points correspondance file not found at {points_correspondance_file}. 
                        check the file and load again."""
                )
            else:
                with open(points_correspondance_file, 'r') as file:
                    self.points_correspondance = json.load(file)
        else :
            self.method = method.lower()
            if self.method not in ["sift", "orb"]:
                raise ValueError("Method must be either 'sift' or 'orb'.")
            
            self.detector = self._initialize_detector(method)
            self.matcher  = self._initialize_matcher()

    def _initialize_detector(self, method):
        if method == "sift":
            return cv2.SIFT_create()
        elif method == "orb":
            return cv2.ORB_create()
    
    def _initialize_matcher(self):
        """Initialize FLANN based matcher"""
        if self.method == "sift":
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
        elif self.method == "orb":
            # ORB uses a different index_params
            index_params = dict(
                algorithm=6, table_number=6, key_size=12, 
                multi_probe_level=1
            )
            
            search_params = dict(checks=50)                 
        return cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect_and_compute(self, image):
       """Detects keypoints and computes descriptors"""
       keypoints, descriptors = self.detector.detectAndCompute(image, None)
       return keypoints, descriptors
      
    def match_features(self, des1, des2):
       """Performs KNN matching with ratio test"""
       matches = self.matcher.knnMatch(des1, des2, k=2)
       good_matches = []
       for m, n in matches:
           if m.distance < 0.9 * n.distance:
              good_matches.append(m)
       return good_matches
       
    def compute_homography(self, *args):
        """Main pipeline: detect → match → estimate homography"""

        if self.FROM_PTS:

            if len(self.points_correspondance) < 10:
                raise ValueError("Not enough points to compute homography.")
            else :
                pts_correspondance = list( self.points_correspondance.values() )
                # Convert points to numpy arrays
                kp1 = np.float32([pt[0] for pt in pts_correspondance]).reshape(-1, 1, 2) # camera_pts
                kp2 = np.float32([pt[1] for pt in pts_correspondance]).reshape(-1, 1, 2) # bird_eye_pts
                # Compute homography using RANSAC
                H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)
                # return H, None, None, None, mask
        
        else :
            if len(args) != 2:
                raise ValueError("Two images are required to compute homography.")
            img1, img2 = args
            # If not using pre-defined points, proceed with feature detection and matching
            kp1, des1 = self.detect_and_compute(img1)
            kp2, des2 = self.detect_and_compute(img2)

            matches = self.match_features(des1, des2)

            if len(matches) < 10:
                raise ValueError("Not enough matches to compute homography.")

            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # return H, matches, kp1, kp2, mask
        return H, matches, kp1, kp2, mask
        
    def draw_matches(self, img1, kp1, img2, kp2, matches, mask):
        """Draw inliers only"""
        matched_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches,
            None, matchesMask=mask.ravel().tolist(), flags=2
        )
        return matched_img
    
normal_view = Image.open("/home/besttic-rd/Documents/besttic/parkOccupancy/data/IMG_20250711_163734.jpg")
normal_view = normal_view.rotate(-90, expand=True)

top_image_view = Image.open("/home/besttic-rd/Documents/besttic/parkOccupancy/data/IMG_20250711_163700.jpg")
top_image_view = top_image_view.rotate(-90, expand=True)

DetectFeature = DetectFeature(method="sift")   

keypoints_n, descriptors_n = DetectFeature.detect_and_compute( np.array(normal_view) )
print(f"Number of keypoints in normal view: {len(keypoints_n)}. Shape of descriptors: {descriptors_n.shape}")

keypoints_top, descriptors_top = DetectFeature.detect_and_compute( np.array(top_image_view) )
print(f"Number of keypoints in top view: {len(keypoints_top)}. Shape of descriptors: {descriptors_top.shape}")

good_matches = DetectFeature.match_features(descriptors_n, descriptors_top)

H, matches, kp1, kp2, mask = DetectFeature.compute_homography(
   np.array(normal_view), np.array(top_image_view)
)

#H, matches, kp1, kp2, mask = DetectFeature.compute_homography(
#    np.array(top_image_view), np.array(normal_view)
#    )

matched_image = DetectFeature.draw_matches(
    np.array(normal_view), kp1,
    np.array(top_image_view), kp2, matches, mask
)   

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.subplot(1, 2, 1)
plt.imshow(normal_view)
plt.title("normal view")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(top_image_view)
plt.title("Top view")
plt.axis('off')
plt.show()

result_image = cv2.drawMatches(
    np.array(normal_view), keypoints_n,
    np.array(top_image_view), keypoints_top,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)   

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.subplot(1, 2, 1)
plt.imshow(result_image)
plt.title("draw matches")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(matched_image)
plt.title("from class")
plt.axis('off')
plt.show()

# warp the top view image using the computed homography
height, width = normal_view.size
warped_image = cv2.warpPerspective(
    np.array(top_image_view), H, (width, height)
)       

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.subplot(1, 2, 1)
plt.imshow(top_image_view)
plt.title("Normal top view")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(warped_image)
plt.title("Warped top view")
plt.axis('off')
plt.show()

# warping the normal view image using the inverse of the computed homography
H_inv = np.linalg.inv(H)
warped_normal_image = cv2.warpPerspective(
    np.array(normal_view), H_inv, (width, height)
) 

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.subplot(1, 2, 1)
plt.imshow(normal_view)
plt.title("Normal view")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(warped_normal_image)
plt.title("inverse Warped normal view")
plt.axis('off')
plt.show()