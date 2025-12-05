import cv2 
import json
import time
import random
from os import path
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from src.config.config import (points_correspondance_file, HOMAGRAPHY_MATRIX_PATH)

class DetectFeature:
    
    def __init__(self, method="sift", from_pts=False):
        """Class to detect features in images using SIFT or ORB and compute homography."""
        self.FROM_PTS = from_pts
        self.method = method.lower()

        if self.FROM_PTS:
            if not path.exists(points_correspondance_file):
                raise FileNotFoundError(
                    f"""Points correspondance file not found at {points_correspondance_file}. 
                        check the file and load again."""
                )
            else:
                with open(points_correspondance_file, 'r') as file:
                    self.points_correspondance = json.load(file)
                    if len(self.points_correspondance) < 10:
                        raise ValueError("Not enough points to compute homography.")
                    else :
                        self.pts_correspondance = list( self.points_correspondance.values() )
        else :
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

            # Convert points to numpy arrays
            kp1 = np.float32([pt[0] for pt in self.pts_correspondance]).reshape(len(self.pts_correspondance), 2) 
            kp2 = np.float32([pt[1] for pt in self.pts_correspondance]).reshape(len(self.pts_correspondance), 2) #reshape(-1, 1, 2) # bird_eye_pts
            # Compute homography using RANSAC
            H, mask = cv2.findHomography(kp1, kp2, cv2.RHO, 2.0) # cv2.RANSAC, cv2.RHO, cv2.LMEDS, 5.0)
            # return H, None, None, None, mask

            #matches = self.match_features(kp1, kp2)
            matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(self.pts_correspondance))]
        
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
            self.save_homography(H, filename=HOMAGRAPHY_MATRIX_PATH)
        return H, matches, kp1, kp2, mask
    
    def compute_homography_from_points(self, method='linear'):
        """ Invoking this method requires From_PTS to be True.
            Compute homography matrix from world and image points.
        """

        if method not in ['linear', 'svd']:
            raise ValueError("Method must be either 'linear' or 'svd'.")

        world_pts = np.float32([pt[0] for pt in self.pts_correspondance]).reshape(len(self.pts_correspondance), 2) 
        img_pts   = np.float32([pt[1] for pt in self.pts_correspondance]).reshape(len(self.pts_correspondance), 2) 

        if world_pts.shape[1] == 2:
            world_pts_array = np.pad(world_pts, [(0, 0), (0, 1)], mode='constant', constant_values=1)
            img_pts_array = np.pad(img_pts, [(0, 0), (0, 1)], mode='constant', constant_values=1)
        else:
            world_pts_array = world_pts
            img_pts_array = img_pts

        A = []
        for i in range(world_pts_array.shape[0]):
            # Solve the linear system Ax = 0
            # where A is a 2n x 9 matrix and x is the vector of homography parameters
            # A.append([-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp])
            # A.append([0, 0, 0, -x, -y, -1, yp*x, yp*y, yp])
            #A.append([0, 0, 0, *-img_pts_array[i, :], *world_pts_array[i, 1]*img_pts_array[i, :]])
            #A.append([*img_pts_array[i, :], 0, 0, 0, *-world_pts_array[i, 0]*img_pts_array[i, :]])

            A.append( [*-world_pts_array[i, :], 0, 0, 0, *img_pts_array[i, 0]*world_pts_array[i, :] ] )
            A.append( [0, 0, 0, *-world_pts_array[i, :],  *img_pts_array[i, 1]*world_pts_array[i, :]] )
        
        A = np.array(A)
        if method == 'svd':
            # Solve homogenous least squares Ah = 0, when ||h|| = 1
            # Solution is the eigenvector corresponding to minimum eigenvalue
            _, _, Vt = np.linalg.svd(A)
            h = Vt[-1, :] / Vt[-1, -1] # Normalize so that h[8] = 1

        elif method == 'linear':
            # Solve the linear system Ah = 0, where h is the vector of homography
            # parameters. The solution is the eigenvector corresponding to the minimum eigenvalue.
            ATA = np.dot(A.T, A)
            eigenvalues, eigenvectors = np.linalg.eig(ATA)  
            h = eigenvectors[:, np.argmin(eigenvalues)].real

        # Reshape h to a 3x3 matrix
        if h.shape[0] != 9:
            raise ValueError("Homography vector must have 9 elements.")
    
        H = h.reshape((3, 3))
        self.save_homography(H, filename=HOMAGRAPHY_MATRIX_PATH)
        return H
        
    def draw_matches(self, img1, kp1, img2, kp2, matches, mask):
        """Draw inliers only"""
        if self.FROM_PTS :
            # If using pre-defined points, we don't have keypoints to draw
            kp1 = [cv2.KeyPoint(x=kp1[i][0], y=kp1[i][1], size=1) for i in range(kp1.shape[0])]
            kp2 = [cv2.KeyPoint(x=kp2[i][0], y=kp2[i][1], size=1) for i in range(kp2.shape[0])]
            
        matched_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None, 
            matchColor=(0,255,0),
            matchesMask=mask.ravel().tolist(), 
            flags=cv2.DrawMatchesFlags_DEFAULT #2
        )
 
        return matched_img
    
    def draw_matches_canvas(self, image1, kp1, image2, kp2):
        """
        Draws matches between two images.
        
        Parameters:
        - image1: First image.
        - kp1: Keypoints in the first image.
        - image2: Second image.
        - kp2: Keypoints in the second image.
        - matches: Matches between the keypoints.
        - mask: Optional mask to filter matches.
        
        Returns:
        - Image with matches drawn.
        """
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = np.array(image1)
        canvas[:h2, w1:w1 + w2] = np.array(image2)

        for i in range( len(kp1) ) :
            
            #pt1 = [cv2.KeyPoint(x=kp1[i][0], y=kp1[i][1], size=1) for i in range(kp1.shape[0])]
            #pt2 = [cv2.KeyPoint(x=kp2[i][0], y=kp2[i][1], size=1) for i in range(kp2.shape[0])]

            # Draw thick lines and circles
            color = (random.randint(100, 255), random.randint(0, 100), random.randint(0, 255))
            cv2.line(canvas, ( int(kp1[i][0]), int(kp1[i][1]) ),
                    ( int(kp2[i][0]) + w2, int(kp2[i][1]) ), color, thickness=4)  # Line


            cv2.circle(canvas, ( int(kp1[i][0]), int(kp1[i][1]) ), 
                       radius=25, color=(0, 255, 0), thickness=8)  # Filled circle
            cv2.circle(canvas, ( int(kp1[i][0]), int(kp1[i][1]) ), 
                       radius=12, color=(0, 255, 0), thickness=-1)


            cv2.circle(canvas, ( int(kp2[i][0]) + w2, int(kp2[i][1]) ), 
                       radius=25, color=(0, 255, 0), thickness=8)
            cv2.circle(canvas, ( int(kp2[i][0]) + w2, int(kp2[i][1]) ), 
                       radius=25, color=(0, 255, 0), thickness=-1)

        return canvas
    
    def save_homography(self, H, frame,  filename=HOMAGRAPHY_MATRIX_PATH):
        """Save the homography matrix to a .npy file."""
        if not filename.endswith('.npy'):
            raise ValueError("Filename must end with .npy")
        
        # add time stamp to the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = filename.replace('.npy', f'_{frame}_{timestamp}.npy')
        
        np.save(filename, H)
        print("=======================================================================")
        print(f"Homography matrix saved to {filename}")
        print("=======================================================================")
        

def show_outputs(camera_image_view_path, bird_view_path):   
    """ This function to show outputs of the homography estimation process the as wirped images with homography."""
    normal_view = Image.open(camera_image_view_path)
    normal_view = normal_view.rotate(-90, expand=True)

    top_image_view = Image.open(bird_view_path)
    top_image_view = top_image_view.rotate(-90, expand=True)

    # Initialize the DetectFeature class
    FeatureDetector = DetectFeature(method="sift", from_pts=True)   

    H, matches, kp1, kp2, mask = FeatureDetector.compute_homography(
        np.array(normal_view), np.array(top_image_view))
    # Compute homography using different methods
    H_linear = FeatureDetector.compute_homography_from_points(method='linear')
    H_svd    = FeatureDetector.compute_homography_from_points(method='svd')

    print(f"Homography matrix:\n{H}")
    print(f"Homography matrix linear:\n{H_linear}")
    print(f"Homography matrix svd:\n{H_svd}")

    matched_image = FeatureDetector.draw_matches(
        np.array(normal_view), kp1,
        np.array(top_image_view), kp2, matches, mask
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(normal_view)
    plt.title("normal view of class match")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(top_image_view)
    plt.title("Top view of class match")
    plt.axis('off')
    plt.show()

    # Now we can draw the matches using the class method
    # draw matches using the native cv2.drawMatches

    result_canvas = FeatureDetector.draw_matches_canvas(
        np.array(normal_view), kp1,
        np.array(top_image_view), kp2,
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(matched_image)
    plt.title("draw matches from class")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result_canvas)
    plt.title("from native custom class method")
    plt.axis('off')
    plt.show()

    # warp the top view image using the computed homography
    height, width = normal_view.size
    warped_image = cv2.warpPerspective(
        np.array(normal_view), H, (width, height)
    )       

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(normal_view)
    plt.title("Normal view")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(warped_image)
    plt.title("Warped view with homography from cv2.findHomography")
    plt.axis('off')
    plt.show()

    # warp the top view image using the computed homography
    height, width = normal_view.size
    warped_image = cv2.warpPerspective(
        np.array(normal_view), H_linear, (width, height)
    )       

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(normal_view)
    plt.title("Normal view")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(warped_image)
    plt.title("Warped view with linear homography")
    plt.axis('off')
    plt.show()

    warped_image = cv2.warpPerspective(
        np.array(normal_view), H_svd, (width, height)
    )       

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(normal_view)
    plt.title("Normal view")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(warped_image)
    plt.title("Warped view with SVD homography")
    plt.axis('off')
    plt.show()

    # warping the normal view image using the inverse of the computed homography
    H_inv = np.linalg.inv(H)
    warped_normal_image = cv2.warpPerspective(
        np.array(top_image_view), H_inv, (width, height)
    ) 

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(top_image_view)
    plt.title("Normal view")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(warped_normal_image)
    plt.title("inverse Warped normal view")
    plt.axis('off')
    plt.show()

    print("=======================================================================")
    print("Homography estimation and visualization completed.")
    print("=======================================================================")        

# show_outputs(camera_image_view_path, bird_view_path)
