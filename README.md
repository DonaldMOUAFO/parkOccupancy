# parkOccupancy
`parkOccupancy` is a parking management software. It integrates two main functionalities related to parking place availabitlity. Its evelalutes the availability by either counting the number of detected vehicules and deduices the number of empty places from the total number of parking places available in the camera field. The second functionality consistes of detecting the occupancy of each parking place individualy and number of available/occupied parking place is deduices directely from the number of positive/negative predictions. 

# 1. Parking place availability

The availability is evaluted in 

The first fonctionaly consiste of counting parking places availabitity based on occupancy detection. The repository presents a parking occupancy classification. It consists of two steps. First, the localization of parking place. Second, the binary classification of the parking place as occupied or unoccupied.

# 2. parkingOccupancy set up
For the software to classify the occupancy of parking places, the coordinates of such places has to be known. Therefore, we implemented a module to manualy collect of coordinates of parking place location. This module is based on `OpenCV`'s mouse callback function.

## 2.1. Collection of parking place ROIs 
To collect the parking place ROIs, run the application with the following code. 
```
  python main.py -c false -sp false -hm true -cpc True spc True -fr focal_0 -i ../../data/camera_frame.png
```
The code loads the image of the parking indicated by the argument `-i`. The collection of ROIs is done by conducting counter clockwise on the rois corners.

### 2.1.1 Arguments
Here is the description of the arguments used to control the behavior of the application differents arguments are used including :
 
 - `i` : 
  - desc : Provide the path to the image to locate the parking place on
  - value : Path/to/image/location. 
  - Default : `../../data/camera_frame.png`
     
 - `cpc` :
  - desc : This arguments is used to explicitely state if yes or not the user intent to collect the parking places coordinates for the camera view.
  - Value : False or True. 
  - Default : False
	
- `spc` : 
  - desc : This arguments is used to explicitely state if yes or not the collected parking place coordinates should be saved. This is to leave the latitude to the user to show case of ROI collections without saving. 
  - value : False or True. If True the collected parking coordinates are save to the default file `../../data/scale/parking_place_coordinates.json` which can be modify with the parameter `-pp` as follow `python main.py -c false -sp false -hm true -cpc True spc True -pp path/to/rois/file.json -i ../../data/camera_frame.png`.  
  - Default : False
  
- `pp` : 
  - desc : Path to the file to save the collected slot_coordinates.
  - Default : `../../data/scale/parking_place_coordinates.json`
  - value : path/to/rois_file.json
 
- `fr` :
  - desc : Arguments used to precise the Frame identifier to differentiate saving files.
  - Value : string. The focale name. 
  - Default : focale_0

## 2.2. Points correspondance for Homography estimation 
To run the software to collect points correspondance in camera view and bird's eye view, run the following code.
```
  python main.py -c True -sp True -pcf ../../data/scale/homography/points_correspondance_focal_0.json -fr focal_0 -hm True
```
Before, to run the code, make sure to prepare the frame and bird's eyes view images with the respectives names "frame_focale_0.jpg" and "frame_focale_0_bird_view.png" place both images in the directory `~/data/scale/homography/`.
Running the previous code open the two images. To collect the points, clics at different locations of each images making sure each points in one image has its exact corresponding in the other image. Those points are thus saved in the file `~/data/scale/parking_place_coordinates.json`.

### 2.2.1 Arguments
Here is the description of the arguments used to control the behavior of the application to colect points correspondance :
- `c` : 
  - desc : This arguments is used to explicitely state if true or false the user intents to collect corresponding points for Homography estimation.
  - value : False or True.
  - Default : False
 
- `sp` : 
  - desc : Used to explicitely state if true or false the user intents to save the collected points corresponding to a file.
  - value : False or True.   
  - Default : False

- `pcf`:
  - desc : This aguments specified the file path to save the collected points correspondance
  - value : path/to/file.json
  - default : config.points_correspondance_file = ~/data/parking_points_correspondance.json

- `hm` : 
  - desc : Estimate homagraphy matrix from the collected points.
  - value : False or True. If the value is True, the homography is estimated and saved in a file "../../data/homography_matrix.npy"
  - Default : False "../../data/homography_matrix.npy"
  
- `fr` :
  - desc : Arguments used to precise the Frame identifier to differentiate saving files.
  - Value : string. The focale name : 0, 1 or 2. 
  - Default : 0

# 3. Parking place counting

The implementation of parking place counting is implemented in two way as previously mentionned. The first approach consist of vehicle detection tracking and motion evaluation. While the second approach is based on occupancy classification. 

# 3.1. Counting based un detection tracking and motion evaluation

In the first appraoch, the detected vahicucle are tracked and frame by frame movement evaluation is conducted and each vahicle is labeled as parked if it is static, meaning the `MOTION_THRESHOLD < 3 px`.
To execute the application in this mode, run the following code  :
```
  python main.py -ppc true -fr 0
```
- `ppc` :
  - desc : Run parking place counting based on vehicle detection tracking and motion evaluation.
  - Value : string.  
  - Default : True
Running this code opens a IU to show the video flux together with predictions. Here, the argument `-fr` with values 0, 1 or 2 enables to choose the focal to run the model on. 
This predictions are saved at one minute frequence in a file located at `~/data/data_file.json` or `~/data/data_file.csv`. There are options to save data either in a single file or multiples files, this can be controled with the argument `muliple_files=False` in the function `save_prediction`. 
Together with the data file the illustration image of the frame state at the moment of saving moment is saved at the path `~/data/data_file_timestamps.jpg`.

***** TO DO *****
Write the fuction to joblib the homography. check the transformation matrice.
