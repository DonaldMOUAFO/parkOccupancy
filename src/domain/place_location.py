import cv2
import numpy as np
import json
#from application.utils import draw_contours
#from os import path
#from config import coordinate_file

coordinate_file = "../../data/parking_place_coordinates.json"
image_path = str("../../data/camera_frame.png")

class ParkingCoordinatesGenerator :

    KEY_RESET = ord("r")
    KEY_QUIT = ord("q")

    def __init__(self, image_path, output_file_object):
        self.camera_view_path = image_path
        self.ouput = output_file_object

        self.camera_view = cv2.imread(image_path).copy()
        self.click_count = 0
        self.ids = 0

        self.coordinates = []

        cv2.namedWindow(self.camera_view_path, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.camera_view_path, self.__mouse_callback)

    def __mouse_callback(self, envent, x, y):

        if envent == cv2.EVENT_LBUTTONDOWN :
            self.coordinates.append( [x, y] )
            self.click_count += 1

            if self.click_count > 1 :
                self.__draw_line()
            
            elif self.click_count >= 4 :
                self.__write_to_file()

        cv2.imshow(self.camera_view_path, self.camera_view)

    def __draw_line(self) :
        cv2.line(
            self.camera_view, 
            self.coordinates[-2], self.coordinates[-1], 
            (255, 0, 0), 1)
        
    def __write_to_file(self):

        cv2.line(self.image, self.coordinates[2], self.coordinates[3],
            (255, 0, 0), 1)
        cv2.line(self.image, self.coordinates[3], self.coordinates[0],
            (255, 0, 0), 1)
        
        self.click_count = 0

        coordinates = np.array( self.coordinates )

        place_dict = {
            "id" : self.id,
            "coordinates" : self.coordinates           
        }

        json.dump(place_dict, self.ouput, indent=4)

        draw_contours(self.camera_view, coordinates, str(self.ids+1), (255, 255, 255))

        self.id += 1
        self.coordinates = []

    #def mouse_call_back(self):
    #    cv2.namedWindow(self.camera_view_path, cv2.WINDOW_GUI_EXPANDED)
    #    cv2.setMouseCallback(self.camera_view_path, self.__mouse_callback)

    def generate(self):

        while True :

            cv2.imshow("Frame", self.camera_view)
            key = cv2.waitKey(0)

            if key == ParkingCoordinatesGenerator.KEY_RESET :
                self.image = self.image.copy()

            if key == ParkingCoordinatesGenerator.KEY_QUIT :
                break
        
        cv2.destroyAllWindows("Frame")


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

with open(coordinate_file, 'w') as output:
    Generator = ParkingCoordinatesGenerator(image_path, output)
    # Generator.mouse_call_back()
    Generator.generate()