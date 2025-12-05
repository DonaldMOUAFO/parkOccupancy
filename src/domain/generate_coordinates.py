import cv2
import json
import numpy as np
from src.config.config import coordinate_file, camera_frame_path, coordinate_file2

class CoordinateGenerator :
    
    KEY_QUIT = ord("q")

    def __init__(self, image_path):
        """image_path : path to the image on which the parking coordinates should be generated
        """

        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"""Image at {image_path} could not be loaded. 
                Please check the path.""")
        
        if self.image.dtype != 'uint8':
            print(f"""self.image.dtype = {self.image.dtype}, and size = {self.image.shape} """)
            self.image = ( 255 * self.image ).astype('uint8')

        self.image_processed = self.image.copy()
        
        self.window_name = "Parking Frame" 
        self.FRAME_OPEN = True

        self.slot_id = 0
        self.click_count = 0
        self.coordinates = []
        self.json_coordinate_format = {}
    
    def generate_coordinate(self, save="true", file_path=coordinate_file) :

        print(f"========save={save}, file_path={file_path}========")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) #cv2.WINDOW_GUI_EXPANDED)

        while self.FRAME_OPEN :

            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            key = cv2.waitKey(0)

            if key == CoordinateGenerator.KEY_QUIT:

                if save.lower() == "true" :
                    self._save_coordinate_to_json(file_path)

                self.FRAME_OPEN = False
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
    
    def _mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN :

            self.coordinates.append( [x, y] ) 
            self.click_count += 1

            if self.click_count > 1 :
                self._draw_line()
            
            if self.click_count == 4 :
                self._complete_polygone()
                self._add_text()
                self._save_coordinates()
                
                self.click_count = 0
                self.coordinates = []
                self.slot_id += 1
        
        cv2.imshow(self.window_name, self.image_processed) 
                
    def _handle_rectangle(self) :

        cv2.rectangle(
            self.image_processed, self.coordinates[-4], self.coordinates[-2], 
            (0, 255, 0), 2
        )
        
    def _draw_line(self):

        # Draw a line between the last two coordinates
        cv2.line(
            self.image_processed, self.coordinates[-2], 
            self.coordinates[-1], (0, 0, 255), 2
        ) 

    def _complete_polygone(self):

        # Draw a line between the first and the last coordinates
        # to close the rectangle
        cv2.line(
            self.image_processed, self.coordinates[-4], 
            self.coordinates[-1], (0, 0, 255), 2
        )
    
    def _add_text(self):

        pts = np.array( self.coordinates[-4:] )
        center = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            self.image_processed, f"{self.slot_id}",
            (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

    def _save_coordinates(self, file_path=coordinate_file, save="true"):

        coordinates_array = np.array(self.coordinates)
        
        coordinates_str = "{ id: " + str( self.slot_id ) + ", coordinates: [" + \
            "[" + str(coordinates_array[0][0]) + "," + str(coordinates_array[0][1]) + "]," + \
            "[" + str(coordinates_array[1][0]) + "," + str(coordinates_array[1][1]) + "]," + \
            "[" + str(coordinates_array[2][0]) + "," + str(coordinates_array[2][1]) + "]," + \
            "[" + str(coordinates_array[3][0]) + "," + str(coordinates_array[3][1]) + "]] } \n"
        
        if save.lower() == "true" :
            with open(file_path, "a") as f:             
                f.write( coordinates_str )

        print("=========================================================================")
        print(f"======== Parking slots Coordinates saved to {file_path}. =========")
        print("=========================================================================")

        # coordinates_str_json = { 
        #     "id":  self.slot_id,
        #     "coordinates" : [
        #         [ coordinates_array[0][0], coordinates_array[0][1] ], 
        #         [ coordinates_array[1][0], coordinates_array[1][1] ],
        #         [ coordinates_array[2][0], coordinates_array[2][1] ],
        #         [ coordinates_array[3][0], coordinates_array[3][1] ] 
        #     ]
        # }   
        
        # self.json_coordinate_format.append(
        #     { "id":  self.slot_id,
        #       "coordinates" : 
        #       f"[[{coordinates_array[0][0]}, {coordinates_array[0][1]}],\
        #         [{coordinates_array[1][0]}, {coordinates_array[1][1]} ],\
        #         [{coordinates_array[2][0]}, {coordinates_array[2][1]} ],\
        #         [{coordinates_array[3][0]}, {coordinates_array[3][1]} ] \
        #         ]"
        #     }
        # )

        # self.json_coordinate_format[self.slot_id] = str( 
        #     [
        #         [coordinates_array[0][0], coordinates_array[0][1]],
        #         [coordinates_array[1][0], coordinates_array[1][1]],
        #         [coordinates_array[2][0], coordinates_array[2][1]],
        #         [coordinates_array[3][0], coordinates_array[3][1]] 
        #     ]
        # )

        self.json_coordinate_format[self.slot_id] = str( self.coordinates ) 
    
    def _save_coordinate_to_json(self, file_path):

        print(f"self.json_coordinate_format = {self.json_coordinate_format}")
        
        with open(file_path, "w") as f:
            json.dump(self.json_coordinate_format, f, indent=4)
        
        print("=========================================================================")
        print(f"==== Parking slots Coordinates saved to {coordinate_file} completed. ===")
        print("=========================================================================")

#GeneratorCoordinate = CoordinateGenerator(camera_frame_path)
#GeneratorCoordinate.generate_coordinate()
