import torch
import os, requests
from torch import nn
from src.model.rcnn import RCNN
from src.model import transforms
from src.config import config
from src.model import visualize as vis
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class Prediction:

    def __init__(self, weights_path):
      self.model = RCNN(roi_res=100, pooling_type='square')
      self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    def rois_tensor(self, rois_file_path):

        with open(rois_file_path) as f:
            self.coordinates = json.load(f) 
        rois_tensor = self._formated_coordinates()

        return rois_tensor

    def _formated_coordinates(self):
        
        coordinate_list  = [self._parse_coordinates(val) for _, val in self.coordinates.items()]
        return torch.tensor(coordinate_list, dtype=torch.float32)

    def _parse_coordinates(self,  coordinates):

        coord_str = coordinates.replace('[', '').replace(']', '')
        points = coord_str.split('), (')
        points = points[0].split(", ")
        coordinates = [[int(points[i]), int(points[i+1])] for i in range(0, len(points), 2)]

        return coordinates
    
    def normalize_coordinates(self, rois_tensor, w, h):
        rois_tensor[:, :, 0] /= (w - 1)
        rois_tensor[:, :, 1] /= (h - 1)
        return rois_tensor
    
    def preprocess(self, image, res=None):
        """
        Resizes, normalizes, and converts image to float32.
        """
        # resize image to model input size
        if res is not None:
            image = TF.resize(image, res)

        # convert image to float
        image = image.to(torch.float32) / 255

        # normalize image to default torchvision values
        image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        return image
    
    def predictions(self, image, rois_tensor): 
        class_logits = self.model(image, rois_tensor) 
        class_scores = class_logits.softmax(1)[:, 1]
        return class_scores
    
    def tensor_pred_label(self, scores, threshold, image: torch.Tensor) -> torch.Tensor:
        pred_labels = [ 1 if s > threshold else 0 for s in scores ]
        tensor_pred_label = torch.tensor(pred_labels)
        return tensor_pred_label
    
    def show_prediction(self, image_path, parking_place_coordinates_path, show=True):
        
        image = torchvision.io.read_image(image_path)
        w, h = image.shape[2], image.shape[1]

        rois_tensor = self.rois_tensor(parking_place_coordinates_path)
        rois_tensor_normalized = self.normalize_coordinates(rois_tensor, w, h)

        image = self.preprocess(image)
        pred_scores = self.predictions(image, rois_tensor_normalized)
        tensor_pred_label = self.tensor_pred_label(pred_scores, 0.5, image)
        if show:
            vis.plot_ds_image(image, rois_tensor_normalized, tensor_pred_label, show=show)

    def save_prediction_to_file(self, tensor_pred_label, rois_tensor, file_path, format="json"):
        
        pred_dict = {"focal_name": os.path.basename(file_path)}
        for idx, label in enumerate(tensor_pred_label):
            pred_dict[f"slot_{idx}"] = {
                "coordinates": rois_tensor[idx].tolist(),
                "predicted_occupancy": int(label.item())
            }
        
        with open(file_path, 'w') as f:
            if format == "json":
                json.dump(pred_dict, f, indent=4)
            else:
                for slot, data in pred_dict.items():
                    f.write(f"{slot}: {data}\n")
           
# def parse_coordinates(coord_str):
#     coord_str = coord_str.replace('[', '').replace(']', '')
#     points = coord_str.split('), (')
#     points = points[0].split(", ")
#     coordinates = [[int(points[i]), int(points[i+1])] for i in range(0, len(points), 2)]
#     return coordinates

# with open("/home/besttic-rd/Documents/besttic/parkOccupancy/data/parking_place_coordinates2.json") as f:
#     custum_coordinates = json.load(f)
    
weights_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/src/notebook/weights.pt"  
Predictor = Prediction(weights_path=weights_path)

# image = torchvision.io.read_image( "/home/besttic-rd/Documents/besttic/parkOccupancy/data/camera_frame.png" )
# w, h = image.shape[2], image.shape[1]

# rois_tensor = Predictor.rois_tensor("/home/besttic-rd/Documents/besttic/parkOccupancy/data/parking_place_coordinates2.json")
# rois_tensor_normalized = Predictor.normalize_coordinates(rois_tensor, w, h)

# image = Predictor.preprocess(image)
# pred_scores = Predictor.predictions(image, rois_tensor_normalized)

# tensor_pred_label = Predictor.tensor_pred_label(pred_scores, 0.5, image)

# split_list = [parse_coordinates(val) for _, val in custum_coordinates.items()]
# torch_tensor_coordinates = torch.tensor(split_list, dtype=torch.float32)
# torch_tensor_coordinates[:, :, 0] /= (w - 1)
# torch_tensor_coordinates[:, :, 1] /= (h - 1)

# vis.plot_ds_image(image, torch_tensor_coordinates, tensor_pred_label, show=True)

image = torchvision.io.read_image( "/home/besttic-rd/Documents/besttic/parkOccupancy/data/camera_frame.png" )
w, h = image.shape[2], image.shape[1]
rois_tensor = Predictor.rois_tensor("/home/besttic-rd/Documents/besttic/parkOccupancy/data/parking_place_coordinates2.json")
rois_tensor_normalized = Predictor.normalize_coordinates(rois_tensor, w, h)
image = Predictor.preprocess(image)
pred_scores = Predictor.predictions(image, rois_tensor_normalized)
tensor_pred_label = Predictor.tensor_pred_label(pred_scores, 0.5, image)
vis.plot_ds_image(image, rois_tensor, tensor_pred_label, show=False, fname="custum_image_result.png")

image_pred_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/camera_frame_1.png"
parking_place_coordinates_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/parking_place_coordinates2.json"
Predictor.show_prediction(image_pred_path, parking_place_coordinates_path)
image_focal_0_pred_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/frame_focale_0.jpg"
parking_place_coordinates_focal_0_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/parking_place_coordinate_focale_0_1.json"

rois_tensor = Predictor.rois_tensor(parking_place_coordinates_focal_0_path)
rois_tensor_focale_0_normalized = Predictor.normalize_coordinates(rois_tensor, w, h)
Predictor.show_prediction(image_focal_0_pred_path, parking_place_coordinates_focal_0_path)
image_focal_0_pred_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/frame_focale_0_1.jpg"
parking_place_coordinates_focal_0_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/parking_place_coordinate_focale_0_1.json"
Predictor.show_prediction(image_focal_0_pred_path, parking_place_coordinates_focal_0_path, show=False)
# =============================================================================================================================================
image_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/frame_focale_0_1.jpg"
rois_path  = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/parking_place_coordinate_focale_0.json"

image      = torchvision.io.read_image(image_path)
w, h       = image.shape[2], image.shape[1]

image      = Predictor.preprocess(image)

rois_tensor = Predictor.rois_tensor(rois_path)
rois_tensor_normalized = Predictor.normalize_coordinates(rois_tensor, w, h) 

scores = Predictor.predictions(image, rois_tensor)
tensor_pred_label = Predictor.tensor_pred_label(scores, 0.5, image)

Predictor.show_prediction(image_path, rois_path, show=False)

# =============================================================================================================================================
image_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/frame_focale_1.jpg"
rois_path  = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/parking_place_coordinate_focale_1.json"

image      = torchvision.io.read_image(image_path)
w, h       = image.shape[2], image.shape[1]

image      = Predictor.preprocess(image)

rois_tensor = Predictor.rois_tensor(rois_path)
rois_tensor_normalized = Predictor.normalize_coordinates(rois_tensor, w, h) 

scores = Predictor.predictions(image, rois_tensor)
tensor_pred_label = Predictor.tensor_pred_label(scores, 0.5, image)

Predictor.show_prediction(image_path, rois_path, show=False)

# =============================================================================================================================================
image_path = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/camera_frame_focal_2_20250919_100411.jpg"
rois_path  = "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/parking_place_coordinate_focale_2.json"

image      = torchvision.io.read_image(image_path)
w, h       = image.shape[2], image.shape[1]

image      = Predictor.preprocess(image)

rois_tensor = Predictor.rois_tensor(rois_path)
rois_tensor_normalized = Predictor.normalize_coordinates(rois_tensor, w, h) 

scores = Predictor.predictions(image, rois_tensor)
tensor_pred_label = Predictor.tensor_pred_label(scores, 0.5, image)

Predictor.show_prediction(image_path, rois_path, show=False)
Predictor.save_prediction_to_file(tensor_pred_label, rois_tensor, "/home/besttic-rd/Documents/besttic/parkOccupancy/data/scale/predictions_focale_2.json", format="json")
# =============================================================================================================================================