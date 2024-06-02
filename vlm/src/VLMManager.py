from typing import List
import os
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
# import torchvision

from PIL import Image
# from torchvision import models, transforms as T
# from torchvision.utils import save_image
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import AutoTokenizer, AutoImageProcessor, VisionTextDualEncoderProcessor, VisionTextDualEncoderConfig, VisionTextDualEncoderModel
from ultralytics import RTDETR, YOLOWorld

# class FRCNNModelWrapper:
#     def __init__(self, model, n_classes, device="cpu", weights=None):
#         self.model = model
#         self.device = device
#         self.n_classes = n_classes
#         self.weights = weights
#         self.config_model(n_classes, weights)

#     def config_model(self, out_classes, weights):
#         # configurate last layer of bbox predictor
#         in_feats_bbox = self.model.roi_heads.box_predictor.cls_score.in_features
#         self.model.roi_heads.box_predictor = FastRCNNPredictor(in_feats_bbox, out_classes)

#         if weights is not None:
#             self.model.load_state_dict(weights)

#         self.model.to(self.device)

#     def predict(self, img):
#         # set model to evaluation mode to get detections
#         self.model.eval()

#         # do not record computations for computing the gradient
#         with torch.no_grad():
#             img = img.to(self.device)
#             output = self.model(img.unsqueeze(0))

#         return output

class VLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.transform = T.Compose([
#             T.Resize(224),
#             T.ToTensor(),
#             T.Normalize([0.4986, 0.5428, 0.5563], [0.2721, 0.2717, 0.3078]),
#         ])
        
#         # init faster rcnn model
#         self.frcnn_model = FRCNNModelWrapper(
#             model=torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT"),
#             n_classes=7,
#             device=self.device,
#             weights=torch.load("rcnn_model.pt", map_location=self.device),
#         )
        
        # init detr model
        self.detr_model = RTDETR("models/detr_model.pt")
        # self.detr_model = YOLOWorld("models/yoloworld_model.pt")
        
        # init clip model
        tokenizer = AutoTokenizer.from_pretrained("models/clip_model")
        image_processor = AutoImageProcessor.from_pretrained("models/clip_model")
        self.clip_processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
        self.clip_model = VisionTextDualEncoderModel.from_pretrained("models/clip_model")
        self.clip_model.to(self.device)

    def identify(self, image: bytes, caption: str) -> List[int]:
        # open image
        img = Image.open(io.BytesIO(image)).convert("RGB")
        # img_wh = img.size
        
        # transform and predict on image
        # inputs = self.transform(img)
        # outputs = self.frcnn_model.predict(inputs)[0]
        
        outputs = self.detr_model.predict(img, conf=0.1)[0]
        bboxes = outputs.boxes
        
        xywh = bboxes.xywh.cpu().numpy()
        xyxy = bboxes.xyxy.cpu().numpy()
        classes = bboxes.cls.cpu().numpy()
        confidence = bboxes.conf.cpu().numpy()
        
#         # get resize scale 
#         x_scale = img_wh[0] / inputs.shape[2]
#         y_scale = img_wh[1] / inputs.shape[1]
                
#         # get rescale bboxes to original resolution
#         bboxes = []
#         for bbox in outputs["boxes"]:
#             x1, y1, x2, y2 = bbox
#             scaled_bbox = [int(x1 * x_scale), int(y1 * y_scale), int(x2 * x_scale), int(y2 * y_scale)]
            
#             if scaled_bbox[2] - scaled_bbox[0] > 0 and scaled_bbox[3] - scaled_bbox[1] > 0:
#                 bboxes.append(scaled_bbox) 
                
        if len(xywh) == 0:
            return [0, 0, 0, 0]
        
        crop_imgs = [img.crop(bbox) for bbox in xyxy]
        
        # preprocess and predict with clip
        inputs = self.clip_processor(
            text=[caption],
            images=crop_imgs,
            return_tensors="pt",
            padding=True
        )
        inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        
        logits_per_img = outputs.logits_per_image.to("cpu")
        prob, idx = logits_per_img.softmax(dim=0).squeeze(1).max(dim=0)
        # print(logits_per_img)
        
        # x1, y1, x2, y2 = bboxes[idx]
        # x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        # crop_imgs[idx].save("img.jpg")
        x, y, w, h = xywh[idx]
        x = x - w / 2
        y = y - h / 2
        return [int(x), int(y), int(w), int(h)]

if __name__ == "__main__":
    with open("../../../advanced/images/image_1000.jpg", "rb") as file:
        image_bytes = file.read()
    
    manager = VLMManager()
    manager.identify(image_bytes, "grey helicopter")