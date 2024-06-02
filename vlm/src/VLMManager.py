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
from ultralytics import RTDETR

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
        
        # init detr model
        self.detr_model = RTDETR("models/detr_model.pt")
        
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
        
        outputs = self.detr_model.predict(img)[0]
        bboxes = outputs.boxes
        
        xywh = bboxes.xywh.cpu().numpy()
        xyxy = bboxes.xyxy.cpu().numpy()
        classes = bboxes.cls.cpu().numpy()
        confidence = bboxes.conf.cpu().numpy()
                
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