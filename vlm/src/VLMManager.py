from typing import List
import os
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision


from ultralytics import YOLOWorld 
from PIL import Image



class VLMManager:
    def __init__(self):
        self.best_model_path = os.path.join("best_model", "best.pt")
        self.pretrained_model_path = os.path.join("best_model", "yolov8s-worldv2.pt")
        self.model = YOLOWorld(self.best_model_path)
        self.original_classes = ['grey missile', 'red, white, and blue light aircraft', 'green and black missile', 'white and red helicopter', 'grey camouflage fighter jet', 'grey and white fighter plane', 'white and black drone', 'white and black fighter jet', 'white missile', 'black and white commercial aircraft', 'grey drone', 'yellow, red, and grey helicopter', 'yellow commercial aircraft', 'black cargo aircraft', 'yellow helicopter', 'white and black light aircraft', 'grey and black fighter plane', 'red fighter plane', 'blue helicopter', 'white, red, and green fighter plane', 'black camouflage fighter jet', 'green light aircraft', 'blue and yellow fighter jet', 'white fighter jet', 'red fighter jet', 'blue and white light aircraft', 'white and black helicopter', 'white and blue fighter plane', 'grey commercial aircraft', 'blue and grey fighter jet', 'green and brown camouflage fighter jet', 'red and grey missile', 'red and white fighter jet', 'orange light aircraft', 'yellow light aircraft', 'white and red light aircraft', 'white and grey helicopter', 'blue, yellow, and green fighter plane', 'yellow and red light aircraft', 'blue and white missile', 'green and white fighter plane', 'blue missile', 'grey, red, and blue commercial aircraft', 'white light aircraft', 'grey and white light aircraft', 'blue and yellow helicopter', 'white fighter plane', 'white and blue fighter jet', 'blue camouflage fighter jet', 'yellow and green helicopter', 'silver fighter plane', 'blue and red light aircraft', 'white and black cargo aircraft', 'green and yellow fighter plane', 'white and blue cargo aircraft', 'blue and red commercial aircraft', 'blue, yellow, and white cargo aircraft', 'white and yellow commercial aircraft', 'white and red missile', 'white cargo aircraft', 'grey helicopter', 'grey and red commercial aircraft', 'white drone', 'yellow, black, and red helicopter', 'white and blue helicopter', 'green and grey helicopter', 'black and brown camouflage helicopter', 'blue and green fighter plane', 'green missile', 'grey cargo aircraft', 'yellow fighter jet', 'yellow, red, and blue fighter plane', 'grey and red missile', 'orange and black fighter jet', 'white and blue light aircraft', 'white and black fighter plane', 'grey and green cargo aircraft', 'blue commercial aircraft', 'grey fighter jet', 'black fighter plane', 'white, black, and red drone', 'blue and white commercial aircraft', 'red, white, and blue fighter jet', 'white, black, and grey missile', 'black fighter jet', 'red and white missile', 'white and orange light aircraft', 'white and red commercial aircraft', 'yellow fighter plane', 'silver and blue fighter plane', 'grey and red fighter jet', 'red helicopter', 'black and white missile', 'grey and black helicopter', 'red and white light aircraft', 'green and black camouflage helicopter', 'black and orange drone', 'grey and yellow fighter plane', 'green camouflage helicopter', 'black drone', 'white and blue commercial aircraft', 'blue and white helicopter', 'green fighter plane', 'red and black drone', 'white and orange commercial aircraft', 'green helicopter', 'black helicopter', 'white, red, and blue commercial aircraft', 'black and yellow missile', 'yellow and black fighter plane', 'white, blue, and red commercial aircraft', 'grey fighter plane', 'red light aircraft', 'green and brown camouflage fighter plane', 'blue, yellow, and black helicopter', 'grey light aircraft', 'white commercial aircraft', 'green and brown camouflage helicopter', 'white and red fighter plane', 'red and white fighter plane', 'red and white helicopter', 'black and white cargo aircraft', 'white helicopter', 'black and yellow drone', 'yellow missile', 'white and red fighter jet']

        # self.model.to('cpu')
    
    def process_result(self, results, caption) -> List[int]:
        results = results[0]
        bboxes_result = results.boxes
        xywh = bboxes_result.xywh.cpu().numpy()
        classes = bboxes_result.cls.cpu().numpy()
        confidence = bboxes_result.conf.cpu().numpy()
        
        if len(confidence) == 0:      
            return [0,0,0,0]
        
        max_conf_idx = np.argmax(confidence)
        x1, y1, w, h = xywh[max_conf_idx]
        x1 = x1 - w / 2
        y1 = y1 - h / 2
        output = [int(x1), int(y1), int(w), int(h)]
        print(f"Output for {caption}: {output}")
        
        return output
    
    def process_result_again(self, results, caption) -> List[int]:
        type_of_aircraft = caption.split(" ")[-1]
        results = results[0]
        bboxes_result = results.boxes
        xywh = bboxes_result.xywh.cpu().numpy()
        classes = bboxes_result.cls.cpu().numpy()
        confidence = bboxes_result.conf.cpu().numpy()
        classes_with_name = [self.model.names[class_id] for class_id in classes]
        
        output = [0,0,0,0]
        
        if len(confidence) == 0:      
            return [0,0,0,0]
        
        max_conf = 0
        max_conf_idx = 0
        for i, (x, y, w, h) in enumerate(xywh):
            if type_of_aircraft in classes_with_name[i]:
                if confidence[i] > max_conf:
                    max_conf = confidence[i]
                    max_conf_idx = i
                    
        if max_conf == 0:
            max_conf_idx = np.argmax(confidence)
            x1, y1, w, h = xywh[max_conf_idx]
            x1 = x1 - w / 2
            y1 = y1 - h / 2
            output = [int(x1), int(y1), int(w), int(h)]
        else:
            x, y, w, h = xywh[max_conf_idx]
            x = x - w / 2
            y = y - h / 2
            output = [int(x), int(y), int(w), int(h)]
        
        return output

        

    def identify(self, image: bytes, caption: str) -> List[int]:
        # open image
        img = Image.open(io.BytesIO(image)).convert("RGB")
        img_wh = img.size
        self.model.set_classes([caption])
        results = self.model.predict(img, conf=0.1)
        bbox = self.process_result(results, caption)
        
        if bbox == [0,0,0,0]:
            self.model.set_classes(self.original_classes)
            results = self.model.predict(img, conf=0.1)
            bbox = self.process_result_again(results, caption)
        
        return bbox
        
        
        

if __name__ == "__main__":
    with open("../../../advanced/images/image_1000.jpg", "rb") as file:
        image_bytes = file.read()
    
    manager = VLMManager()
    print(manager.identify(image_bytes, "grey missile"))
