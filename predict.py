import asyncio
import sys
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import json
import os

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load Detectron2 model
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file("config.yaml")
    cfg.MODEL.DEVICE='cpu'  # Load your Detectron2 config
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detections
    cfg.MODEL.WEIGHTS = "model_final.pth"  # Load your trained model
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

predictor, cfg = load_model()

# Function to perform inference
async def predict(image_path, cfg):
    # Convert base64 image to numpy array
    image = Image.open(image_path)
    image = np.array(image)[:, :, ::-1]  # Convert to BGR

    # Perform inference
    outputs = predictor(image)["instances"]

    # Visualize predictions on the image
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs.to("cpu"))
    processed_image = v.get_image()[:, :, ::-1]

    # Save the processed image to a file
    processed_image_path = "processed_image.jpg"
    Image.fromarray(processed_image).save(processed_image_path)

    # Process output to extract bounding boxes
    return {"processedImagePath": processed_image_path}

# Perform inference and print results as JSON
async def main():
    image_path = sys.argv[1]  # Assuming the image path is passed as argument
    response = await predict(image_path, cfg)
    print(json.dumps(response))

# Run the event loop to execute the async function
asyncio.run(main())
