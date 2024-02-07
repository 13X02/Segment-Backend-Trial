from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import os
import io
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    image_path = "uploads/" + image_file.filename
    image_file.save(image_path)

    # Perform inference
    processed_result = predict_image(image_path)

    # Delete the uploaded image file
    os.remove(image_path)

    return jsonify(processed_result)

# Initialize the model once outside the function to avoid loading it on every request
cfg = get_cfg()
cfg.merge_from_file("config.yaml")
cfg.MODEL.DEVICE = 'cpu'  # Set to 'cuda' if you have GPU
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detections
cfg.MODEL.WEIGHTS = "model_final.pth"  # Load your trained model
predictor = DefaultPredictor(cfg)

def predict_image(image_path):
    # Read the image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Perform inference
    outputs = predictor(image)

    # Get the detected instances
    instances = outputs["instances"].to("cpu")

    # Get the metadata
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Get the category names
    category_names = metadata.get("thing_classes", None)

    # Initialize a dictionary to store the count of each instance
    instance_counts = {}

    # Loop through each detected instance
    for i in range(len(instances)):
        label = instances.pred_classes[i].item()
        if category_names:
            label_name = category_names[label]
        else:
            label_name = f"Instance {label}"
        # Increment the count for this instance type
        instance_counts[label_name] = instance_counts.get(label_name, 0) + 1

    # Visualize predictions on the image (optional)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_image = v.get_image()[:, :, ::-1]

    # Save the processed image on the server
    processed_image_path = os.path.join("static", "processed_images", "processed_image.jpg")
    Image.fromarray(processed_image).save(processed_image_path)

    # Generate the URL for the processed image
    processed_image_url = url_for("static", filename="processed_images/processed_image.jpg")

    # Process output to extract bounding boxes
    predicted_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    return {"numInstances": len(instances),            "instanceCounts": instance_counts,

        "boundingBoxes": predicted_boxes.tolist(), "processedImageURL": processed_image_url}


if __name__ == '__main__':
    app.run(debug=True)
