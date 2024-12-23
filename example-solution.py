"""Count the number of people in a room"""
import cv2
import supervision as sv
from roboflow import Roboflow
import numpy as np
import json
import time

# load config
with open('roboflow-info.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_MODEL_VERSION = config["ROBOFLOW_MODEL_VERSION"]

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_MODEL)
model = project.version(ROBOFLOW_MODEL_VERSION).model

#change input number from 0 to change the camera in use
video = cv2.VideoCapture(0)
time.sleep(5)

def infer():
    """ Use roboflow to get the JSON with predictions from the NN"""
    # Get the current image from the webcam
    _, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, _ = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # infer on one local frame
    predictions = model.predict(img, confidence=40, overlap=30).json()['predictions']

    if len(predictions) > 0:

        # Convert predictions to the sv.Detections format
        xyxy = []
        class_ids = []
        confidences = []

        for pred in predictions:
            x0 = pred['x'] - pred['width'] / 2
            y0 = pred['y'] - pred['height'] / 2
            x1 = pred['x'] + pred['width'] / 2
            y1 = pred['y'] + pred['height'] / 2
            xyxy.append([x0, y0, x1, y1])
            class_ids.append(pred['class_id'])
            confidences.append(pred['confidence'])

        # Convert lists to numpy arrays
        xyxy = np.array(xyxy)
        class_ids = np.array(class_ids)
        confidences = np.array(confidences)

        # Create sv.Detections object
        detections = sv.Detections(
            xyxy=xyxy,
            class_id=class_ids,
            confidence=confidences
        )

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = bounding_box_annotator.annotate(
            scene=img.copy(),
            detections=detections
        )

        cv2.imshow("Live footage", annotated_frame)

    else:
        cv2.imshow('Live footage', img)


# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if cv2.waitKey(1) == ord('q'):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    infer()

# Release resources when finished
video.release()
cv2.destroyAllWindows()