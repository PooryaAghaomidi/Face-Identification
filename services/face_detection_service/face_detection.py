import logging
import numpy as np
from time import time
from ultralytics import YOLO
from typing import List, Tuple, Any


class FaceDetection:
    def __init__(self, model_path: str = 'models/yolov8n-face.pt', confidence: float = 0.8):
        """
        Initializes the FaceDetection object with the YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence (float): Confidence threshold for detecting faces.
        """
        logging.info("Initializing the Face Detection Service ...")
        print("Initializing the Face Detection Service ...")

        self.model = YOLO(model_path)
        self.model.conf = confidence

    def detect(self, image: np.ndarray) -> tuple[list[tuple[Any, Any, Any, Any]], float]:
        """
        Detects faces in the provided image.

        Args:
            image (np.ndarray): The input image in which to detect faces.

        Returns:
            List[Tuple[int, int, int, int]]: A list of tuples, each containing the coordinates
                                             (y1, x2, y2, x1) of detected faces.
        """
        detection_time = time()
        results = self.model(image, verbose=False)
        detected_faces = []

        # Iterate through detected bounding boxes and convert to integer coordinates
        for bbox in results[0].boxes.xyxy:
            # Extract and convert bounding box coordinates to integers
            x1, y1, x2, y2 = map(int, bbox.cpu().numpy())  # Ensures the bounding box is in CPU and as integers

            # Append the coordinates in (y1, x2, y2, x1) order to detected_faces list
            detected_faces.append((y1, x2, y2, x1))

        return detected_faces, round(time() - detection_time, 3)
