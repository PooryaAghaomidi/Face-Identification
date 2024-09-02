import logging
import numpy as np
import face_recognition
from time import time
from typing import List, Dict, Tuple


class FrameIdentification:
    def __init__(self, known_face_names: List[str], known_face_encodings: List[np.ndarray], threshold: float):
        """
        Initialize the FrameIdentification class with known faces and a verification threshold.

        Args:
            known_face_names (List[str]): List of names corresponding to the known face encodings.
            known_face_encodings (List[np.ndarray]): List of face encodings for known individuals.
            threshold (float): Threshold for face verification (e.g., 0.6 is commonly used).
        """
        logging.info("Initializing the Face Recognition Service ...")
        print("Initializing the Face Recognition Service ...")

        self.known_face_names = known_face_names
        self.known_face_encodings = known_face_encodings
        self.threshold = threshold

    def verify_frame(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> dict:
        """
        Verify all faces in a video frame.

        Args:
            frame (np.ndarray): The video frame to process.
            face_locations (List[Tuple[int, int, int, int]]): List of face locations in the frame.

        Returns:
            Dict[Any]: Dictionary containing verification results, including the number of detected
                       and identified people, as well as details for each face.
        """
        identification_start = time()

        # Encode all faces found in the provided locations within the frame
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        detected_people = len(face_encodings)
        identified_people = 0
        all_result = []

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare the detected face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, self.threshold)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Determine if the face is verified based on the closest match
            if matches[best_match_index]:
                identified_people += 1
                verification_result = {
                    "Verified": True,
                    "ID": self.known_face_names[best_match_index],
                    "Coordinates": (left * 2, top * 2, right * 2, bottom * 2)
                }
            else:
                verification_result = {
                    "Verified": False,
                    "ID": None,
                    "Coordinates": (left * 2, top * 2, right * 2, bottom * 2)
                }

            all_result.append(verification_result)

        final_result = {
            "timestamp": 0.0,  # Placeholder, should be filled with the actual timestamp
            "frame_number": 0,  # Placeholder, should be filled with the actual frame number
            "extraction_time": 0.0,  # Placeholder, should be filled with the frame extraction time
            "detection_time": 0.0,  # Placeholder, should be filled with the detection time
            "identification_time": round(time() - identification_start, 3),
            "overall_time": 0.0,  # Placeholder, should be filled with the time taken to process the frame
            "detected_people": detected_people,
            "identified_people": identified_people,
            "all_result": all_result,
        }

        return final_result
