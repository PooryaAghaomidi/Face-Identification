import os
import json
import logging
from video_input_service.capture_video import extract_frames
from face_detection_service.face_detection import FaceDetection
from logging_service.logger import log_helper, summary_helper
from face_identification_service.identify_people import FrameIdentification


class FaceIdentification:
    def __init__(self, configs_path, log_file):
        # Initialize logging
        if os.path.exists(log_file):
            os.remove(log_file)
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s %(levelname)s:%(message)s')

        with open(configs_path, 'r') as file:
            CFG = json.load(file)

        self.face_detection_module = FaceDetection(model_path=CFG['detection_model_path'],
                                                   confidence=CFG['detection_threshold'])
        self.frame_identification_module = FrameIdentification(people_dictionaries=CFG['people'],
                                                               threshold=CFG['verification_threshold'])

    def run(self, video_path):
        all_info = []

        # 1. Video Input Service:
        frame_generator = extract_frames(video_path)

        for frame_info in frame_generator:
            # Check if all frames have been extracted
            if isinstance(frame_info, str):
                print(frame_info)
                break

            # 2. Face Detection Service:
            detections, detection_time = self.face_detection_module.detect(frame_info['frame'])

            # 3. Face Recognition Service:
            identifications = self.frame_identification_module.verify_frame(frame_info['frame'], detections)

            # 4. Logging Service:
            log_info = log_helper(frame_info, identifications, detection_time)

            all_info.append(log_info)

        summary_helper(all_info)


if __name__ == "__main__":
    FaceIdentification('configs.JSON', '../logs/face-identification.log').run('../data/task-video.mp4')
