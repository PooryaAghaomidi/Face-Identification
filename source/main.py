import os
import logging
from configs.cfg import CFG
from capture.capture_video import extract_frames
from targets.target_people import create_dataset
from detection.face_detection import FaceDetection
from forwarding.logger import log_helper, summary_helper
from identification.identify_people import FrameIdentification


class FaceIdentification:
    def __init__(self, log_file):
        # Initialize logging
        if os.path.exists(log_file):
            os.remove(log_file)
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s %(levelname)s:%(message)s')

        known_face_names, known_face_encodings = create_dataset(people_dictionaries=CFG['people'])

        self.face_detection_module = FaceDetection(model_path=CFG['detection_model_path'],
                                                   confidence=CFG['detection_threshold'])
        self.frame_identification_module = FrameIdentification(known_face_names=known_face_names,
                                                               known_face_encodings=known_face_encodings,
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
    FaceIdentification('logs/face-identification.log').run('../data/task-video.mp4')
