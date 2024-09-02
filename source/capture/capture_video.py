import cv2
import logging
from time import time
from typing import Generator, Dict, Union
from .image_enhancement import automatic_adjust


def extract_frames(video_path: str) -> Generator[Union[Dict[str, any], str], None, None]:
    """
    Extract frames from a video file along with their metadata.

    Args:
        video_path (str): Path to the video file.

    Yields:
        dict or str: A dictionary containing the frame, frame number, and timestamp,
                     or a string message indicating the completion of frame extraction.
    """
    logging.info(f"Initializing the Video Input Service ...\n\n")
    print(f"Initializing the Video Input Service ...\n\n")

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        logging.error(f"Failed to open video file at {video_path}")
        raise ValueError(f"Invalid video path: {video_path}")

    frame_counter = 0

    while True:
        extraction_start = time()
        ret, frame = video_capture.read()
        timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)  # Get timestamp in milliseconds

        if not ret:
            break  # Exit loop if there are no more frames

        frame_counter += 1

        frame_info = {
            "frame": automatic_adjust(image=frame, clip_hist_percent=15, return_pill=False),
            "frame_number": frame_counter,
            "timestamp": round(timestamp / 1000.0, 3),
            "extraction_time": round(time() - extraction_start, 3),
            "start_time": extraction_start
        }

        yield frame_info

    video_capture.release()
    logging.info("All frames have been successfully extracted.")
    print("All frames have been successfully extracted.")

    # Indicate the completion of frame extraction
    yield "All frames have been successfully extracted."
