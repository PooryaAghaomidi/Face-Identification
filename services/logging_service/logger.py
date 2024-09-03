import logging
from time import time


def log_helper(frame_info, identifications, detection_time):
    identifications["timestamp"] = round(frame_info["timestamp"], 3)
    identifications["frame_number"] = frame_info["frame_number"]
    identifications["extraction_time"] = frame_info["extraction_time"]
    identifications["detection_time"] = round(detection_time, 3)
    identifications["identification_time"] = identifications["identification_time"]
    identifications["overall_time"] = round(time() - frame_info["start_time"], 3)

    print(
        f'{identifications["timestamp"]}: {identifications["detected_people"]} people detected, {identifications["identified_people"]} people verified.')
    print(
        f'extraction: {identifications["extraction_time"]}s - detection: {identifications["detection_time"]}s - identification: {identifications["identification_time"]}s - overall: {identifications["overall_time"]}s\n')

    logging.info(
        f' {identifications["timestamp"]}: {identifications["detected_people"]} people detected, {identifications["identified_people"]} people verified.')
    logging.info(
        f' extraction: {identifications["extraction_time"]}s - detection: {identifications["detection_time"]}s - identification: {identifications["identification_time"]}s - overall: {identifications["overall_time"]}s')
    logging.info(f' all results: {identifications["all_result"]}\n')

    return identifications


def summary_helper(video_results):
    processing_time = 0
    detection_time = 0
    identification_time = 0
    frame_time = 0
    id_timestamps = {}

    frame_numbers = len(video_results)

    for frame in video_results:
        processing_time += frame['extraction_time']
        detection_time += frame['detection_time']
        identification_time += frame['identification_time']
        frame_time += frame['overall_time']

        if frame['identified_people'] > 0:
            for person in frame['all_result']:
                if person['Verified']:
                    person_id = person['ID']
                    time = frame['timestamp']

                    if person_id in id_timestamps:
                        id_timestamps[person_id].append(time)
                    else:
                        id_timestamps[person_id] = [time]

    final_info = {
        'extraction_time': round(processing_time / frame_numbers, 3),
        'detection_time': round(detection_time / frame_numbers, 3),
        'identification_time': round(identification_time / frame_numbers, 3),
        'overall_time': round(frame_time / frame_numbers, 3),
    }

    logging.info("\n\n\n   ---------------------------------------------------------\n\n\n")
    logging.info(f"   Average time for detection people in a frame: {final_info['detection_time']}\n")
    logging.info(f"   Average time for identifying people in a frame: {final_info['identification_time']}\n")
    logging.info(f"   Average overall time for a frame: {final_info['overall_time']}\n")

    for person in id_timestamps:
        logging.info(f"   Times when {person} was in the frame: {id_timestamps[person]}")

    print("\n\n\n---------------------------------------------------------\n\n\n")
    print(f"Average time for detection people in a frame: {final_info['detection_time']}\n")
    print(f"Average time for identifying people in a frame: {final_info['identification_time']}\n")
    print(f"Average overall time for a frame: {final_info['overall_time']}\n")

    for person in id_timestamps:
        print(f"Times when {person} was in the frame: {id_timestamps[person]}")
