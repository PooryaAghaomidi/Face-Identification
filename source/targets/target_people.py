import logging
import numpy as np
import face_recognition
from time import time
from PIL import Image, ImageOps
from typing import Dict, List, Tuple


def create_dataset(people_dictionaries: Dict[str, List[str]]) -> Tuple[List[str], List[np.ndarray]]:
    """
    Creates a dataset of face encodings and their corresponding names.

    :param people_dictionaries: A dictionary where the keys are person names and the values are lists of image paths.
    :return: A tuple containing a list of names and a list of corresponding face encodings.
    """

    logging.info("Initializing a database out of provided images ...")
    print("\nInitializing a database out of provided images ...")

    known_face_names = []  # List to store names corresponding to each face encoding
    image_paths = []       # List to store the paths of images to process

    # Populate known_face_names and image_paths from the input dictionary
    for person, images in people_dictionaries.items():
        for image in images:
            known_face_names.append(person)
            image_paths.append(image)

    # List to store the face encodings
    known_face_encodings = []

    # Process each image and extract face encodings
    for img_path in image_paths:
        try:
            # Open the image and correct its orientation using EXIF data
            my_img = Image.open(img_path)
            my_img = ImageOps.exif_transpose(my_img)

            # Convert the image to a NumPy array and extract face encodings
            face_encoding = face_recognition.face_encodings(np.array(my_img))[0]
            known_face_encodings.append(face_encoding)

        except IndexError:
            # Skip images where no face encodings are found
            pass

        except Exception as e:
            # Handle other exceptions (e.g., file not found, image format issues)
            print(f"Error processing image {img_path}: {e}")
            continue

    return known_face_names, known_face_encodings

