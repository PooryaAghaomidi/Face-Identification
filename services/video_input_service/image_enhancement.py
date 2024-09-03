import cv2
import numpy as np
from PIL import Image


def automatic_adjust(image, clip_hist_percent=15, return_pill=False):
    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Ensure image is in BGR format as expected by OpenCV
    if image_np.ndim == 2:  # Grayscale image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:  # RGBA image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    # Convert image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # Apply the brightness and contrast adjustment
    auto_result = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)

    # Resize frame of video to 1/2 size for faster face recognition processing
    small_frame = cv2.resize(auto_result, (0, 0), fx=0.5, fy=0.5)

    # Convert the result back to a PIL image
    if return_pill:
        auto_result = Image.fromarray(auto_result)
    else:
        auto_result = small_frame[:, :, ::-1]

    return auto_result
