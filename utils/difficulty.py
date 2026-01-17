import cv2
import numpy as np

def image_difficulty(image_path, labels):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    brightness = np.mean(img) / 255.0
    sizes = []
    for lbl in labels:
        _, _, _, w, h = lbl
        sizes.append(w * h)
    size_score = 1 - np.mean(sizes) if sizes else 1.0
    lighting_score = abs(brightness - 0.5)
    return 0.6 * size_score + 0.4 * lighting_score
