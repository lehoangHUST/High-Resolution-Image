import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def nearest_interpolation(image: np.ndarray, factor: float):
    assert factor > 0, f"Value factor should bigger than zero."
    if len(image.shape) == 3:
        height, width, _ = image.shape
    elif len(image.shape) == 2:
        height, width = image.shape
    height_new, width_new = int(height * factor), int(width * factor)
    image_new = np.zeros((height_new, width_new, 3)) if len(image.shape) == 3 \
        else np.zeros((height_new, width_new, 1))
    for h in range(height_new - 1):
        for w in range(width_new - 1):
            image_new[h, w, :] = image[int(round(h / factor)), int(round(w / factor)), :]
    return image_new.astype(dtype=np.uint8)


def bilinear_interpolation(image: np.ndarray, factor: float):
    assert factor > 0, f"Value factor should bigger than zero."
    if len(image.shape) == 3:
        height, width, _ = image.shape
    elif len(image.shape) == 2:
        height, width = image.shape
    height_new, width_new = int(height * factor), int(width * factor)
    image_new = np.zeros((height_new, width_new, 3)) if len(image.shape) == 3 \
        else np.zeros((height_new, width_new, 1))
    img = np.zeros((height + 1, width + 1, 3))
    img[0:-1, 0:-1, :] = image

    # R1 = (x2 - x)/(x2 - x1) * Q11 + (x - x1)/(x2 - x1) * Q21
    # R2 = (x2 - x)/(x2 - x1) * Q12 + (x - x1)/(x2 - x1) * Q22
    # P = (y2 - y)/(y2 - y1) * R1 + (y - y1)/(y2 - y1) * R2
    for h in range(height_new):
        for w in range(width_new):
            x = w / factor - int(w / factor)
            y = h / factor - int(h / factor)
            R1 = (1 - x) * img[int(h / factor), int(w / factor), :] \
                + x * img[int(h / factor), int(w / factor) + 1, :]
            R2 = (1 - x) * img[int(h / factor) + 1, int(w / factor), :] \
                + x * img[int(h / factor) + 1, int(w / factor) + 1, :]
            image_new[h, w, :] = (1 - y) * R1 + y * R2
    return image_new.astype(dtype=np.uint8)


def bicubic_interpolation(image: np.ndarray, factor: float):
    pass


img = cv2.imread('D:\\Machine_Learning\\High-Resolution-Image\\80.png')
img_new = bilinear_interpolation(img, 1.3)
print(img_new.shape)
plt.imshow(img_new)
plt.show()