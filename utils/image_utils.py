import cv2
import numpy as np


def load_image(image_path, dynamic=False):
    image = cv2.imread(image_path)
    if not dynamic:
        resize_scale = (100, 32)
        # Note: opencv.resize scale format is different from tf.image.resize
        # new_width, new_height
    else:
        image_h, image_w = image.shape[:2]
        new_height = 32
        new_width = (new_height * image_w) / image_h
        resize_scale = (int(new_width), new_height)
    image = cv2.resize(image, resize_scale)
    # print(image_path, " resize to: ", image.shape)
    image = image[:, :, ::-1]  # From BRG to RGB format
    image = image.astype(np.float32) / 128.0 - 1
    return image

