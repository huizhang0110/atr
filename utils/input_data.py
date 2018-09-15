import tensorflow as tf
import cv2
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


def random_rotate_image_func(image):
    angle = np.random.uniform(low=0, high=360)
    image = misc.imrotate(image, angle, 'bicubic')
    return image


def read_tfrecord_use_queue_runner(filename, batch_size=32):
    filequeue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, example_tensor = reader.read(filequeue)  # return ther next key/value pair produced by a reader
    example_features = tf.parse_single_example(
        example_tensor,
        features={
            'image/transcript': tf.FixedLenFeature([], dtype=tf.string),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string)
        }
    )
    height = tf.cast(example_features['image/height'], tf.int32)
    width = tf.cast(example_features['image/width'], tf.int32)

    image = tf.image.decode_jpeg(example_features['image/encoded'], channels=3)
    image = tf.reshape(image, [height, width, 3])

    # Dynamic resize image to [32, new_width]
    h = tf.cast(height, tf.float32)
    w = tf.cast(width, tf.float32)
    new_h = tf.cast(32, tf.float32)
    new_w = tf.cast(new_h * w / h, tf.int32)
    scale_resize = tf.cast([32, new_w], tf.int32)
    image = tf.image.resize_images(image,
                                   size=scale_resize,
                                   method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, dtype=tf.float32) / 128.0 - 1

    groundtruth_text = tf.cast(example_features['image/transcript'], tf.string)
    filename = tf.cast(example_features['image/filename'], tf.string)

    min_after_dequeue = 2000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, groundtruth_text_batch, filename_batch= tf.train.batch(
        [image, groundtruth_text, filename],
        batch_size=batch_size, 
        capacity=capacity,
        dynamic_pad=True,
        num_threads=64,
    )

    batch_tensor_dict = {
        'filenames': filename_batch,
        'images': image_batch,
        'groundtruth_text': groundtruth_text_batch,
    }
    return batch_tensor_dict


def get_batch_data(tfrecord_path, batch_size=32, mode='train'):
    if mode=='train':
        return read_tfrecord_use_queue_runner(tfrecord_path, batch_size=batch_size)
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))
