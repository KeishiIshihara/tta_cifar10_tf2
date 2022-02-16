import numpy as np
import tensorflow as tf

def identity(image: np.asarray):
    return image

def horizontal_flip(image: np.ndarray):
    return tf.image.flip_left_right(image)

def vertical_flip(image: np.ndarray):
    return tf.image.flip_up_down(image)

def horizontal_flip_np(image: np.ndarray):
    if len(image.shape) == 4: # (batch, height, width, channel)
        image = image[:, :, ::-1, :]
    elif len(image.shape) == 3: # (hieght, width, channel)
        image = image[:, ::-1, :]
    else:
        raise ValueError()
    return image

def vertical_flip_np(image: np.ndarray):
    if len(image.shape) == 4: # (batch, height, width, channel)
        image = image[:, ::-1, :, :]
    elif len(image.shape) == 3: # (hieght, width, channel)
        image = image[::-1, :, :]
    else:
        raise ValueError()
    return image

