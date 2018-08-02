# -*- coding: utf-8 -*-

import pathlib
from keras.utils import np_utils
import numpy as np
from skimage.io import imread


def load_data(image_dir: str, one_hot=True):
    """ Load dataset of keyakizaka

    Arguments
    ---------
    image_dir: str
        A path of the directory that images are saved in.

    one_hot: bool
        if True then the labels will be convert to one hot
        representation, default is True.

    Returns:
        a tuple of images and labels
    """
    X = []
    Y = []

    for member_dir in pathlib.Path(image_dir).iterdir():
        member_id = int(member_dir.name)

        for i, image_path in enumerate(member_dir.iterdir()):
            X.append(imread(image_path))
            Y.append(member_id)

    X = np.array(X)
    if one_hot:
        Y = np_utils.to_categorical(Y)

    return (X, Y)

