from functools import partial
from pathlib import Path, PurePath

from cytoolz import compose, curry
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.misc
import scipy.stats

from sklearn import svm

training_raw_path = Path('training_images_raw')
training_processed_path = Path('training_images_processed')

#
# training_files = {
#     'ovc1.jpg': 'overcast',
#     'ovc2.jpg': 'overcast',
#     'ovc3.jpg': 'overcast',
#     'ovc4.jpg': 'overcast',
#     'sct1.jpg': 'scattered',
#     'sct2.jpg': 'scattered',
#     'sct3.jpg': 'scattered',
#     'sct4.jpg': 'scattered',
# }


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Uses a common algorithm."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

#
# def rgb_to_gray2(rgb: np.ndarray) -> np.ndarray:
#     """A WIP attempt at a less-human-centered, more balanced version?"""
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     return (r + g + b) // 3


def contrast(im: np.ndarray) -> np.ndarray:
    # todo broken
    im = scipy.stats.scoreatpercentile(im, 50)
    return im
    plt.imshow(im, cmap=cm.Greys_r)
    plt.show()


# todo @curry  # Note: Curry is currently incompatible with annotations.
def resize(new_width: int, im: np.ndarray) -> np.ndarray:
    """Scale to new_width, maintaining constant aspect ratio."""
    aspect_ratio = im.shape[1] / im.shape[0]
    return cv2.resize(im, (new_width, int(new_width // aspect_ratio)),
                      interpolation=cv2.INTER_AREA)


# todo @curry  # Note: Curry is currently incompatible with annotations.
def crop(aspect_ratio: float, im: np.ndarray) -> np.ndarray:
    """Crop an image to an aspect ratio on its longer side, leaving its shorter
     side the same."""
    # An aspect_ratio greater than 1 means wider than tall.
    height, width = im.shape[:2]

    if height < width:
        crop_size = width - height * aspect_ratio
        if not crop_size:  # Trying to crop with 0 size will cause problems.
            return im
        return im[:, crop_size/2:-crop_size/2, :]
    else:
        crop_size = height - width * aspect_ratio
        if not crop_size:
            return im
        return im[crop_size/2:-crop_size/2, :, :]


process = compose(rgb_to_gray, partial(resize, 100), partial(crop, 4/3))


def make_training_set() -> None:
    """Creates training images, in a format suitable for machine learning.
    Reduces to a low resolution, and converts to greyscale."""
    for training_image in training_raw_path.iterdir():
        im = scipy.misc.imread(training_image)

        new_filename = 'post_' + training_image.name[:-3] + 'png'
        new_file = training_processed_path.joinpath(new_filename).open('wb')

        scipy.misc.imsave(new_file, process(im), format='png')


def train() -> svm.SVC:
    clf = svm.SVC()
