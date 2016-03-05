from cytoolz import compose
import cv2
from  matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy

from sklearn import svm


training_files = {
    'ovc1.jpg': 'overcast',
    'ovc2.jpg': 'overcast',
    'ovc3.jpg': 'overcast',
    'ovc4.jpg': 'overcast',
    'sct1.jpg': 'scattered',
    'sct2.jpg': 'scattered',
    'sct3.jpg': 'scattered',
    'sct4.jpg': 'scattered',
}


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def rgb_to_gray2(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return (r + g + b) // 3


def contrast(im: np.ndarray) -> np.ndarray:
    im = scipy.stats.scoreatpercentile(im, 50)
    return im
    plt.imshow(im, cmap=cm.Greys_r)
    plt.show()


def resize(im: np.ndarray) -> np.ndarray:
    return cv2.resize(im, (100, 100), interpolation=cv2.INTER_AREA)


def square_crop(im: np.ndarray) -> np.ndarray:
    """Crops to a centered square."""
    # image is wider than long.
    height, width = im.shape[:2]

    if height < width:
        crop_size = width - height
        return im[:,crop_size/2:-crop_size/2,:]
    # image is longer than wide
    else:
        crop_size = height - width
        return im[crop_size/2:-crop_size/2,:,:]


process = compose(rgb_to_gray, resize, square_crop)

# todo perhaps use a wide aspect rather than square.

def make_training_set() -> None:
    for filename, cat in training_files.items():
        im = scipy.misc.imread(filename)
        # processed = pipe(im, square_crop, rgb_to_gray)

        processed = process(im)
        scipy.misc.imsave('post_' + filename, processed)


def train() -> svm.SVC:
    clf = svm.SVC()
