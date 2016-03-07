from functools import partial
from pathlib import Path, PurePath
from typing import Tuple

from cytoolz import compose, curry, count
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.misc
import scipy.stats

from sklearn import cross_validation, svm
from sklearn.grid_search import RandomizedSearchCV
from skimage import exposure, filters, feature

training_raw_path = Path('training_images_raw')
training_processed_path = Path('training_images_processed')

# Dimensions for the processed image
WIDTH = 48
ASPECT_RATIO = 4/3
HEIGHT = 36
assert HEIGHT == WIDTH / ASPECT_RATIO


def show_im(im: np.ndarray) -> None:
    """Display a greyscale image."""
    plt.imshow(im, cmap=cm.Greys_r)
    plt.show()


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Uses a common algorithm."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    result = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return result.astype(np.uint8)

#
# def rgb_to_gray2(rgb: np.ndarray) -> np.ndarray:
#     """A WIP attempt at a less-human-centered, more balanced version?"""
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     return (r + g + b) // 3


def threshold(im: np.ndarray) -> np.ndarray:
    im = filters.threshold_otsu(im)
    # im = cv2.cvtColor(im, cv.CV_BGR2GRAY)
    # v, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
    return im


# todo @curry  # Note: Curry is currently incompatible with annotations.
def resize(new_width: int, im: np.ndarray) -> np.ndarray:
    """Scale to new_width, maintaining constant aspect ratio."""
    new_height = int(new_width // (im.shape[1] / im.shape[0]))

    return cv2.resize(im, (new_width, new_height),
                      interpolation=cv2.INTER_AREA)


# todo @curry  # Note: Curry is currently incompatible with annotations.
def crop(aspect_ratio: float, im: np.ndarray) -> np.ndarray:
    """Crop an image to an aspect ratio on its longer side, leaving its shorter
     side the same."""
    # An aspect_ratio greater than 1 means wider than tall.
    height, width = im.shape[:2]
    # todo clean this func
    # aspect_ratio is  w/h; w is the ratio, h is 1.
    w_rat, h_rat = width / aspect_ratio, height / 1

    if h_rat < w_rat or h_rat == w_rat:
        # Trying to crop with 0 size will raise indexing problems when cropping to -0.
        crop_size = width - height * aspect_ratio
        if crop_size < 2:
            return im
        return im[:, crop_size/2:-crop_size/2]
    else:
        crop_size = height - width / aspect_ratio
        if crop_size < 2:
            return im
        return im[crop_size/2:-crop_size/2]


def edge(im: np.ndarray) -> np.ndarray:
    """"""
    # return cv2.Canny(im, 100, 100)
    return feature.canny(im)

process = compose(partial(resize, WIDTH), edge, rgb_to_gray, partial(crop, ASPECT_RATIO))


def make_training_images() -> None:
    """Creates training images, in a format suitable for machine learning.
    Reduces to a low resolution, and converts to greyscale."""
    for training_image in training_raw_path.iterdir():
        im = scipy.misc.imread(training_image)

        new_filename = 'post_' + training_image.name[:-3] + 'png'
        new_file = training_processed_path.joinpath(new_filename).open('wb')

        scipy.misc.imsave(new_file, process(im), format='png')


def build_data() -> Tuple[np.ndarray, np.ndarray]:
    """Creates a set of scikit-learn-formatted training data, from processed
    images."""
    num_samples = int(count(training_processed_path.iterdir()))
    data = np.zeros([num_samples, HEIGHT, WIDTH])
    target = np.zeros(num_samples)

    for i, file in enumerate(training_processed_path.iterdir()):
        im = scipy.misc.imread(file)
        # todo temporary workaroudn for some images ending up 74 wide; add
        # todo extra column

        if im.shape[0] == HEIGHT - 1:
            extra_col = np.array([np.zeros(WIDTH)])
            im = np.concatenate([im, extra_col])
        data[i] = im

        if 'sct' in file.name:
            cat = 0
        elif 'ovc' in file.name:
            cat = 1
        else:
            raise AttributeError

        target[i] = cat

    # Flatten each image to a 1d array.
    data = data.reshape((num_samples, -1))
    return data, target


def learn() -> svm.SVC:
    clf = svm.SVC()

    data, target = build_data()

    clf.fit(data[:-1], target[:-1])
    return clf.predict(data[-1:])


def class_validation():
    """Scores the classifier using cross-validation"""
    # http://scikit-learn.org/stable/modules/cross_validation.html
    X, y = build_data()

    clf = svm.SVC(kernel='rbf', C=1)
    preds = cross_validation.cross_val_predict(clf, X, y, cv=5)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return preds, scores


def grid_search(symbol='MSFT'):
    """Find optimal SVC parameters"""
    from scipy.stats import randint as sp_randint
    X, y = build_data()

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.4, random_state=0)

    param_grid = [
        {'C': [.1, 1, 10, 100, 1000], 'gamma': [1e-2, 1e-3, 1e-4],
         'kernel': ['linear', 'rbf']},
        # {'C': [1, 10, 100, 1000], 'gamma': [.001, .0001], 'kernel': ['linear', 'rbf']}
    ]

    param_dist = {'C': [.001, .01, .1, 1, 10, 100],
                  'gamma': [1e2, 1e-1, 1e-2, 1e-3, 1e-4],
                  'kernel': ['linear', 'rbf']
                  }
    n_iter_search = 20
    clf = svm.SVC()
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    #
    # clf = GridSearchCV(estimator=svm.SVC(C=1), param_grid=param_grid, cv=5)
    # clf.fit(X_train, y_train)
    # return clf

    random_search.fit(X_train, y_train)
    return random_search
