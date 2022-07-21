
import cv2 as cv # opencv stores images natively in numpy
import logging
import matplotlib.pyplot as plt # alternative show_img method plt.matshow()
import numpy as np
import os
from pathlib import Path, PureWindowsPath
import random

try:
    from . import config
except:
    import config

logging.basicConfig(level=config.LOG_LEVEL)

try:
    from . import utils
    logging.info("ImageProcessing.py imports 0")
except:
    import utils
    logging.info("ImageProcessing.py imports 1")

# Test images for ease of use
test_imgs = []
for i in range(5):
    test_imgs.append(str(utils.THISPATH / f"Images\\Cats\\{i}.jpg"))
    test_imgs.append(str(utils.THISPATH / f"Images\\Dogs\\{i}.jpg"))

# Asserts that img is actually an image
def assert_img(img, msg="argument is not an image!"):
    assert (type(img) == np.ndarray), msg

# loads image from path/string/whatever
def load_img(path, grayscale=True):
    path = str(path)
    assert (Path(path).is_file()), f"path is not file: {path}"
    img = None
    if grayscale:
        img = cv.imread(path, 0)
    else:
        img = cv.imread(path)
    assert_img(img, f"load_img failed with path {path}")
    return img

# saves an image to path/string/whatever
def save_img(img, path):
    path = str(path)
    assert (cv.imwrite(path, img)), f"Image save failed to path {path}"

# displays image using opencv's imshow and list comprehension
def show_img(window_name="Testyboiiii", *args):
    assert (type(window_name) == str), "first arg must be string"
    imgs = []
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            imgs.append(arg)
        elif type(arg) == list:
            imgs.extend(arg)
        else:
            raise Exception(f"Invalid type {type(arg)} for arg {i}")
    for img in imgs:
        cv.imshow(window_name, img)
        cv.waitKey(0)

# standardizes fetching dimensions from imgs
def img_dims(img):
    width = img.shape[1]
    height = img.shape[0]
    return width, height

# Crops a region from an image
def crop_img(img, x, y, w, h):
    return img[y:y+h, x:x+w]

# Crops an image according to margins
def crop_margins(img, top, bottom, left, right):
    img_width, img_height = img_dims(img)
    y = top
    x = left
    h = img_height - (top + bottom)
    w = img_width - (left + right)
    crop_img(img, x, y, w, h)

# Crops an image to square
def crop_square(img, mode=0):
    if mode == 0:
        # reduce the larger dimension to the length of the
        # smaller dimension by cropping about the center
        # ties favor cutting from bottom and right
        width, height = img_dims(img)
        if width > height: # reduce width to size of height
            diff = width - height
            left = diff // 2
            right = diff - left
            return crop_margins(img, 0, 0, left, right)
        elif height > width: # reduce height to size of width
            diff = height - width
            top = diff // 2
            bottom = diff - top
            return crop_margins(img, top, bottom, 0, 0)
        else: # img is already square
            return img
    else:
        raise Exception("No modes other than 0 are allowed right now")

# Crops a randomly-selected region of size dims from img
def crop_random(img, dims):
    width, height = img_dims(img)
    w, h = dims
    assert (width >= w), "region width is greater than image width"
    assert (height >= h), "region height is greater than image height"
    x = random.randint(0, width - dims[0])
    y = random.randint(0, height - dims[1])
    return crop_img(img, x, y, w, h)

# resizes an image to width and height irrespective of current dims
def resize_img(img, width, height):
    return cv.resize(img, (width, height))

# crops to square then resizes to (dim x dim)
def set_square(img, dim=500):
    return resize_img(crop_square(img), dim, dim)

# flips an image vertically (0), horiztonal (1), or both (-1)
def flip_img(img, mode=None):
    if mode is None:
        mode = random.randrange(-1,2)
    assert (mode in [-1,0,1]), "Invalid mode!"
    return cv.flip(img, mode)

# rotates an image by 90, 180, or 270 degrees
ROTATE_ACCEPTED_ANGLES = (90, 180, 270, -90, -180, -270)
def rotate_img(img, angle):
    assert (angle in ROTATE_ACCEPTED_ANGLES), f"Angle {angle} is not valid"
    width, height = img_dims(img)
    center = (width//2, height//2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(img, M, (height, width)) 

# blurs an image with a guassian kernel
def blur_img(img, kernel_size):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), cv.BORDER_DEFAULT)

# applies sobel filter indicated direction or both
def edge_filter_sobel_x(img, kernel_size):
    return cv.Sobel(img, cv.CV_64F, 1, 0, ksize=kernel_size)
def edge_filter_sobel_y(img, kernel_size):
    return cv.Sobel(img, cv.CV_64F, 0, 1, ksize=kernel_size)
def edge_filter_sobel(img, kernel_size):
    return cv.Sobel(img, cv.CV_64F, 1, 1, ksize=kernel_size)

# normalizes an img
def normalize_img(img):
    return cv.normalize(img, img, 0, 255, cv.NORM_MINMAX, dtype=-1).astype(np.uint8)

# Main
if __name__ == "__main__":
    for img in (load_img(path) for path in test_imgs):
        show_img('Original Image', img)
        func = blur_img
        for i in range(1, 16, 2):
            show_img(f"{func.__name__}'d Image", func(img, i))
     