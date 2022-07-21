
import copy
import logging
import math
import numpy as np
import random

try:
    from . import config
except:
    import config

logging.basicConfig(level=config.LOG_LEVEL)

try:
    from . import ImageProcessing as IP
    from . import utils
    logging.info("Augmentation.py imports 0")
except:
    import ImageProcessing as IP
    import utils
    logging.info("Augmentation.py imports 1")

# if the images are of size 500x500, then even crop/cutout/etc factors
# are helpful to avoid unecessary artifacts in the augmentations

# the augmentations are kinda fragile so let's try not to stray from
# 500x500 for now, but we may want to try a number of different sizes
# at some point...

####################################################################

###################################
#                                 #
#   Base Augmentation Functions   #
#                                 #
###################################

logging.basicConfig(level=logging.warning)

# if the images are of size 500x500, then even crop/cutout/etc factors
# are helpful to avoid unecessary artifacts in the augmentations

# the augmentations are kinda fragile so let's try not to stray from
# 500x500 for now, but we may want to try a number of different sizes
# at some point...

####################################################################

###################################
#                                 #
#   Base Augmentation Functions   #
#                                 #
###################################

# Randomly crops a square region from an image (no stretching)
# TODO - figure out more appropriate factors
CROP_RESIZE_FACTORS = (0.5,) # should be tuple/list
def CropResize(img, rs):
    width, height = IP.img_dims(img)
    crop_factor = CROP_RESIZE_FACTORS if type(CROP_RESIZE_FACTORS) not in [tuple, list] else rs.choice(CROP_RESIZE_FACTORS)
    logging.debug(f"Augmentation CropResize: crop_factor={crop_factor}")
    crop = IP.crop_random(img, (int(width * crop_factor), int(height * crop_factor)))
    resize = IP.resize_img(crop, width, height)
    return resize

# Flips an image vertically (0), horiztonal (1), or both (-1)
# TODO - figure out how to allow for flips on non-cardinal axes
FLIP_MODES = {-1:'both', 0:'vertical', 1:'horizontal'}
def Flip(img, rs):
    mode = rs.choice(list(FLIP_MODES.keys()))
    logging.debug(f"Augmentation Flip: mode={FLIP_MODES[mode]}")
    return IP.flip_img(img, mode)

# Rotates img by 90 degrees some number of times
# TODO - figure out a good way to allow for rotation at any angle
ROTATE_ACCEPTED_ANGLES = (90, 180, 270)
def Rotate(img, rs):
    angle = rs.choice(ROTATE_ACCEPTED_ANGLES)
    logging.debug(f"Augmentation Rotate: angle={angle}")
    return IP.rotate_img(img, angle)

# Replaces a randomly selected rectangular region with black pixels
# TODO - figure out what factors are appropriate
CUTOUT_SIZE_FACTORS = (3/10, 4/10)
def Cutout(img, rs):
    width, height = IP.img_dims(img)
    cutout_factor = rs.choice(CUTOUT_SIZE_FACTORS)
    w, h = int(width * cutout_factor), int(height * cutout_factor)
    x = random.randint(0, width - w)
    y = random.randint(0, height - h)
    logging.debug(f"Augmentation Cutout: x={x}, y={y}, w={w}, h={h}")
    new_img = img.copy()
    new_img[y:y+h, x:x+w] = 0
    return new_img

# Adds gaussian noise to the image
# https://stackoverflow.com/a/30609854/13747259
# TODO - are these good hyperparams?
NOISE_MEAN = 0
NOISE_VAR = 40
NOISE_SIGMA = NOISE_VAR ** 0.5
def Noise(img, rs):
    logging.debug("Augmentation Noise: [no params]")
    gauss = rs.normal(NOISE_MEAN, NOISE_SIGMA, img.shape)
    result = img + gauss
    result[np.where(result > 255)] = 255.0
    result[np.where(result < 0)] = 0.0
    result = result.astype(np.uint8)
    return result

# Blurs an image using a gaussian kernel
# TODO - figure out what kernel sizes are appropriate
BLUR_KERNEL_SIZES = (9, 11, 13)
def Blur(img, rs):
    size = rs.choice(BLUR_KERNEL_SIZES)
    logging.debug(f"Augmentation Blur: size={size}")
    return IP.blur_img(img, size)

# Filters for edges using the Sobel filter
EDGE_FILTER_KERNEL_SIZES = (1, 3)
EDGE_FILTER_DIRECTIONS = ('x', 'y', 'both')
def EdgeFilter(img, rs):
    size = rs.choice(EDGE_FILTER_KERNEL_SIZES)
    direction = rs.choice(EDGE_FILTER_DIRECTIONS)
    logging.debug(f"Augmentation EdgeFilter: size={size}, direction={direction}")
    if direction == 'x':
        return IP.edge_filter_sobel_x(img, size)
    elif direction == 'y':
        return IP.edge_filter_sobel_y(img, size)
    elif direction == 'both':
        return IP.edge_filter_sobel(img, size)
    else:
        raise Exception(f"Invalid direction {direction}")

DEFAULT_BASE_FUNCS = [Cutout, Rotate, Flip, CropResize, Noise, Blur, EdgeFilter]

#################################################################

#######################################
#                                     #
#   Compound Augmentation Functions   #
#                                     #
#######################################

def CAF(img, rs, CAF_id, base_funcs):
    logging.debug(f"CAF #{CAF_id} is begin called:")
    result = img
    for func in base_funcs:
        result = func(result, rs)
    logging.debug(f"CAF #{CAF_id} is done!")
    return result

CAFG_ID_COUNT = 0
class CAFGenerator:
    def __init__(self, base_funcs, seed=config.DEFAULT_SEED, **kwargs):
        self.base_funcs = base_funcs # list of base augmentation functions that are combined 
        self.RS = np.random.RandomState(seed=seed)
        self.max = kwargs.get('max_CAF_size') or len(self.base_funcs) # maximum number of base funcs in a CAF
        self.replacement = kwargs.get('replacement') or False # allows for an CAF of [noise, noise, blur] if true
        # for keeping track in logs
        global CAFG_ID_COUNT
        self.CAFG_id = CAFG_ID_COUNT
        CAFG_ID_COUNT += 1
        self.CAF_count = 0
    def getCAF(self):
        func_count = self.RS.randint(1, self.max + 1)
        compound = self.RS.choice(self.base_funcs, func_count, self.replacement)
        new_CAF_id = f"{self.CAFG_id}.{self.CAF_count}"
        logging.debug(f"CAFG #{self.CAFG_id} --> CAF {new_CAF_id} = {[f.__name__ for f in compound]}")
        new_CAF = lambda img: CAF(img, self.RS, new_CAF_id, compound)
        self.CAF_count += 1
        return new_CAF

#################################################################

# Example use
def test_CAFG(**kwargs):
    test_img_num = kwargs.get('test_img_num') or 1
    CAFG_num = kwargs.get('CAFG_num') or 0
    CAFGs = [
        CAFGenerator( # most basic
            DEFAULT_BASE_FUNCS,
            config.DEFAULT_SEED,
        ),
        CAFGenerator( # each CAF only has one base func
            DEFAULT_BASE_FUNCS,
            config.DEFAULT_SEED,
            max_CAF_size = 1
        ),
        CAFGenerator(
            DEFAULT_BASE_FUNCS,
            config.DEFAULT_SEED,
            max_CAF_size = 3
        )
    ]
    img = IP.load_img(IP.test_imgs[test_img_num])
    IP.show_img('Original Image', img)
    while True:
        IP.show_img('Augmented Image', CAFGs[CAFG_num].getCAF()(img))

if __name__ == "__main__":
    # to go to the next image, press [ENTER] in the window
    # og img persists for reference while CAFed imgs cycle
    test_CAFG(CAFG_num = 2, test_img_num = 1)
