
import numpy as np
from pathlib import Path, PureWindowsPath
import random
from sklearn.metrics.pairwise import cosine_similarity

THISPATH = PureWindowsPath(__file__).parents[0]
IMGSPATH = THISPATH / 'Images'
CATSPATH = IMGSPATH / 'Cats'
DOGSPATH = IMGSPATH / 'Dogs'

def check_positive(min, max, min_bound=0):
    assert (min >= min_bound), f"whoops {min} is not greater than or equal to {min_bound}"
    assert (max > min), f"whoops {max} is not greater than {min}"

def randrange(min, max):
    assert (min >= 0 and max > 0 and max > min), "Only generates positive numbers"
    return random.random() * (max-min) + min

def make_column_vector(v):
    if len(v.shape) == 1: # ravel array
        return v.reshape(len(v), 1)
    elif len(v.shape) == 2: # vector has 2 declared dimensions            
        h, w = v.shape
        if h == 1:
            return v.reshape(w, h)
        elif w == 1:
            return v # already a column vector
        else:
            raise Exception(f"v must be a 1-dimensional vector... v.shape={v.shape}")
    else:
        raise Exception(f"v must be a 2D array, v.shape={v.shape}")

def make_row_vector(v):
    if len(v.shape) == 1: # ravel array
        return v.reshape(1, len(v))
    elif len(v.shape) == 2: # vector has 2 declared dimensions            
        h, w = v.shape
        if h == 1:
            return v # already a row vector
        elif w == 1:
            return v.reshape(w, h)
        else:
            raise Exception(f"v must be a 1-dimensional vector... v.shape={v.shape}")
    else:
        raise Exception(f"v must be a 2D array, v.shape={v.shape}")

def cosim(v1, v2):
    assert (v1.shape == v2.shape), f"Images must be the same shape, not {v1.shape} and {v2.shape}"
    v1 = make_row_vector(v1)
    v2 = make_row_vector(v2)
    # NOTE - enforcing similarity complement to reflect similarity according to smaller 'distance'
    return 1 - cosine_similarity(v1, v2)

if __name__ == "__main__":
    a1 = np.array([[1, 2, 3]])
    a2 = np.array([[2,3,4]])
    print(cosim(a1, a2))

def make_row_vector(v):
    if len(v.shape) == 1: # ravel array
        return v.reshape(1, len(v))
    elif len(v.shape) == 2: # vector has 2 declared dimensions            
        h, w = v.shape
        if h == 1:
            return v # already a row vector
        elif w == 1:
            return v.reshape(w, h)
        else:
            raise Exception(f"v must be a 1-dimensional vector... v.shape={v.shape}")
    else:
        raise Exception(f"v must be a 2D array, v.shape={v.shape}")

def cosim(v1, v2):
    assert (v1.shape == v2.shape), f"Images must be the same shape, not {v1.shape} and {v2.shape}"
    v1 = make_row_vector(v1)
    v2 = make_row_vector(v2)
    # NOTE - enforcing similarity complement to reflect similarity according to smaller 'distance'
    return 1 - cosine_similarity(v1, v2)

if __name__ == "__main__":
    a1 = np.array([[1, 2, 3]])
    a2 = np.array([[2,3,4]])
    print(cosim(a1, a2))
