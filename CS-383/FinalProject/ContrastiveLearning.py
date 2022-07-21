
from ImageProcessing import load_img
from datetime import datetime
import json
import logging
import itertools
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torchvision import models

import contrastive_learner

try:
    from . import config
except:
    import config

logging.basicConfig(level=config.LOG_LEVEL)

try:
    from . import ImageProcessing as ip
    from . import Augmentation as aug
    from . import utils
    logging.info("ContrastiveLearning.py imports 0")
except:
    import ImageProcessing as ip
    import Augmentation as aug
    import utils
    logging.info("ContrastiveLearning.py imports 1")


#################################################################

# splits the labeled data into training and testing images
TRAIN_TEST_RATIO = 4
def split_data(seed=config.DEFAULT_SEED, max_data=None):
    logging.debug("Starting split_data()")
    cats = [str(utils.CATSPATH / path) for path in os.listdir(str(utils.CATSPATH))]
    logging.debug(f"found {len(cats)} cat images")
    dogs = [str(utils.DOGSPATH / path) for path in os.listdir(str(utils.DOGSPATH))]
    logging.debug(f"found {len(dogs)} dogs images")
    cats_train, cats_test, dogs_train, dogs_test = \
        train_test_split(cats, dogs, test_size=(1/(TRAIN_TEST_RATIO+1)), random_state=seed)
    if max_data: # check to see if the data needs to be cut down at all
        assert (type(max_data) == int), "Needs a maximum size of the dataset including both train and test data"
        rs = np.random.RandomState(seed=seed)
        train_size = TRAIN_TEST_RATIO * (max_data // (TRAIN_TEST_RATIO+1))
        if len(cats_train)+len(dogs_train) > (train_size):
            cats_train = rs.choice(cats_train, train_size // 2).tolist()
            dogs_train = rs.choice(dogs_train, train_size // 2).tolist()
        test_size = max_data - train_size
        if len(cats_test)+len(dogs_test) > (test_size):
            cats_test = rs.choice(cats_test, test_size // 2).tolist()
            dogs_test = rs.choice(dogs_test, test_size // 2).tolist()
    logging.debug("Returning from split_data()")
    return cats_train + dogs_train, cats_test + dogs_test

# gets class from path
def cat_or_dog_path(path):
    return 'Cat' if 'cats' in str(path).lower() else 'Dog'

# returns a generator which preserves ram
def img_reader(path_list):
    # example use: for img in img_reader(training_data): ...
    # generates (img_data, path) tuples
    return ((ip.load_img(path), path) for path in path_list)

ENCODER_COUNT = 0
class Encoder():

    DEFAULT_SEED = config.DEFAULT_SEED
    DEFAULT_BASE_FUNCS = aug.DEFAULT_BASE_FUNCS
    DEFAULT_CAF_MAX = 3 # kinda arbitrary
    DEFAULT_REPLACEMENT = False
    DEFAULT_MINIBATCH_SIZE = 1
    DEFAULT_MINIBATCH_COUNT = lambda N, minibatch_size: (N // minibatch_size)
    DEFAULT_KNN_K = 3

    def __init__(self, seed=DEFAULT_SEED):
        self.seed = seed
        self.RS = np.random.RandomState(self.seed)
        self.model = None
        global ENCODER_COUNT
        self.Encoder_id = ENCODER_COUNT
        ENCODER_COUNT += 1
        # trying out model stuff
        # right from https://github.com/lucidrains/contrastive-learner
        self.net = models.resnet50(pretrained=True)
        self.learner = contrastive_learner.ContrastiveLearner(
            self.net,
            image_size = 500,
            hidden_layer = 'avgpool',  # layer name where output is hidden dimension. this can also be an integer specifying the index of the child
            project_hidden = True,     # use projection head
            project_dim = 25,          # projection head dimensions, 128 from paper
            use_nt_xent_loss = True,   # the above mentioned loss, abbreviated
            temperature = 0.1,         # temperature
            augment_both = True        # augment both query and key
        )
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=3e-4)
        # log ya boi
        logging.info(f"Created Encoder #{self.Encoder_id}")

    def encode(self, input):
        logging.debug(f"Encoder #{self.Encoder_id} -> encoding!")
        img_t = None
        if type(input) == str: # one path
            img_t = torch.zeros(1, 3, 500, 500)
            img = torch.from_numpy(ip.load_img(input))
            img_t[0,0] = img
            img_t[0,1] = img
            img_t[0,2] = img
        elif type(input) == np.ndarray: # one img in np array
            img_t = torch.zeros(1, 3, 500, 500)
            input = torch.from_numpy(input)
            img_t[0,0] = input
            img_t[0,1] = input
            img_t[0,2] = input
        elif type(input) == list: # multiple imgs in a list
            if type(input[0]) == str:
                img_t = torch.zeros(len(input), 3, 500, 500)
                for i, img in enumerate(input):
                    img = torch.from_numpy(ip.load_img(img))
                    img_t[i,0] = img
                    img_t[i,1] = img
                    img_t[i,2] = img
            else: raise Exception(f"if input is a list, it should be a list of paths to the image files")
        # img_t holds the input
        return self.net(img_t)

    def train(self, training_data, **kwargs):
        # Create a the CAF Generator
        base_funcs = kwargs.get('base_funcs') or Encoder.DEFAULT_BASE_FUNCS
        max_CAF_size = kwargs.get('max_CAF_size') or Encoder.DEFAULT_CAF_MAX
        replacement = kwargs.get('replacement') or Encoder.DEFAULT_REPLACEMENT
        #CAFG = aug.CAFGenerator(base_funcs, max_CAF_size=max_CAF_size, replacement=replacement)
        # Start iterating over the training data
        minibatch_size = kwargs.get('minibatch_size') or Encoder.DEFAULT_MINIBATCH_SIZE
        minibatch_count = kwargs.get('minibatch_count') or Encoder.DEFAULT_MINIBATCH_COUNT(len(training_data), minibatch_size)
        logging.info(f"Encoder #{self.Encoder_id} -> Training: base_funcs={[f.__name__ for f in base_funcs]}, max_CAF_size={max_CAF_size}, " + \
            f"replacement={replacement}, minibatch_size={minibatch_size}, minibatch_count={minibatch_count}")
        new_minibatch = lambda s=minibatch_size: self.RS.choice(training_data, minibatch_size)
        #def augment_minibatch(m):
        #    aug_m = []
        #    for img in m:
        #        # each augmentated copy of the image uses the same CAF because the CAF performs randomly each time it's called
        #        CAF= CAFG.getCAF()
        #        aug_m.append( (CAF(img), CAF(img)) )
        #    return aug_m
        for minibatch_num in range(minibatch_count):
            logging.debug(f"Encoder #{self.Encoder_id} -> Training: minibatch {minibatch_num}/{minibatch_count}")
            minibatch = [ip.load_img(img) for img in new_minibatch()]
            # also from https://github.com/lucidrains/contrastive-learner
            images_tensor = torch.zeros(minibatch_size, 3, 500, 500)
            for i in range(minibatch_size):
                images_tensor[i, 0] = torch.from_numpy(minibatch[i])
                images_tensor[i, 1] = torch.from_numpy(minibatch[i])
                images_tensor[i, 2] = torch.from_numpy(minibatch[i])
            loss = self.learner(images_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # the stuff below here is not being used for testing stuff from lucidrains

            ## augment minibatch
            #augmented_minibatch = augment_minibatch([ip.load_img(img) for img in minibatch])
            ## encode minibatch
            #encoded_minibatch = \
            #    [(self.encode(augmented_minibatch[i][0]), self.encode(augmented_minibatch[i][1]))
            #     for i in range(len(augmented_minibatch))]
            ## update encoder
            ## TODO - this!
            ## TODO - include logging here?
        logging.info(f"Encoder #{self.Encoder_id} -> Done training!")

    def test(self, test_data, train_data, k=DEFAULT_KNN_K):
        logging.info(f"Encoder #{self.Encoder_id} -> Testing: len(test_data)={len(test_data)}, len(train_data)={len(train_data)}, k={k}")
        assert (l := len(test_data)), f"no test data! {l}"
        assert (l := len(train_data)), f"no train data! {l}"
        assert (k % 2 == 1), "k-value must be odd to avoid ties in nearest-neighbors classification"
        # neighbors will be stored as (distance, class) because comparing tuples compares first element, then second if first is equal
        # NOTE - extremely slight bias towards Cats being nearer as two otherwise equidistant neighbors will be compared by class1 < class2, and always 'Cat' < 'Dog
        # nearest_neighbors[0] is the nearest, [-1] is the farthest
        nearest_neighbors = None
        def new_neighbor(n): # n is (distance, class)
            # assert (isinstance(n[0], float) and isinstance(n[1], str)), f"{type(n[0])}, {type(n[1])}"
            if len(nearest_neighbors) < k: # just append if there aren't enough neighbors yet
                logging.debug(f"Encoder #{self.Encoder_id} -> initial nearest neighbor")
                nearest_neighbors.append(n)
                nearest_neighbors.sort() # always keep nearest_neighbors sorted
            elif n < nearest_neighbors[-1]: # replace fartherst nn if n is nearer
                logging.debug(f"Encoder #{self.Encoder_id} -> new nearest neighbor")
                nearest_neighbors[-1] = n
                nearest_neighbors.sort() # always keep nearest_neighbors sorted
            else: pass # nothing needs to be done if n isn't one of the nearest
        # iterate over test_data
        logging.debug(f"Encoder #{self.Encoder_id} -> Testing: starting big loops")
        classifications = []
        encoded_bois = {}
        for test_img_num, test_img_path in enumerate(test_data): # test_img is a path
            # Encode this test_img and get it's class
            test_img = ip.load_img(test_img_path)
            encoded_test_img = self.encode(test_img)
            test_img_class = cat_or_dog_path(test_img_path)
            logging.debug(f"Encoder #{self.Encoder_id} -> {test_img_num}/{len(test_data)}  -> test_img = {test_img_path}")
            # Create an image pipeline
            # NOTE - this is slow because its running encode() on each train img for each test img, but the alternative would be keeping len(train_data) encoded imgs in memory
            # NOTE - ^^^ might not be that bad if encoded imgs are only 25-long bois
            img_tap = ((self.encode(img[0]),cat_or_dog_path(img[1]), img[1]) for img in img_reader(train_data))
            # initialize test_img's nearest_neighbors
            nearest_neighbors = []
            # start iterating over the train imgs
            for i, (encoded_train_img, train_img_class, train_img_path) in enumerate(img_tap):
                logging.debug(f"Encoder #{self.Encoder_id} -> train_img #{i} = {train_img_path}")
                similarity = utils.cosim(encoded_test_img.detach().numpy(), encoded_train_img.detach().numpy())
                assert (similarity.shape == (1,1)), "Similarity was calculated wrong"
                new_neighbor((similarity[0,0], train_img_class))
            # nearest neighbors now contains the nearest k neighbors
            # figure out the classification!
            cat_votes = sum([int(n[1]=='Cats') for n in nearest_neighbors])
            dog_votes = sum([int(n[1]=='Dogs') for n in nearest_neighbors])
            test_img_model_class = 'Cat' if cat_votes > dog_votes else 'Dog'
            test_img_confidence_differential = abs(cat_votes - dog_votes)
            classifications.append({
                'path': test_img_path, 
                'real_class': test_img_class, 
                'predicted_class': test_img_model_class, 
                'confidence_differential': test_img_confidence_differential
            })
        logging.debug(f"Encoder #{self.Encoder_id} -> Testing: done with big loops")
        # Done with classifying test imgs
        # now claculate stats
        confusion_matrix = [[0,0],[0,0]] # 0 is cat, 1 is dog
        for c in classifications:
            confusion_matrix[int(c['real_class']=='Dog')][int(c['predicted_class']=='Dog')] += 1
        logging.debug(f"Encoder #{self.Encoder_id} -> returning from testing")
        # TODO - any other stats?
        return {
            'confusion_matrix_description':
                "first index is real class, second is predicted, whereby 0 = Cat and 1 = Dog",
            'confusion_matrix': confusion_matrix,
            'classifications': classifications
        }

#################################################################

if __name__ == "__main__":

    # Randomizer seed for reproducability
    seed = 42

    # Split 4:1 training and testing images by Encoder.default
    example_dataset_size_limit = 100
    train_imgs, test_imgs = split_data(seed, max_data=example_dataset_size_limit)
    
    # Initialize new encoder
    MyEncoder = Encoder(seed)
    
    # Train encoder on train_imgs
    start_train_time = datetime.now()
    MyEncoder.train(train_imgs, minibatch_size = 10)
    end_train_time = datetime.now()
    logging.info(f"training took {end_train_time - start_train_time}")
    
    # Test Encoder on test_imgs
    start_test_time = datetime.now()
    print(len(test_imgs), len(train_imgs))
    results = MyEncoder.test(test_imgs, train_imgs)
    end_test_time = datetime.now()
    logging.info(f"testing took {end_test_time - start_test_time}")
    
    # Save results to a json file
    save_file = utils.THISPATH / 'save_file.json'
    try:
        save_file = open(save_file, 'w')
    except:
        save_file = open(save_file, 'x')
    json.dump(results, save_file)
    save_file.close()
    logging.info(f"\n\n\nSaved results from testing to save_file.json")
    logging.info(f"Training took {end_train_time-start_train_time}")
    logging.info(f"Testing took {end_test_time-start_test_time}")
