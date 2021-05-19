# import the necessary packages
import os

# define the base path to the *original* input dataset and then use
# the base path to derive the image and annotations directories
ORIG_BASE_PATH = "cells"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])
ORIG_IMAGES_THRESHOLD = os.path.sep.join([ORIG_BASE_PATH, "threshold_images"])

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
BASE_PATH = "dataset_cells"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "cell"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_cell"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 3000
MAX_PROPOSALS_INFER = 600

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 100
MAX_NEGATIVE = 100

# initialize the input dimensions to the network
INPUT_DIMS = (64, 64)

# define the path to the output model and label binarizer

MODEL_NAME = 'cells_detector.h5'
ENCODER_NAME = 'cells_label_encoder.pickle'

MODEL_RCNN = 'model_rcnn'
LABEL_ENCODER = 'label_encoder'

MODEL_PATH = os.path.sep.join([MODEL_RCNN, MODEL_NAME])
ENCODER_PATH = os.path.sep.join([LABEL_ENCODER, ENCODER_NAME])

#MODEL_PATH = 'model_rcnn/ant_detector.h5'
#ENCODER_PATH = 'label_encoder/ant_label_encoder.pickle'

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.8