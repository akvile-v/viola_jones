from PIL import Image, ImageOps
import numpy as np
import random
import os
import glob
from sklearn.metrics import *
from typing import *

SUB_WINDOW = 17
TRAIN_FACES_PATH = 'testing_face'
TRAIN_FACE = glob.glob(
    os.path.join(TRAIN_FACES_PATH, '**', '*.jpg'), recursive=True)
TRAIN_NONFACES_PATH = 'testing_non_face'
TRAIN_NONFACE = glob.glob(
    os.path.join(TRAIN_NONFACES_PATH, '**', '*.jpg'), recursive=True)

#########################PROCESSING FUNCTIONS##################################

# Function for reading image in provided path
# and resizing for given width and height
def read_resize(path_to_image, width, height):
    # Open image
    image = Image.open(path_to_image)
    # Convert image to greyscale
    image = image.convert("L")
    # Making original photo copy for resizing
    resized_image = image.copy()
    # Resizing image and upscaling quality 
    resized_image.thumbnail((width, height), Image.Resampling.LANCZOS)
    return resized_image

# Function to revert image pixels to array in range 0...1
def to_float_array(image):
    return np.array(image).astype(np.float32) / 255.

# Function to revert image  from array to pixels
def to_image(array):
    return Image.fromarray(np.uint8(array * 255.))

# Function for reading test images and adapting for testing
def open_test(image_path, resize = True):
    # Open image from path
    image = Image.open(image_path)
    # Convert image to array, crop top and convert back to image
    image = to_image(to_float_array(image))
    # Selecting if height or width is smaler
    min_side = np.min(image.size)
    # Resizing image and upscaling quality 
    image = ImageOps.fit(image, (min_side, min_side), Image.Resampling.LANCZOS)
    # Resizing image to sub-window size and upscaling quality 
    if resize:
        image = image.resize((SUB_WINDOW, SUB_WINDOW), Image.Resampling.LANCZOS)
    # Savin image in grayscale for smaller dimension
    image = ImageOps.grayscale(image)    
    return image

# Data normalisation function
def normalize(data):
    return (data - data.mean()) / data.std()

# Function that takes random p positive and n negative
# samples from given data set
def sample_data(p, n, norma = False):
    xs = []
    # Random non-repeated samples ar taken
    xs.extend([to_float_array(open_test(f)) for f in 
                    random.sample(TRAIN_FACE, p)])
    # Random samples, that can be repeated few times are taken (replace)
    xs.extend([to_float_array(open_test(f)) for f in 
                    random.sample(TRAIN_NONFACE, n)])
    # Saving number of positive and negative examples in data set
    ys = np.hstack([np.ones((p,)), np.zeros((n,))])
    xs = np.array(xs)
    if norma:
        xs = normalize(xs)
    return xs, ys
