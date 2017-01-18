import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import os
from keras.models import Model
from keras.layers import Convolution2D, Activation, Input, merge

from scipy.ndimage.filters import convolve

import sol5_utils

FINALE_OUTPUT_CHANNELS = 1

DEF_RES_BLOCKS = 5

IM_NORM_FACTOR = 0.5

IDENTITY_KERNEL_SIZE = 1
BINOMIAL_MAT = [0.5, 0.5]
GRAY = 1
RGB = 2
NORM_PIX_FACTOR = 255
ROWS = 0
COLS = 1
LARGEST_IM_INDEX = 0
DIM_RGB = 3

HEIGHT = 0
WIDTH = 1
DEF_KER_CONVE_HEIGHT = 3
DEF_KER_CONVE_WIDTH = 3


def read_image(filename, representation):
    """this function reads a given image file and converts it into a given
    representation:
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the
                     output should be either a grayscale image (1) or an
                     RGB image (2).
    output - the image in the given representation when the pixels are
             of type np.float32 and normalized"""
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return
    im = imread(filename)
    if im.dtype == np.float32:
        '''I don't handle this case, we asume imput in uint8 format'''
        return
    if representation == GRAY:
        im = rgb2gray(im).astype(np.float32)
        if np.max(im) > 1:
            '''not suppose to happened'''
            im /= NORM_PIX_FACTOR
        return im
    im = im.astype(np.float32)
    im /= NORM_PIX_FACTOR
    return im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    '''
    outputs data_generator, a Python’s generator object which outputs random
    tuples of the form (source_batch, target_batch), where each output variable
    is an array of shape (batch_size, 1, height, width), target_batch is made
    of clean images, and source_batch is their respective randomly
    corrupted version according to corruption_func(im)
    :param filenames: A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration
    of Stochastic Gradient Descent
    :param corruption_func: A function receiving a numpy’s array representation
    of an image as a single argument, and returns a randomly corrupted version
    of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the
    patches to extract.
    :return: data_generator, a Python’s generator object which outputs random
    tuples of the form (source_batch, target_batch)
    '''

    def generator():
        # todo check if need to store values filenames, batch_size, corruption_func, crop_size
        dic = {}
        num_files = len(filenames)
        height = crop_size[HEIGHT]
        width = crop_size[WIDTH]
        while True:
            batch_filenames = []
            source_batch, target_batch = np.zeros((batch_size, 1, height,
                                                   width), dtype=np.float32), np.zeros((batch_size, 1, height,
                                                                                        width), dtype=np.float32)
            counter = 0
            while len(batch_filenames) != batch_size:
                filename_ind = np.random.permutation(num_files)[:1][0]
                cur_filename = filenames[filename_ind]
                if cur_filename not in batch_filenames:
                    batch_filenames.append(cur_filename)
                    if cur_filename not in dic:
                        cur_image = read_image(cur_filename, 1)
                        dic[cur_filename] = cur_image
                    else:
                        cur_image = dic[cur_filename]
                    cur_corrupt_image = corruption_func(cur_image)
                    cur_height = cur_image.shape[HEIGHT]
                    cur_width = cur_image.shape[WIDTH]

                    start_height = np.random.permutation(cur_height - height)[
                                   :1][0]
                    start_width = np.random.permutation(cur_width - width)[:1][0]

                    source_batch[counter, ...] = cur_corrupt_image[
                                                    start_height:start_height + height,
                                                    start_width:start_width + width] - IM_NORM_FACTOR

                    target_batch[counter, ...] = cur_image[
                                                    start_height:start_height + height,
                                                    start_width:start_width + width] - IM_NORM_FACTOR
                    counter += 1
            yield (source_batch.astype(np.float32), target_batch.astype(np.float32))

    return generator()


def resblock(input_tensor, num_channels):
    '''
    creating a residual block
    :param input_tensor: symbolic input tensor
    :param num_channels: number of channels for each of its
    convolutional layers
    :return: a residual block
    '''
    conv1 = Convolution2D(num_channels, DEF_KER_CONVE_HEIGHT, DEF_KER_CONVE_WIDTH, border_mode='same')(input_tensor)
    relu1 = Activation('relu')(conv1)
    conv2 = Convolution2D(num_channels, DEF_KER_CONVE_HEIGHT, DEF_KER_CONVE_WIDTH, border_mode='same')(relu1)
    return merge([input_tensor, conv2], mode='sum')


def build_nn_model(height, width, num_channels):
    '''
    return an untrained Keras model
    :param height: input dimension for model
    :param width: input dimension for model
    :param num_channels: output channels for all convolution layers except the very last
    :return: an untrained Keras model
    '''
    input_tensor = Input(shape=(height, width, 1)) # todo change to: input_tensor = Input(shape=(1, height, width)) # after changing file mentioned in forum
    conv1 = Convolution2D(num_channels, DEF_KER_CONVE_HEIGHT, DEF_KER_CONVE_WIDTH, border_mode='same')(input_tensor)
    relu1 = Activation('relu')(conv1)
    after_resblocks = relu1
    for i in range(DEF_RES_BLOCKS):
        after_resblocks = resblock(after_resblocks, num_channels)
    last_addition = merge([relu1, after_resblocks], mode='sum')
    last_conv = Convolution2D(FINALE_OUTPUT_CHANNELS, DEF_KER_CONVE_HEIGHT, DEF_KER_CONVE_WIDTH, border_mode='same')(last_addition)

    model = Model(input=input_tensor, output=last_conv)
    return model
