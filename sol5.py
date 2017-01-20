import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
import os
from keras.models import Model
from keras.layers import Convolution2D, Activation, Input, merge
from keras.optimizers import Adam

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

DEF_CROP_SIZE_DENOISE = (24, 24)
DEF_NUM_CHANNELS_DENOISE = 48
DEF_BATCH_SIZE = 100
DEF_SAMPLE_PER_EPOCH = 10000  # todo change
DEF_NUM_EPOCHS_DENOISE = 5
DEF_NUM_VALID_SAMPLES = 100  # todo change
# DEF_NUM_VALID_SAMPLES = 1000  # todo change

DEF_MIN_SIGMA = 0.0
DEF_MAX_SIGMA = 0.2

DEF_CROP_SIZE_DEBLUR = (16, 16)
DEF_NUM_CHANNELS_DEBLUR = 32
DEF_NUM_EPOCHS_DEBLUR = 10

DEF_KER_LIST = [7]


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

                    start_height = np.random.permutation(cur_height - height)[:1][0]
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
    input_tensor = Input(shape=(1, height, width))
    conv1 = Convolution2D(num_channels, DEF_KER_CONVE_HEIGHT, DEF_KER_CONVE_WIDTH, border_mode='same')(input_tensor)
    relu1 = Activation('relu')(conv1)
    after_resblocks = relu1
    for i in range(DEF_RES_BLOCKS):
        after_resblocks = resblock(after_resblocks, num_channels)
    last_addition = merge([relu1, after_resblocks], mode='sum')
    last_conv = Convolution2D(FINALE_OUTPUT_CHANNELS, DEF_KER_CONVE_HEIGHT, DEF_KER_CONVE_WIDTH, border_mode='same')(
        last_addition)

    model = Model(input=input_tensor, output=last_conv)
    return model


def train_model(model, images, corruption_func, batch_size,
                samples_per_epoch, num_epochs, num_valid_samples):
    '''
    train our model
    :param model: a general neural network model for image restoration
    :param images: a list of file paths pointing to image files.
    :param corruption_func: A function receiving a numpy’s array representation
    of an image as a single argument, and returns a randomly corrupted version
    of the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: The number of samples in each epoch (actual samples, not batches!).
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    '''
    num_images = len(images)
    inputShape = model.get_input_shape_at(0)
    crop_size = (inputShape[-2], inputShape[-1])  # todo check if 1 and 2 is good
    num_train_ims = int(0.8 * num_images)

    train_ims_names = images[:num_train_ims]
    test_ims_name = images[num_train_ims:]

    train_gen = load_dataset(train_ims_names, batch_size, corruption_func, crop_size)
    test_gen = load_dataset(test_ims_name, batch_size, corruption_func, crop_size)

    model.compile(optimizer=Adam(beta_2=0.9), loss='mse')
    model.fit_generator(train_gen, samples_per_epoch, num_epochs,
                        validation_data=test_gen, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model, num_channels):
    '''
    return restored full images of any size
    :param corrupted_image:  a grayscale image of shape (height, width) and with values in the [0, 1] range of
    type float32
    :param base_model: a neural network trained to restore small patches
    :param num_channels: the number of channels used in the base model. Use it to construct the larger model
    :return: restored full images of any size
    '''
    corrupted_image_shape = corrupted_image.shape
    height, width = corrupted_image_shape[HEIGHT], corrupted_image_shape[WIDTH]

    corrupted_image -= IM_NORM_FACTOR

    full_im_model = build_nn_model(height, width, num_channels)
    full_im_model.set_weights(base_model.get_weights())

    restored_im = full_im_model.predict(corrupted_image[np.newaxis, np.newaxis, ...])[0][0, :, :]
    return (restored_im + IM_NORM_FACTOR).clip(0, 1).astype(np.float32)


def add_gaussian_noise(image, min_sigma, max_sigma):
    '''
    add random noise to image
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution
    :return: the image with random noise
    '''

    sigma = np.random.uniform(min_sigma, max_sigma)
    noisy = image + np.random.normal(loc=0.0, scale=sigma, size=image.shape)

    return noisy.clip(0, 1).astype(np.float32)


def learn_denoising_model(quick_mode=False):
    '''
    return a trained denoising model
    :param quick_mode: If true, much lower values arguments for model
    :return: a trained denoising model
    '''
    height, width = DEF_CROP_SIZE_DENOISE
    num_channels = DEF_NUM_CHANNELS_DENOISE
    images = sol5_utils.images_for_denoising()

    if quick_mode == False:
        batch_size = DEF_BATCH_SIZE
        samples_per_epoch = DEF_SAMPLE_PER_EPOCH
        num_epochs = DEF_NUM_EPOCHS_DENOISE
        num_valid_samples = DEF_NUM_VALID_SAMPLES
    else:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30

    denoising_model = build_nn_model(height, width, num_channels)

    if os.path.exists(sol5_utils.relpath('./model_den.h')):  # todo rm before submission
        denoising_model.load_weights('./model_den.h')
    else:
        train_model(denoising_model, images, lambda im: add_gaussian_noise(im, DEF_MIN_SIGMA, DEF_MAX_SIGMA),
                    batch_size, samples_per_epoch, num_epochs, num_valid_samples)
        denoising_model.save_weights(sol5_utils.relpath('./model_den.h'))
    return denoising_model, num_channels


def add_motion_blur(image, kernel_size, angle):
    '''
    simulate motion blur on the given image
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle:an angle in radians in the range [0, π)
    :return: simulate motion blur on the given image
    '''
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    corrupt = convolve(image, kernel)
    return corrupt.astype(np.float32)


def random_motion_blur(image, list_of_kernel_sizes):
    '''
    simulate motion blur on the given image
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param list_of_kernel_sizes:  a list of odd integers
    :return: simulate motion blur on the given image
    '''
    num_kernels = len(list_of_kernel_sizes)
    kernel_ind = np.random.permutation(num_kernels)
    kernel_size = list_of_kernel_sizes[kernel_ind]
    angle = np.random.uniform(0, np.pi)
    return add_motion_blur(image, kernel_size, angle)


def learn_deblurring_model(quick_mode=False):
    '''
    return a trained deblurring model, and the number of channels used in its construction
    :param quick_mode: If true, much lower arguments for model
    :return: a trained deblurring model, and the number of channels used in its construction
    '''
    height, width = DEF_CROP_SIZE_DEBLUR
    num_channels = DEF_NUM_CHANNELS_DEBLUR
    images = sol5_utils.images_for_deblurring()

    if quick_mode == False:
        batch_size = DEF_BATCH_SIZE
        samples_per_epoch = DEF_SAMPLE_PER_EPOCH
        num_epochs = DEF_NUM_EPOCHS_DEBLUR
        num_valid_samples = DEF_NUM_VALID_SAMPLES
    else:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30

    debluring_model = build_nn_model(height, width, num_channels)

    if os.path.exists(sol5_utils.relpath('./model_den.h')):  # todo rm before submission
        debluring_model.load_weights('./model_deb.h')
    else:
        train_model(debluring_model, images, lambda im: random_motion_blur(im, DEF_KER_LIST), batch_size,
                    samples_per_epoch, num_epochs, num_valid_samples)
        debluring_model.save_weights(sol5_utils.relpath('./model_deb.h'))
    return debluring_model, num_channels
