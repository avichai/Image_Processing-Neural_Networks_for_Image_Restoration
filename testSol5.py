import sol5
import sol5_utils

import numpy as np
import matplotlib.pyplot as plt
import os

DEF_NUM_VALID_SAMPLES = 1000  # todo change

DEF_NUM_EPOCHS = 5

DEF_SAMPLE_PER_EPOCH = 10000  # todo change

DEF_NUM_CHANNELS = 48

DEF_CROP_SIZE = (24, 24)

DEF_BATCH_SIZE = 100


def corruption_func(im):
    noisy = im + 0.4 * np.std(im) * np.random.random(im.shape)  # todo check if correct
    return noisy


def add_gaussian_noise(image):
    '''
    add random noise to image
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution
    :return: the image with random noise
    '''

    sigma = np.random.uniform(0, 1)
    noisy = image + np.random.normal(loc=0.0, scale=sigma)

    return noisy.clip(0, 1).astype(np.float32)


def testLoadDataSet():
    filenames = sol5_utils.images_for_denoising()
    batch_size = DEF_BATCH_SIZE
    crop_size = DEF_CROP_SIZE

    data_generator = sol5.load_dataset(filenames, batch_size, corruption_func, crop_size)

    source, target = next(data_generator)

    plt.figure()
    plt.imshow(source[0, 0, ...], cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(target[0, 0, ...], cmap=plt.cm.gray)
    plt.show()


# def testResBlock(): #todo still need to test
#     output_tensor = sol5.resblock(input_tensor, num_channels)


def test_build_nn_model():
    height, width = DEF_CROP_SIZE
    num_channels = DEF_NUM_CHANNELS
    model = sol5.build_nn_model(height, width, num_channels)


def test_train_model():
    height, width = DEF_CROP_SIZE
    num_channels = DEF_NUM_CHANNELS
    batch_size = DEF_BATCH_SIZE
    images = sol5_utils.images_for_denoising()
    samples_per_epoch = DEF_SAMPLE_PER_EPOCH
    num_epochs = DEF_NUM_EPOCHS
    num_valid_samples = DEF_NUM_VALID_SAMPLES

    model = sol5.build_nn_model(height, width, num_channels)

    corruption_function = lambda im: sol5.add_gaussian_noise(im, 0.0, 0.2)

    sol5.train_model(model, images, corruption_function, batch_size,
                     samples_per_epoch, num_epochs, num_valid_samples)

    model.save_weights(sol5_utils.relpath('./model.h4'), overwrite=False)


def test_restored_image():
    height, width = DEF_CROP_SIZE
    num_channels = DEF_NUM_CHANNELS
    images = sol5_utils.images_for_denoising()

    base_model = sol5.build_nn_model(height, width, num_channels)
    base_model.load_weights('./model_den.h')

    # print(base_model.get_weights()[0][0][0][0])

    im = sol5.read_image(images[2], 1)
    corrupted_image = corruption_func(im)
    restored_image = sol5.restore_image(corrupted_image, base_model, num_channels)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(corrupted_image, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(restored_image, cmap=plt.cm.gray)
    plt.show()


def test_add_gaussian_noise():
    images = sol5_utils.images_for_denoising()
    im = sol5.read_image(images[2], 1)
    corrupted_image = sol5.add_gaussian_noise(im, 0, 1)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(corrupted_image, cmap=plt.cm.gray)
    plt.show()


def test_learn_denoising_model():
    model, num_channels = sol5.learn_denoising_model(quick_mode=False)  # todo this one is needed
    # model, num_channels = sol5.learn_denoising_model(quick_mode=True)

    images = sol5_utils.images_for_denoising()
    im = sol5.read_image(images[2], 1)
    corrupted_image = sol5.add_gaussian_noise(im, 0, 0.2)
    restored_image = sol5.restore_image(corrupted_image, model, num_channels)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(corrupted_image, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(restored_image, cmap=plt.cm.gray)
    plt.show()


def test_add_motion_blur():
    images = sol5_utils.images_for_denoising()
    im = sol5.read_image(images[2], 1)

    kernel_size = 3
    angle = 2.2

    corrupted = sol5.add_motion_blur(im, kernel_size, angle)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(corrupted, cmap=plt.cm.gray)
    plt.show()


def test_random_motion_blur():
    images = sol5_utils.images_for_denoising()
    im = sol5.read_image(images[2], 1)

    list_of_kernel_sizes = [3, 5, 7, 9]
    corrupted = sol5.random_motion_blur(im, list_of_kernel_sizes)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(corrupted, cmap=plt.cm.gray)
    plt.show()


def test_learn_deblurring_model():
    # model, num_channels = sol5.learn_deblurring_model(quick_mode=False) # todo this one is needed
    model, num_channels = sol5.learn_deblurring_model(quick_mode=True)

    images = sol5_utils.images_for_deblurring()
    im = sol5.read_image(images[2], 1)
    corrupted_image = sol5.random_motion_blur(im, [7])
    restored_image = sol5.restore_image(corrupted_image, model, num_channels)

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(corrupted_image, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(restored_image, cmap=plt.cm.gray)
    plt.show()


# tests
# testLoadDataSet
# testResBlock
# test_build_nn_model
# test_train_model
# test_restored_image
# test_add_gaussian_noise
# test_add_motion_blur
# test_random_motion_blur
# test_learn_deblurring_model

tests = [test_learn_denoising_model]


def main():
    try:
        for test in tests:
            test()
    except Exception as e:
        print('Failed due to: {0}'.format(e))
        exit(-1)


if __name__ == '__main__':
    main()
