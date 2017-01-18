import sol5
import sol5_utils

import numpy as np
import matplotlib.pyplot as plt

DEF_NUM_CHANNELS = 10

DEF_CROP_SIZE = (200, 200)

DEF_BATCH_SIZE = 10


def corruption_func(im):
    noisy = im + 0.4 * np.std(im) * np.random.random(im.shape)  # todo check if correct
    return noisy


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


# tests
# testLoadDataSet
# testResBlock
# test_build_nn_model

tests = [test_build_nn_model]

def main():
    try:
        for test in tests:
            test()
    except Exception as e:
        print('Failed due to: {0}'.format(e))
        exit(-1)


if __name__ == '__main__':
    main()
