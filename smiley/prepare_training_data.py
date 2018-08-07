from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy
from scipy import ndimage
import category_manager
import tensorflow as tf
import configparser

# parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = len(category_manager.update())
VALIDATION_RATIO = 0.20  # split training data into 80% training data and 20% validation data

EXPAND_DISPLAY_STEP = 5  # image augmentation is logged every EXPAND_DISPLAY_STEP images
TRAIN_RATIO = 0.8 # split generated data into 80& for training and 20% for testing

# get images from category folders, add them to training/test images
def add_data(model, train_images, train_labels, test_images, test_labels, train_ratio):
    from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        os.path.join(os.path.dirname(__file__), category_manager.CATEGORIES_LOCATION),
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=1,
        class_mode='binary')

    number_of_images = generator.samples
    number_of_categories = generator.num_classes
    number_processed = 0
    images = []
    labels = []

    # is there any data?
    if number_of_images == 0:
        return train_images, train_labels, test_images, test_labels

    # stores how many images of each category are present
    number_per_category = {c: 0.0 for c in range(number_of_categories)}

    while number_processed < number_of_images:
        item = next(generator)
        image = numpy.array(item[0], dtype=numpy.uint8).reshape(1, 28, 28, 1)
        if model == "regression":
            image = ((255 - image) / 255.0)
        elif model == "CNN":
            image = (((255 - image) / 255.0) - 0.5)
        image = numpy.reshape(image, [1, -1])
        label = int(item[1][0])
        number_per_category[label] += 1.0
        labels.append(label)
        images.append(numpy.reshape(image, 784))
        number_processed += 1

    # stores how many images of each category are in the training set
    number_per_category_in_training = {c: 0.0 for c in range(NUM_LABELS)}

    for i, x in enumerate(images):
        category = labels[i]
        if number_per_category_in_training[category] < number_per_category[category] * train_ratio:
            number_per_category_in_training[category] += 1.0
            train_images.append(x)
            train_labels.append(labels[i])
        else:
            test_images.append(x)
            test_labels.append(labels[i])

    train_images = numpy.array(train_images)
    test_images = numpy.array(test_images)

    # transform labels into one-hot vectors
    one_hot_encoding = numpy.zeros((len(train_images), number_of_categories))
    one_hot_encoding[numpy.arange(len(train_images)), train_labels] = 1
    train_labels = numpy.reshape(one_hot_encoding, [-1, number_of_categories])
    one_hot_encoding = numpy.zeros((len(test_images), number_of_categories))
    one_hot_encoding[numpy.arange(len(test_images)), test_labels] = 1
    test_labels = numpy.reshape(one_hot_encoding, [-1, number_of_categories])

    if sum([1 for c in number_per_category.items() if
            c[0] not in [str(n) for n in range(10)] and
                            c[1] != 0.0 and c[1] == number_per_category_in_training[c[0]]]) == 0:

        return train_images, train_labels, test_images, test_labels
    else:
        # at least one category has all examples in the training set (meaning there are not
        # enough examples for a training set and a testing set)
        return None


# create a validation set from part of the training data
def create_validation_set(train_data, train_labels, VALIDATION_PROPORTION):
    train_data_result = []
    train_labels_result = []
    validation_data_result = []
    validation_labels_result = []

    number_per_category = {c: 0.0 for c in range(NUM_LABELS)}
    for i, x in enumerate(train_data):
        category = [z for z in range(len(train_labels[i])) if train_labels[i][z] == 1.0][0]
        number_per_category[category] += 1.0

    number_per_category_in_validation = {c: 0.0 for c in range(NUM_LABELS)}
    for i, x in enumerate(train_data):
        category = [z for z in range(len(train_labels[i])) if train_labels[i][z] == 1.0][0]
        if number_per_category_in_validation[category] < number_per_category[category] * VALIDATION_PROPORTION:
            number_per_category_in_validation[category] += 1.0
            validation_data_result.append(x)
            validation_labels_result.append(train_labels[i])
        else:
            train_data_result.append(x)
            train_labels_result.append(train_labels[i])

    if min(number_per_category_in_validation.values()) == 0:
        # at least one of the categories has no items in the validation set (not enough training examples)
        return None
    else:
        return numpy.array(train_data_result), numpy.array(train_labels_result), \
               numpy.array(validation_data_result), numpy.array(validation_labels_result)


# augment training data
def expand_training_data(images, labels):
    expanded_images = []
    expanded_labels = []
    j = 0
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'trainConfig.ini'))
    for x, y in zip(images, labels):
        j += 1
        if j % EXPAND_DISPLAY_STEP == 0:
            print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x)  # this is regarded as background's value
        image = numpy.reshape(x, (-1, 28))

        num_augm_per_img = int(config['DEFAULT']['NUMBER_AUGMENTATIONS_PER_IMAGE'])
        max_angle = int(config['DEFAULT']['MAX_ANGLE_FOR_AUGMENTATION'])
        for i in range(num_augm_per_img):
            # rotate the image with random degree
            angle = numpy.random.randint(-max_angle, max_angle, 1)
            new_img = ndimage.rotate(image, angle, reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data


# prepare training data (generated images)
def prepare_data(model, use_data_augmentation=True):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # add  data from category folders
    train_data, train_labels, test_data, test_labels = add_data(model, train_data, train_labels, test_data, test_labels, TRAIN_RATIO)

    # create a validation set
    train_data, train_labels, validation_data, validation_labels = create_validation_set(train_data, train_labels, VALIDATION_RATIO)

    # concatenate train_data and train_labels for random shuffle
    if use_data_augmentation:
        # augment training data by random rotations etc.
        train_total_data = expand_training_data(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)
        numpy.random.shuffle(train_total_data)

    train_size = train_total_data.shape[0]  # size of training set

    return NUM_LABELS, train_total_data, train_size, validation_data, validation_labels, test_data, test_labels
