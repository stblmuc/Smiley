from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import numpy
from scipy import ndimage
from six.moves import urllib
import category_manager
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data/mnist_data"

# parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = len(category_manager.update())
VALIDATION_RATIO = 0.20  # split training data into 80% training data and 20% validation data
MNIST_TRAIN_DATA_SIZE = 60000
MNIST_TEST_DATA_SIZE = 10000

USE_MNIST = True # True if and only if mnist dataset should be used to create a base model
EXPAND_DISPLAY_STEP = 5  # image augmentation is logged every EXPAND_DISPLAY_STEP images
NUM_AUGM_PER_IMAGE = 5 # number of random augmentations per image


# download MNIST data
def maybe_download(filename):
    # download the data from Yann's website, unless it's already here.
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


# extract the images and labels
def extract_data(model, filename_images, filename_labels, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename_images)
    with gzip.open(filename_images) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        if model == "regression":
            data = data / PIXEL_DEPTH
        elif model == "CNN":
            data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = numpy.reshape(data, [num_images, -1])

    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename_labels)
    with gzip.open(filename_labels) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        labels_filtered = []
        data_filtered = []
        for i in range(len(labels) - 1, -1, -1):
            if str(labels[i]) in category_manager.CATEGORIES:
                # replace label value with correct index for category
                # only keep data belonging to one of the categories in the categories folder
                labels_filtered.append(category_manager.CATEGORIES[str(labels[i])])
                data_filtered.append(data[i])

        labels_filtered = numpy.array(labels_filtered)
        data_filtered = numpy.array(data_filtered)

        num_labels_data = len(labels_filtered)

        # transform labels into one-hot vectors
        one_hot_encoding = numpy.zeros((num_labels_data, NUM_LABELS))
        if num_labels_data > 0:
            one_hot_encoding[numpy.arange(num_labels_data), labels_filtered] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])

    return data_filtered, one_hot_encoding


# get images from category folders, add them to training/test images
def add_extra_data(model, train_images, train_labels, test_images, test_labels, train_ratio):
    from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        category_manager.CATEGORIES_LOCATION,
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=1,
        class_mode='binary')

    number_of_images = generator.samples
    number_of_categories = generator.num_classes
    number_processed = 0
    images = []
    labels = []

    # is there any extra data?
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

    extra_train_images = []
    extra_test_images = []
    extra_train_labels = []
    extra_test_labels = []
    for i, x in enumerate(images):
        category = labels[i]
        if number_per_category_in_training[category] < number_per_category[category] * train_ratio:
            number_per_category_in_training[category] += 1.0
            extra_train_images.append(x)
            extra_train_labels.append(labels[i])
        else:
            extra_test_images.append(x)
            extra_test_labels.append(labels[i])

    extra_train_images = numpy.array(extra_train_images)
    extra_test_images = numpy.array(extra_test_images)

    # transform labels into one-hot vectors
    one_hot_encoding = numpy.zeros((len(extra_train_images), number_of_categories))
    one_hot_encoding[numpy.arange(len(extra_train_images)), extra_train_labels] = 1
    extra_train_labels = numpy.reshape(one_hot_encoding, [-1, number_of_categories])
    one_hot_encoding = numpy.zeros((len(extra_test_images), number_of_categories))
    one_hot_encoding[numpy.arange(len(extra_test_images)), extra_test_labels] = 1
    extra_test_labels = numpy.reshape(one_hot_encoding, [-1, number_of_categories])

    if sum([1 for c in number_per_category.items() if
            c[0] not in [str(n) for n in range(10)] and
                            c[1] != 0.0 and c[1] == number_per_category_in_training[c[0]]]) == 0:

        # concatenate new and existing data
        if len(extra_train_images) > 0 and len(train_images) > 0 and \
                        len(extra_test_images) > 0 and len(test_images) > 0:
            train_images = numpy.concatenate((extra_train_images, train_images))
            test_images = numpy.concatenate((extra_test_images, test_images))
            train_labels = numpy.concatenate((extra_train_labels, train_labels))
            test_labels = numpy.concatenate((extra_test_labels, test_labels))

            return train_images, train_labels, test_images, test_labels
        else:
            return extra_train_images, extra_train_labels, extra_test_images, extra_test_labels
    else:
        # at least one category other than the MNIST categories (0-9)
        # has all examples in the training set (meaning there are not
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

        for i in range(NUM_AUGM_PER_IMAGE):
            # rotate the image with random degree
            angle = numpy.random.randint(-15, 15, 1)
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


# prepare training data (MNIST + generated images)
def prepare_MNIST_data(model, use_data_augmentation=True):
    any_in = lambda a, b: any(i in b for i in a)
    digit_categories_exist = any_in(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], category_manager.CATEGORIES)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    if USE_MNIST and digit_categories_exist:
        # get the data files
        train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
        train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
        test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

        # extract data and labels into vectors
        train_data, train_labels = extract_data(model, train_data_filename, train_labels_filename, MNIST_TRAIN_DATA_SIZE)
        test_data, test_labels = extract_data(model, test_data_filename, test_labels_filename, MNIST_TEST_DATA_SIZE)

    # add extra data from category folders
    train_ratio = float(MNIST_TRAIN_DATA_SIZE) / float(MNIST_TEST_DATA_SIZE + MNIST_TRAIN_DATA_SIZE)
    train_data, train_labels, test_data, test_labels = add_extra_data(model, train_data, train_labels, test_data, test_labels, train_ratio)

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
