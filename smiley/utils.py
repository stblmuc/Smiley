import configparser
import os
import sys
import png
import math
import shutil

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

MODELS_DIRECTORY = os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['MODELS'],
                                config['DEFAULT']['IMAGE_SIZE'])
CATEGORIES_LOCATION = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['CATEGORIES'],
                                       config['DEFAULT']['IMAGE_SIZE'] + "/")
CATEGORIES = None
PROGRESS = {
    'value': 100,
    'num_processes': 2,
    'previous_value': 0,
    'stop': False
}


def get_progress():
    global PROGRESS

    return PROGRESS['value']


def update_progress(value):
    global PROGRESS

    PROGRESS['value'] = PROGRESS['previous_value'] + (100*value/PROGRESS['num_processes'])
    
    # if proccess is completed, add its contribution to previous_value
    if (value == 1):
        PROGRESS['previous_value'] += 100/PROGRESS['num_processes']

    return PROGRESS['value']


def train_should_stop(stop='MISSING'):
    global PROGRESS

    if stop is not 'MISSING':
        PROGRESS['stop'] = stop

    return PROGRESS['stop']


def reset_progress():
    global PROGRESS

    PROGRESS['value'] = 100
    PROGRESS['previous_value'] = 0


# Class for log handling
class Logger(object):
    def __init__(self):
        self.buffer = ""

    def start(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def end(self):
        sys.stdout = self.stdout

    def write(self, data):
        self.buffer += data
        self.stdout.write(data)

    def flush(self):
        pass

    def pop(self):
        length = len(self.buffer)
        out = self.buffer[:length]
        self.buffer = self.buffer[length:]
        return out


# logger object
LOGGER = Logger()


# Decorator to capture standard output
def capture(f):
    def captured(*args, **kwargs):
        LOGGER.start()
        try:
            result = f(*args, **kwargs)
        finally:
            LOGGER.end()
        return result  # captured result from decorated function
    return captured


def delete_all_models():
    for f in os.listdir(MODELS_DIRECTORY):
        os.remove(os.path.join(MODELS_DIRECTORY, f))


def update_categories():
    global CATEGORIES

    # create folder for categories if it doesn't exist:
    if not os.path.exists(CATEGORIES_LOCATION):
        os.makedirs(CATEGORIES_LOCATION)

    # dictionary with keys = category names (from folders), values = category indices
    CATEGORIES = {x: i for (i, x) in
                  enumerate(sorted([z[0].split("/")[-1] for z in os.walk(CATEGORIES_LOCATION) if
                                    z[0].count("/") == CATEGORIES_LOCATION.count("/") and len(
                                        z[0].split("/")[-1]) > 0]))}
    return CATEGORIES


def add_training_example(image, category):
    # create folder for category if it doesn't exist:
    path = os.path.join(CATEGORIES_LOCATION, category)
    if not os.path.exists(path):
        os.makedirs(path)

    save_image(image, path)
    update_categories()


def save_image(image, path):
    # name for new training example image
    image_name = max([0] + [int(x.split(".")[0]) for x in os.listdir(path)]) + 1

    # store new training example image
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    w = png.Writer(image_size, image_size, greyscale=True)
    w.write(open(path + "/" + str(image_name) + ".png", "wb"), image)


# Deletes the folder of the category
def delete_category(category):
    path = os.path.join(CATEGORIES_LOCATION, category)
    shutil.rmtree(path)
    update_categories()


def get_category_names():
    return list(CATEGORIES.keys())


# Returns dictionary with keys = category names (from folders), values = number of images for the category
def get_number_of_images_per_category():
    cat_images = {}
    for z in os.walk(CATEGORIES_LOCATION):
        if len(str(z[0].split("/")[-1])) > 0:
            for x in os.walk(os.path.join(CATEGORIES_LOCATION, str(z[0].split("/")[-1]))):
                # the number of files in the category folder is used for the number of images for that category
                cat_images[str(x[0].split("/")[-1])] = len(x[-1])
    return cat_images


# Returns the number of images required for each category
def get_number_of_images_required():
    # calculating number of images required for each category (-0.000001 for float precision errors)
    # (for with test set add 1)
    return math.ceil((1.0 / (1.0 - float(config['DEFAULT']['train_ratio']))) - 0.000001)


# Returns a string error message that a category has to be added
def get_no_cat_error():
    req_images_per_cat = get_number_of_images_required()
    return "Please add at least one category (by adding at least <b>%d</b> images in that category)." % req_images_per_cat


# Checks if at least one category has not the least required number of images
def not_enough_images():
    req_images_per_cat = get_number_of_images_required()
    cat_img = get_number_of_images_per_category()
    return any(cat_img[cat] < req_images_per_cat for cat in cat_img.keys())


# Returns a string error message with the number of images for each category which is below the minimum images required
def get_not_enough_images_error():
    msg = ""
    req_images_per_cat = get_number_of_images_required()
    cat_img = get_number_of_images_per_category()
    for cat in cat_img.keys():
        if cat_img[cat] < req_images_per_cat:
            img = "images" if cat_img[cat] > 1 else "image"
            msg += "category '<b>" + cat + "</b>' has just <b>" + str(cat_img[cat]) + "</b> " + img + ", "
    if len(msg) > 0:
        msg += "but at least <b>%d</b> images are required for each category. Please add the missing images and (re-)train the classifiers." % req_images_per_cat
    return msg
