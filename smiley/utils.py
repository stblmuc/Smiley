import configparser
import os
import sys
import png
import math

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

CATEGORIES_LOCATION = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['CATEGORIES'],
                                       config['DEFAULT']['IMAGE_SIZE'] + "/")
CATEGORIES = None


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
    if not os.path.exists(os.path.join(CATEGORIES_LOCATION, category)):
        os.makedirs(os.path.join(CATEGORIES_LOCATION, category))

    # name for new training example image
    image_name = max([0] + [int(x.split(".")[0]) for x in os.listdir(os.path.join(CATEGORIES_LOCATION, category))]) + 1

    # store new training example image
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    w = png.Writer(image_size, image_size, greyscale=True)
    w.write(open(os.path.join(CATEGORIES_LOCATION, category) + "/" + str(image_name) + ".png", "wb"), image)


def get_category_names():
    category_names = ["" for _ in range(len(CATEGORIES))]
    for ind in range(len(category_names)):
        category_names[ind] = [x for x in CATEGORIES.keys() if CATEGORIES[x] == ind][0]
    return category_names


# returns dictionary with keys = category names (from folders), values = number of images for the category
def get_number_of_images_per_category():
    cat_images = {}
    for z in os.walk(CATEGORIES_LOCATION):
        if len(str(z[0].split("/")[-1])) > 0:
            for x in os.walk(os.path.join(CATEGORIES_LOCATION, str(z[0].split("/")[-1]))):
                # the number of files in the category folder is used for the number of images for that category
                cat_images[str(x[0].split("/")[-1])] = len(x[-1])
    return cat_images


# returns the number of images required for each category
def get_number_of_images_required():
    # calculating number of images required for each category (-0.000001 for float precision errors)
    # (for with test set add 1)
    return math.ceil((1.0 / (1.0 - float(config['DEFAULT']['train_ratio']))) - 0.000001)


# Returns a string error message that a category has to be added
def get_no_cat_error():
    req_images_per_cat = get_number_of_images_required()
    return "Please add at least one category (by adding at least <b>%d</b> images in that category)." % req_images_per_cat


def not_enough_images():
    req_images_per_cat = get_number_of_images_required()
    cat_img = get_number_of_images_per_category()
    return all(cat_img[cat] < req_images_per_cat for cat in cat_img.keys())


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
        msg += "but at least <b>%d</b> images are required for each category. Please add at least the required images." % req_images_per_cat
    return msg
