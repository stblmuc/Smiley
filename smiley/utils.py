import configparser
import os
import png
import math

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

CATEGORIES_LOCATION = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['CATEGORIES'],
                                       config['DEFAULT']['IMAGE_SIZE'] + "/")
CATEGORIES = None


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
def get_no_cat_error_msg():
    req_images_per_cat = get_number_of_images_required()
    return "Please add at least one category (by adding at least <b>%d</b> images in that category)." % req_images_per_cat


# Returns a string error message with the number of images for each category which is below the minimum images required
def get_too_less_images_error_msg():
    msg = ""
    req_images_per_cat = get_number_of_images_required()
    cat_img = get_number_of_images_per_category()
    for cat in cat_img.keys():
        if cat_img[cat] < req_images_per_cat:
            msg += "category '<b>" + cat + "</b>' has just <b>%d</b> images, " % cat_img[cat]
    if len(msg) > 0:
        msg += "but at least <b>%d</b> images are required for each category." % req_images_per_cat
    return msg