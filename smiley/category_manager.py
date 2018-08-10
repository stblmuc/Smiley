import configparser
import os
import png

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'trainConfig.ini'))

CATEGORIES_LOCATION = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['CATEGORIES'],
                                   config['DEFAULT']['IMAGE_SIZE'] + "/")
CATEGORIES = None


def update():
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
