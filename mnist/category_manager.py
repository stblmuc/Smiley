import os
import png

CATEGORIES_LOCATION = "data/categories/"
CATEGORIES = None


def update():
    global CATEGORIES

    # find path to category_manager.py
    dir = os.path.dirname(__file__)

    # create folder for categories if it doesn't exist:
    if not os.path.exists(os.path.join(dir, CATEGORIES_LOCATION)):
        os.makedirs(os.path.join(dir, CATEGORIES_LOCATION))

    # dictionary with keys = category names (from folders), values = category indices
    CATEGORIES = {x: i for (i, x) in
                  enumerate(sorted([z[0].split("/")[-1] for z in os.walk(os.path.join(dir, CATEGORIES_LOCATION)) if
                                    z[0].count("/") == os.path.join(dir, CATEGORIES_LOCATION).count("/") and len(
                                        z[0].split("/")[-1]) > 0]))}
    return CATEGORIES


def add_training_example(image, category):
    # find path to category_manager.py
    dir = os.path.dirname(__file__)

    # create folder for category if it doesn't exist:
    if not os.path.exists(os.path.join(dir, CATEGORIES_LOCATION, category)):
        os.makedirs(os.path.join(dir, CATEGORIES_LOCATION, category))

    # name for new training example image
    image_name = max(
        [0] + [int(x.split(".")[0]) for x in os.listdir(os.path.join(dir, CATEGORIES_LOCATION, category))]) + 1

    # store new training example image
    w = png.Writer(28, 28, greyscale=True)
    w.write(open(os.path.join(dir, CATEGORIES_LOCATION, category) + "/" + str(image_name) + ".png", "wb"), image)
