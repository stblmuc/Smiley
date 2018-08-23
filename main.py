import configparser
import math
import os
import sys
import webbrowser
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError

sys.path.append('smiley')
import regression_model, cnn_model, regression_train, cnn_train, utils

config = configparser.ConfigParser()
config.file = os.path.join(os.path.dirname(__file__), 'smiley/config.ini')
config.read(config.file)

MODELS_DIRECTORY = os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['MODELS'],
                                config['DEFAULT']['IMAGE_SIZE'])
IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])

# create folder for models if it doesn't exist
if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)

# updates the models if the number of classes changed
def maybe_update_models():
    global y1, variables, saver_regression, y2, saver_cnn, x, is_training, sess, num_categories
    if 'num_categories' not in globals() or num_categories != len(utils.update_categories()):
        # close (old) tensorflow if existent
        if 'sess' in globals():
            sess.close()

        num_categories = len(utils.CATEGORIES)

        # Model variables
        x = tf.placeholder("float", [None, IMAGE_SIZE * IMAGE_SIZE])  # image input placeholder
        is_training = tf.placeholder("bool")  # used for activating the dropout

        # Tensorflow session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        # Regression model
        y1, variables = regression_model.regression(x, nCategories=num_categories)
        saver_regression = tf.train.Saver(variables)

        # CNN model
        y2, variables = cnn_model.convolutional(x, nCategories=num_categories, is_training=is_training)
        saver_cnn = tf.train.Saver(variables)


# Initialize the categories mapping, the tensorflow session and the models
maybe_update_models()


# Regression prediction
def regression_predict(input):
    saver_regression.restore(sess, os.path.join(MODELS_DIRECTORY, config['REGRESSION']['MODEL_FILENAME']))
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


# CNN prediction
def cnn_predict(input):
    saver_cnn.restore(sess, os.path.join(MODELS_DIRECTORY, config['CNN']['MODEL_FILENAME']))
    return sess.run(y2, feed_dict={x: input, is_training: False}).flatten().tolist()


# Webapp definition
app = Flask(__name__)

# Root
@app.route('/')
def main():
    numAugm = config['DEFAULT']['NUMBER_AUGMENTATIONS_PER_IMAGE']
    batchSize = config['DEFAULT']['train_batch_size']
    srRate = config['REGRESSION']['LEARNING_RATE']
    srEpochs = config['REGRESSION']['EPOCHS']
    cnnRate = config['CNN']['LEARNING_RATE']
    cnnEpochs = config['CNN']['EPOCHS']
    data = {'image_size': IMAGE_SIZE, 'numAugm': numAugm, 'batchSize': batchSize, 'srRate': srRate,
            'srEpochs': srEpochs, 'cnnRate': cnnRate, 'cnnEpochs': cnnEpochs,
            'categories': list(utils.CATEGORIES.keys())}
    return render_template('index.html', data=data)


# Predict
@app.route('/api/smiley', methods=['POST'])
def smiley():
    maybe_update_models()

    # input with pixel values between 0 (black) and 255 (white)
    data = np.array(request.json, dtype=np.uint8)

    # transform pixels to values between 0 (white) and 1 (black)
    regression_input = ((255 - data) / 255.0).reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    # transform pixels to values between -0.5 (white) and 0.5 (black)
    cnn_input = (((255 - data) / 255.0) - 0.5).reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    err = ""  # string with error messages

    if num_categories == 0:
        err = utils.get_no_cat_error()

    # if too less images are added, print an error message
    elif utils.not_enough_images():
        err = utils.get_not_enough_images_error()
        regression_output = []
        cnn_output = []

    else:
        retrain_error = "Models not found or incompatible number of categories or incompatible image size. Please (re-)train the classifiers."

        try:
            regression_output = regression_predict(regression_input)
            regression_output = [-1.0 if math.isnan(b) else b for b in regression_output]
        except (NotFoundError, InvalidArgumentError):
            regression_output = []
            err = retrain_error

        try:
            cnn_output = cnn_predict(cnn_input)
            cnn_output = [-1.0 if math.isnan(f) else f for f in cnn_output]
        except (NotFoundError, InvalidArgumentError):
            cnn_output = []
            err = retrain_error

    return jsonify(classifiers=["Softmax Regression", "CNN"], results=[regression_output, cnn_output],
                   error=err, categories=utils.get_category_names())


# Add training example
@app.route('/api/generate-training-example', methods=['POST'])
def generate_training_example():
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    image = np.array(request.json["img"], dtype=np.uint8).reshape(image_size, image_size, 1)
    category = request.json["cat"]
    utils.add_training_example(image, category)

    if utils.not_enough_images():
        err = utils.get_not_enough_images_error()
        return jsonify(error=err)
    else:
        return "ok"


# Update config parameters
@app.route('/api/update-config', methods=['POST'])
def update_config():
    config.set("CNN", "LEARNING_RATE", request.json["cnnLearningRate"])
    config.set("REGRESSION", "LEARNING_RATE", request.json["srLearningRate"])
    config.set("CNN", "EPOCHS", request.json["cnnEpochs"])
    config.set("REGRESSION", "EPOCHS", request.json["srEpochs"])
    config.set("DEFAULT", "number_augmentations_per_image", request.json["numberAugmentations"])
    config.set("DEFAULT", "train_batch_size", request.json["batchSize"])

    # Write config back to file
    with open(config.file, "w") as f:
        config.write(f)

    return "ok"


# Train model
@app.route('/api/train-models', methods=['POST'])
@utils.capture
def train_models():
    maybe_update_models()

    # if no categories are added, print error
    if num_categories == 0:
        err = utils.get_no_cat_error()
        return jsonify(error=err)
    elif utils.not_enough_images():
        err = utils.get_not_enough_images_error()
        return jsonify(error=err)
    try:
        regression_train.train()
        cnn_train.train()
    except:
        err = "Unknown error."
        return jsonify(error=err)

    return "ok"


# Delete all saved models
@app.route('/api/delete-all-models', methods=['POST'])
def delete_all_models():
    filelist = [f for f in os.listdir(MODELS_DIRECTORY)]
    for f in filelist:
        os.remove(os.path.join(MODELS_DIRECTORY, f))

    return "ok"

@app.route('/api/get-console-output')
def console_output():
    output = utils.LOGGER.pop()
    return jsonify(out=output)


# main
if __name__ == '__main__':
    # Open webbrowser tab for the app
    webbrowser.open_new_tab("http://localhost:5000")

    app.run()
