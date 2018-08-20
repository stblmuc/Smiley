import configparser
import math
import os
import sys
import webbrowser
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError

sys.path.append("smiley")
import regression_model, cnn_model, category_manager, regression_train, cnn_train

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'smiley/config.ini'))

MODELS_DIRECTORY = os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['MODELS'],
                                config['DEFAULT']['IMAGE_SIZE'])
IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])

# Initialize the mapping between categories and indices in the prediction vectors
category_manager.update()
num_categories = len(category_manager.CATEGORIES)

# create folder for models if it doesn't exist
if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)

# Model variables
x = tf.placeholder("float", [None, IMAGE_SIZE * IMAGE_SIZE])  # image placeholder
is_training = tf.placeholder("bool")

# Tensorflow session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Regression model
y1, variables = regression_model.regression(x, nCategories=num_categories)
saver_regression = tf.train.Saver(variables)

# CNN model
y2, variables = cnn_model.convolutional(x, nCategories=num_categories, is_training=is_training)
saver_cnn = tf.train.Saver(variables)

# Webapp definition
app = Flask(__name__)


# updates the models if the number of classes changed
def maybe_update_models():
    global y1, variables, saver_regression, y2, saver_cnn, x, is_training, sess, num_categories
    if num_categories != len(category_manager.update()):
        num_categories = len(category_manager.CATEGORIES)
        x = tf.placeholder("float", [None, IMAGE_SIZE * IMAGE_SIZE])  # image placeholder
        is_training = tf.placeholder("bool")

        # Tensorflow session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        y1, variables = regression_model.regression(x, nCategories=num_categories)
        saver_regression = tf.train.Saver(variables)

        y2, variables = cnn_model.convolutional(x, nCategories=num_categories, is_training=is_training)
        saver_cnn = tf.train.Saver(variables)


# Regression prediction
def regression_predict(input):
    saver_regression.restore(sess, os.path.join(MODELS_DIRECTORY, config['REGRESSION']['MODEL_FILENAME']))
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


# CNN prediction
def cnn_predict(input):
    saver_cnn.restore(sess, os.path.join(MODELS_DIRECTORY, config['CNN']['MODEL_FILENAME']))
    return sess.run(y2, feed_dict={x: input, is_training: False}).flatten().tolist()


# Root
@app.route('/')
def main():
    numAugm = config['DEFAULT']['NUMBER_AUGMENTATIONS_PER_IMAGE']
    batchSize = config['DEFAULT']['train_batch_size']
    lrRate = config['REGRESSION']['LEARNING_RATE']
    lrEpochs = config['REGRESSION']['EPOCHS']
    cnnRate = config['CNN']['LEARNING_RATE']
    cnnEpochs = config['CNN']['EPOCHS']
    data = {'image_size': IMAGE_SIZE, 'numAugm': numAugm, 'batchSize': batchSize, 'lrRate': lrRate,
            'lrEpochs': lrEpochs, 'cnnRate': cnnRate, 'cnnEpochs': cnnEpochs,
            'categories': list(category_manager.CATEGORIES.keys())}
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

    try:
        regression_output = regression_predict(regression_input)
        regression_output = [-1.0 if math.isnan(b) else b for b in regression_output]
    except (NotFoundError, InvalidArgumentError):
        regression_output = []
    try:
        cnn_output = cnn_predict(cnn_input)
        cnn_output = [-1.0 if math.isnan(f) else f for f in cnn_output]
    except (NotFoundError, InvalidArgumentError):
        cnn_output = []

    err = ""  # string with error messages
    if len(regression_output) == 0:
        if len(cnn_output) == 0:
            # error loading both models
            err = "Models not found or incompatible number of categories or incompatible image size. Please (re-)train the classifiers."
        else:
            # error loading regression model
            err = "Model not found or incompatible number of categories or incompatible image size. Please (re-)train the regression classifier."
    elif len(cnn_output) == 0:
        # error loading CNN model
        err = "Model not found or incompatible number of categories or incompatible image size. Please (re-)train the CNN classifier."

    if len(err) > 0:
        print(err)

    category_names = ["" for _ in range(len(category_manager.CATEGORIES))]
    for ind in range(len(category_names)):
        category_names[ind] = [x for x in category_manager.CATEGORIES.keys() if category_manager.CATEGORIES[x] == ind][
            0]

    return jsonify(classifiers=["Linear Regression", "CNN"], results=[regression_output, cnn_output],
                   error=err, categories=category_names)


# Add training example
@app.route('/api/generate-training-example', methods=['POST'])
def generate_training_example():
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    image = np.array(request.json["img"], dtype=np.uint8).reshape(image_size, image_size, 1)
    category = request.json["cat"]
    category_manager.add_training_example(image, category)

    return "ok"


# Update config parameters
@app.route('/api/update-config', methods=['POST'])
def update_config():
    config.set("CNN", "LEARNING_RATE", request.json["cnnLearningRate"])
    config.set("REGRESSION", "LEARNING_RATE", request.json["lrLearningRate"])
    config.set("CNN", "EPOCHS", request.json["cnnEpochs"])
    config.set("REGRESSION", "EPOCHS", request.json["lrEpochs"])
    config.set("DEFAULT", "number_augmentations_per_image", request.json["numberAugmentations"])
    config.set("DEFAULT", "train_batch_size", request.json["batchSize"])

    # Write config back to file
    with open(os.path.join(os.path.dirname(__file__), 'smiley/config.ini'), "w") as f:
        config.write(f)

    return "ok"


# Train model
@app.route('/api/train-models', methods=['POST'])
def train_models():
    category_manager.update()
    try:
        regression_train.train()
        cnn_train.train()
    except Exception as inst:
        if len(inst.args) > 0:
            err = inst.args[0]
        else:
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


# main
if __name__ == '__main__':
    # Open webbrowser tab for the app
    webbrowser.open_new_tab("http://localhost:5000")

    app.run()
