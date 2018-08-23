import configparser
import math
import os
import sys
import webbrowser
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_file
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError
from io import StringIO
from functools import wraps

sys.path.append('smiley')
import regression_model, cnn_model, utils, regression_train, cnn_train

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'smiley/config.ini'))

MODELS_DIRECTORY = os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['MODELS'],
                                config['DEFAULT']['IMAGE_SIZE'])
IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])

# create folder for models if it doesn't exist
if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)

# updates the models if the number of classes changed
def maybe_update_models():
    global y1, variables, saver_regression, y2, saver_cnn, x, is_training, sess, num_categories
    if 'num_categories' not in globals() or num_categories != len(utils.update()):
        num_categories = len(utils.CATEGORIES)

        # Model variables
        x = tf.placeholder("float", [None, IMAGE_SIZE * IMAGE_SIZE])  # image placeholder
        is_training = tf.placeholder("bool") # used for activating the dropout

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

# Delete last console output
if os.path.isfile(os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['LOGS'], 'console.txt')):
    os.remove(os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['LOGS'], 'console.txt'))

# Decorator to capture standard output
def capture(f):
    @wraps(f)
    def captured(*args, **kwargs):
        backup = sys.stdout # setup the environment

        try:
            sys.stdout = StringIO()     # capture output
            result = f(*args, **kwargs)
            out = sys.stdout.getvalue() # release output
        finally:
            sys.stdout.close()  # close the stream 
            sys.stdout = backup # restore original stdout

        with open(os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['LOGS'], 'console.txt'), "a") as file:
            file.write(out) # write print output to file
            file.close()

        return result # captured result from function
    return captured

# Root
@app.route('/')
@capture
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
@capture
def smiley():
    maybe_update_models()

    # input with pixel values between 0 (black) and 255 (white)
    data = np.array(request.json, dtype=np.uint8)

    # transform pixels to values between 0 (white) and 1 (black)
    regression_input = ((255 - data) / 255.0).reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    # transform pixels to values between -0.5 (white) and 0.5 (black)
    cnn_input = (((255 - data) / 255.0) - 0.5).reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    err = ""  # string with error messages
    err_retrain = "Models not found or incompatible number of categories or incompatible image size. Please (re-)train the classifiers."
    
    try:
        regression_output = regression_predict(regression_input)
        regression_output = [-1.0 if math.isnan(b) else b for b in regression_output]
    except (NotFoundError, InvalidArgumentError):
        regression_output = []
        err = err_retrain

    try:
        cnn_output = cnn_predict(cnn_input)
        cnn_output = [-1.0 if math.isnan(f) else f for f in cnn_output]
    except (NotFoundError, InvalidArgumentError):
        cnn_output = []
        err = err_retrain

    # if no categories are added, print error
    if (num_categories == 0):
        err = "Please add at least one category (by adding N images in that category)."

    return jsonify(classifiers=["Softmax Regression", "CNN"], results=[regression_output, cnn_output],
                   error=err, categories=utils.get_category_names())


# Add training example
@app.route('/api/generate-training-example', methods=['POST'])
@capture
def generate_training_example():
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])
    image = np.array(request.json["img"], dtype=np.uint8).reshape(image_size, image_size, 1)
    category = request.json["cat"]
    utils.add_training_example(image, category)

    return "ok"

# Update config parameters
@app.route('/api/update-config', methods=['POST'])
@capture
def update_config():
    config.set("CNN", "LEARNING_RATE", request.json["cnnLearningRate"])
    config.set("REGRESSION", "LEARNING_RATE", request.json["srLearningRate"])
    config.set("CNN", "EPOCHS", request.json["cnnEpochs"])
    config.set("REGRESSION", "EPOCHS", request.json["srEpochs"])
    config.set("DEFAULT", "number_augmentations_per_image", request.json["numberAugmentations"])
    config.set("DEFAULT", "train_batch_size", request.json["batchSize"])

    # Write config back to file
    with open(os.path.join(os.path.dirname(__file__), 'smiley/config.ini'), "w") as f:
        config.write(f)

    return "ok"

# Train model
@app.route('/api/train-models', methods=['POST'])
@capture
def train_models():
    maybe_update_models()

    # if no categories are added, print error
    if (num_categories == 0):
        err = "Please add at least one category (by adding images in that category)."
        return jsonify(error=err)

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
@capture
@app.route('/api/delete-all-models', methods=['POST'])
def delete_all_models():
    filelist = [f for f in os.listdir(MODELS_DIRECTORY)]
    for f in filelist:
        os.remove(os.path.join(MODELS_DIRECTORY, f))

    return "ok"

@app.route('/api/get-console-output')
@capture
def console_output():
    console_file = os.path.join(config['DIRECTORIES']['LOGIC'], config['DIRECTORIES']['LOGS'], 'console.txt')
    if (os.path.isfile(console_file) and os.path.getsize(console_file) > 0):
        print(os.path.getsize(console_file))
        return send_file(console_file)
    else:
        return 'No entries (yet)'

# main
if __name__ == '__main__':
    # Open webbrowser tab for the app
    webbrowser.open_new_tab("http://localhost:5000")

    app.run()
