import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError
import webbrowser
import os, sys
import math
import configparser

sys.path.append("smiley")
import regression_model, cnn_model, category_manager, regression_train, cnn_train

config = configparser.ConfigParser()
config.read('./smiley/trainConfig.ini')

MODELS_DIRECTORY = config['DEFAULT']['LOGIC_DIRECTORY'] + config['DEFAULT']['MODELS_DIRECTORY']

# Initialize the mapping between categories and indices in the prediction vectors
category_manager.update()

# create folder for models if it doesn't exist
if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)

# Model variables
x = tf.placeholder("float", [None, 784])
is_training = tf.placeholder("bool")

# Tensorflow session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Regression model
y1, variables = regression_model.regression(x, categories=len(category_manager.CATEGORIES))
saver_regression = tf.train.Saver(variables)

# CNN model
y2, variables = cnn_model.convolutional(x, nCategories=len(category_manager.CATEGORIES), is_training=is_training)
saver_cnn = tf.train.Saver(variables)

# Webapp definition
app = Flask(__name__)

# Regression prediction
def regression_predict(input):
    saver_regression.restore(sess, MODELS_DIRECTORY + config['REGRESSION']['MODEL_FILENAME'])  # load saved model
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()

# CNN prediction
def cnn_predict(input):
    saver_cnn.restore(sess, MODELS_DIRECTORY + config['CNN']['MODEL_FILENAME'])  # load saved model
    result = sess.run(y2, feed_dict={x: input, is_training: False}).flatten().tolist()
    return result

# Root
@app.route('/')
def main():
    return render_template('index.html')

# Predict
@app.route('/api/smiley', methods=['POST'])
def smiley():
    # input with pixel values between 0 (black) and 255 (white)
    data = np.array(request.json, dtype=np.uint8)

    # transform pixels to values between 0 (white) and 1 (black)
    regression_input = ((255 - data) / 255.0).reshape(1, 784)

    # transform pixels to values between -0.5 (white) and 0.5 (black)
    cnn_input = (((255 - data) / 255.0) - 0.5).reshape(1, 784)

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
            err = "Models not found or incompatible number of categories. Please retrain the classifiers or restart the server"
        else:
            # error loading regression model
            err = "Model not found or incompatible number of categories. Please retrain the regression classifier or restart the server"
    elif len(cnn_output) == 0:
        # error loading CNN model
        err = "Model not found or incompatible number of categories. Please retrain the CNN classifier or restart the server"

    if len(err) > 0:
        print(err)

    category_names = ["" for x in range(len(category_manager.CATEGORIES))]
    for ind in range(len(category_names)):
        category_names[ind] = [x for x in category_manager.CATEGORIES.keys() if category_manager.CATEGORIES[x] == ind][0]

    return jsonify(classifiers=["Linear Regression", "CNN"], results=[regression_output, cnn_output],
                   error=err, categories=category_names)


# Add training example
@app.route('/api/generate-training-example', methods=['POST'])
def generate_training_example():
    image = np.array(request.json["img"], dtype=np.uint8).reshape(28, 28, 1)
    category = request.json["cat"]

    category_manager.add_training_example(image, category)

    return "ok"

# Train model
@app.route('/api/train-models', methods=['POST'])
def train_models():
    regression_train.train()
    cnn_train.train()

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
    new = 2 # open in a new tab, if possible
    webbrowser.open("http://localhost:5000", new=new)
    
    app.run()
