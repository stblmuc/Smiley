import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from smiley import regression_model, cnn_model, category_manager
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError
import os
import math

MODELS_DIRECTORY = "smiley/data/models/"

# Initialize the mapping between categories and indices in the prediction vectors
category_manager.update()

# create folder for models if it doesn't exist
if not os.path.exists("smiley/data/models/"):
    os.makedirs("smiley/data/models/")

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
    saver_regression.restore(sess, MODELS_DIRECTORY + "regression.ckpt")  # load saved model
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()

# CNN prediction
def cnn_predict(input):
    saver_cnn.restore(sess, MODELS_DIRECTORY + "convolutional1.ckpt")  # load saved model
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
        category_names[ind] = [x for x in category_manager.CATEGORIES.keys() if category_manager.CATEGORIES[x] == ind][
            0]

    return jsonify(classifiers=["Linear Regression", "CNN"], results=[regression_output, cnn_output],
                   error=err,
                   categories=category_names)


# Add training example
@app.route('/api/generate-training-example', methods=['POST'])
def generate_training_example():
    image = np.array(request.json["img"], dtype=np.uint8).reshape(28, 28, 1)
    category = request.json["cat"]

    category_manager.add_training_example(image, category)

    return "ok"

# main
if __name__ == '__main__':
    app.run()
