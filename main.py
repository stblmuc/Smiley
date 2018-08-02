import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from mnist import regression_model, cnn_model, category_manager, keras_model
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError
import os
import math
import matplotlib as mp
import matplotlib.pyplot as plt

MODELS_DIRECTORY = "mnist/data/models/"

# Initialize the mapping between categories and indices in the prediction vectors
category_manager.update()

# create folder for models if it doesn't exist
if not os.path.exists("mnist/data/models/"):
    os.makedirs("mnist/data/models/")

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
y2, variables = cnn_model.CNN(x, categories=len(category_manager.CATEGORIES), is_training=is_training)
saver_cnn = tf.train.Saver(variables)

#############  put your keras learned .h5-filename here  ####################
keras = keras_model.Keras("first_try_keras_10.h5", numCats=10)

# Webapp definition
app = Flask(__name__)


# Regression prediction
def regression_predict(input):
    saver_regression.restore(sess, MODELS_DIRECTORY + "regression.ckpt")  # load saved model
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


# CNN prediction
def cnn_predict(input):
    saver_cnn.restore(sess, MODELS_DIRECTORY + "convolutional.ckpt")  # load saved model
    result = sess.run(y2, feed_dict={x: input, is_training: False}).flatten().tolist()
    return [z - min(result) for z in result]  # shift predictions to avoid negative values


# Keras prediction
def keras_predict(input):
    pred = keras.predict(input)
    output = np.zeros(12)  # up to 12 classes
    fillUptoIdx = len(pred[0])
    output[:fillUptoIdx] = pred
    output = output.tolist()
    return output


# Root
@app.route('/')
def main():
    return render_template('index.html')


# Predict
@app.route('/api/mnist', methods=['POST'])
def mnist():
    # input with pixel values between 0 (black) and 255 (white)
    data = np.array(request.json, dtype=np.uint8)

    # transform pixels to values between 0 (white) and 1 (black)
    regression_input = ((255 - data) / 255.0).reshape(1, 784)

    # transform pixels to values between -0.5 (white) and 0.5 (black)
    cnn_input = (((255 - data) / 255.0) - 0.5).reshape(1, 784)

    keras_input = (data / 255.0).reshape(1, 28, 28, 1)
    #keras_output = keras_predict(keras_input)

    # get_activations(sess.graph.get_tensor_by_name("conv2/Relu:0"), cnn_input)

    try:
        regression_output = regression_predict(regression_input)
        regression_output = [-1.0 if math.isnan(b) else b for b in regression_output]
    except (NotFoundError, InvalidArgumentError):
        regression_output = []
    try:
        cnn_output = cnn_predict(cnn_input)
        cnn_output = [-1.0 if math.isnan(f) else f for f in cnn_output]
        print(cnn_output)
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

    return jsonify(classifiers=["regression", "CNN"], results=[regression_output, cnn_output],
                   error=err,
                   categories=category_names)


# Add training example
@app.route('/api/generate-training-example', methods=['POST'])
def generate_training_example():
    image = np.array(request.json["img"], dtype=np.uint8).reshape(28, 28, 1)
    category = request.json["cat"]

    category_manager.add_training_example(image, category)

    return "ok"


# functions for plotting activations of different layers:
def plot_nn_filter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
    print("done plotting")


def get_activations(layer, stimuli):
    units = sess.run(layer, feed_dict={x: stimuli, is_training: False})
    plot_nn_filter(units)


# main
if __name__ == '__main__':
    app.run()
