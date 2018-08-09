import numpy
import regression_model
import tensorflow as tf
import prepare_training_data, category_manager
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError
import os
import configparser


def train():
    print("\nLINEAR REGRESSION TRAINING STARTED.")

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'trainConfig.ini'))

    BATCH_SIZE = int(config['DEFAULT']['TRAIN_BATCH_SIZE'])
    MODEL_DIRECTORY = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['MODELS'] + config['REGRESSION']['MODEL_FILENAME'])
    IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])

    # get training/validation/testing data
    try:
        curr_number_of_categories, train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = prepare_training_data.prepare_data(
            "regression", True)
    except TypeError:
        raise Exception("Error preparing training/validation/test data. Create more training examples.")

    # regression model
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE], name="image")  # regression input
    y_ = tf.placeholder(tf.float32, [None, curr_number_of_categories], name="labels")  # regression output
    y, variables = regression_model.regression(x, nCategories=curr_number_of_categories)
    # training variables
    with tf.name_scope("Loss"):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    with tf.name_scope("GradientDescent"):
        train_step = tf.train.GradientDescentOptimizer(float(config['REGRESSION']['LEARNING_RATE'])).minimize(cross_entropy)
    with tf.name_scope("Acc"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # merge training data and validation data
    validation_total_data = numpy.concatenate((validation_data, validation_labels), axis=1)
    new_train_total_data = numpy.concatenate((train_total_data, validation_total_data))
    train_size = new_train_total_data.shape[0]

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(variables)

    # training cycle
    total_batch = int(train_size / BATCH_SIZE)

    # restore stored regression model if it exists and has the correct number of categories
    max_acc = maybe_restore_model(MODEL_DIRECTORY, saver, sess, accuracy, validation_data, x, validation_labels, y_)

    # loop for epoch
    for epoch in range(int(config['REGRESSION']['EPOCHS'])):

        # random shuffling
        numpy.random.shuffle(train_total_data)
        train_data_ = new_train_total_data[:, :-curr_number_of_categories]
        train_labels_ = new_train_total_data[:, -curr_number_of_categories:]

        # loop over all batches
        for i in range(total_batch):
            # compute the offset of the current minibatch in the data.
            offset = (i * BATCH_SIZE) % (train_size)
            batch_xs = train_data_[offset:(offset + BATCH_SIZE), :]
            batch_ys = train_labels_[offset:(offset + BATCH_SIZE), :]

            _, train_accuracy = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})

            validation_accuracy = computeAccuracy(MODEL_DIRECTORY, saver, sess, accuracy, train_accuracy, i, total_batch, epoch, validation_data, x, 
                validation_labels, y_, int(config['LOGS']['TRAIN_ACCURACY_DISPLAY_STEP']), int(config['LOGS']['VALIDATION_STEP']))

            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_DIRECTORY, write_meta_graph=False, write_state=False)
                print("Model updated and saved in file: %s" % save_path)

    print("Optimization Finished!")

    # restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)

    # calculate accuracy for all test images
    test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
    print("test accuracy for the stored model: %g" % test_accuracy)

    sess.close()

    print("LINEAR REGRESSION TRAINING END.")

def maybe_restore_model(model_path, saver, sess, accuracy, validation_data, x, validation_labels, y_):    
    try:
        saver.restore(sess, model_path)
        # save the current maximum accuracy value for validation data
        max_acc = sess.run(accuracy, feed_dict={x: validation_data, y_: validation_labels})
    except (NotFoundError, InvalidArgumentError):
        # initialize the maximum accuracy value for validation data
        max_acc = 0.
    return max_acc

def computeAccuracy(MODEL_DIRECTORY, saver, sess, accuracy, train_accuracy, i, total_batch, epoch, validation_data, x, validation_labels, y_, DISPLAY_STEP, VALIDATION_STEP):
    if i % DISPLAY_STEP == 0:
        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

    # get accuracy for validation data
    validation_accuracy = 0
    if i % VALIDATION_STEP == 0:
        # calculate accuracy
        validation_accuracy = sess.run(accuracy, feed_dict={x: validation_data, y_: validation_labels})

        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))
    return validation_accuracy

if __name__ == '__main__':
    train()