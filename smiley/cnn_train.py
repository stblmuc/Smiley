import numpy
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError, NotFoundError
import prepare_training_data, cnn_model
import os
import configparser


def train():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'trainConfig.ini'))

    MODEL_DIRECTORY = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['MODELS'], config['DEFAULT']['IMAGE_SIZE'] + "/" + config['CNN']['MODEL_FILENAME'])
    LOGS_DIRECTORY = os.path.join(os.path.dirname(__file__), config['DIRECTORIES']['LOGS'])
    IMAGE_SIZE = int(config['DEFAULT']['IMAGE_SIZE'])

    print("\nCNN TRAINING STARTED.")

    # get training/validation/testing data
    try:
        curr_number_of_categories, train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = prepare_training_data.prepare_data(
            "CNN", True)
    except TypeError:
        raise Exception("Error preparing training/validation/test data. Create more training examples.")

    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'trainConfig.ini'))

    batch_size = int(config['DEFAULT']['TRAIN_BATCH_SIZE'])    
    is_training = tf.placeholder(tf.bool)

    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE], name="image")  # CNN input
    y_ = tf.placeholder(tf.float32, [None, curr_number_of_categories], name="labels")  # CNN output
    # CNN model
    y, variables = cnn_model.convolutional(x, nCategories=curr_number_of_categories)

    # loss function
    with tf.name_scope("Loss"):
        loss = tf.losses.softmax_cross_entropy(y_, y)

    # create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # define optimizer
    with tf.name_scope("ADAM"):
        # optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            float(config['CNN']['LEARNING_RATE']),  # base learning rate.
            batch * batch_size,  # current index into the dataset.Sav
            train_size,  # decay step.
            0.95,  # decay rate.
            staircase=True)

        # use simple momentum for the optimization
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

    # create a summary to monitor learning_rate tensor
    tf.summary.scalar('learning_rate', learning_rate)

    # get accuracy of model
    with tf.name_scope("Acc"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})
    saver = tf.train.Saver(variables)

    # training cycle
    total_batch = int(train_size / batch_size)

    # op to write logs to Tensorboard
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # restore stored CNN model if it exists and has the correct number of categories
    max_acc = maybe_restore_model(MODEL_DIRECTORY, saver, sess, accuracy, validation_data, x, validation_labels, y_, is_training)

    # loop for epoch
    for epoch in range(int(config['CNN']['EPOCHS'])):

        # random shuffling
        numpy.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-curr_number_of_categories]
        train_labels_ = train_total_data[:, -curr_number_of_categories:]

        # loop over all batches
        for i in range(total_batch):

            # compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (train_size)
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            # run optimization op (backprop), loss op (to get loss value) and summary nodes
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op],
                                                  feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            validation_accuracy = computeAccuracy(MODEL_DIRECTORY, saver, sess, accuracy, train_accuracy, i, total_batch, epoch, validation_data, x, 
                validation_labels, y_, is_training, int(config['LOGS']['TRAIN_ACCURACY_DISPLAY_STEP']), int(config['LOGS']['VALIDATION_STEP']))
            # save the current model if the maximum accuracy is updated
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_DIRECTORY, write_meta_graph=False, write_state=False)
                print("Model updated and saved in file: %s" % save_path)

                # saver.save(sess, LOGS_DIRECTORY + "CNN", epoch)

    print("Optimization Finished!")

    # restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)

    # calculate accuracy for all test images
    test_size = test_labels.shape[0]
    batch_size = min(test_size, int(config['DEFAULT']['TEST_BATCH_SIZE']))
    total_batch = int(test_size / batch_size)

    acc_buffer = []

    # loop over all batches
    for i in range(total_batch):
        # compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size), :]
        batch_ys = test_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
        acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))

    sess.close()

    print("CNN TRAINING END.")

def maybe_restore_model(model_path, saver, sess, accuracy, validation_data, x, validation_labels, y_, is_training):
    try:
        saver.restore(sess, model_path)
        # save the current maximum accuracy value for validation data
        max_acc = sess.run(accuracy,
                           feed_dict={x: validation_data, y_: validation_labels,
                                      is_training: False})
    except (NotFoundError, InvalidArgumentError):
        # initialize the maximum accuracy value for validation data
        max_acc = 0.
    return max_acc

def computeAccuracy(MODEL_DIRECTORY, saver, sess, accuracy, train_accuracy, i, total_batch, epoch, validation_data, x, validation_labels, y_, is_training, DISPLAY_STEP, VALIDATION_STEP):
    if i % DISPLAY_STEP == 0:
        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

    # get accuracy for validation data
    validation_accuracy = 0
    if i % VALIDATION_STEP == 0:
        # calculate accuracy
        validation_accuracy = sess.run(accuracy,
                                       feed_dict={x: validation_data, y_: validation_labels,
                                                  is_training: False})

        print("Epoch:", '%04d,' % (epoch + 1),
              "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))
    return validation_accuracy

if __name__ == '__main__':
    train()
