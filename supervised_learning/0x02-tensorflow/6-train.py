#!/usr/bin/env python3
"""  Train """
import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"
          ):
    """ function that builds, trains,
    and saves a neural network classifier"""
    classes = Y_train.shape[1]
    nx = X_train.shape[1]
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    ac = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train)
    tf.add_to_collection('accuracy', ac)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    initialize = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(initialize)
        for i in range(iterations + 1):
            cost_train, ac_train = sess.run((loss, ac), feed_dict={
                x: X_train,
                y: Y_train
            })
            cost_valid, ac_valid = sess.run((loss, ac), feed_dict={
                x: X_valid,
                y: Y_valid
            })
            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(ac_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(ac_valid))
            if(i != iterations):
                sess.run(train, {x: X_train, y: Y_train})
        return saver.save(sess, save_path)
