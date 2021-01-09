#!/usr/bin/env python3
"""   Evaluate  """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """function that evaluates the output of a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        x = tf.get_collection('x')
        y = tf.get_collection('y')
        y_pred = tf.get_collection('y_pred')
        pred_y = sess.run(y_pred[0], feed_dict={x[0]: X, y[0]: Y})
        loss = tf.get_collection('loss')
        loss_train = sess.run(loss[0], feed_dict={x[0]: X, y[0]: Y})
        ac = tf.get_collection('accuracy')
        ac_train = sess.run(ac[0], feed_dict={x[0]: X, y[0]: Y})

        return pred_y, ac_train, loss_train
