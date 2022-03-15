#!/usr/bin/env python3
"""module for evaluate function"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """loads tf network from file, evaluates for x and y"""
    sess = tf.Session()
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(sess, save_path)
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    pred = tf.get_collection('y_pred')[0]
    acc = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    return sess.run([pred, acc, loss], feed_dict={x: X, y: Y})[0:3]

    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        acc = tf.get_collection("accuracy")[0]
        vars = sess.run([y_pred, acc, loss], feed_dict={x: X, y: Y})
        return vars[0], vars[1], vars[2]
    """
