#!/usr/bin/env python3
"""module for normalization constants"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """trains a loaded neural network model using mini-batch grad desc"""
    with tf.Session() as sess:
        load = tf.train.import_meta_graph(load_path + '.meta')
        load.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        train_op = tf.get_collection("train_op")
        batches = X_train.shape[0]//batch_size
        if X_train.shape[0] % batch_size != 0:
            batches += 1
        for i in range(epochs+1):
            tLoss, tAccuracy = sess.run((loss, accuracy),
                                        feed_dict={x: X_train, y:  Y_train})
            vLoss, vAccuracy = sess.run((loss, accuracy),
                                        feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(tLoss))
            print("\tTraining Accuracy: {}".format(tAccuracy))
            print("\tValidation Cost: {}".format(vLoss))
            print("\tValidation Accuracy: {}".format(vAccuracy))
            if i < epochs:
                X_shuff, Y_shuff = shuffle_data(X_train, Y_train)
                for batch in range(batches):
                    feed = {x: X_shuff[batch_size*batch:batch_size*(batch+1)],
                            y: Y_shuff[batch_size*batch:batch_size*(batch+1)]}
                    sess.run((train_op), feed_dict=feed)
                    if (batch + 1) % 100 == 0 and batch != 0:
                        batch_loss = loss.eval(feed)
                        batch_acc = accuracy.eval(feed)
                        print("\tStep {}:".format(batch + 1))
                        print("\t\tCost: {}".format(batch_loss))
                        print("\t\tAccuracy: {}".format(batch_acc))
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
