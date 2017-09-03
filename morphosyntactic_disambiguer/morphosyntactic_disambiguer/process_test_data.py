import tensorflow as tf
import pandas as pd
import numpy as np
import progressbar
from common import *
import itertools

### MODEL EVALUATION ON TEST DATA ###
def write_tags_input_data(test_file, output_tags):
    merged = list(itertools.chain(*output_tags))
    out = pd.DataFrame(merged)
    out.to_csv("output_tags.csv")


def transform_test_data(test_file):
    tf.reset_default_graph()
    chunksize = 10000
    test = pd.read_csv(test_file,chunksize=chunksize, delimiter=',')
    columns = 0
    with open('in-out_correct.csv') as f:
        columns = len(f.readline().split(','))

    # first layer
    hid_layer = 140
    x = tf.placeholder(tf.float32, shape=[None, columns])
    y_train = tf.placeholder(tf.float32, shape=[None,2])

    W_1 = weight_variable([columns, hid_layer])
    b_1 = bias_variable([hid_layer])

    h_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,W_1),b_1))

    # second layer
    W_2 = weight_variable([hid_layer, 2])
    b_2 = bias_variable([2])

    h_2 =  tf.nn.sigmoid(tf.add(tf.matmul(h_1,W_2),b_2))

    # Train and evaluate the model
    cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=h_2))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(h_2, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    chunksize = 10000
    epochs = 2

    bar = progressbar.ProgressBar(max_value=epochs)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "./model.ckpt")
        print("Model restored.")
        output = []
        #prepare_test_data(test_file)
        test = pd.read_csv('input_test.csv',chunksize=chunksize, delimiter=',')

        for chunk in test:
            del chunk['Unnamed: 0']
            del chunk['0']
            columns = None
            with open('all_columns') as f:
                columns = f.read().split(', ')

            fix_columns( chunk, columns )
            print(chunk.columns)
            del chunk[u'100']
#            del[u'disamb]']
            del chunk[u'[0']
            test_x = chunk.values
            print("Size of test_x for chunk {0}".format(len(test_x)))
            test_out = h_2.eval(feed_dict={x: test_x})
            print("Test output size: {0}".format(len(test_out)))
            output.append(test_out)

        write_tags_input_data(test_file, output)
