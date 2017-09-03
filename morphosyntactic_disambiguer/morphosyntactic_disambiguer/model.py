import tensorflow as tf
import pandas as pd
import numpy as np
import progressbar
from common import *

### MODEL CREATION AND TEACHING ###
def create_model():
    print("[create model] Create model")
    tf.reset_default_graph()
    accu = []
    with open('./in-out_correct.csv') as f:
        columns = len(f.readline().split(','))
    print("COLUMNSSS: {0}".format(columns))
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
    epochs = 30

    bar = progressbar.ProgressBar(max_value=epochs)

    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(epochs):
            print("Step: {0}".format(i))
            corr_chunks = pd.read_csv('./in-out_correct.csv', chunksize=10000, delimiter=',')
            non_corr_chunks = pd.read_csv('./in-out_non-correct.csv', chunksize=10000, delimiter=',')
            for chunk_corr, chunk_non_corr in zip(corr_chunks, non_corr_chunks):
#                print("CHUNK CORR {0}".format(np.shape(chunk_corr)))
#                print("CHUNK NON-CORR {0}".format(np.shape(chunk_non_corr)))

                batch_y = pd.concat([pd.DataFrame(np.ones(len(chunk_corr))), pd.DataFrame(np.zeros(len(chunk_non_corr)))]).values
                batch_y_neg = pd.concat([pd.DataFrame(np.zeros(len(chunk_corr))), pd.DataFrame(np.ones(len(chunk_non_corr)))]).values
                batch_y = np.hstack((batch_y, batch_y_neg))

                batch_x = pd.concat([chunk_corr,chunk_non_corr], ignore_index=True,axis=0)
#                print(batch_x.columns)
#                del batch_x[u'103']

#                print(batch_x.head())

                batch_x = batch_x.values
#                print("BATCH SIZE: {0}".format(np.shape(batch_x)))
                #shuffle
                combined = list(zip(batch_x, batch_y))
                np.random.shuffle(combined)

                batch_x[:], batch_y[:] = zip(*combined)
            
#                print("BATCH SIZE: {0}".format(np.shape(batch_x)))


                train_step.run(feed_dict={x: batch_x, y_train: batch_y})

#                print("Step accuracy.")
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_x, y_train: batch_y})
                print('step {0}, train accuracy {1:.10}'.format(i, train_accuracy))
            bar.update(i)

        test_corr = pd.read_csv('in-out_correct_test.csv',chunksize=chunksize, delimiter=',')
        test_non_corr = pd.read_csv('in-out_non-correct_test.csv', chunksize=chunksize, delimiter=',')

        output = []
        for chunk_corr_tst, chunk_non_corr_tst in zip(test_corr, test_non_corr):
            test_y = pd.concat([pd.DataFrame(np.ones(len(chunk_corr_tst))), pd.DataFrame(np.zeros(len(chunk_non_corr_tst)))]).values
            test_y_neg = pd.concat([pd.DataFrame(np.zeros(len(chunk_corr_tst))), pd.DataFrame(np.ones(len(chunk_non_corr_tst)))]).values

            test_y = np.hstack((test_y, test_y_neg))

            tmp_1 = chunk_corr_tst.values
            tmp_2 = chunk_non_corr_tst.values
            test_x = np.vstack((tmp_1, tmp_2))
            accu.append(accuracy.eval(feed_dict={x: test_x, y_train: test_y}))
            print('test accuracy {0:.10f}'.format(accu[-1]))
            output.append(h_2.eval(feed_dict={x: test_x}))

        df = pd.DataFrame(output)
        df.to_csv('output_test')
        # Save the variables to disk.
        save_path = saver.save(sess, "./model.ckpt")
        print("Model saved in file: %s" % save_path)


