######################################################
# Multi-Layer Perceptron Classifier for MNIST dataset
# Mark Harvey
# Dec 2018
######################################################
import tensorflow as tf


#####################################################
# Dataset preparation
#####################################################
# download of dataset, will only run if doesn't already exist in disk
mnist_dataset = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist_dataset

# flatten the images
x_train = x_train.reshape(len(x_train), 784)
x_test = x_test.reshape(len(x_test), 784)

# The image pixels are 8bit integers (uint8)
# scale them from range 0:255 to range 0:1
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


###############################################
# Hyperparameters
###############################################
BATCHSIZE=50
LEARNRATE=0.0001
STEPS=int(len(x_train) / BATCHSIZE)


#####################################################
# Create the Computational graph
#####################################################


# define placeholders for the input data & labels
x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32', [None,10])


# dense, fully-connected layer of 196 nodes, reLu activation
input_layer = tf.layers.dense(inputs=x, units=196, activation=tf.nn.relu)
# dense, fully-connected layer of 10 nodes, softmax activation
logits = tf.layers.dense(inputs=input_layer, units=10, activation=None)
prediction = tf.nn.softmax(logits)


# Define a cross entropy loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

# Define the optimizer function
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNRATE).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# variable initialization
init = tf.initializers.global_variables()


#####################################################
# Create & run the graph in a Session
#####################################################
with tf.Session() as sess:

    sess.run(init)

    # Training cycle with training data
    for i in range(STEPS):
           
            # fetch a batch from training dataset
            batch_x, batch_y = x_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE], y_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE]
            
            # calculate training accuracy & display it every 100 steps
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            if i % 100 == 0:
                print ("Train Step:", i, ' Training Accuracy: ', train_accuracy)

            # Run graph for optimization - i.e. do the training
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    print("Training Finished!")

    # Evaluation with test data
    print ("Accuracy of trained network with test data:", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
