import pandas as pd
import re
import sys
import time
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.preprocessing.sequence import pad_sequences

import json
from ast import literal_eval

STROKE_COUNT = 196

"""
This model's architecture is as follows: product descriptions' bag of words encodings go into
a 2-layer FC network. Output is a softmax vector of length 12, corresponding to 12 distinct
price buckets. Baseline model.
"""

def main():
    # Usage is as follows: python model.py <train_enc>.csv <dev_enc>.csv

    X_test = []
    Y_test = []
    devCSV = None
    trainCSV = None

    if (len(sys.argv) >=2):
        trainCSV = sys.argv[1]
    if (len(sys.argv) >= 3):
        devCSV = sys.argv[2]
    trainDF = pd.read_csv(trainCSV, header = 0)
    trainDF = trainDF.values
    X_train = trainDF[:, 0:-5]
    Y_train = trainDF[:, -5:].T
    #X_train, Y_train = StrokesAndLabels(trainDF)
    devDF = pd.read_csv(devCSV, header = 0)
    devDF = devDF.values
    X_dev = devDF[:, 0:-5]
    Y_dev = devDF[:, -5:].T
    #X_dev, Y_dev = StrokesAndLabels(devDF)
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))
    parameters = model(X_train, Y_train, X_dev, Y_dev, num_epochs = 5)

def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector of shape (numbuckets, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- so that everyone in our group will get same minibatches permutations
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = len(X)                 # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def StrokesAndLabels(data):
    """
    Function looks at the dataframe and returns matrix of description
    condensed encodings for all samples (each sample has variable length vectors
        of indices corresponding to vocab indices of words that appear in each
        description)
    Arguments:
    df -- dataframe of data
    Returns:
    X -- product descriptions
    Y -- price of item placed in pre-determined buckets
    """
    print('dim of data: ', data.shape)
    data = data.values
    m, n = data.shape
    #print('data: ', data)

    # Key: col 1: airplane, col 2: campfire, col 3: key, col 4: moon, col 5: palm tree
    #print('data: ', data)

    labels = np.zeros((m, 5))
    for row in range(m):
        #print('data: ', data[row, 0])
        if data[row, 2] == 'airplane':
            labels[row, 0] = 1
        elif data[row, 2] == 'campfire':
            labels[row, 1] = 1
        elif data[row, 2] == 'key':
            labels[row, 2] = 1
        elif data[row, 2] == 'moon':
            labels[row, 3] = 1
        else:
            labels[row, 4] = 1
    Y = labels
    print('labels: ', labels)

    print('X row: ', data[0, 0])
    temp = []
    for i in range(data.shape[0]):
        temp.append(stack_it(data[i, 0]))
    X = np.asarray(temp)

    #Y= np.array(Y)
    X = np.asarray(X)
    print('X: ', X)
    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)
    return X, Y

def stack_it(raw_strokes):
    """preprocess the string and make
    a standard Nx3 stroke vector"""

    stroke_vec = literal_eval(raw_strokes)# string->list
    #print('stroke_vec: ', stroke_vec)
    # unwrap the list
    in_strokes = [(xi,yi,i)
    for i,(x,y) in enumerate(stroke_vec)
    for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1),
                         maxlen=STROKE_COUNT,
                         padding='post').swapaxes(0, 1)


def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 128, print_cost = True):
    """
    Implements a two-layer tensorflow neural network: LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = vocab_length, number of training examples = 1452885)
    Y_train -- test set, of shape (output size = 12, number of training examples = 1452885)
    X_test -- training set, of shape (input size = vocab_length, number of test examples)
    Y_test -- test set, of shape (output size = 12, number of test examples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) =  X_train.shape[1], len(X_train)                        # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_x, m, n_y)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z2 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z2, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                # Turns index array of variable length to one-hot array of length vocab-length
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X.T, Y:minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)


        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        # Add print operation
        Y = tf.Print(Y, [Y], message="This is Y: ")
        Z2 = tf.Print(Z2, [Z2], message="this is Z2")
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

        # Calculate accuracy on the train set
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        overallRight = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            overallRight = overallRight + accuracy.eval({X: minibatch_X, Y: minibatch_Y})
        print ('Train accuracy = ', str(overallRight/m))
        # Calculate accuracy on the dev set
        minibatches = random_mini_batches(X_dev, Y_dev, minibatch_size, seed)
        overallRight = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            overallRight = overallRight + accuracy.eval({X: minibatch_X, Y: minibatch_Y})
        print ('Dev accuracy = ', str(overallRight/len(X_dev)))
        # Writing out trained parameters onto an uotput file called parameters.json
        for val in parameters:
            parameters[val] = parameters[val].tolist()
        fileout = open('parameters.json', 'w')
        json.dump(parameters, fileout)
        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()
        return parameters

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_x -- scalar, size of vocab
    n_y -- scalar, number of classes
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    - Use None because it let's us be flexible on the number of examples you will for the placeholders
    """
    X = tf.placeholder('float32', [n_x, None], name = "X")
    Y = tf.placeholder('float32', [n_y, None], name = "Y")

    return X, Y

def initialize_parameters(n_x, m, n_y):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, n_x]
                        b1 : [25, 1]
                        W2 : [n_y, 25]
                        b2 : [n_y, 1]
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [25,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer= tf.constant_initializer(0.0))
    W2 = tf.get_variable("W2", [n_y, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [n_y, 1], initializer = tf.constant_initializer(0.0))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2"
                  the shapes are given in initialize_parameters
    Returns:
    Z2 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2

    return Z2


def compute_cost(Z2, Y):
    """
    Computes the cost
    Arguments:
    Z2 -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as Z2
    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

if __name__ == "__main__":
    main()
