'''
'''
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

THIS_DIR = pathlib.Path(__file__).parent

def targetfunction(array):
    '''Compute the value given a array.

    Args:
        array: A list of floats, representing[x1, x2, x3, x4, x5]

    Returns:
        The value calculated by the targetfuntion.
    '''
    matrix = [[1, 0.1, 0.2, 0.5, 0.2],
              [0.1, 1, 0.4, 0.3, 0.1],
              [0.2, 0.4, 1, 0.4, 0.1],
              [0.5, 0.3, 0.4, 1, 0.1],
              [0.2, 0.1, 0.1, 0.1, 1]]
    value = matrix @ array @ array
    value = max(0, value)
    return value

def add_layer(inputs, in_size, out_size, activation=None):
    '''Add layer to the neural networks.
    
    Args:
        inputs: Input for each neuron in this layer.
        in_size: 
    '''
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation is None:
        outputs = wx_plus_b
    else:
        outputs = activation(wx_plus_b)
    return outputs, weights, biases

def main():
    ''''''
    # Read the training data from 'x_train.txt'.
    e = []
    with open(THIS_DIR/ 'x_train.txt', 'r') as f:
        c = []
        train_data = np.loadtxt(f)
        for i in range(0, 50000, 5):
            b = [train_data[i + l] for l in range(5)]
            d = np.array(b)
            e.append(b)
            result = targetfunction(d)
            c.append(result)
        y_train = np.array(c)
    x_train = np.array(e)
    x_train = np.reshape(x_train, (10000, 5))
    y_train = np.reshape(y_train, (10000, 1))

    xs = tf.placeholder(tf.float32, [None, 5])
    ys = tf.placeholder(tf.float32, [None, 1])

    layer1, weight1, biases1 = add_layer(xs, 5, 10, activation=tf.nn.relu)
    layer2, weight2, biases2 = add_layer(layer1, 10, 10, activation=tf.nn.relu)
    layer3, weight3, biases3 = add_layer(layer2, 10, 10, activation=tf.nn.relu)
    layer4, weight4, biases4 = add_layer(layer3, 10, 10, activation=tf.nn.relu)
    layer5, weight5, biases5 = add_layer(layer4, 10, 10, activation=tf.nn.relu)
    layer6, weight6, biases6 = add_layer(layer5, 10, 10, activation=tf.nn.relu)
    layer7, weight7, biases7 = add_layer(layer6, 10, 10, activation=tf.nn.relu)
    layer8, weight8, biases8 = add_layer(layer7, 10, 5, activation=tf.nn.relu)
    prediction, weight9, biases9 = add_layer(layer8, 5, 1, activation=None)

    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(ys - prediction), reduction_indices=[1]))


    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    loss_list = []
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_train, ys: y_train})

        f_loss = sess.run(loss, feed_dict={xs: x_train, ys: y_train})
        loss_list.append(f_loss)

    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Time in epochs")
    plt.ylabel("loss")

    c = []
    e = []
    with open(THIS_DIR/ 'x_test.txt', 'r') as f:
        a = np.loadtxt(f)
        for i in range(0, 100, 5):
            b = [a[i + l] for l in range(5)]
            d = np.array(b)
            e.append(b)
            result = targetfunction(d)
            c.append(result)
        y = np.array(c)

    x_test = np.array(e)
    x_test = np.reshape(x_test, (20, 5))
    y_test = np.reshape(y, (20, 1))
    test_prediction = sess.run(prediction, feed_dict={xs: x_test
    , ys: y_test})
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_line = [x_test[i][0] for i in range(20)]
    y_line = [x_test[i][1] for i in range(20)]
    z_line = np.array(test_prediction.reshape(1, 20)[0])
    y_test = np.reshape(y_test, (1, 20))
    ax.scatter3D(x_line, y_line, z_line, color='b')
    ax.scatter3D(x_line, y_line, y_test, color='r')
    plt.title("blue:prediction red:actual")
    plt.show()


    x1 = np.arange(0, 1, 0.01)
    x2 = np.arange(0, 1, 0.01)
    x1, x2 = np.meshgrid(x1, x2)
    # test_prediction = sess.run(prediction, feed_dict={xs: x1, ys: x2})
    z = x1**2 + x2**2 + 0.2*x1*x2
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, z, rstride=1, cstride= 1,cmap=cm.viridis)
    plt.title("targetfunction")

    plt.show()


if __name__ == "__main__":
    main()
