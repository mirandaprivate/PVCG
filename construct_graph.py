import tensorflow as tf
import math

n_players = 10
lambda_1 = 0.4
lambda_2 = 0.3
lambda_3 = 0.3

alpha = math.sqrt(n_players)


def layer_sigmoid(
    inputs,
    kernel_shape,
    bias_shape,
    activation=None,
    scope='layer',
    name='single_layer',
    simple_mul=False,
    output_scope='layer_output'
):
    with tf.compat.v1.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', kernel_shape)
        print(weights)

    if simple_mul == True:
        with tf.name_scope(output_scope):
            return tf.matmul(inputs, weights, name=name)

    with tf.compat.v1.variable_scope(scope, reuse=tf.AUTO_REUSE):
        biases = tf.get_variable('biases', bias_shape)
        print(biases)

    with tf.name_scope(output_scope):
        if activation == None:
            return tf.math.add(tf.matmul(inputs, weights), biases, name=name)
        if activation != None:
            return tf.sigmoid(tf.matmul(inputs, weights) + biases, name=name)


def construct_h(effect_contrib, cost_type, i):
    effect_contrib_before_i = tf.strided_slice(
        effect_contrib, [
            0,
        ], [
            i,
        ], [
            1,
        ]
    )
    effect_contrib_after_i = tf.strided_slice(
        effect_contrib, [
            i + 1,
        ], [
            n_players,
        ], [
            1,
        ]
    )
    effect_contrib_except_i = tf.concat([
        effect_contrib_before_i,
        effect_contrib_after_i,
    ], 0)

    cost_type_before_i = tf.strided_slice(cost_type, [
        0,
    ], [
        i,
    ], [
        1,
    ])
    cost_type_after_i = tf.strided_slice(
        cost_type, [
            i + 1,
        ], [
            n_players,
        ], [
            1,
        ]
    )
    cost_type_except_i = tf.concat([cost_type_before_i, cost_type_after_i], 0)

    input_except_i = tf.reshape(
        tf.concat([effect_contrib_except_i, cost_type_except_i], 0),
        (1, 2 * n_players - 2)
    )

    layer1 = layer_sigmoid(
        input_except_i, [2 * n_players - 2, 10], [10],
        activation=tf.sigmoid,
        scope='h_layer1',
        name='h_layer1' + str(i)
    )
    layer2 = layer_sigmoid(
        layer1, [10, 10], [10],
        activation=tf.sigmoid,
        scope='h_layer2',
        name='h_layer2_' + str(i)
    )
    layer3 = layer_sigmoid(
        layer2,
        [10, 10],
        [10],
        activation=tf.sigmoid,
        scope='h_layer3',
        name='h_layer3_' + str(i),
    )
    layer4 = layer_sigmoid(
        layer3, [10, 1], [1],
        scope='h_layer4',
        name='h_' + str(i),
        output_scope='h_' + str(i)
    )
    print(layer4)
    return layer4


def construct_g(effect_contrib, i):
    effect_i = tf.reshape(effect_contrib[i], (1, 1))
    layer_1 = layer_sigmoid(
        effect_i, [1, 50], [50],
        activation=tf.sigmoid,
        scope='g_layer1',
        name='g_layer1_' + str(i)
    )
    layer_2 = layer_sigmoid(
        layer_1, [50, 1], [1],
        scope='g_layer2',
        name='g_' + str(i),
        output_scope='g_' + str(i),
        simple_mul=True
    )
    print(layer_2)
    return layer_2


def construct_pi(effect_contrib, cost_type, name='pi'):
    pi_list = []
    for i in range(n_players):
        pi_i_tensor = construct_h(effect_contrib, cost_type, i) + \
        construct_g(effect_contrib, i)
        #here we may require pi to be greater than 0
        #        pi_i = tf.nn.relu(tf.reshape(pi_i_tensor, (1, )))
        pi_i = tf.reshape(pi_i_tensor, (1, ))
        pi_list.append(pi_i)
    return tf.concat(pi_list, 0, name=name)


def construct_graph():

    #initiate placeholders
    effect_contrib = tf.placeholder(
        tf.float32, shape=[n_players], name='effect_contrib'
    )
    cost_type = tf.placeholder(tf.float32, shape=[n_players], name='cost_type')
    tau = tf.placeholder(tf.float32, shape=[n_players], name='tau')
    accepted_contrib = tf.placeholder(
        tf.float32, shape=[n_players], name='accepted_contrib'
    )
    social_surplus = tf.placeholder(tf.float32, shape=(), name='social_surplus')
    social_surplus_except_i = tf.placeholder(
        tf.float32, shape=[n_players], name='social_surplus_except_i'
    )

    social_surplus_tile = tf.tile(
        tf.reshape(social_surplus, (1, )), [n_players]
    )

    pi = construct_pi(effect_contrib, cost_type)
    payment = tf.math.add(tau, pi, name='payment')

    #print(pi)
    #print(effect_contrib)
    #print(cost_type)
    #print(tau)
    #print(social_surplus_except_i)
    print(social_surplus_tile)

    loss_1 = tf.math.reduce_variance(
        effect_contrib / (effect_contrib + payment), name='loss_1'
    )
    print(loss_1)

    loss_2 = tf.reduce_sum(
        tf.nn.relu(
            -tf.math.add_n([pi, social_surplus_tile, -social_surplus_except_i])
        ),
        name='loss_2'
    )
    print(loss_2)

    loss_3 = tf.nn.relu(
        tf.reduce_sum(
            tf.math.add_n([pi, social_surplus_tile, -social_surplus_except_i])
        ) - social_surplus,
        name='loss_3'
    )
    print(loss_3)

    loss = tf.math.add_n([
        lambda_1 * loss_1, lambda_2 * loss_2, lambda_3 * loss_3
    ],
                         name='loss')
    print(loss)

    return (loss)
