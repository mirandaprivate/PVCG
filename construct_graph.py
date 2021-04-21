import tensorflow as tf
import math

n_players = 10
m_consumers = 10

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


def construct_h(effect_contrib, cost_type, valuation_type, i):
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
        tf.concat([effect_contrib_except_i, cost_type_except_i, valuation_type], 0),
        (1, 2 * n_players - 2 + m_consumers)
    )

    layer1 = layer_sigmoid(
        input_except_i, [2 * n_players - 2 + m_consumers, 10], [10],
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


def construct_adjustment(effect_contrib, cost_type, valuation_type, name='adj'):
    adj_list = []
    for i in range(n_players):
        adj_i_tensor = construct_h(effect_contrib, cost_type, valuation_type, i)
        adj_i = tf.reshape(adj_i_tensor, (1, ))
        adj_list.append(adj_i)
    return tf.concat(adj_list, 0, name=name)


def construct_graph():

    #initiate placeholders
    effect_contrib = tf.placeholder(
        tf.float32, shape=[n_players], name='effect_contrib'
    )
    cost_type = tf.placeholder(tf.float32, shape=[n_players], name='cost_type')
    valuation_type = tf.placeholder(tf.float32, shape=[m_consumers], name='valuation_type')
   
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

    adj = construct_adjustment(effect_contrib, cost_type, valuation_type)
    payment = tf.math.add(tau, adj, name='payment')

    print(social_surplus_tile)

    loss_2 = tf.reduce_sum(
        tf.nn.relu(
            -tf.math.add_n([adj, social_surplus_tile, -social_surplus_except_i])
        ),
        name='loss_2'
    )
    print(loss_2)

    loss_1 = tf.nn.relu(
        tf.reduce_sum(
            tf.math.add_n([adj, social_surplus_tile, -social_surplus_except_i])
        ) - social_surplus,
        name='loss_1'
    )
    print(loss_1)

    loss = tf.math.add_n([
        loss_2, loss_1
    ],
        name='loss')
    print(loss)

    return (loss)
