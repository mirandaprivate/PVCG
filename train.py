import tensorflow as tf
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import pathlib
from construct_graph import construct_graph, n_players, m_consumers, alpha
# import psutil

from compute_tau import compute_tau
from compute_acceptance import compute_acceptance, compute_acc
from compute_social_surplus import compute_social_surplus, compute_social_surplus_except_i

min_cost_type = 0.
max_cost_type = 1.
min_valuation_type = 0.
max_valuation_type = 0.2
min_effect = 0.
max_effect = 5.
batch_size = 100
learning_rate = 0.001
n_training_loop = 100
bias_increment = 0

THIS_DIR = pathlib.Path(__file__).parent

save_path = './model/model.ckpt'
save_path_initialize = './model_initialize/model_initialize.ckpt'


def compute_feed_dict(
    effect_contrib_value, cost_type_value, valuation_type_value, effect_contrib, cost_type, valuation_type, tau,
    accepted_contrib, social_surplus, social_surplus_except_i
):

    tau_value = compute_tau(effect_contrib_value, cost_type_value, valuation_type_value, alpha=alpha)
    acceptance_value = compute_acceptance(
        effect_contrib_value, cost_type_value, valuation_type_value, alpha=alpha
    )
    accepted_contrib_value = [
        effect_contrib_value[i] * acceptance_value[i] for i in range(n_players)
    ]
    social_surplus_value = compute_social_surplus(
        effect_contrib_value, cost_type_value, valuation_type_value, alpha=alpha
    )
    social_surplus_except_i_value = [
        compute_social_surplus_except_i(
            effect_contrib_value, cost_type_value, valuation_type_value, i, alpha=alpha
        ) for i in range(n_players)
    ]

    feed_dict = {
        effect_contrib: effect_contrib_value,
        cost_type: cost_type_value,
        valuation_type: valuation_type_value,
        tau: tau_value,
        accepted_contrib: accepted_contrib_value,
        social_surplus: social_surplus_value,
        social_surplus_except_i: social_surplus_except_i_value
    }

    return feed_dict


def generate_feed_dict(seed=0):
    random.seed(seed)
    effect_contrib_value = [
        random.uniform(min_effect, max_effect) for i in range(n_players)
    ]
    cost_type_value = [
        random.uniform(min_cost_type, max_cost_type) for i in range(n_players)
    ]
    valuation_type_value = [
        random.uniform(min_valuation_type, max_valuation_type) for i in range(m_consumers)
    ]

    return compute_feed_dict(
        effect_contrib_value, cost_type_value, valuation_type_value, effect_contrib, cost_type, valuation_type, tau,
        accepted_contrib, social_surplus, social_surplus_except_i
    )


def main():
    ### create the graph and set up training environment
    loss = construct_graph()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #saver.save(sess, save_path_initialize)
        #return
        saver.restore(sess, save_path_initialize)
        #saver.restore(sess,save_path)
        saver.save(sess, save_path)

    graph = tf.compat.v1.get_default_graph()

    global effect_contrib, cost_type, valuation_type, tau, accepted_contrib
    global social_surplus, social_surplus_except_i

    effect_contrib = graph.get_tensor_by_name('effect_contrib:0')
    cost_type = graph.get_tensor_by_name('cost_type:0')
    valuation_type = graph.get_tensor_by_name('valuation_type:0')
    tau = graph.get_tensor_by_name('tau:0')
    accepted_contrib = graph.get_tensor_by_name('accepted_contrib:0')
    social_surplus = graph.get_tensor_by_name('social_surplus:0')
    social_surplus_except_i = graph.get_tensor_by_name(
        'social_surplus_except_i:0'
    )

    bias = [
        graph.get_tensor_by_name('h_layer4' + '/biases:0')
        for i in range(n_players)
    ]
    payment = graph.get_tensor_by_name('payment:0')

    loss_2 = graph.get_tensor_by_name('loss_2:0')
    loss_1 = graph.get_tensor_by_name('loss_1:0')

    variables = tf.trainable_variables()
    delta = tf.gradients(loss, variables, name='gradients')

    var_length = len(variables)
    total_loss_list = []
    total_loss_2_list = []
    total_loss_1_list = []
 
    for loop in range(n_training_loop):
        # tf.reset_default_graph()
        ###generate data for each batch
        updates = []
        delta_value_list = []
        loss_value_list = []
        loss_2_value_list = []
        loss_1_value_list = []

        inv_batch_size = 1. / batch_size

        with tf.compat.v1.Session() as sess:

            saver.restore(sess, save_path)

            flag_negative_payment =0

            for t in range(batch_size):
                feed_dict = generate_feed_dict(seed=loop*0.1+t*0.01)

                delta_value = sess.run(delta, feed_dict=feed_dict)
                delta_value_list.append(delta_value)

                loss_value = sess.run(loss, feed_dict=feed_dict)
                loss_value_list.append(loss_value)
                total_loss_list.append(loss_value)

                loss_2_value = sess.run(loss_2, feed_dict=feed_dict)
                loss_2_value_list.append(loss_2_value)
                total_loss_2_list.append(loss_2_value)

                loss_1_value = sess.run(loss_1, feed_dict=feed_dict)
                loss_1_value_list.append(loss_1_value)
                total_loss_1_list.append(loss_1_value)

                payment_value = sess.run(payment, feed_dict=feed_dict)
                
                for i in range(n_players):
                    if payment_value[i] < 0:
                        flag_negative_payment = 1

            print(
                'Loss 2: ',
                np.mean(loss_2_value_list), ', Loss 1: ',
                np.mean(loss_1_value_list), ', Loss:', np.mean(loss_value_list)
            )

            # write loss1, loss2 and total loss to 'loss.txt'
            with open(THIS_DIR / 'loss.txt', 'a') as f:

                f.write(
                    '%10.5f' % (np.mean(loss_2_value_list)) + '   ' +
                    '%10.5f' % (np.mean(loss_1_value_list)) + '   ' +
                    '%10.5f' % np.mean(loss_value_list) + '\n'
                )

            delta_value_list_reshape = [[
                delta_value_list[t][i] for t in range(batch_size)
            ] for i in range(var_length)]

            # calculate the update for each parameter
            for i in range(var_length):
                update = inv_batch_size * sum(delta_value_list_reshape[i])
                updates.append(update)

            # update each parameter
            for i in range(len(variables)):
                var_name = variables[i].name

                if 'g_' in var_name and 'weight' in var_name:
                    sess.run(
                        tf.assign(
                            variables[i],
                            tf.nn.
                            relu(variables[i] - learning_rate * updates[i])
                        )
                    )
                else:
                    sess.run(
                        tf.assign(
                            variables[i],
                            variables[i] - learning_rate * updates[i]
                        )
                    )

            saver.save(sess, save_path)

            print(
                'Loop ', loop, ' Loss ', sess.run(loss, feed_dict=feed_dict),
                datetime.datetime.now().strftime('%H:%M:%S')
            )

    # plot the loss curve
    x = [i for i in range(len(total_loss_list))]
    plt.plot(x, total_loss_list)
    plt.plot(x, total_loss_2_list, 'b')
    plt.plot(x, total_loss_1_list, 'y')
    plt.show()


if __name__ == '__main__':
    main()
