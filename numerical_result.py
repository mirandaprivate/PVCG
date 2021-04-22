import tensorflow as tf
import math
import pathlib

from construct_graph import construct_graph, alpha, n_players, construct_h, construct_g
from compute_tau import compute_tau
from compute_acceptance import compute_acceptance
from compute_social_surplus import compute_social_surplus, compute_social_surplus_except_i

from train import save_path, compute_feed_dict

n_column = 4
n_row = n_players

THIS_DIR = pathlib.Path(__file__).parent

test_effect_contrib = [[0.5 + 0.5 * i] * n_row for i in range(n_column)]

# print payment when effect_contrib is fixed
test_effect_contrib = [[1.0 + 1.0 * i] * n_row for i in range(4)]

test_cost_type = [[0.1 + 0.1 * i for i in range(n_row)]
                  for _ in range(n_column)]

# #print payment when cost_type is fixed

# test_effect_contrib = [[0.5 + 0.5 * i for i in range(n_row)] for _ in range(n_column)]

# #test_cost_type = [[0.1 + 0.1 * i] * n_row for i in range(n_column)]
# test_cost_type = [[0.2 + 0.2 * i] * n_row for i in range(4)]

with tf.compat.v1.Session() as sess:

    # set up the graph environment

    loss = construct_graph()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, save_path)
    graph = tf.compat.v1.get_default_graph()

    effect_contrib = graph.get_tensor_by_name('effect_contrib:0')
    cost_type = graph.get_tensor_by_name('cost_type:0')
    tau = graph.get_tensor_by_name('tau:0')
    accepted_contrib = graph.get_tensor_by_name('accepted_contrib:0')
    social_surplus = graph.get_tensor_by_name('social_surplus:0')
    social_surplus_except_i = graph.get_tensor_by_name(
        'social_surplus_except_i:0'
    )

    loss = graph.get_tensor_by_name('loss:0')
    payment = graph.get_tensor_by_name('payment:0')

    #compute for each column
    result_h = []
    result_g = []
    result_p = []
    result_tau = []
    result_acceptance = []

    for l in range(n_column):
        acc = compute_acceptance(test_effect_contrib[l], test_cost_type[l], alpha=alpha)
        result_acceptance.append(acc)

    for l in range(n_column):

        feed_dict = compute_feed_dict(
            test_effect_contrib[l], test_cost_type[l], effect_contrib,
            cost_type, tau, accepted_contrib, social_surplus,
            social_surplus_except_i
        )

        h_value_list = []
        g_value_list = []
        tau_value_list = []

        for i in range(n_row):
            h = graph.get_tensor_by_name('h_' + str(i) + '/h_' + str(i) + ':0')
            g = graph.get_tensor_by_name('g_' + str(i) + '/g_' + str(i) + ':0')
            h_value = sess.run(h, feed_dict=feed_dict)
            g_value = sess.run(g, feed_dict=feed_dict)
            h_value_list.append(h_value[0][0])
            g_value_list.append(g_value[0][0])

        result_h.append(h_value_list)
        result_g.append(g_value_list)

        result_tau.append(feed_dict[tau])

    #if we require pi >= 0, use the following code
    #result_pi = [ [max(0, result_h[j][i] + result_g[j][i] ) for i in range(n_row)] for j in range(n_column)]
    result_pi = [[result_h[j][i] + result_g[j][i] for i in range(n_row)]
                 for j in range(n_column)]
    result_p = [[result_tau[j][i] + result_pi[j][i] for i in range(n_row)]
                for j in range(n_column)]

    with open(THIS_DIR / 'numerical_result.txt', 'a') as f:

        f.write('effect_contrib\n')
        for i in range(n_row):
            f.write(
                ''.join([
                    '%8.4f' % test_effect_contrib[j][i]
                    for j in range(n_column)
                ] + ['\n'])
            )

        f.write('cost_type\n')
        for i in range(n_row):
            f.write(
                ''.
                join(['%8.4f' % test_cost_type[j][i]
                      for j in range(n_column)] + ['\n'])
            )

        f.write('h\n')

        for i in range(n_row):
            f.write(
                ''.join(['%8.4f' % result_h[j][i]
                         for j in range(n_column)] + ['\n'])
            )

        f.write('g\n')

        for i in range(n_row):
            f.write(
                ''.join(['%8.4f' % result_g[j][i]
                         for j in range(n_column)] + ['\n'])
            )

        # f.write('diff_g\n')

        # for i in range(n_row):
        #     f.write(
        #         ''.join([
        #             '%8.4f' % (result_g[j + 1][i] - result_g[j][i])
        #             for j in range(n_column - 1)
        #         ] + ['\n'])
        #     )

        f.write('tau\n')

        for i in range(n_row):
            f.write(
                ''.join(['%8.4f' % result_tau[j][i]
                         for j in range(n_column)] + ['\n'])
            )

        f.write('pi\n')

        for i in range(n_row):
            f.write(
                ''.join(['%8.4f' % result_pi[j][i]
                         for j in range(n_column)] + ['\n'])
            )

        f.write('p\n')

        for i in range(n_row):
            f.write(
                ''.join(['%8.4f' % result_p[j][i]
                         for j in range(n_column)] + ['\n'])
            )
        
        f.write('acc\n')

        for i in range(n_row):
            f.write(''.join(
                ['%8.4f' % result_acceptance[j][i]
                for j in range(n_column)] + ['\n']
            ))