import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import math
import pathlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from construct_graph import construct_graph, alpha, n_players, construct_h, construct_g
from train import min_cost_type, max_cost_type, min_effect, max_effect
from compute_tau import compute_tau
from compute_acceptance import compute_acceptance
from compute_social_surplus import compute_social_surplus, compute_social_surplus_except_i

from train import save_path, compute_feed_dict

n_splits_contrib = 200
n_splits_cost = 200
n_row = n_players
gap_contrib = (max_effect - min_effect) / n_splits_contrib
gap_cost = (max_cost_type - min_cost_type) / n_splits_cost

THIS_DIR = pathlib.Path(__file__).parent

test_effect_contrib = [[[min_effect + gap_contrib * (i + 1)] +
                        [2.5 for _ in range(9)]
                        for i in range(n_splits_contrib)]
                       for j in range(n_splits_cost)]

test_cost_type = [[[min_cost_type + gap_cost * (j + 1)] +
                   [0.5 for _ in range(9)] for i in range(n_splits_contrib)]
                  for j in range(n_splits_cost)]

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

    result_p_value = []
    result_pi_value = []
    for j in range(n_splits_cost):

        h_value_list = []
        g_value_list = []
        tau_value_list = []

        for l in range(n_splits_contrib):

            feed_dict = compute_feed_dict(
                test_effect_contrib[j][l], test_cost_type[j][l], effect_contrib,
                cost_type, tau, accepted_contrib, social_surplus,
                social_surplus_except_i
            )

            h = graph.get_tensor_by_name('h_' + str(0) + '/h_' + str(0) + ':0')
            g = graph.get_tensor_by_name('g_' + str(0) + '/g_' + str(0) + ':0')
            h_value = sess.run(h, feed_dict=feed_dict)
            g_value = sess.run(g, feed_dict=feed_dict)
            h_value_list.append(h_value[0][0])
            g_value_list.append(g_value[0][0])
            tau_value_list.append(feed_dict[tau][0])

        result_h.append(h_value_list)
        result_g.append(g_value_list)
        result_tau.append(tau_value_list)

    #print(result_h)
    #print(result_tau)

    #if we require pi >= 0, use the following code
    #result_pi = [ [max(0, result_h[j][i] + result_g[j][i] ) for i in range(n_row)] for j in range(n_column)]
    result_pi = [[
        result_h[j][i] + result_g[j][i] for i in range(n_splits_contrib)
    ] for j in range(n_splits_cost)]
    result_p = [[
        result_tau[j][i] + result_pi[j][i] for i in range(n_splits_contrib)
    ] for j in range(n_splits_cost)]

    print(result_p)

test_cost_type = [[test_cost_type[i][l][0] for l in range(n_splits_contrib)]
                  for i in range(n_splits_cost)]
test_effect_contrib = [[
    test_effect_contrib[i][l][0] for l in range(n_splits_contrib)
] for i in range(n_splits_cost)]

x = np.reshape(test_cost_type, (n_splits_cost, n_splits_contrib))
y = np.reshape(test_effect_contrib, (n_splits_cost, n_splits_contrib))
z = np.reshape(result_p, (n_splits_cost, n_splits_contrib))
zero = np.reshape([0] * n_splits_contrib * n_splits_cost,
                  (n_splits_cost, n_splits_contrib))
z_1 = np.reshape(result_pi, (n_splits_cost, n_splits_contrib))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, zero, color='gray')
ax.plot_surface(x, y, z)
ax.set_xlabel('cost type')
ax.set_ylabel('dataset metrics')
ax.set_zlabel('payment')
ax.set_xlim((0., 1.))
ax.set_zlim((0., 5.))
ax.set_zlim((0., 4.))
fake2Dline = mpl.lines.Line2D([0], [0], linestyle='none', c='b', marker='o')
fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle='none', c='gray', marker='o')
plt.title('PVCG Payment v.s. Dataset Metrics & Cost Type')
ax.legend([fake2Dline, fake2Dline2],
          ['PVCG payment surface', 'zero payment surface'],
          numpoints=1,P
          loc='upper right',
          bbox_to_anchor=(0.4, 0., 0.5, 0.9))
plt.show()
