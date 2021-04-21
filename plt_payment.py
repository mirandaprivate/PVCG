import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import math
import pathlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from construct_graph import construct_graph, alpha, n_players, construct_h, m_consumers
from train import min_cost_type, max_cost_type, min_effect, max_effect, min_valuation_type, max_valuation_type
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
                        [0.5*(min_effect + max_effect) for _ in range(9)]
                        for i in range(n_splits_contrib)]
                       for j in range(n_splits_cost)]

test_cost_type = [[[min_cost_type + gap_cost * (j + 1)] +
                   [0.5*(min_cost_type + max_cost_type) for _ in range(9)] for i in range(n_splits_contrib)]
                  for j in range(n_splits_cost)]
test_valuation_type = [0.5*(min_valuation_type + max_valuation_type)]*m_consumers

with tf.compat.v1.Session() as sess:

    # set up the graph environment

    loss = construct_graph()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, save_path)
    graph = tf.compat.v1.get_default_graph()

    effect_contrib = graph.get_tensor_by_name('effect_contrib:0')
    cost_type = graph.get_tensor_by_name('cost_type:0')
    valuation_type = graph.get_tensor_by_name('valuation_type:0')
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
    result_p = []
    result_tau = []

    result_p_value = []
    result_adj_value = []
    for j in range(n_splits_cost):

        h_value_list = []
        tau_value_list = []

        for l in range(n_splits_contrib):

            feed_dict = compute_feed_dict(
                test_effect_contrib[j][l], test_cost_type[j][l], test_valuation_type, effect_contrib,
                cost_type, valuation_type, tau, accepted_contrib, social_surplus,
                social_surplus_except_i
            )

            h = graph.get_tensor_by_name('h_' + str(0) + '/h_' + str(0) + ':0')
            h_value = sess.run(h, feed_dict=feed_dict)
            h_value_list.append(h_value[0][0])
            tau_value_list.append(feed_dict[tau][0])

        result_h.append(h_value_list)
        result_tau.append(tau_value_list)

    result_adj = [[
        result_h[j][i]  for i in range(n_splits_contrib)
    ] for j in range(n_splits_cost)]
    result_p = [[
        result_tau[j][i] + result_adj[j][i] for i in range(n_splits_contrib)
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
z_1 = np.reshape(result_adj, (n_splits_cost, n_splits_contrib))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z)
ax.set_xlabel('reported cost type')
ax.set_ylabel('reported capacity limit')
ax.set_zlabel('payment')
ax.set_xlim((0., 1.))
ax.set_zlim((0., 5.))
ax.set_zlim((0., 2.5))
fake2Dline = mpl.lines.Line2D([0], [0], linestyle='none', c='b', marker='o')
plt.title('PVCG Payment v.s. \n Reported Capacity Limit & Reported Cost Type\n')
plt.show()
