import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import math
import pathlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# sharply
from itertools import combinations
import bisect

import pandas as pd
import itertools

import random


from construct_graph import construct_graph, alpha, n_players, construct_h, construct_g
from train import min_cost_type, max_cost_type, min_effect, max_effect
from compute_tau import compute_tau
from compute_acceptance import compute_acceptance
from compute_social_surplus import compute_social_surplus, compute_social_surplus_except_i

from train import save_path, compute_feed_dict

# set up the graph environment
sess = tf.compat.v1.Session()

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

def compute_payment(values_qk, values_v):

    effect_contrib_value = values_qk
    cost_type_value = values_v

    tau_value = compute_tau(effect_contrib_value, cost_type_value, alpha=alpha)
    acceptance_value = compute_acceptance(
        values_qk, values_v, alpha=alpha
    )
    accepted_contrib_value = [
        effect_contrib_value[i] * acceptance_value[i] for i in range(n_players)
    ]
    social_surplus_value = compute_social_surplus(
        effect_contrib_value, cost_type_value, alpha=alpha
    )
    social_surplus_except_i_value = [
        compute_social_surplus_except_i(
            effect_contrib_value, cost_type_value, i, alpha=alpha
        ) for i in range(n_players)
    ]

    feed_dict = {
        effect_contrib: effect_contrib_value,
        cost_type: cost_type_value,
        tau: tau_value,
        accepted_contrib: accepted_contrib_value,
        social_surplus: social_surplus_value,
        social_surplus_except_i: social_surplus_except_i_value
    }
    
    p_list = []

    for i in range(n_players): 
        h = graph.get_tensor_by_name('h_' + str(i) + '/h_' + str(i) + ':0')
        g = graph.get_tensor_by_name('g_' + str(i) + '/g_' + str(i) + ':0')
        h_value = sess.run(h, feed_dict=feed_dict)
        g_value = sess.run(g, feed_dict=feed_dict)
        p_list.append(tau_value[i] + h_value[0][0] + g_value[0][0])
    
    return p_list, acceptance_value

def revenue_f(values_q, values_k, values_a):
    q = np.asarray(values_q)
    k = np.asarray(values_k)
    a = np.asarray(values_a)

    revenue = np.sqrt(10) * np.sqrt(np.sum(q * k * a))
    return revenue

def total_cost(cost):
    c = np.asarray(cost)

    cost = np.sum(c)
    return cost

def main_f():
    data_train = pd.read_csv('x_train_10.txt')
    data_train = pd.DataFrame(data_train)
    data_train['g'].astype(int)

    grouped = data_train.groupby('g')
    
    count_group = 0

    for name, group in grouped:
        count_group = count_group + 1
        print("Group:", count_group)
        q_list, v_list, k_list = [], [], []
        p_list, c_list = [], []

        for j in range(len(group)):   # i is 'g q v

            q_value = group['q'].iloc[j]
            v_value = group['v'].iloc[j]

            q_list.append(q_value)
            v_list.append(v_value)

            N = j+1
            min_k_old = 0
        
        random.seed(0)
        v_list_report =  [random.uniform(min_cost_type, max_cost_type) for i in range(n_players)]
        #v_list_report = [v for v in v_list]

        k_list = [0.1] * 10
        qk_list = [q_list[i]*k_list[i] for i in range(n_players)]
        
        # judgement
        count = 0
        fl_is_ok = True

        while fl_is_ok == True and count < 1000: 
            
            count = count+1
            qk_list = [q_list[i] * k_list[i] for i in range(n_players)]
            p_list, a_list = compute_payment(qk_list,v_list_report)
            v_list_report_old = [v for v in v_list_report]
            c_list = [v_list[i] * qk_list[i] * a_list[i] for i in range(n_players)]
            k_list_old = [k for k in k_list]
            fl_is_ok = False

            for i in range(n_players):

                # each player first adjust the reported cost type    
                v_count = 0
                v_update = True    
                while v_update == True and v_count<10:
                    v_update == False
                    v_count = v_count + 1
                    v_list_report_change = [v for v in v_list_report_old]
                    v_list_report_change[i] = (v_list_report_old[i]+v_list[i])/2
                    p_list_change, a_list_change = compute_payment(qk_list, v_list_report_change)
                    c_list_change = [v_list[i]*qk_list[i]*a_list_change[i] for i in range(n_players)]
                    if p_list_change[i] - c_list_change[i] >= p_list[i] - c_list[i]:
                        v_list_report_old[i] = v_list_report[i]
                        v_list_report[i] = v_list_report_change[i]
                        #print("Change V")
                        v_update = True

                # then, each player decides whether to contribute more data or withdraw data
                p_list, a_list = compute_payment(qk_list,v_list_report)
                c_list = [v_list[i] * qk_list[i] * a_list[i] for i in range(n_players)]

                if p_list[i] - c_list[i] < 0:
                    k_list[i] = 0
                    fl_is_ok = True
                
                if k_list_old[i] <= 0.9:
                    k_list_increase = [k for k in k_list_old]
                    k_list_increase[i] = k_list_old[i] + 0.1
                    qk_list_increase = [q_list[i] * k_list_increase[i] for i in range(n_players)]
                    p_list_increase, a_list_increase = compute_payment(qk_list_increase, v_list_report)
                    c_list_increase = [v_list[i]*qk_list_increase[i]*a_list_increase[i] for i in range(n_players)] 
                    if p_list_increase[i] - c_list_increase[i] >= p_list[i] - c_list[i]:
                        k_list[i] = k_list_old[i] + 0.1
                        fl_is_ok = True

                if k_list_old[i] >= 0.1:
                    k_list_decrease = [k for k in k_list_old]
                    k_list_decrease[i] = k_list_decrease[i] - 0.1
                    qk_list_decrease = [q_list[i]*k_list_decrease[i] for i in range(n_players)]
                    p_list_decrease, a_list_decrease = compute_payment(qk_list_decrease, v_list_report)
                    c_list_decrease = [v_list[i]*qk_list_decrease[i]*a_list_decrease[i] for i in range(n_players)]
                   
                    if p_list_decrease[i] - c_list_decrease[i] >= p_list[i] - c_list[i]:
                        k_list[i] = k_list_old[i] - 0.1
                        fl_is_ok = True

                
            #print("ROUND: ", count)
            #print("k: ", k_list)
            #print("a: ", a_list)
            #print("v: ", v_list)
            #print("p: ", p_list)
            #print("v: ", v_list_report)


            qk_list = [q_list[i] * k_list[i]  for i in range(n_players)]
            c_list = [qk_list[i] * v_list[i] * a_list[i] for i in range(n_players) ]
            ka_list = [k_list[i] * a_list[i] for i in range(n_players)]
            u_list = [p_list[i] - c_list[i] for i in range(n_players)]
            # R - C
            C_ = total_cost(c_list)
            R_ = revenue_f(q_list, k_list, a_list)
            surplus = R_ - C_
            budget = R_ - np.sum(np.asarray(p_list))

        #print("s: ", surplus)


        #print("s: ", surplus)
        print("k: ", k_list)
        #print("ka: ", ka_list)
        #print("budget: ", budget)
        #print("utility: ", u_list)

        summary_s = pd.DataFrame([surplus])
        summary_s.to_csv('output/fvcg_summary_s.csv', mode='a', header=False)
    
        summary_b = pd.DataFrame([budget])
        summary_b.to_csv('output/fvcg_summary_b.csv', mode='a', header=False)

        summary_r = pd.DataFrame([R_])
        summary_r.to_csv('output/fvcg_summary_r.csv', mode='a', header=False)

        summary_k = pd.DataFrame([k_list])
        summary_k.to_csv('output/fvcg_summary_k.csv', mode='a', header=False) 

        summary_p = pd.DataFrame([p_list])
        summary_p.to_csv('output/fvcg_summary_p.csv', mode='a', header=False)

        summary_a = pd.DataFrame([a_list])
        summary_a.to_csv('output/fvcg_summary_a.csv', mode='a', header=False)

        summary_u = pd.DataFrame([u_list])
        summary_u.to_csv('output/fvcg_summary_u.csv', mode='a', header=False)



if __name__ == "__main__":

    # test main f function 
    
    main_f()
