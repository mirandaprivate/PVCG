# pylint: disable=too-many-locals
''' This function is to compute acceptance.

    The acceptance is a list to weight effective_contrib and cost type.

'''
import numpy as np
import math
import tensorflow as tf


def compute_acceptance(effective_contrib, cost_type, valuation_type, alpha=1.0):
    r'''Compute the acceptance vector given the quality and cost type.

    Args:
        effective_contrib: A list of floats, each of whose element represents
            the capacity limit of each agent.
        cost_type: A list of floats, each of whose element represents the
            cost type of each agent.
        valuation_type: A list of floats, each of whose element represents the
            valuation type of each agent.
        alpha: A float.  A hyper-parameter, the linear coefficient in the
            individual valuation function:
                    v_i = \alpha \theta_i \sqrt{\sum{k=0}{n-1}x_k}

    Returns:
        A list of floats between 0 and 1, each representing the recommended
        acceptance ratio of the input resources contributed by the corresponding supplier.
        The acceptance ratio is controlled by the coordinator.  This
        function computes the recommended value based on reported capacity limits
        , cost types and valuation types.
    '''
    alpha =  alpha * sum(valuation_type)
    contrib_len = len(effective_contrib)
    index = [i for i in range(contrib_len)]
    # Packing the input and their subscripts to get the one-to-one correspondence
    # of the input, and then sort it from small to large according to cost_type.
    input_tuples = (zip(index, effective_contrib, cost_type))
    sorted_input_tuples = sorted(input_tuples, key=lambda x: x[2])
    sorted_index, sorted_effect, sorted_cost_type = zip(*sorted_input_tuples)
    sorted_acceptance = []
    for i in range(2, contrib_len + 1):
        boundary1 = alpha / (0.0000001 + 2 * math.sqrt(abs(sum(sorted_effect[:i]))))
        boundary2 = alpha / (0.0000001 + 2 * math.sqrt(abs(sum(sorted_effect[:i - 1]))))
        threshold = sorted_cost_type[i - 1]
        if boundary1 <= threshold:
            if threshold <= boundary2:
                # acceptance_k is the value of the kth acceptance,the acceptance
                # value before it is 1.0 and the acceptance value behind it is 0.0.
                # acceptance_k = 0.25 \alpha \alpha \threshold \threshold \
                # sum{i=0}{i-1}q_i \q_i-1
                acceptance_k = (
                    0.25 * alpha * alpha /
                    (threshold * threshold) - sum(sorted_effect[:i - 1])
                ) / sorted_effect[i - 1]
                sorted_acceptance = [1.0] * (i - 1) + [acceptance_k] + [0.0] * \
                    (contrib_len - i)
                break
            else:
                sorted_acceptance = [1.0] * (i - 1) + [0.0] * (contrib_len - i + 1)
                break
        if i == contrib_len:
            sorted_acceptance = contrib_len * [1.0]
    # Restore acceptance to an unsorted state according to the corresponding
    # relationship.
    sorted_acc_tuples = zip(sorted_index, sorted_acceptance)
    resort_tuples = sorted(sorted_acc_tuples, key=lambda x: x[0])
    for idx, dummy in enumerate(zip(*resort_tuples)):
        acceptance = dummy
    return list(acceptance)


def compute_acc(effective_contrib, cost_type, valuation_type, *,  alpha=1.0):
    ''' Compute huge sets of data's acceptance.
    '''
    acc_list = []
    length = len(effective_contrib)
    for dummy in range(length):
        e_c = effective_contrib[dummy]
        c_t = cost_type[dummy]
        result = compute_acceptance(e_c, c_t, valuation_type, alpha=alpha)
        acc_list.append(list(result))
    return acc_list
