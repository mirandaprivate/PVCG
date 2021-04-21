'''
'''
import numpy as np
from compute_acceptance import compute_acceptance
from compute_social_surplus import compute_social_surplus, compute_social_surplus_except_i


def tau(effective_contrib, cost_type, valuation_type, i, alpha=1.0):
    '''
    Args:
        effective_contrib: A list of floats, each of whose element represents
            the capacity limit of each agent.
        cost_type: A list of floats, each of whose element represents the
            cost type of each agent.
        valuation_type: A list of floats, each of whose element represents the
            valuation type of each agent.
        i: The ith element of tau.
    '''
    length = len(effective_contrib)
    acceptance = compute_acceptance(effective_contrib, cost_type, valuation_type, alpha)
    social_surplus = compute_social_surplus(effective_contrib, cost_type, valuation_type, alpha)

    social_surplus_except_i = compute_social_surplus_except_i(
        effective_contrib, cost_type, valuation_type,i, alpha=alpha
    )

    #print(social_surplus_except_i)
    #print(cost_type[i] * acceptance[i] * effective_contrib[i])

    ## for the case there is no info asymmetry on the demand side

    tau = cost_type[i] * acceptance[i] * effective_contrib[i] \
    + social_surplus - social_surplus_except_i
    #print(tau)

    return tau


def compute_tau(effective_contrib, cost_type, valuation_type, alpha=1.0):
    ''' Computing all data in groups.
    '''
    tau_vector = []
    length = len(effective_contrib)
    for dummy in range(length):
        tau_i = tau(effective_contrib, cost_type, valuation_type, dummy,alpha)
        tau_vector.append(tau_i)
    return tau_vector
