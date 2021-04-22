import numpy as np
from compute_acceptance import compute_acceptance
import math


def compute_social_surplus(effective_contrib, cost_type, alpha=1.0):

    length = len(effective_contrib)
    acceptance = compute_acceptance(effective_contrib, cost_type, alpha)

    #print(acceptance)

    accepted_contrib = np.multiply(effective_contrib, acceptance)

    #print(accepted_contrib)

    cost_list = np.multiply(cost_type, accepted_contrib)

    #print(cost_list)

    return alpha * math.sqrt(sum(accepted_contrib)) - sum(cost_list)


def compute_social_surplus_except_i(effective_contrib, cost_type, i, alpha=1.0):

    length = len(effective_contrib)
    acceptance = compute_acceptance(effective_contrib, cost_type, alpha)
    #print("------------------FLAG---------------")
    #print(acceptance)
    social_surplus = compute_social_surplus(effective_contrib, cost_type, alpha)
    #print(social_surplus)

    #compute the parameters when data owner i is excluded
    #acc_minus_i = acceptance[:i] + acceptance[i+1:]
    #print(acc_minus_i)

    effect_except_i = effective_contrib[:i] + effective_contrib[i + 1:]
    cost_type_except_i = cost_type[:i] + cost_type[i + 1:]
    acc_except_i = compute_acceptance(
        effect_except_i, cost_type_except_i, alpha
    )
    #print(acc_except_i)

    social_surplus_except_i = compute_social_surplus(
        effect_except_i, cost_type_except_i, alpha
    )

    return social_surplus_except_i
