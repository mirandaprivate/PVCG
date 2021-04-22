from compute_tau import compute_tau
import math

effect = [[1.0] * 10, [3.0] * 10, [5.0] * 10, [7.0] * 10, [10] * 10]
cost_type = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
             for i in range(5)]

for dummy in range(5):
    print(compute_tau(effect[dummy], cost_type[dummy], math.sqrt(10)))
