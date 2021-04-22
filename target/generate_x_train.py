'''Generate input training data.
'''
import random
import pathlib
import numpy as np

THIS_DIR = pathlib.Path(__file__).parent

with open(THIS_DIR/ 'x_test.txt', 'w') as g:
    y = [0 for x in range(5)]
    for i in range(20):
        for l in range(5):
            y[l] = random.uniform(0, 1)
        np.savetxt(g, y)
        g.write('# New\n')

# with open(THIS_DIR/ 'x_test.txt', 'w') as f:
#     y =[0 for x in range(5)]
#     for i in range(1000):
#         y[0]=0.01 * i
#         np.savetxt(f, y)
#         f.write('# New\n')