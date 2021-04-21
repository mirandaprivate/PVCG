# This file is to plot loss function in the iterations

import matplotlib.pyplot as plt
import pathlib
import math
import string

THIS_DIR = pathlib.Path(__file__).parent

loss_2_list = []
loss_1_list = []
total_loss_list = []


def change_str_to_float(stri):
    ''' change the string to the float which in the list accept or list cost.
    '''
    test_data = []
    for i in stri:
        index = [float(l.strip()) for l in i.split('    ')]
        loss_2_list.append(index[0])
        loss_1_list.append(index[1])
        total_loss_list.append(index[2])

    x = [i for i in range(len(loss_2_list))]
    fig, ax1 = plt.subplots(figsize = (4.5,4.5))

    # ax2 = ax1.twinx()
    ax1.plot(x, loss_1_list, linestyle ='--',linewidth=3.0, color='gray', alpha = 0.7, label='Loss1')
    ax1.plot(x, loss_2_list, linestyle ='-',linewidth=3.0, color='green', alpha = 0.5, label='Loss2')
    # ax1.plot(x, total_loss_list, color='b',linewidth=2.0, label='LOSS')
    ax1.set_ylim(0,)
    ax1.set_xlim(0,100)
    ax1.legend(loc='upper left', fontsize = 12)
    ax1.set_ylabel("Loss1 and Loss2", fontsize = 12)
    ax1.set_xlabel("No. of iterations", fontsize = 12)

    plt.title('Training Loss v.s. Iterations\n', fontsize = 14)
    plt.show()


with open(THIS_DIR / 'loss.txt', 'r') as f:
    lines = f.read().splitlines()
    change_str_to_float(lines[1:300])
