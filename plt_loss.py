# This file is to plot loss function in the iterations

import matplotlib.pyplot as plt
import pathlib
import math
import string

from construct_graph import lambda_1, lambda_2, lambda_3

THIS_DIR = pathlib.Path(__file__).parent

loss_1_list = []
loss_2_list = []
loss_3_list = []
total_loss_list = []


def change_str_to_float(stri):
    ''' change the string to the float which in the list accept or list cost.
    '''
    test_data = []
    for i in stri:
        index = [float(l.strip()) for l in i.split('    ')]
        loss_1_list.append(index[0] / lambda_1)
        loss_2_list.append(index[1] / lambda_2)
        loss_3_list.append(index[2] / lambda_3)
        total_loss_list.append(index[3])

    x = [i for i in range(len(loss_1_list))]
    fig, ax1 = plt.subplots(figsize = (6.5,4))

    ax2 = ax1.twinx()
    ax1.plot(x, total_loss_list, color='b',linewidth=3.0, label='LOSS')
    ax1.plot(x, loss_3_list, linestyle ='dashdot',linewidth=2.0, color='gray', label='Loss1')
    ax1.set_ylim(0,)
    ax1.set_xlim(0,300)
    ax1.legend(loc='upper left')
    ax1.set_ylabel("LOSS & Loss1", fontsize = 12)
    ax1.set_xlabel("No. of iterations", fontsize = 12)

    ax2.plot(x, loss_2_list, color='r', linewidth=2.0, linestyle = 'dashdot',label='Loss2')
    ax2.plot(x, loss_1_list, color='g',linewidth=2.0, linestyle = 'dashdot', label='Loss3')
    ax2.set_ylim(0,0.1)
    ax2.legend(loc='best')
    ax2.set_ylabel("Loss2 & Loss3", fontsize = 12)

    # plt.fill_between(x, loss_3_list_new, color='blue')
    # plt.fill_between(x, loss_1_list_new,color='red')

    # plt.fill_between(x, loss_1_list, loss_2_list,where=loss_1_list>0, color='bule')
    # plt.fill_between(x, loss_2_list, loss_3_list, color='red')
    # plt.fill_between(x, loss_3_list, loss_1_list, color='yellow')
    plt.title('Training Loss v.s. Iterations', fontsize = 16)
    plt.show()


with open(THIS_DIR / 'loss.txt', 'r') as f:
    lines = f.read().splitlines()
    change_str_to_float(lines[1:300])

# x = [i for i in range(len(lines[0]))]
# plt.plot(x, lines[0])
# plt.show()
