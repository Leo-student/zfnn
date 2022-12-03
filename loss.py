import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import torch
import numpy as np
import config as cfg

def plot_loss():
    y = []
    y1 = []
    #for i in range(0,n):
    enc = np.load('./gloss.npy')
    #enc1 = np.load('./gloss.npy')
    

    # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
    tempy = list(enc)
    #tempy1 = list(enc1)
    y += tempy
    #y1 += tempy1
    x = range(0,len(y))

    plt.plot(x, y, '.-')
    #plt.plot(x, y1, '.-')
    plt.legend( [' loss'], loc = 'upper right')
    plt_title = 'batch_size = {} learning_rate = {}'.format(cfg.batch_size,cfg.learning_rate)
    
    plt.title(plt_title)
    plt.xlabel('per epoch')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    plot_loss()
