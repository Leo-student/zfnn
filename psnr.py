import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import torch
import numpy as np
import config as cfg

def plot_psnr():
    y = []
    y1 = []
    #for i in range(0,n):
    eva = np.load('./train_psnr.npy')
    tra = np.load('./evaluate_psnr.npy')
    
    # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
    evalu = list(eva)
    train = list(tra)

    y   += evalu
    y1  += train
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    plt.plot(x, y1, '.-')
    
    plt.legend( [' train_psnr','eval_psnr'], loc = 'upper left')
    plt_title = 'BS = {} learning_rate = {}'.format(cfg.batch_size,cfg.learning_rate)
    
    plt.title(plt_title)
    plt.xlabel('per epoch')
    plt.ylabel('psnr')
    # plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    plot_psnr()
