import torch
import numpy as np
import config as cfg
import torch.nn as nn

def pack_gbrg_raw(raw):
    black_level = 240
    white_level = 2 ** 12 - 1 
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],          # r
                          im[1:H:2, 1:W:2, :],          # gr
                          im[0:H:2, 1:W:2, :],          # b
                          im[0:H:2, 0:W:2, :]), axis=2) # gb
    #print( out.shape )
    
    return out

def depack_gbrg_raw(raw):
    H = raw.shape[1]
    W = raw.shape[2]
    output = np.zeros((H*2,W*2))
    for i in range(H):
        for j in range(W):
            output[2 * i     , 2 * j    ] = raw[0, i , j , 3]        # gb
            output[2 * i     , 2 * j + 1] = raw[0, i , j , 2]      # b
            output[2 * i + 1 , 2 * j    ] = raw[0, i , j , 0]      # r
            output[2 * i + 1 , 2 * j + 1] = raw[0, i , j , 1]    # gr
    return output