# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import cv2
import glob

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
#checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_Sony/'
result_dir_in = './result_Sony/in/'
# get test IDs
#test_fns = glob.glob(gt_dir + '/*.tiff')
test_fns = glob.glob(gt_dir + '/10203_*.ARW') 
#
#The file name contains the image information. For example, 
# in '10019_00_0.033s.RAF', 
# the first digit 1' means it is from the test set 
# ('0' for training set and '2' for validation set);
#  '0019' is the image ID; the following 
# '00' is the number in the sequence/burst;
#  '0.033s' is the exposure time 1/30 seconds.
#test_ids = [int(os.path.basename(test_fn)[5:6]) for test_fn in test_fns]
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
print (f'{len(test_ids)} raw images were found. in {gt_dir}')
DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:1]



def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # out = np.concatenate((im[0:H:2, 0:W:2, :],
    #                       im[0:H:2, 1:W:2, :],
    #                       im[1:H:2, 1:W:2, :],
    #                       im[1:H:2, 0:W:2, :]), axis=2)
    
    out = np.concatenate((im[1:H:2, 1:W:2, :],          # r
                          im[0:H:2, 1:W:2, :],          # gr
                          im[0:H:2, 0:W:2, :],         # b
                          im[1:H:2, 0:W:2, :]), axis=2) # gb
    return out

if not os.path.isdir(result_dir_in + 'final/'):
    os.makedirs(result_dir_in + 'final/')



if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids:
    
    # test the first image in each sequence
    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(gt_files)):
        in_path = gt_files[k]
        in_fn = os.path.basename(in_path)
  
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
  

        gt_raw = rawpy.imread(gt_path)
        image_visible = gt_raw.raw_image_visible

        H = image_visible.shape[0]
        W = image_visible.shape[1]
    
        print(gt_path, H, W)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
       
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_full = gt_full[0, :, :, :]
        gt_full =cv2.cvtColor(gt_full, cv2.COLOR_BGR2RGB)
        cv2.imwrite(result_dir_in + 'final/%5d_00__gt.png' % (test_id),gt_full * 255)
