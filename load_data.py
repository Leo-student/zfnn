
import os
import cv2
import numpy as np 
import config as cfg
from torch.utils.data import Dataset

from dataset import *




iso_list = [1600, 3200, 6400, 12800, 25600]
a_list   = [3.513262,   6.955588,  13.486051,  26.585953, 52.032536  ]
b_list   = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
def generate_file_list(scene_list):
    file_num = 0
    data_name = []
    data_random_list = []
    for scene_ind in scene_list:
        for iso in iso_list:
            for frame_ind in range(1,8):
                gt_name = os.path.join('ISO{}/scene{}_frame{}_gt_sRGB.png'
                .format(iso, scene_ind, frame_ind-1))
                data_name.append(gt_name)
                file_num += 1
    #### ? 
    random_index = np.random.permutation(file_num)
    for i,idx in enumerate(random_index):
        data_random_list.append(data_name[idx])
    
    return data_random_list



def decode_data(data_name):
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
    H = 1080
    W = 1920

    # xx = np.random.randint(0, (W - cfg.image_width * 2 + 1) / 2) * 2
    # yy = np.random.randint(0, (H - cfg.image_height * 2 + 1) / 2) * 2
    xx = np.random.randint(0, (W - cfg.image_width * 2 - 2) ) // 2 * 2
    yy = np.random.randint(0, (H - cfg.image_height * 2 - 2) ) // 2 * 2
    xx = 200
    yy = 200
    #print(yy)
    scene_ind       =     data_name.split('/')[1].split('_')[0]
    frame_ind       = int(data_name.split('/')[1].split('_')[1][5:])
    iso_ind         =     data_name.split('/')[0]

    noisy_level_ind = iso_list.index(int(iso_ind[3:]))
    noisy_level     = [ a_list[noisy_level_ind] , 
                        b_list[noisy_level_ind] ]

    noise_map = np.zeros((2, cfg.image_height, cfg.image_width))
    noise_map[0,:,:] = ( np.sqrt(noisy_level[0]) / (2**12-1.0) ) ** 2
    noise_map[1,:,:] = ( np.sqrt(noisy_level[1]) / (2 ** 12 - 1.0) ) ** 2
    #print(noise_map.shape)
    
    gt_name_list    = []
    noisy_name_list = []
    xx_list         = []
    yy_list         = []

    for shift in range(0, cfg.frame_num):
    #for shift in range(0, 1):
        gt_name = os.path.join(cfg.data_root[1],'indoor_raw_gt/{}/{}/frame{}_clean_and_slightly_denoised.tiff'.format(
                               scene_ind,
                               iso_ind,
                               frame_list[frame_ind + shift]))

        noisy_frame_index_for_current = np.random.randint(0, 10)
        noisy_name = os.path.join(cfg.data_root[1],
								  'indoor_raw_noisy/{}/{}/frame{}_noisy{}.tiff'.format(
                                      scene_ind, 
                                      iso_ind, 
                                      frame_list[frame_ind + shift], 
                                      noisy_frame_index_for_current))

        gt_name_list.append(gt_name)
        noisy_name_list.append(noisy_name)

        xx_list.append(xx)
        yy_list.append(yy)


    gt_raw_data_list = []
    noisy_data_list  = []
    for ii in range(len(xx_list)): #read data using the neme list
        
        #print(ii)
        #print(len(xx_list))
        #print(noisy_name_list[ii])
        noise_raw = read_img(noisy_name_list[ii], int(xx_list[ii]), int(yy_list[ii]))
        gt_raw    = read_img(gt_name_list[ii],    int(xx_list[ii]), int(yy_list[ii]))
        
        gt_raw_data_list.append(gt_raw)
        noisy_data_list.append(noise_raw)

    gt_raw_batch    = np.concatenate(gt_raw_data_list, axis=2)
    noisy_raw_batch = np.concatenate(noisy_data_list,  axis=2)
    gt_raw_batch    = gt_raw_batch.transpose(2,0,1)
    noisy_raw_batch = noisy_raw_batch.transpose(2, 0, 1)
    #print("noisy_raw_batch {}".format(noisy_raw_batch.shape))
    
    #print(gt_raw_batch.shape)
    return noisy_raw_batch, gt_raw_batch, noise_map

def read_img(img_name, xx, yy):
    raw           = cv2.imread(img_name, -1)
    #print("img_name, xx, yy {} {} {} {}\n".format(xx, yy,xx + cfg.image_width  * 2,yy + cfg.image_height * 2))
    raw_full      = raw
    #print(raw_full.shape)
    raw_patch     = raw_full[yy:yy + cfg.image_height * 2,xx:xx + cfg.image_width  * 2]  
    
    raw_pack_data = pack_gbrg_raw(raw_patch) 

    return raw_pack_data

class loadImgs(Dataset):
    def __init__(self, filelist):
        self.filelist = filelist

    def __len__(self):
        return len( self.filelist)

    def __getitem__(self, item):
        self.data_name = self.filelist[ item ]
        image, label ,noise_level = decode_data( self.data_name )
        self.image          = image
        self.label          = label
        self.noise_level    = noise_level
        #print(noise_level)
        return self.image, self.label, self.noise_level 




