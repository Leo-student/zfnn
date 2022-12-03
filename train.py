import os
import cv2
import sys
import time
import argparse
sys.path.append('../')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: 4096"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


import torch
import stat
import numpy       as np
import torch.nn    as nn
import torch.optim as optim
import config      as cfg
import util        as util
import torch.backends.cudnn as cudnn

#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


import warnings
warnings.filterwarnings('ignore')

#from torchstat import stat
from random    import random
from dataset   import *
from load_data import *

from model import fastdvd_3
from util  import evaluate, evaluate_train, L1_Charbonnier_loss
from skimage.measure.simple_metrics import compare_psnr as PSNR


# from thop import profile
# from thop import clever_format


# Define GPU devices
#device_ids = [4,5,6,7]
cudnn.benchmark = True  # CUDNN optimization for faster training yet taking more time at first

def initialize():
	"""
	# clear some dir if necessary
	make some dir if necessary
	make sure training from scratch
	:return:
	"""
	##
	if not os.path.exists(cfg.model_name):
		os.mkdir(cfg.model_name)

	if not os.path.exists(cfg.debug_dir):
		os.mkdir(cfg.debug_dir)

	if not os.path.exists(cfg.log_dir):
		os.mkdir(cfg.log_dir)


	if cfg.checkpoint == None:
		s = input('Are you sure training the model from scratch? y/n \n')
		if not (s=='y'):
			return 


def main():
    checkpoint    = cfg.checkpoint
    start_epoch   = cfg.start_epoch
    start_iter    = cfg.start_iter
    ngpu          = cfg.ngpu
    learning_rate = cfg.learning_rate
    device        = cfg.device
    epoch         = cfg.epoch
    image_height  = cfg.image_height
    image_width   = cfg.image_width
    print("checkpoint    = {}".format(cfg.checkpoint   ))
    
    print("ngpu          = {}".format(cfg.ngpu         ))
    print("learning_rate = {}".format(cfg.learning_rate))
    print("device        = {}".format(cfg.device       ))
    
    print("epoch         = {}".format(cfg.epoch        ))
    print("image_height  = {}".format(cfg.image_height ))
    print("image_width   = {}\n".format(cfg.image_width  ))
    print("batch_size    = {}\n".format(cfg.batch_size  ))
    # Load dataset
    train_CRVD_list =  generate_file_list(['1', '2', '3', '4', '5', '6'])
    train_dataset = loadImgs(train_CRVD_list)   
    # train_loader [0]: noise_image
    # train_loader [1]: gt_image
    # train_loader [2]: noise_level
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size  = cfg.batch_size,
                                                num_workers = 4 ,
                                                shuffle     = True,
                                                pin_memory  = True)
    
    #device_ids = [4,5,6,7]
    device_ids = [0,1]
    torch.backends.cudnn.benchmark = True  # CUDNN optimization
    model      = fastdvd_3()
    model      = nn.DataParallel(model, device_ids=device_ids).cuda()


    compsnr    = util.PSNR().cuda()
    criterion  = L1_Charbonnier_loss().cuda()
    #criterion  = nn.L1Loss().cuda()
    #criterion = nn.MSELoss().cuda()
    # if torch.cuda.is_available() and ngpu > 1:
    #     model = nn.DataParallel(model, device_ids=list(range(ngpu))).cuda()
    #     print("list of ngpu{}".format(list(range(ngpu))))
    frame_psnr = 0
    g_loss     = 0
    best_psnr  = 0
    #stat(model, inputs=((12 ,128,128),(2,128,128)))
    


    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    '''
    contentloss = L1_Charbonnier_loss().cuda()
    msecriterion = nn.MSELoss()
    '''

    ## load pretrained model
    if checkpoint is not None:
        print('--- Loading Pretrained Model ---')
        checkpoint  = torch.load(checkpoint)
        
        
        start_epoch = checkpoint['epoch']
       
        print(checkpoint['iter'])
        start_iter  = checkpoint['iter']
        print("start_epoch   = {}".format(start_epoch      ))
        print("start_iter    = {}".format(start_iter       ))
        
        
        
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # loss_arr = np.array([])
        # psnr_t_arr = np.array([])
        # psnr_v_arr = np.array([])
        loss_arr   = np.load("./gloss.npy")
        psnr_t_arr = np.load("./train_psnr.npy")
        psnr_v_arr = np.load("./evaluate_psnr.npy")
    else :
        loss_arr    = np.array([])
        psnr_t_arr  = np.array([])
        psnr_v_arr  = np.array([])
        start_epoch = 0
        iter = start_iter
   
    eval_psnr = 0
    optimizer.zero_grad()
    #model.train()
    #print(train_loader)
    ## tensorboard --logdir runs
    #writer = SummaryWriter(cfg.log_dir)
    
    # Training
    start_time = time.time()
   
    for epoch in range(start_epoch, cfg.epoch*5):
        print('------------------------------------------------')
        print('Epoch                |   ', ('%08d' % epoch))
        
        g_loss = 0
        for iter, data in enumerate(train_loader):
            img_trains = data[0].float().cuda() 
            
            gt_trains = data[1].float().cuda()  
            #print(gt_trains.shape)
            noise_maps = data[2].float().cuda()  
            #print(noise_maps.shape)
            # model.train()
            loss = 0
            optimizer.zero_grad()
            for frame_idx in range (cfg.frame_num):
                #print("{} {} {}".format(epoch,iter,frame_idx))
                gt_train  = gt_trains [:, frame_idx * 4:(frame_idx + 1)*4, :, :]
                cur       = img_trains[:, frame_idx * 4:(frame_idx + 1)*4 ,:, :]
                noise_map = noise_maps[:]
                if frame_idx == 0:
                    pre = img_trains[:,(frame_idx + 1)*4:(frame_idx + 2)*4,:,:]
                    nxt = img_trains[:,(frame_idx + 2)*4:(frame_idx + 3)*4,:,:]
                elif frame_idx == 6:
                    pre = img_trains[:,(frame_idx - 2)*4:(frame_idx - 1)*4,:,:]
                    nxt = img_trains[:,(frame_idx - 3)*4:(frame_idx - 2)*4,:,:]
                else:
                    pre = img_trains[:,(frame_idx - 1)*4: frame_idx*4     ,:,:]
                    nxt = img_trains[:,(frame_idx + 1)*4:(frame_idx + 2)*4,:,:]
                inputn = torch.cat([pre, cur, nxt], 1)
                # optimization
                #print(inputn.shape)
                #print(inputn.size)
                
                out_train = model(inputn, noise_map)
                

                out_train = torch.clamp(out_train,0,1)
                #frame_psnr += compsnr(out_train, gt_train)

                
                loss += criterion(out_train, gt_train)

                g_loss += loss.item()
                
                
            loss.backward()
            optimizer.step()
            del gt_train, cur
            
            if epoch % 2 == 0 and iter == 0:
                torch.save({
                    'epoch': epoch,
                    'iter': iter,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    cfg.model_save_root)
            if iter % 10 == 0 :
                    eval_psnr = evaluate(model, compsnr)
                    eval_psnr = eval_psnr.item()
                    if eval_psnr > best_psnr + 0.0:
                        best_psnr = eval_psnr
                        torch.save({
                            'epoch': epoch,
                            'iter': iter,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_psnr': best_psnr},
                            cfg.best_model_save_root)
        model.eval()
        
        train_psnr = evaluate_train(model, compsnr)
        train_psnr = train_psnr.item()
        print('epoch {}, iter {}'.format(epoch, iter))

        loss_arr = np.append(loss_arr, g_loss)
        psnr_t_arr = np.append(psnr_t_arr, train_psnr)
        psnr_v_arr = np.append(psnr_v_arr, eval_psnr)

        np.save(file='gloss.npy', arr=loss_arr)
        np.save(file='train_psnr.npy', arr=psnr_t_arr)
        np.save(file='evaluate_psnr.npy', arr=psnr_v_arr)
        print('   gloss :{} and psnr eval {}   train {}: '.format(g_loss, eval_psnr, train_psnr))
        torch.cuda.empty_cache()



    #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)


    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    print('Training done.')

    # # Close logger file
    # close_logger(logger)

if __name__ == "__main__":

    initialize()
    main ( )
