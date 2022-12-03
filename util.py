import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataset import *
from load_data import *

def evaluate(model, compsnr):
    print('Evaluate...\n')
    cnt = 0
    total_psnr = 0
    model.eval()

    test_name_queue = generate_file_list(['7', '8', '9', '10', '11'])
    test_dataset = loadImgs(test_name_queue)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1,
                                               shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            ''' load data and add noise '''
            img_trains = data[0].float().cuda()  # b，w，h, c*frame
            gt_trains = data[1].float().cuda()  # b,w,h,c*frame
            noise_maps = data[2].float().cuda()  # b,2
            frame_psnr = 0
            for tt in range(7):
                gt_train = gt_trains[:, tt * 4:tt * 4 + 4, :, :]
                cur = img_trains[:, tt * 4:tt * 4 + 4, :, :]
                noise_map = noise_maps[:]
                if tt == 0:# [ 1  0 2]
                    pre = img_trains[:,tt*4+4 * 1:tt*4 + 4 * 2,:,:]
                    nxt = img_trains[:,tt*4+4 * 2:tt*4 + 4 * 3,:,:]
                elif tt == 6: #[ 4 6 5]
                    pre = img_trains[:,tt*4-4 * 1:tt*4 + 4 * 0,:,:]
                    nxt = img_trains[:,tt*4-4 * 2:tt*4 - 4 * 1,:,:]
                else: #[ x-1 x x +1]
                    pre = img_trains[:,tt*4-4 * 1:tt*4 + 4 * 0,:,:]
                    nxt = img_trains[:,tt*4+4 * 1:tt*4 + 4 * 2,:,:]
                inputn = torch.cat([pre, cur, nxt], 1)

                
                x = model(inputn, noise_map)
                x = torch.clamp(x,0,1)

                frame_psnr += compsnr(x, gt_train)
                #pre = x.detach()
                del gt_train, cur
            frame_psnr = frame_psnr / (7.0)
            # print('---------')
            # print('Scene: ', ('%02d' % scene_ind), 'Noisy_level: ', ('%02d' % noisy_level), 'PSNR: ', '%.8f' % frame_psnr.item())
            total_psnr += frame_psnr


            cnt += 1
        total_psnr = total_psnr / cnt
    print('Eval_Total_PSNR              |   ', ('%.8f' % total_psnr.item()))
    torch.cuda.empty_cache()
    return	total_psnr
def evaluate_train(model, compsnr):
    print('Train...\n')
    cnt = 0
    total_psnr = 0
    model.eval()

    rand_int = np.random.randint(0, 6)
    print('Evaluate on part of training set{}...'.format(rand_int+1))
    file_list = ['1', '2', '3', '4', '5', '6']
    rand_filelist = []
    rand_filelist.append(file_list[rand_int])

    train_data_name_queue1 = generate_file_list(['1', '2', '3', '4', '5', '6']) #str
    train_dataset = loadImgs(train_data_name_queue1) #crvd
    #test_name_queue = generate_file_list(['7', '8', '9', '10', '11'])
    #test_dataset = loadImgs(test_name_queue,trainflag,fullres)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1,
                                               shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iter, data in enumerate(train_loader):
            ''' load data and add noise '''
            img_trains = data[0].float().cuda()  # b，w，h, c*frame
            gt_trains = data[1].float().cuda()  # b,w,h,c*frame
            noise_maps = data[2].float().cuda()  # b,2
            frame_psnr = 0
            for tt in range(7):
                gt_train = gt_trains[:, tt * 4:tt * 4 + 4, :, :]
                cur = img_trains[:, tt * 4:tt * 4 + 4, :, :]
                noise_map = noise_maps[:]
                if tt == 0:
                    pre = img_trains[:,tt*4+4 * 1:tt*4 + 4 * 2,:,:]
                    nxt = img_trains[:,tt*4+4 * 2:tt*4 + 4 * 3,:,:]
                elif tt == 6:
                    pre = img_trains[:,tt*4-4 * 1:tt*4 + 4 * 0,:,:]
                    nxt = img_trains[:,tt*4-4 * 2:tt*4 - 4 * 1,:,:]
                else:
                    pre = img_trains[:,tt*4-4 * 1:tt*4 + 4 * 0,:,:]
                    nxt = img_trains[:,tt*4+4 * 1:tt*4 + 4 * 2,:,:]
                inputn = torch.cat([pre, cur, nxt], 1)

                x = model(inputn, noise_map)
                x = torch.clamp(x,0,1)

                frame_psnr += compsnr(x, gt_train)
                #pre = x.detach()
                del gt_train, cur

            frame_psnr = frame_psnr / (7.0)
                # print('---------')
                # print('Scene: ', ('%02d' % scene_ind), 'Noisy_level: ', ('%02d' % noisy_level), 'PSNR: ', '%.8f' % frame_psnr.item())
            total_psnr += frame_psnr
            cnt += 1
        total_psnr = total_psnr / cnt
    print('Train_Total_PSNR              |   ', ('%.8f' % total_psnr.item()))
    torch.cuda.empty_cache()
    return	total_psnr

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, image, label):
        MSE = (image - label) * (image - label)
        MSE = torch.mean(MSE)
        PSNR = 10 * torch.log(1 / MSE) / torch.log(torch.Tensor([10.])).cuda()  # torch.log is log base e

        return PSNR

class L1_Charbonnier_loss(torch.nn.Module):
	"""L1 Charbonnierloss."""

	def __init__(self):
		super(L1_Charbonnier_loss, self).__init__()
		self.eps = 1e-6

	def forward(self, X, Y):
		# weight = torch.ones_like(X)
		# weight[:,:,2:-2,4:-4] = 2
		diff = torch.add(X, -Y)
		error = torch.sqrt(diff * diff + self.eps)
		loss = torch.mean(error)
		return loss