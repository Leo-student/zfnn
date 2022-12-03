import torch
import torch.nn as nn
#[base]
class CvBlock(nn.Module):
    '''(Conv2d  => ReLU) x 2'''
    def __init__(self, in_ch, out_ch,fil_lrelu=['']):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1 ),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1 ),
            nn.LeakyReLU(0.2, inplace=False),
        )
    def forward(self, x):
        return  (self.convblock(x) )

#[1]
class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups  => ReLU) + (Conv  => ReLU)'''
    def __init__(self, num_in_frames, out_ch,filter_lrelu_flag=['']):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 16 * 2
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*( 4 + 2 ), num_in_frames*self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames ),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1 )
        )
        self.act = nn.LeakyReLU(0.2,inplace=False)

    def forward(self, x):
        # print("input  = {}".format( x.shape))
        # print("output = {}".format((self.act( self.convblock(x))).shape))
        return self.act( self.convblock(x) )
#[2 - 3]
class DownBlock(nn.Module):
    ''' Downscale + (Conv2d  => ReLU)*2 '''
    def __init__(self, in_ch, out_ch, filter_lrelu_flag=['']):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=False),
            CvBlock(out_ch, out_ch, filter_lrelu_flag)
        )
    def forward(self, x):
        return self.convblock(x)
#[4 - 5]
class UpBlock(nn.Module):
    '''(Conv2d  => ReLU)*2 + Upscale'''
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch,['']),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)
#[6]
class OutputCvBlock(nn.Module):
    '''Conv2d  => ReLU => Conv2d'''
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.convblock(x)



class DenBlock(nn.Module):
    '''
    define the denosing block of fastdvdnet

    '''
    def __init__( self, num_input_frames = 3):
        super(DenBlock, self).__init__()
        self.chNo_32 = 32 
        self.chNo_64 = 64
        self.chNo_128 = 128

        self.inc    = InputCvBlock(num_in_frames=num_input_frames, out_ch = self.chNo_32)
        self.downc0 = DownBlock(    in_ch = self.chNo_32         , out_ch = self.chNo_64)
        self.downc1 = DownBlock(    in_ch = self.chNo_64         , out_ch = self.chNo_128)
        self.upc2   = UpBlock(      in_ch = self.chNo_128        , out_ch = self.chNo_64)
        self.upc1   = UpBlock(      in_ch = self.chNo_64         , out_ch = self.chNo_32)
        self.outc   = OutputCvBlock(in_ch = self.chNo_32         , out_ch = 4)

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # def reset_params(self):
    #     for _, m in enumerate(self.modules()):
    #         self.weight_init(m)
    
    def forward(self, in0, in1, in2, noise_map):
        #3 input  + noise map
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim = 1))
        # print("x0 {}".format(x0.shape))
        #Downsampling  
        x1 = self.downc0(x0)
        # print("x1 {}".format(x1.shape))
        x2 = self.downc1(x1)
        # print("x2 {}".format(x2.shape))
        #Upsampling
        x2 = self.upc2(x2)

        x1 = self.upc1(x1 + x2)

        x  = self.outc(x0 +x1)
        x  = in1 - x
        return x

class fastdvd_3(nn.Module):
    '''
    define the main model of fastdvdnet
    input of forward():
        xn        : input frame of dim [N, C, H, W], (C = 4 rggb )
        noise_map : array with noise map of dim [N, 1, H, W]
    '''
    def __init__(self, num_input_frames = 3):
        super(fastdvd_3, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp1            = DenBlock( num_input_frames = 3)
        self.temp2            = DenBlock( num_input_frames = 3)
         
    def forward(self, x, noise_map):
        #(x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))
        
        (x0, x1, x2) = tuple(x[:, 4 * m : 4 * m + 4, :, :] for m in range(self.num_input_frames))
        
        x20          = self.temp1(x0, x1, x2, noise_map)
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(noise_map.shape)
        
        # print(x20.shape)
        # print(type(x20))
        #print(x20.size())
        #x21 
        #x22
        x            = x20 

        return x
        # return 0