import thop
from thop import profile
from thop import clever_format
from model import fastdvd_3
import torch
model = fastdvd_3()
inputn = torch.randn(8, 12, 128, 128)
noimap = torch.randn(8,  2, 128, 128)
flops, params = profile(model, inputs=(inputn, noimap))
print(flops, params)
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)