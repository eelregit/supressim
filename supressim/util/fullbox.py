import numpy as np
import torch
import torch.nn as nn
from supressim.srsgan import models
from supressim.srsgan.datasets import *
from collections import OrderedDict

generator = models.GeneratorResNet(n_residual_blocks=2)
gpath = '/home/yueying/scratch1/dmo-srsgan/test3/saved_models/generator_14.pth'
d1 = (torch.load(gpath,map_location=lambda storage, loc: storage))
d2 = OrderedDict([(k[7:], v) if k.startswith('module.') else (k, v) for k, v in d1.items()]) # strip module wrapped from data parallel
generator.load_state_dict(d2)

device = torch.device('cpu')
generator = generator.to(device)
generator.eval()

lrfile = "/home/yueying/scratch1/dmo-srsgan/full/lr_set4_004.npy"

lr_box = np.load(lrfile)
lr_box = lr_box[0:150,0:150,0:150,:]
lr_box = np.moveaxis(lr_box, -1, 0)
lr_box = lr_box[:3]
normalize(lr_box)

lr_box = np.expand_dims(lr_box, axis=0)
lr_box = torch.from_numpy(lr_box).float()
# print (lr_box.size())

lr_box = lr_box.to(device)

with torch.no_grad():
    sr_box = generator(lr_box)

np.save("/home/yueying/scratch1/dmo-srsgan/full/sr_set4_004",sr_box.detach().cpu().numpy())
