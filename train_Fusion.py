# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())
import time
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')  
import logging
logging.basicConfig(level=logging.CRITICAL)

from nets.Unet5 import UNet5
from nets.Ufuser import Ufuser
from utils import loss_fusion,Transformer,H5Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_epochs = 40
lr = 1e-4
alpha=0.1
batch_size = 4

shift_num=3
rotate_num=3
flip_num=3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=Ufuser().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model.train()

F2V_path=r'/home/liuyong/code/MMIF-EMMA-main_111/model/Av.pth'
F2I_path=r'/home/liuyong/code/MMIF-EMMA-main_111/model/Ai.pth'

F2Vmodel = UNet5().to(device)
# F2V_weights_dict = torch.load(F2V_path, map_location=device)
# F2V_load_weights_dict = {k: v for k, v in F2V_weights_dict.items()
#                      if k in F2Vmodel.state_dict().keys() and F2Vmodel.state_dict()[k].numel() == v.numel()}
# F2Vmodel.load_state_dict(F2V_load_weights_dict, strict=False)
F2Vmodel.load_state_dict(torch.load(F2V_path))
F2Vmodel.eval()

F2Imodel = UNet5().to(device)
# F2I_weights_dict = torch.load(F2I_path, map_location=device)
# F2I_load_weights_dict = {k: v for k, v in F2I_weights_dict.items()
#                      if k in F2Imodel.state_dict().keys() and F2Imodel.state_dict()[k].numel() == v.numel()}
# F2Imodel.load_state_dict(F2I_load_weights_dict, strict=False)
F2Imodel.load_state_dict(torch.load(F2I_path))
F2Imodel.eval()

trainloader = DataLoader(H5Dataset(r"data/MSRS_train_128_200.h5"),batch_size=batch_size, shuffle=True, num_workers=0)
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

loss=loss_fusion()  
tran = Transformer(shift_num, rotate_num, flip_num)

torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    
    for i, (data_IR, data_VIS, index) in enumerate(trainloader):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        F=model(data_IR,data_VIS)  # F
        Ft = tran.apply(F)
        Ft_caret= model(F2Imodel(Ft),F2Vmodel(Ft)) # Ft_caret
        optimizer.zero_grad()
        loss_total=loss(F2Vmodel(F),data_VIS)+loss(F2Imodel(F),data_IR)+alpha*loss(Ft,Ft_caret)
        loss_total.backward()
        optimizer.step()

        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [loss_total: %f] ETA: %.10s"
            % (
                epoch+1,
                num_epochs,
                i,
                len(trainloader),
                loss_total.item(),
                time_left,
            )
        )

    scheduler.step()  
torch.save(model.state_dict(), os.path.join("model/EMMA_"+timestamp+'.pth'))


