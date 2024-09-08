# -*- coding: utf-8 -*-
import sys
import os

import cv2

sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import torch
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.CRITICAL)
import numpy as np

from utils import image_read_cv2,img_save,Evaluator
from nets.Ufuser import Ufuser

# MSRS dataset
# path_ir=r"/home/liuyong/dataset/MSRS-main/test/ir"
# path_vi=r"/home/liuyong/dataset/MSRS-main/test/vi"
# path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_MSRS_result"
# rgb_path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_MSRS_rgb_result"
# ori_img_folder = "/home/liuyong/dataset/MSRS-main/test"

# RoadScene dataset
path_ir=r"/home/liuyong/dataset/RoadScene/ir"
path_vi=r"/home/liuyong/dataset/RoadScene/vi"
path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_RoadScene_result"
rgb_path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_RoadScene_rgb_result"
ori_img_folder = "/home/liuyong/dataset/RoadScene"

# public roadscene dataset
# path_ir=r"/home/liuyong/dataset/public_roadscene/ir"
# path_vi=r"/home/liuyong/dataset/public_roadscene/vi"
# path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_public_roadscene_result"
# rgb_path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_public_roadscene_rgb_result"
# ori_img_folder = "/home/liuyong/dataset/public_roadscene"

# M3FD
# path_ir=r"/home/liuyong/dataset/M3FD_Fusion/ir"
# path_vi=r"/home/liuyong/dataset/M3FD_Fusion/vi"
# path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_M3FD_result"
# rgb_path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_M3FD_rgb_result"
# ori_img_folder = "/home/liuyong/dataset/M3FD_Fusion"

# MRI_PET dataset
# path_ir=r"/home/liuyong/dataset/MRI_PET/ir"
# path_vi=r"/home/liuyong/dataset/MRI_PET/vi"
# path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_MRI_PET_result"
# rgb_path_save=r"/home/liuyong/code/MMIF-EMMA-main_111/test_MRI_PET_rgb_result"
# ori_img_folder = "/home/liuyong/dataset/MRI_PET"


path_model=r"/home/liuyong/code/MMIF-EMMA-main_111/model/EMMA_selfpromter_crossatt_best_08-03-15-51.pth"
# path_model=r"model/EMMA.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=Ufuser().to(device)
model.load_state_dict(torch.load(path_model))
model.eval()

with torch.no_grad():
    for imgname in tqdm(os.listdir(path_ir)):

        # 源码2行
        IR = image_read_cv2(os.path.join(path_ir, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255
        VI = image_read_cv2(os.path.join(path_vi, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255

        # 后加的4行
        IR = image_read_cv2(os.path.join(path_ir, imgname), 'GRAY')[np.newaxis, np.newaxis,...]/255
        VI = cv2.split(image_read_cv2(os.path.join(path_vi, imgname), mode='YCrCb'))[0][np.newaxis,np.newaxis,...]/255
        data_vis_bgr = cv2.imread(os.path.join(path_vi, imgname))
        _, data_vis_cr, data_vis_cb = cv2.split(cv2.cvtColor(data_vis_bgr, cv2.COLOR_BGR2YCrCb))

        h, w = IR.shape[2:]
        h1 = h - h % 32
        w1 = w - w % 32
        h2 = h % 32
        w2 = w % 32

        if h1==h and w1==w: # Image size can be divided by 32
            ir = ((torch.FloatTensor(IR))).to(device)
            vi = ((torch.FloatTensor(VI))).to(device)
            data_Fuse=model(ir,vi)
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fused_image = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)

            # 后加的3行
            ycrcb_fi = np.stack((fused_image, data_vis_cr, data_vis_cb))
            ycrcb_fi = np.transpose(ycrcb_fi, (1, 2, 0))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            # img_save(fused_image, imgname.split(sep='.')[0], path_save)
            # 将fused_image改为了rgb_fi
            img_save(rgb_fi, imgname.split(sep='.')[0], rgb_path_save)
        else:
            # Upper left 
            fused_temp=np.zeros((h,w),dtype=np.float32)
            ir_temp = ((torch.FloatTensor(IR))[:,:,:h1,:w1]).to(device)
            vi_temp = ((torch.FloatTensor(VI))[:,:,:h1,:w1]).to(device)
            data_Fuse=model(ir_temp,vi_temp)
            fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
            fused_temp[:h1,:w1]=fused_image

            # upper right
            if w1!=w:
                ir_temp = ((torch.FloatTensor(IR))[:,:,:h1,-w1:]).to(device)
                vi_temp = ((torch.FloatTensor(VI))[:,:,:h1,-w1:]).to(device)
                data_Fuse=model(ir_temp,vi_temp)
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_temp[:h1,-w2:]=fused_image[:,-w2:]

            # lower left
            if h1!=h:    
                ir_temp = ((torch.FloatTensor(IR))[:,:,-h1:,:w1]).to(device)
                vi_temp = ((torch.FloatTensor(VI))[:,:,-h1:,:w1]).to(device)
                data_Fuse=model(ir_temp,vi_temp)
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_temp[-h2:,:w1]=fused_image[-h2:,:]

            
            # lower right
            if h1!=h and w1!=w:
                ir_temp = ((torch.FloatTensor(IR))[:,:,-h1:,-w1:]).to(device)
                vi_temp = ((torch.FloatTensor(VI))[:,:,-h1:,-w1:]).to(device)
                data_Fuse=model(ir_temp,vi_temp)
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_temp[-h2:,-w2:]=fused_image[-h2:,-w2:]

            fused_temp=(fused_temp-np.min(fused_temp))/(np.max(fused_temp)-np.min(fused_temp))
            # 下面这行代码后面需要将fused_temp转换为uint类型，参考fused_image = np.squeeze((data_Fuse * 255).cpu().numpy()).astype(np.uint8)
            fused_temp=((fused_temp)*255).astype(np.uint8)

            # 后加的3行
            ycrcb_fi = np.stack((fused_temp, data_vis_cr, data_vis_cb))
            ycrcb_fi = np.transpose(ycrcb_fi, (1, 2, 0))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)

            # 源码
            # img_save(fused_temp, imgname.split(sep='.')[0], path_save)
            # 将fused_temp改为了rgbfi
            img_save(rgb_fi, imgname.split(sep='.')[0], rgb_path_save)


# 验证指标结果代码部分
eval_folder = path_save
print("\n"*2+"="*80)
dataset_name = "MSRS"
model_name="CDDFuse    "
print("The test result of "+dataset_name+' :')
metric_result = np.zeros((9))
for img_name in tqdm(os.listdir(os.path.join(ori_img_folder, "ir"))):
    ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
    vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
    fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
    metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                  , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                  , Evaluator.AG(fi), Evaluator.SCD(fi, ir, vi)
                                  , Evaluator.VIFF(fi, ir, vi), Evaluator.Qabf(fi, ir, vi)
                                  , Evaluator.SSIM(fi, ir, vi)])

metric_result /= len(os.listdir(eval_folder))
print("\t\t EN\t SD\t SF\t MI\t AG\tSCD\tVIF\tQabf\tSSIM")
print(model_name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
      + str(np.round(metric_result[1], 2)) + '\t'
      + str(np.round(metric_result[2], 2)) + '\t'
      + str(np.round(metric_result[3], 2)) + '\t'
      + str(np.round(metric_result[4], 2)) + '\t'
      + str(np.round(metric_result[5], 2)) + '\t'
      + str(np.round(metric_result[6], 2)) + '\t'
      + str(np.round(metric_result[7], 2)) + '\t'
      + str(np.round(metric_result[8], 2))
      )
