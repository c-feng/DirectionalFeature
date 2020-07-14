import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import cv2
from collections import namedtuple
from utils.metrics import dice, cal_hausdorff_distance
from utils.vis_utils import batchToColorImg, masks_to_contours

def model_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])
    
    def model_fn(model, data, criterion, perfermance=False, vis=False, device="cuda", epoch=0, num_class=4):
        # imgs, gts, _ = data
        imgs, gts = data[:2]

        imgs = imgs.to(device)
        gts = torch.squeeze(gts, 1).to(device)

        net_out = model(imgs)

        loss = criterion(net_out[0], gts.long())

        tb_dict = {}
        disp_dict = {}
        tb_dict.update({"loss": loss.item()})
        disp_dict.update({"loss": loss.item()})

        if perfermance:
            gts_ = gts.unsqueeze(1)
            
            net_out = F.softmax(net_out[0], dim=1)
            _, preds = torch.max(net_out, 1)
            preds = preds.unsqueeze(1)
            cal_perfer(make_one_hot(preds, num_class), make_one_hot(gts_, num_class), tb_dict)
        

        return ModelReturn(loss, tb_dict, disp_dict)
    
    return model_fn

def model_DF_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])
    
    def model_fn(model, data, criterion=None, perfermance=False, vis=False, device="cuda", epoch=0, num_class=4):
        imgs, gts = data[:2]
        gts_df, dist_maps = data[2:]

        imgs = imgs.to(device)
        gts = torch.squeeze(gts, 1).to(device).long()
        gts_df = gts_df.to(device)

        net_out = model(imgs)
        seg_out, df_out = net_out[:2]

        # add Auxiliary Segmentation
        if len(net_out) >= 3 and net_out[2] is not None:
            auxseg_out = net_out[2]
            auxseg_loss = F.cross_entropy(auxseg_out, gts)
        else:
            auxseg_loss =  torch.tensor([0.], dtype=torch.float32, device=device)


        # loss = criterion(net_out, gts.long())
        # segmentation Loss
        seg_loss = F.cross_entropy(seg_out, gts)

        # direction field Loss
        df_loss, boundary_loss = criterion(seg_out, dist_maps, df_out, gts_df, gts)

        alpha = 1.0 
        loss = alpha*(seg_loss + 1. * df_loss + 0.1*auxseg_loss) + (1.-alpha)*boundary_loss

        tb_dict = {}
        disp_dict = {}
        tb_dict.update({"loss": loss.item(), "seg_loss": alpha*seg_loss.item(), "df_loss": alpha*1.*df_loss.item(),
                        "boundary_loss": (1.-alpha)*boundary_loss.item(), "auxseg_loss": alpha*0.1*auxseg_loss.item()})
        disp_dict.update({"loss": loss.item()})

        if perfermance:
            gts_ = gts.unsqueeze(1)
            
            seg_out = F.softmax(seg_out, dim=1)
            _, preds = torch.max(seg_out, 1)
            preds = preds.unsqueeze(1)
            cal_perfer(make_one_hot(preds, num_class), make_one_hot(gts_, num_class), tb_dict)

        if vis:
            # 可视化 方向场
            # vis_dict = {}
            gt_df = gts_df.cpu().numpy()
            _, angle_gt = cv2.cartToPolar(gt_df[:, 0,...], gt_df[:, 1,...])
            angle_gt = batchToColorImg(angle_gt, minv=0, maxv=2*math.pi).transpose(0, 3, 1, 2)

            df_map = df_out.cpu().numpy()
            mag, angle_df = cv2.cartToPolar(df_map[:, 0,...], df_map[:, 1,...])
            angle_df = batchToColorImg(angle_df, minv=0, maxv=2*math.pi).transpose(0, 3, 1, 2)
            mag = batchToColorImg(mag).transpose(0, 3, 1, 2)

            tb_dict.update({"vis": [angle_gt, mag, angle_df]})


        return ModelReturn(loss, tb_dict, disp_dict)
    
    return model_fn



def cal_perfer(preds, masks, tb_dict):
    LV_dice = []  # 1
    MYO_dice = []  # 2
    RV_dice = []  # 3
    LV_hausdorff = []
    MYO_hausdorff = []
    RV_hausdorff = []

    for i in range(preds.shape[0]):
        LV_dice.append(dice(preds[i,1,:,:],masks[i,1,:,:]))
        RV_dice.append(dice(preds[i, 3, :, :], masks[i, 3, :, :]))
        MYO_dice.append(dice(preds[i, 2, :, :], masks[i, 2, :, :]))
        
        LV_hausdorff.append(cal_hausdorff_distance(preds[i,1,:,:],masks[i,1,:,:]))
        RV_hausdorff.append(cal_hausdorff_distance(preds[i,3,:,:],masks[i,3,:,:]))
        MYO_hausdorff.append(cal_hausdorff_distance(preds[i,2,:,:],masks[i,2,:,:]))
    
    tb_dict.update({"LV_dice": np.mean(LV_dice)})
    tb_dict.update({"RV_dice": np.mean(RV_dice)})
    tb_dict.update({"MYO_dice": np.mean(MYO_dice)})
    tb_dict.update({"LV_hausdorff": np.mean(LV_hausdorff)})
    tb_dict.update({"RV_hausdorff": np.mean(RV_hausdorff)})
    tb_dict.update({"MYO_hausdorff": np.mean(MYO_hausdorff)})

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).scatter_(1, input.cpu().long(), 1)
    # result = result.scatter_(1, input.cpu(), 1)

    return result
