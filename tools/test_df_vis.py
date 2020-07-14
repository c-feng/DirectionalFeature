import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse
import numpy as np
import cv2
import logging
import math

import _init_paths
from libs.network import U_Net, U_NetDF 
from libs.datasets import AcdcDataset
import libs.datasets.joint_augment as joint_augment
import libs.datasets.augment as standard_augment
from libs.datasets.collate_batch import BatchCollator

from libs.configs.config_acdc import cfg
from train_utils.train_utils import load_checkpoint
from utils.metrics import dice
from utils.vis_utils import mask2png, masks_to_contours, apply_mask, img_mask_png

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--used_df', type=str, default=False, help='whether to use df')
parser.add_argument('--selfeat', action='store_true', default=False, help='whether to use feature select')
parser.add_argument('--mgpus', type=str, default=None, required=True, help='whether to use multiple gpu')
parser.add_argument('--model_path1', type=str, default=None, help='whether to train with evaluation')
parser.add_argument('--model_path2', type=str, default=None, help='whether to train with evaluation')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
parser.add_argument('--output_dir', type=str, default=None, required=True, help='specify an output directory if needed')
parser.add_argument('--log_file', type=str, default="../log_evalation.txt", help="the file to write logging")
parser.add_argument('--vis', action='store_true', default=False, help="weather to save test result images")
args = parser.parse_args()

if args.mgpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.mgpus

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def create_dataloader():
    eval_transform = joint_augment.Compose([
                    joint_augment.To_PIL_Image(),
                    joint_augment.FixResize(256),
                    joint_augment.To_Tensor()])
    evalImg_transform = standard_augment.Compose([
                        standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD])])

    if cfg.DATASET.NAME == "acdc":
        test_set = AcdcDataset(cfg.DATASET.TEST_LIST, df_used=True, joint_augment=eval_transform,
                            augment=evalImg_transform)

    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                             num_workers=args.workers, shuffle=False,
                             collate_fn=BatchCollator(size_divisible=32, df_used=True))
    return test_loader, test_set

def cal_perfer(preds, masks, tb_dict):
    LV_dice = []  # 1
    MYO_dice = []  # 2
    RV_dice = []  # 3

    for i in range(preds.shape[0]):
        LV_dice.append(dice(preds[i,1,:,:],masks[i,1,:,:]))
        RV_dice.append(dice(preds[i, 3, :, :], masks[i, 3, :, :]))
        MYO_dice.append(dice(preds[i, 2, :, :], masks[i, 2, :, :]))
        # LV_dice.append(dice(preds[i, 3,:,:],masks[i,1,:,:]))
        # RV_dice.append(dice(preds[i, 1, :, :], masks[i, 3, :, :]))
        # MYO_dice.append(dice(preds[i, 2, :, :], masks[i, 2, :, :]))
    
    tb_dict["LV_dice"].append(np.mean(LV_dice))
    tb_dict["RV_dice"].append(np.mean(RV_dice))
    tb_dict["MYO_dice"].append(np.mean(MYO_dice))
    return np.mean(LV_dice), np.mean(RV_dice), np.mean(MYO_dice)

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

def test_it(model, data, device="cuda"):
    model.eval()
    imgs, gts = data[:2]
    gts_df = data[2]

    imgs = imgs.to(device)
    gts = gts.to(device)

    net_out = model(imgs)
    if len(net_out) > 1:
        preds_out = net_out[0]
        preds_df = net_out[1]
    else:
        preds_out = net_out[0]
        preds_df = None
    preds_out = nn.functional.softmax(preds_out, dim=1)
    _, preds = torch.max(preds_out, 1)
    preds = preds.unsqueeze(1)  # (N, 1, *)

    return preds, preds_df

def vis_it(pred, gt, img=None, filename=None, infos=None):
    h, w = pred.shape
    # gt_contours = masks_to_contours(gt)
    # mask = np.hstack([pred, np.zeros((h, 1)), gt])
    # gt_contours = np.hstack([gt_contours, np.zeros((h, 1)), np.zeros_like(gt)])
    # im_rgb = mask2png(mask).astype(np.int16)
    # im_rgb[:, w, :] = [255, 255, 255]
    # im_rgb = apply_mask(im_rgb, gt_contours, [255, 255, 255], 0.8)
    pred_im = mask2png(pred).astype(np.int16)
    gt_im = mask2png(gt).astype(np.int16)

    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = np.stack([img, img, img], axis=2)
    # img = img_mask_png(img, gt, alpha=0.1)

    # im_rgb = np.hstack([im_rgb, 255*np.ones((h, 1, 3)), img])

    # cv2.putText(im_rgb, "prediction", (2,h-4),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1)
    # cv2.putText(im_rgb, "ground truth", (w, h-4),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1)

    # st_pos = 15
    # if infos is not None:
    #     for info in infos:
    #         cv2.putText(im_rgb, info+": {}".format(infos[info]), (2, st_pos),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1)
    #         st_pos += 10

    cv2.imwrite(filename+"_img.png", img[:,:,::-1])
    cv2.imwrite(filename+"_pred.png", pred_im[:,:,::-1])
    cv2.imwrite(filename+"_gt.png", gt_im[:,:,::-1])

def vis_df(pred_df, gt_df, filename, infos=None):
    _, h, w = pred_df.shape

    # save .npy files
    np.save(filename+'.npy', [pred_df, gt_df])
    
    theta = np.arctan2(gt_df[1,...], gt_df[0,...])
    degree_gt = (theta - theta.min()) / (theta.max() - theta.min()) * 255
    # degree_gt = theta * 360
    mag_gt = np.sum(gt_df ** 2, axis=0, keepdims=False)
    mag_gt = mag_gt / mag_gt.max() * 255

    theta = np.arctan2(pred_df[1,...], pred_df[0,...])
    degree_df = (theta - theta.min()) / (theta.max() - theta.min()) * 255
    # degree_df = theta * 360
    magnitude = np.sum(pred_df ** 2, axis=0, keepdims=False)
    magnitude = magnitude / magnitude.max() * 255

    im = np.hstack([magnitude, np.zeros((h, 1)), mag_gt, np.zeros((h, 1)), degree_df, np.zeros((h, 1)), degree_gt]).astype(np.uint8)
    im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    cv2.imwrite(filename+"_df_pred_mag.png", im[:h, :w, ...])
    cv2.imwrite(filename+"_df_gt_mag.png", im[:h, w+1:2*w+1, ...])
    cv2.imwrite(filename+"_df_pred_degree.png", im[:h, 2*w+2:3*w+2, ...])
    cv2.imwrite(filename+"_df_gt_degree.png", im[:h, 3*w+3:, ...])


def test():
    root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, args.log_file)
    logger = create_logger(log_file)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    # create dataset & dataloader & network
    if args.used_df == 'U_NetDF':
        model = U_NetDF(selfeat=args.selfeat, num_class=4, auxseg=True)
    elif args.used_df == 'U_Net':
        model = U_Net(num_class=4)

    if args.mgpus is not None and len(args.mgpus) > 2:
        model = nn.DataParallel(model)
    model.cuda()

    test_loader, test_set = create_dataloader()
    
    checkpoint = torch.load(args.model_path1, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])

    dice_dict = {"LV_dice": [],
                    "RV_dice": [],
                    "MYO_dice": []}
    for i, data in enumerate(test_loader):
        if i != 23: continue
    # i = 5405
    # data = test_set[5405]
        # data = [data[0][None], data[1][None], data[2][None]]

        pred, pred_df = test_it(model, data[:3])

        _, gt, gt_df = data[:3]
        gt = gt.to("cuda")

        L, R, MYO = cal_perfer(make_one_hot(pred, 4), make_one_hot(gt, 4), dice_dict)

        data_info = test_set.data_infos[i]
        if args.vis:
        # if 0.7 <= (L + R + MYO) / 3 < 0.8:
            vis_it(pred.cpu().numpy()[0, 0], gt.cpu().numpy()[0, 0], data[0].cpu().numpy()[0, 0],
                    filename=os.path.join(root_result_dir, str(i)))
            if pred_df is not None:
                vis_df(pred_df.detach().cpu().numpy()[0], gt_df.cpu().numpy()[0], 
                        filename=os.path.join(root_result_dir, str(i)))

        print("\r{}/{} {:.0%}   {}".format(i, len(test_set), i/len(test_set), 
                                np.mean(list(dice_dict.values()))), end="")
    print()

    logger.info("2D Dice Metirc:")
    logger.info("Total {}".format(len(test_set)))
    logger.info("LV_dice: {}".format(np.mean(dice_dict["LV_dice"])))
    logger.info("RV_dice: {}".format(np.mean(dice_dict["RV_dice"])))
    logger.info("MYO_dice: {}".format(np.mean(dice_dict["MYO_dice"])))
    logger.info("Mean_dice: {}".format(np.mean(list(dice_dict.values()))))


if __name__ == "__main__":
    test()