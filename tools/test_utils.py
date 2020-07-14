import torch
import torch.nn as nn
import h5py
import numpy as np
import math

from utils.image_list import to_image_list

# acdc data
def personTo4Ddata(personname, test_list):
    sliceofp = []
    for tl in test_list:
        if '/'.join(personname.split('-')) in tl:
            sliceofp.append(tl)

    imgs = [[], []]
    gts = [[], []]
    for ti, time_i in enumerate(["ES", "ED"]):
        time_path = []
        
        for sp in sliceofp:
            if time_i in sp:
                time_path.append(sp)
        
        for tp in time_path:
            imgs[ti].append(h5py.File(tp, 'r')['image'])
            gts[ti].append(h5py.File(tp, 'r')['label'])

    imgs = np.array(imgs).transpose(1,2,3,0)
    gts = np.array(gts).transpose(1,2,3,0)
    return imgs, gts

def test_it(model, data, device="cuda", used_df=False):
    model.eval()
    imgs = data

    imgs = imgs.to(device)
    # gts = gts.to(device)

    net_out = model(imgs)
    if used_df:
        preds_out = net_out[0]
        preds_df = net_out[1]
    else:
        preds_out = net_out[0]
        preds_df = None
    preds_out = nn.functional.softmax(preds_out, dim=1)
    _, preds = torch.max(preds_out, 1)
    preds = preds.unsqueeze(1)  # (N, 1, *)

    return preds, preds_df

def test_person(model, imgs, multi_batches=False, used_df=False):
    """ imgs: (times, slices, H, W)
        preds: (times, slices, H, W)
    """
    preds = []
    for i in range(len(imgs)):
        preds_timei = np.zeros([imgs[i].size(0), imgs[i].size(2), imgs[i].size(3)])

        if multi_batches:
            batch_size = 32
            for bs in range(math.ceil(len(imgs[i]) / batch_size)):
                st = batch_size * bs
                end = st + batch_size if (st+batch_size) <= len(imgs[i]) else len(imgs[i])
                # data, origin_shape = to_image_list(imgs[i][st:st+batch_size], size_divisible=32, return_size=True)
                data = imgs[i][st:end]
                origin_shape = imgs[i].shape[-2:]

                pred, _ = test_it(model, data, used_df=used_df)
                preds_timei[st:end, ...] = pred.cpu().numpy()[:, 0, :origin_shape[0], :origin_shape[1]]
            # ===========================
            # data, origin_shape = to_image_list(imgs[i], size_divisible=32, return_size=True)
            # pred, _ = test_it(model, data, used_df=used_df)
            
            # for j in range(imgs[i].shape[0]):
            #     preds_timei[j, ...] = pred.cpu().numpy()[j, :, :origin_shape[j][0], :origin_shape[j][1]]
        else:
            for j, pt in enumerate(imgs[i]):
                data = [pt]
                data, origin_shape = to_image_list(data, size_divisible=32, return_size=True)
                pred, _ = test_it(model, data)
                preds_timei[j, ...] = pred.cpu().numpy()[0, 0, :origin_shape[0][0], :origin_shape[0][1]]
        
        preds.append(preds_timei)

    return preds
