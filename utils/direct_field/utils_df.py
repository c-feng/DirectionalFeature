import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from utils.utils_loss import one_hot, simplex, class2one_hot

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2dist(seg: np.ndarray, C=4) -> np.ndarray:
    """ res: (C, H, W)
    """
    if seg.ndim == 2:
        seg_tensor = torch.Tensor(seg)
    elif seg.ndim == 3:
        seg_tensor = torch.Tensor(seg[0])
    elif seg.ndim == 4:
        seg_tensor = torch.Tensor(seg[0, ..., 0])

    seg_onehot = class2one_hot(seg_tensor, C).to(torch.float32)

    assert simplex(seg_onehot)
    res = one_hot2dist(seg_onehot[0].numpy())
    return res

