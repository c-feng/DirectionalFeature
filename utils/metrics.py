import torch
import numpy as np
from hausdorff import hausdorff_distance
from medpy.metric.binary import hd, dc

def dice(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    smooth = 0.00001

    # intersection = (pred * target).sum(dim=2).sum(dim=2)
    pred_flat = pred.view(1, -1)
    target_flat = target.view(1, -1)

    intersection = (pred_flat * target_flat).sum().item()

    # loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    dice = (2 * intersection + smooth) / (pred_flat.sum().item() + target_flat.sum().item() + smooth)
    return dice

def dice3D(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        # volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        # volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        # res += [dice, volpred, volpred-volgt]
        res += [dice]

    return res

def hd_3D(img_pred, img_gt, labels=[3, 1, 2]):
    res = []
    for c in labels:
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        if np.sum(pred_c_i) == 0 or np.sum(gt_c_i) == 0:
            hausdorff = 0
        else:
            hausdorff = hd(pred_c_i, gt_c_i)

        res += [hausdorff]

    return res

def cal_hausdorff_distance(pred,target):

    pred = np.array(pred.contiguous())
    target = np.array(target.contiguous())
    result = hausdorff_distance(pred,target,distance="euclidean")

    return result

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

def match_pred_gt(pred, gt):
    """ pred: (1, C, H, W)
        gt: (1, C, H, W)
    """
    gt_labels = torch.unique(gt, sorted=True)[1:]
    pred_labels = torch.unique(pred, sorted=True)[1:]

    if len(gt_labels) != 0 and len(pred_labels) != 0:
        dice_Matrix = torch.zeros((len(pred_labels), len(gt_labels)))
        for i, pl in enumerate(pred_labels):
            pred_i = torch.tensor(pred==pl, dtype=torch.float)
            for j, gl in enumerate(gt_labels):
                dice_Matrix[i, j] = dice(make_one_hot(pred_i, 2)[0], make_one_hot(gt==gl, 2)[0])

        # max_axis0 = np.max(dice_Matrix, axis=0)
        max_arg0 = np.argmax(dice_Matrix, axis=0)
    else:
        return torch.zeros_like(pred)

    pred_match = torch.zeros_like(pred)
    for i, arg in enumerate(max_arg0):
        pred_match[pred==pred_labels[arg]] = i + 1
    return pred_match

if __name__ == "__main__":
    npy_path = "/home/fcheng/Cardia/source_code/logs/logs_df_50000/eval_pp_test/200.npy"
    pred_df, gt_df = np.load(npy_p)
