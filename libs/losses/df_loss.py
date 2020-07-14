import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def LossSegDF(net_ret, data, device="cuda"):
    net_out, df_out = net_ret

    _, gts, gts_df = data
    gts = torch.squeeze(gts, 1).to(device).long()
    gts_df = gts_df.to(device).long()

    # segmentation Loss
    seg_loss = F.cross_entropy(net_out, gts)

    # direction field Loss
    df_loss = F.mse_loss(df_out, gts_df)

    total_loss = seg_loss + df_loss
    return total_loss

class EuclideanLossWithOHEM(nn.Module):
    def __init__(self, npRatio=3):
        super(EuclideanLossWithOHEM, self).__init__()
        self.npRatio = npRatio
    
    def __cal_weight(self, gt):
        _, H, W = gt.shape  # N=1
        labels = torch.unique(gt, sorted=True)[1:]
        weight = torch.zeros((H, W), dtype=torch.float, device=gt.device)
        posRegion = gt[0, ...] > 0
        posCount = torch.sum(posRegion)
        if posCount != 0:
            segRemain = 0
            for segi in labels:
                overlap_segi = gt[0, ...] == segi
                overlapCount_segi = torch.sum(overlap_segi)
                if overlapCount_segi == 0: continue
                segRemain = segRemain + 1
            segAve = float(posCount) / segRemain
            for segi in labels:
                overlap_segi = gt[0, ...] == segi
                overlapCount_segi = torch.sum(overlap_segi, dtype=torch.float)
                if overlapCount_segi == 0: continue
                pixAve = segAve / overlapCount_segi
                weight = weight * (~overlap_segi).to(torch.float) + pixAve * overlap_segi.to(torch.float)
        # weight = weight[None]
        return weight

    def forward(self, pred, gt_df, gt, weight=None):
        """ pred: (N, C, H, W)
            gt_df: (N, C, H, W)
            gt: (N, 1, H, W)
        """
        # L1 and L2 distance
        N, _, H, W = pred.shape
        distL1 = pred - gt_df
        distL2 = distL1 ** 2

        if weight is None:
            weight = torch.zeros((N, H, W), device=pred.device)
            for i in range(N):
                weight[i] = self.__cal_weight(gt[i])

        # the amount of positive and negtive pixels
        regionPos = (weight > 0).to(torch.float)
        regionNeg = (weight == 0).to(torch.float)
        sumPos = torch.sum(regionPos, dim=(1,2))  # (N,)
        sumNeg = torch.sum(regionNeg, dim=(1,2))

        # the amount of hard negative pixels
        sumhardNeg = torch.min(self.npRatio * sumPos, sumNeg).to(torch.int)  # (N,)

        # set loss on ~(top - sumhardNeg) negative pixels to 0
        lossNeg = (distL2[:,0,...] + distL2[:, 1, ...]) * regionNeg
        lossFlat = torch.flatten(lossNeg, start_dim=1)  # (N, ...)
        arg = torch.argsort(lossFlat, dim=1)
        for i in range(N):
            lossFlat[i, arg[i, :-sumhardNeg[i]]] = 0
        lossHard = lossFlat.view(lossNeg.shape)

        # weight for positive and negative pixels
        weightPos = torch.zeros_like(pred)
        weightNeg = torch.zeros_like(pred)

        weightPos = torch.stack([weight, weight], dim=1)

        weightNeg[:,0,...] = (lossHard != 0).to(torch.float32)
        weightNeg[:,1,...] = (lossHard != 0).to(torch.float32)

        # total loss
        total_loss = torch.sum((distL1 ** 2) * (weightPos + weightNeg)) / N / 2. / torch.sum(weightPos + weightNeg)

        return total_loss



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    criterion = EuclideanLossWithOHEM()
    for i in range(100):
        pred = torch.randn((32, 2, 224, 224)).cuda()
        gt_df = torch.randn((32, 2, 224, 224)).cuda()
        gt = torch.randint(0, 4, (32, 1, 224, 224)).cuda()

        loss = criterion(100*gt_df, gt_df, gt)
        print("{:6} loss:{}".format(i, loss))