import torch
import torch.nn as nn
import math


def cart2polar(coord):
    """ coord: (N, 2, ...)
    """
    x = coord[:, 0, ...]
    y = coord[:, 1, ...]

    theta = torch.atan(y / (x + 1e-12))

    theta = theta + (x < 0).to(coord.dtype) * math.pi
    theta = theta + ((x > 0).to(coord.dtype) * (y < 0).to(coord.dtype)) * 2 * math.pi
    return theta / (2 * math.pi)

class EuclideanAngleLossWithOHEM(nn.Module):
    def __init__(self, npRatio=3):
        super(EuclideanAngleLossWithOHEM, self).__init__()
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

        theta_p = cart2polar(pred)
        theta_g = cart2polar(gt_df)
        angleDistL1 = theta_g - theta_p


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

        # angle loss on ~(top - sumhardNeg) negative pixels to 0
        angleLossNeg = (angleDistL1 ** 2) * regionNeg
        angleLossNegFlat = torch.flatten(angleLossNeg, start_dim=1)  # (N, ...)


        # set loss on ~(top - sumhardNeg) negative pixels to 0
        lossNeg = (distL2[:,0,...] + distL2[:, 1, ...]) * regionNeg
        lossFlat = torch.flatten(lossNeg, start_dim=1)  # (N, ...)
        
        # l2-norm distance and angle distance
        lossFlat = lossFlat + angleLossNegFlat
        arg = torch.argsort(lossFlat, dim=1)
        for i in range(N):
            lossFlat[i, arg[i, :-sumhardNeg[i]]] = 0
        lossHard = lossFlat.view(lossNeg.shape)

        # weight for positive and negative pixels
        weightPos = torch.zeros_like(gt, dtype=pred.dtype)
        weightNeg = torch.zeros_like(gt, dtype=pred.dtype)

        weightPos = weight.clone()

        weightNeg[:,0,...] = (lossHard != 0).to(torch.float32)

        # total loss
        total_loss = torch.sum(((distL2[:,0,...] + distL2[:, 1, ...]) + angleDistL1 ** 2) *
                               (weightPos + weightNeg)) / N / 2. / torch.sum(weightPos + weightNeg)

        return total_loss



if __name__ == "__main__":
    import os
    import torch.nn as nn
    import torch.optim as optim

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    criterion = EuclideanAngleLossWithOHEM()

    # models = nn.Sequential(nn.Conv2d(2, 2, 1),
    #                        nn.ReLU())
    # models.to(device="cuda")

    # epoch_n = 200
    # learning_rate = 1e-4

    # optimizer = optim.Adam(params=models.parameters(), lr=learning_rate)

    # for i in range(100):
    #     pred = torch.randn((32, 2, 224, 224)).cuda()
    #     gt_df = torch.randn((32, 2, 224, 224)).cuda()
    #     gt = torch.randint(0, 4, (32, 1, 224, 224)).cuda()

    #     pred = models(gt_df)
    #     loss = criterion(pred, gt_df, gt)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    #     print("{:6} loss:{}".format(i, loss))

    for i in range(100):
        pred = torch.randn((32, 2, 224, 224)).cuda()
        gt_df = torch.randn((32, 2, 224, 224)).cuda()
        gt = torch.randint(0, 4, (32, 1, 224, 224)).cuda()

        loss = criterion(-gt_df, gt_df, gt)
        print("{:6} loss:{}".format(i, loss))
