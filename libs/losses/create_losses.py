import torch
import torch.nn as nn

from libs.losses.df_loss import EuclideanLossWithOHEM
from libs.losses.mag_angle_loss import EuclideanAngleLossWithOHEM
from libs.losses.surface_loss import SurfaceLoss

class Total_loss():
    def __init__(self, boundary=False):
        self.df_loss = EuclideanAngleLossWithOHEM()
        self.boundary = boundary
        if boundary:
            self.boundary_loss = SurfaceLoss(idc=[1,2,3])
        
    def __call__(self, net_logit, dist_maps, df_out, gts_df, gts):
        df_loss = self.df_loss(df_out, gts_df, gts[:, None, ...])

        if self.boundary:
            net_prob = nn.functional.softmax(net_logit, dim=1)
            b_loss = self.boundary_loss(net_prob, dist_maps, gts)
        else:
            b_loss = torch.tensor([0.], device=net_logit.device)

        return df_loss, b_loss

