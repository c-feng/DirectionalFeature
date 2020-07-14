import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libs.network.unet import conv_block, up_conv

class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=5, n_class=4, auxseg=False):
        super(SelFuseFeature, self).__init__()
        
        self.shift_n = shift_n
        self.n_class = n_class
        self.auxseg = auxseg
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(inplace=True),
                                    )
        if auxseg:
            self.auxseg_conv = nn.Conv2d(in_channels, self.n_class, 1)
        

    def forward(self, x, df):
        N, _, H, W = df.shape
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0

        scale = 1.
        
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        grid = grid + scale * df

        grid = grid.permute(0, 2, 3, 1).transpose(1, 2)
        grid_ = grid + 0.
        grid[...,0] = 2*grid_[..., 0] / (H-1) - 1
        grid[...,1] = 2*grid_[..., 1] / (W-1) - 1

        # features = []
        select_x = x.clone()
        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border')
            # features.append(select_x)
        # select_x = torch.mean(torch.stack(features, dim=0), dim=0)
        # features.append(select_x.detach().cpu().numpy())
        # np.save("/root/chengfeng/Cardiac/source_code/logs/acdc_logs/logs_temp/feature.npy", np.array(features))
        if self.auxseg:
            auxseg = self.auxseg_conv(x)
        else:
            auxseg = None

        select_x = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return [select_x, auxseg]

class U_NetDF(nn.Module):
    def __init__(self,img_ch=1,num_class=4, selfeat=False, shift_n=5, auxseg=False):
        super(U_NetDF,self).__init__()

        self.selfeat = selfeat
        self.shift_n = shift_n
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        # Direct Field
        self.ConvDf_1x1 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)

        if selfeat:
            self.SelDF = SelFuseFeature(64, auxseg=auxseg, shift_n=shift_n)

        self.Conv_1x1 = nn.Conv2d(64,num_class,kernel_size=1,stride=1,padding=0)

    def forward(self, inputs):
        x = inputs

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        # df = self.ConvDf_1x1(d3)
        # # df = F.interpolate(inputs[1], size=d3.shape[-2:], mode='bilinear', align_corners=True)
        # if self.selfeat:
        #     d3 = self.SelDF(d3, df)


        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        # Direct Field
        df = self.ConvDf_1x1(d2)
        # df = None
        if self.selfeat:
            d2_auxseg = self.SelDF(d2, df)
            d2, auxseg = d2_auxseg[:2]
        else:
            auxseg = None

        # df = F.interpolate(df, size=x.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.Conv_1x1(d2)

        return [d1, df, auxseg]



if __name__ == "__main__":

    a = torch.randn(1, 1, 224, 224)

    model = U_NetDF(selfeat=True)

    out = model(a)
    print(out[0].shape, out[1].shape, out[2].shape)

