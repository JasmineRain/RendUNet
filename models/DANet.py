import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter
from torch.nn import Module, Conv2d, Parameter, Softmax

import dilated as resnet


__all__ = ["DANet"]


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, dilated=True, norm_layer=None):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False, dilated=dilated, norm_layer=norm_layer)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False, dilated=dilated, norm_layer=norm_layer)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False, dilated=dilated, norm_layer=norm_layer)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        # print(x.shape)
        x = self.pretrained.conv1(x)
        # print(x.shape)
        x = self.pretrained.bn1(x)
        # print(x.shape)
        x = self.pretrained.relu(x)
        # print(x.shape)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4


class DANet(BaseNet):

    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d):
        super(DANet, self).__init__(nclass, backbone, norm_layer=norm_layer)
        self.head = DANetHead(2048, 512, norm_layer)

        self.seg = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1),
                                 nn.PixelShuffle(upscale_factor=8))

        self.seg1 = nn.Sequential(nn.Dropout(0.1),
                                  nn.Conv2d(512, nclass, 1))

        self.seg2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                  norm_layer(512),
                                  nn.ReLU(inplace=True),
                                  nn.PixelShuffle(upscale_factor=8),
                                  nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
                                  norm_layer(8),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.1),
                                  nn.Conv2d(8, nclass, kernel_size=1, stride=1, padding=0)
                                  )


    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)

        output = self.seg1(x)
        output = upsample(output, imsize, mode='bilinear', align_corners=False)

        return output


class DANetHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU(inplace=True))

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        # sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        return feat_sum


if __name__ == "__main__":
    net = DANet(backbone='resnet101', nclass=1)
    img = torch.rand(1, 3, 384, 384)
    net(img)
