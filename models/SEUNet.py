import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SEUNet"]


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=8):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):

        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):

        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=True):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.ReLU(inplace=True)
        )


def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


class SEUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, useBN=True, useCSE=True, useSSE=False, useCSSE=False):
        super(SEUNet, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.useCSE = useCSE
        self.useSSE = useSSE
        self.useCSSE = useCSSE

        self.conv1 = add_conv_stage(self.in_channels, nb_filter[0], useBN=useBN)
        self.conv2 = add_conv_stage(nb_filter[0], nb_filter[1], useBN=useBN)
        self.conv3 = add_conv_stage(nb_filter[1], nb_filter[2], useBN=useBN)
        self.conv4 = add_conv_stage(nb_filter[2], nb_filter[3], useBN=useBN)
        self.conv5 = add_conv_stage(nb_filter[3], nb_filter[4], useBN=useBN)

        self.conv4m = add_conv_stage(nb_filter[3] + nb_filter[3], nb_filter[3], useBN=useBN)
        self.conv3m = add_conv_stage(nb_filter[2] + nb_filter[2], nb_filter[2], useBN=useBN)
        self.conv2m = add_conv_stage(nb_filter[1] + nb_filter[1], nb_filter[1], useBN=useBN)
        self.conv1m = add_conv_stage(nb_filter[0] + nb_filter[0], nb_filter[0], useBN=useBN)

        self.conv0 = nn.Conv2d(nb_filter[0], self.out_channels, 3, 1, 1)

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(nb_filter[4], nb_filter[3])
        self.upsample43 = upsample(nb_filter[3], nb_filter[2])
        self.upsample32 = upsample(nb_filter[2], nb_filter[1])
        self.upsample21 = upsample(nb_filter[1], nb_filter[0])

        self.cse1 = ChannelSELayer(nb_filter[0], 2)
        self.cse2 = ChannelSELayer(nb_filter[1], 2)
        self.cse3 = ChannelSELayer(nb_filter[2], 2)
        self.cse4 = ChannelSELayer(nb_filter[3], 2)

        self.cse4m = ChannelSELayer(nb_filter[3], 2)
        self.cse3m = ChannelSELayer(nb_filter[2], 2)
        self.cse2m = ChannelSELayer(nb_filter[1], 2)
        self.cse1m = ChannelSELayer(nb_filter[0], 2)

        self.sse1 = SpatialSELayer(nb_filter[0])
        self.sse2 = SpatialSELayer(nb_filter[1])
        self.sse3 = SpatialSELayer(nb_filter[2])
        self.sse4 = SpatialSELayer(nb_filter[3])

        self.sse4m = SpatialSELayer(nb_filter[3])
        self.sse3m = SpatialSELayer(nb_filter[2])
        self.sse2m = SpatialSELayer(nb_filter[1])
        self.sse1m = SpatialSELayer(nb_filter[0])

        self.csse1 = ChannelSpatialSELayer(nb_filter[0], 2)
        self.csse2 = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse3 = ChannelSpatialSELayer(nb_filter[2], 2)
        self.csse4 = ChannelSpatialSELayer(nb_filter[3], 2)

        self.csse4m = ChannelSpatialSELayer(nb_filter[3], 2)
        self.csse3m = ChannelSpatialSELayer(nb_filter[2], 2)
        self.csse2m = ChannelSpatialSELayer(nb_filter[1], 2)
        self.csse1m = ChannelSpatialSELayer(nb_filter[0], 2)

    def forward(self, x):
        if (self.useCSSE):
            conv1_ = self.csse1(self.conv1(x))
            conv2_ = self.csse2(self.conv2(self.max_pool(conv1_)))
            conv3_ = self.csse3(self.conv3(self.max_pool(conv2_)))
            conv4_ = self.csse4(self.conv4(self.max_pool(conv3_)))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.csse4m(self.conv4m(conv5_))

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.csse3m(self.conv3m(conv4_))

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.csse2m(self.conv2m(conv3_))

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.csse1m(self.conv1m(conv2_))

            conv0_ = self.conv0(conv1_)

        elif (self.useCSE):
            conv1_ = self.cse1(self.conv1(x))
            conv2_ = self.cse2(self.conv2(self.max_pool(conv1_)))
            conv3_ = self.cse3(self.conv3(self.max_pool(conv2_)))
            conv4_ = self.cse4(self.conv4(self.max_pool(conv3_)))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.cse4m(self.conv4m(conv5_))

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.cse3m(self.conv3m(conv4_))

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.cse2m(self.conv2m(conv3_))

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.cse1m(self.conv1m(conv2_))

            conv0_ = self.conv0(conv1_)

        elif (self.useSSE):
            conv1_ = self.sse1(self.conv1(x))
            conv2_ = self.sse2(self.conv2(self.max_pool(conv1_)))
            conv3_ = self.sse3(self.conv3(self.max_pool(conv2_)))
            conv4_ = self.sse4(self.conv4(self.max_pool(conv3_)))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.sse4m(self.conv4m(conv5_))

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.sse3m(self.conv3m(conv4_))

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.sse2m(self.conv2m(conv3_))

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.sse1m(self.conv1m(conv2_))

            conv0_ = self.conv0(conv1_)

        else:
            conv1_ = self.conv1(x)
            conv2_ = self.conv2(self.max_pool(conv1_))
            conv3_ = self.conv3(self.max_pool(conv2_))
            conv4_ = self.conv4(self.max_pool(conv3_))
            conv5_ = self.conv5(self.max_pool(conv4_))

            conv5_ = torch.cat((self.upsample54(conv5_), conv4_), 1)
            conv4_ = self.conv4m(conv5_)

            conv4_ = torch.cat((self.upsample43(conv4_), conv3_), 1)
            conv3_ = self.conv3m(conv4_)

            conv3_ = torch.cat((self.upsample32(conv3_), conv2_), 1)
            conv2_ = self.conv2m(conv3_)

            conv2_ = torch.cat((self.upsample21(conv2_), conv1_), 1)
            conv1_ = self.conv1m(conv2_)

            conv0_ = self.conv0(conv1_)

        return conv0_