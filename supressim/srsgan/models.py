import torch
import torch.nn as nn
import torch.nn.functional as F
from supressim.srsgan.pixelshuffle import PixelShuffle


def narrow_like(a, b):
    """Narrow a to be like b.
    """
    for dim in range(2, 5):
        half_width = (a.shape[dim] - b.shape[dim]) // 2
        a = a.narrow(dim, half_width, a.shape[dim] - 2 * half_width)
    return a


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        #self.conv_block = nn.Sequential(
        #    nn.Conv3d(in_channels, in_channels, 3),
        #    # NOTE why is the eps so large in BatchNorm3d?
        #    nn.BatchNorm3d(in_channels, 0.8),
        #    nn.PReLU(),
        #    nn.Conv3d(in_channels, in_channels, 3),
        #    nn.BatchNorm3d(in_channels, 0.8),
        #)
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels, in_channels, 3)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        #y = self.conv_block(x)
        #x = narrow_like(x, y)
        #return x + y  # NOTE activation?

        y = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        y = narrow_like(y, x)
        x += y
        x = self.act(x)

        return x


class GeneratorResNet(nn.Module):
    #def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=4):
        super(GeneratorResNet, self).__init__()

        #self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 32, 5), nn.PReLU())
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 32, 5), nn.BatchNorm3d(32), nn.LeakyReLU())

        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(32))
        self.res_blocks = nn.Sequential(*res_blocks)

        #self.conv2 = nn.Sequential(nn.Conv3d(32, 32, 3), nn.BatchNorm3d(32, 0.8))
        self.conv2 = nn.Sequential(nn.Conv3d(32, out_channels, 3), nn.BatchNorm3d(out_channels), nn.LeakyReLU())

        #upsampling = []
        #for _ in range(1):
        #    upsampling += [
        #        nn.Conv3d(32, 256, 3),
        #        nn.BatchNorm3d(256),
        #        PixelShuffle(upscale_factor=2),
        #        nn.PReLU(),
        #    ]
        #self.upsampling = nn.Sequential(*upsampling)

        #self.conv3 = nn.Conv3d(32, out_channels, 3)
        self.conv3 = nn.Conv3d(out_channels, out_channels, 3)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                # NOTE why Gaby use 1 to init bias
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        #out1 = self.conv1(x)
        #out = self.res_blocks(out1)
        #out2 = self.conv2(out)
        #out1 = narrow_like(out1, out2)
        #out = out1 + out2
        #out = self.upsampling(out)
        #out = self.conv3(out)
        #return out

        x = F.interpolate(x, scale_factor=2, mode='nearest')

        y = x

        x = self.conv1(x)

        x = self.res_blocks(x)

        x = self.conv2(x)

        y = narrow_like(y, x)
        x += y

        x = self.conv3(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, n_blocks=4):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, first_block=False):
            layers = []
            layers.append(nn.Conv3d(in_channels, out_channels, 3))
            if not first_block:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv3d(out_channels, out_channels, 3, stride=2))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        block_out_channels = [(32 << i) for i in range(n_blocks)]
        for i, out_channels in enumerate(block_out_channels):
            layers.extend(discriminator_block(in_channels, out_channels, first_block=(i == 0)))
            in_channels = out_channels

        layers.append(nn.Conv3d(out_channels, 1, 3))

        self.model = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                # NOTE why Gaby use 1 to init bias
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, img):
        return self.model(img)
