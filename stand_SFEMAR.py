import torch.nn as nn
import torch


class SFEMARnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SFEMARnet, self).__init__()
        self.SFE1 = SFEBlock(in_channels=1, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.SFE2 = SFEBlock(in_channels=64, out_channels=192)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.MAR1= MARBlock(in_channels=192, out_channels=96, se_channel=96, cbam_channel=96)
        self.MAR2 = MARBlock(in_channels=288, out_channels=144, se_channel=144, cbam_channel=144)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.MAR3 = MARBlock(in_channels=432, out_channels=216, se_channel=216, cbam_channel=216)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.MAR4 = MARBlock(in_channels=648, out_channels=216, se_channel=216, cbam_channel=216)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(648, num_classes)
    def forward(self, x):
        x = self.SFE1(x)
        x = self.maxpool1(x)
        x = self.SFE2(x)
        x = self.maxpool2(x)
        x = self.MAR1(x)
        x = self.MAR2(x)
        x = self.maxpool3(x)
        x = self.MAR3(x)
        x = self.maxpool4(x)
        x = self.MAR4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class MARBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se_channel, cbam_channel, ratio=16):
        super(MARBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(se_channel, se_channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(se_channel // ratio, se_channel, bias=False),
            nn.Sigmoid())
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, se_channel, kernel_size=1),
            BasicConv2d(se_channel, se_channel, kernel_size=3, padding=1),
        )
        # 分支3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, cbam_channel, kernel_size=1),
            BasicConv2d(cbam_channel, cbam_channel, kernel_size=3, padding=1)
        )
        self.branch3_1 = CBAMLayer(cbam_channel,reduction=ratio)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 3, kernel_size=1, stride=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        b, c, h, w = branch2.size()
        scale = self.gap(branch2).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        branch2 = branch2 * scale.expand_as(branch2)

        branch3 = self.branch3(x)
        branch3 = self.branch3_1(branch3)
        outputs = [branch1, branch2, branch3]
        outputs = torch.cat(outputs, 1)
        x = self.conv1(x)
        return (x + outputs)


class SFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SFEBlock, self).__init__()
        self.conv7x7 = BasicConv2d(kernel_size=7, in_channels=in_channels, out_channels=out_channels, padding=3)
        self.conv3x3_0 = BasicConv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels,padding=1)
        self.conv5x5 = BasicConv2d(kernel_size=7, in_channels=in_channels, out_channels=out_channels, padding=3)
        self.conv3x3_1 = BasicConv2d(kernel_size=3, in_channels=out_channels, out_channels=out_channels,padding=1)
        self.conv3x3_2 = BasicConv2d(kernel_size=3, in_channels=out_channels, out_channels=out_channels,padding=1)
        self.conv1 = BasicConv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels)
    def forward(self, x):
        x1x1= self.conv1(x)
        x5x5 = self.conv5x5(x)
        x7x7 = self.conv7x7(x)
        x = self.conv3x3_0(x)
        x = x + x7x7
        x = self.conv3x3_1(x)
        x = x + x5x5+ x1x1
        x=self.conv3x3_2(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x