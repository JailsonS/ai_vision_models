import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.stride = stride

        #init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        #init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or identity.size(1) != out.size(1):
            identity = F.avg_pool2d(identity, kernel_size=self.stride, stride=self.stride)
            padding = torch.zeros(identity.size(0), out.size(1)-identity.size(1), identity.size(2), identity.size(3))
            identity = torch.cat((identity, padding), dim=1)

        out += identity
        out = self.relu(out)

        return out

class ResNetSemanticSegmentation(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(ResNetSemanticSegmentation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(BasicBlock, in_channels=64, out_channels=64, blocks=3)
        self.layer2 = self.make_layer(BasicBlock, in_channels=64, out_channels=128, blocks=4, stride=2)
        self.layer3 = self.make_layer(BasicBlock, in_channels=128, out_channels=256, blocks=6, stride=2)
        self.layer4 = self.make_layer(BasicBlock, in_channels=256, out_channels=512, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((512,512))
        self.fc = nn.Conv2d(512, num_classes, kernel_size=1)

        #init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='leaky_relu') 

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = torch.sigmoid(x, dim=1)
        return x
