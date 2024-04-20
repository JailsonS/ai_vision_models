import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Calcular o número de canais para a última camada de upsampling
        self.num_channels_upsample1 = 512
        self.num_channels_upsample2 = 256
        self.num_channels_upsample3 = 128
        
        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(self.num_channels_upsample1, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample2 = nn.ConvTranspose2d(self.num_channels_upsample2, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample3 = nn.ConvTranspose2d(self.num_channels_upsample3, num_classes, kernel_size=4, stride=2, padding=1, bias=False)

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

        # Upsampling
        x = self.upsample1(x)
        x = self.relu(x)
        x = self.upsample2(x)
        x = self.relu(x)
        x = self.upsample3(x)
        x = torch.sigmoid(x)

        return x
