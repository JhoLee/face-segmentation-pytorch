# Written by jho.lee@kaka.com (https://githbu.com/jholee)
# Date: 20.04.26

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class Fcn32s(nn.Module):

    def __init__(self, n_classes, n_channels=3, height=224, width=224):
        super(Fcn32s, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.height, self.width = height, width

        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score = nn.Sequential(
            nn.Conv2d(4096, self.n_classes, 1)
        )
        self.up_score = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)

    def forward(self, x):
        o = self.conv_1(x)
        o = self.conv_2(o)
        o = self.conv_3(o)
        o = self.conv_4(o)
        o = self.conv_5(o)
        o = self.conv_6(o)
        o = self.conv_7(o)
        o = self.score(o)
        o = self.up_score(o)

        o = o[:, :, 19:19+self.height, 19:19+self.width]
        # o = o

        return o
