import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np 

from PIL import Image
from torchvision import transforms

class Model(nn.Module):
    def __init__(self,):    
        super(Model, self).__init__()

        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.features[-1] = torchvision.models.mobilenet.ConvBNReLU(320, 256, kernel_size=1)

        self.feature_extractor = model.features
        # While the pretrained models of torchvision are trained using
        # images with RGB channel order, in this repository images are
        # treated as BGR channel order.
        # Therefore, reverse the channel order of the first convolutional
        # layer.
        module = getattr(self.feature_extractor, '0')
        module.weight.data = module.weight.data[:, [2, 1, 0]]

        self.backbone = model.features

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   
        self.fc_final = nn.Linear(512, 2)

        self._initialize_weight()
        self._initialize_bias()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.001)

    def _initialize_bias(self):
        nn.init.constant_(self.conv1.bias, val=0.1)
        nn.init.constant_(self.conv2.bias, val=0.1)
        nn.init.constant_(self.conv3.bias, val=1)

    def forward(self, x):
        x = self.backbone(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        
        x = F.dropout(F.relu(torch.mul(x, y)), 0.5)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        gaze = self.fc_final(x)

        return gaze

    def get_gaze(self, img):
        img = Image.fromarray(img)
        img = self.preprocess(img)[np.newaxis,:,:,:]
        x = self.forward(img.to(self.device))
        return x