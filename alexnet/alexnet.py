import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import requests
from PIL import Image

import efemarai as ef


class AlexNet(nn.Module):
    """docstring for AlexNet"""
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__()
        
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.ll1 = nn.Linear(256 * 6 * 6, 4096)
        self.ll2 = nn.Linear(4096, 4096)
        self.ll3 = nn.Linear(4096, num_classes)
        

    def feature_extraction(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=3, stride=2)
        return x

    def classifier(self, features):
        x = nn.Dropout()(features)
        x = nn.Dropout()(F.relu(self.ll1(x)))
        x = F.relu(self.ll2(x))
        x = self.ll3(x)
        return x

    def forward(self, x):
        features = self.feature_extraction(x)
        features_pool = F.adaptive_avg_pool2d(features, (6, 6))
        features_pool_flat = torch.flatten(features_pool, 1)
        output = self.classifier(features_pool_flat)
        return output


url="https://data.efemarai.com/samples/dino.jpg"
img = Image.open(requests.get(url, stream=True).raw).convert('RGB').resize((224, 224))

img = T.ToTensor()(img).unsqueeze(dim=0)

net = AlexNet(10)

with ef.scan(max_size="10mb"):
    net_out = net(img)
