import requests
from PIL import Image
from torchvision import transforms as T
import torch

import efemarai as ef

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=False)


url="https://data.efemarai.com/samples/dino.jpg"
img = Image.open(requests.get(url, stream=True).raw).convert('RGB').resize((224, 224))

img = T.ToTensor()(img).unsqueeze(dim=0)


with ef.scan(max_size="10mb"):
    output = model(img)
