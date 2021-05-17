import torch
from vit_pytorch import ViT
import requests
from torchvision import transforms as T
from PIL import Image
import efemarai as ef


vitmodel = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 100,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Load an image
url = "https://data.efemarai.com/samples/dino.jpg"
img = Image.open(requests.get(url, stream=True).raw).convert('RGB').resize((256, 256))
img = T.ToTensor()(img).unsqueeze(dim=0)

# Run the model
with ef.scan(max_size="10mb"):
    preds = vitmodel(img)
    print(preds)
