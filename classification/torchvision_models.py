from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from PIL import Image

from torchvision import datasets, transforms, models

import efemarai as ef

parser = ArgumentParser(description="Visualising ImageNet models with Efemarai")
parser.add_argument("-a", "--arch", metavar="ARCH", default="mobilenet_v3_small", help="Choose an architecture")
parser.add_argument("--gpu", default=False, action="store_true", help="Use gpu")
parser.add_argument("-i", "--image", default=None, help="Optional Input image")

def main():
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")

    model = models.__dict__[args.arch](pretrained=True)

    model.transform_input = False
    model.eval()

    model.to(device)

    img_transforms = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    if args.image:
        img = Image.open(args.image)
        img = img_transforms(img).unsqueeze(0).to(device)

        with ef.scan(max_size="10mb"):
            out = model(img)
    else:
        data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                "data/",
                transform=img_transforms,
            ),
            batch_size=1,
            shuffle=True,
        )


        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            ef.add_view(data, ef.View.Image)

            with ef.scan(idx, wait=True, show=True):
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()


if __name__ == "__main__":
    print(models.__dict__.keys())
    main()
