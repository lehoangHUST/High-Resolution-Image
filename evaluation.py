import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import cv2
from SR.models import SRCNN
from utils.loss import calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    model = SRCNN(in_channels=3)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    # Pre-processing data
    image = pil_image.open(args.image).convert('RGB')

    image_width = image.width * args.scale
    image_height = image.height * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image.show()
    image = np.array(image).astype(np.float32) / 255
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)

    with torch.no_grad():
        preds = model(image)

    preds = preds[0].permute(1, 2, 0)
    preds = preds.mul(255.0).cpu().numpy()
    preds = np.array(preds, dtype=np.int16)
    print(preds.shape)
    plt.imshow(preds)
    plt.show()


