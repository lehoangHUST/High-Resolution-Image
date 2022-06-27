import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import cv2
import os
from SR.models import SRCNN
from utils.loss import calc_psnr


def load_model(file_weight: str):
    if os.path.exists(file_weight):
        model = SRCNN(in_channels=3)
        model.load_state_dict(torch.load(file_weight), map_location='cpu')
        model.eval()
    return model


def proc_img(path: str):
    # Pre-processing data
    image = pil_image.open(path).convert('RGB')

    image_width, image_height = image.width * args.scale, image.height * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return image


def run(args):
    model = load_model(args.weights)
    with torch.no_grad():
        if args.image is not None:
            image = proc_img(args.image)
            preds = model(image)
            preds = preds[0].permute(1, 2, 0)
            preds = preds.mul(255.0).cpu().numpy()
            preds = np.array(preds, dtype=np.int16)
        elif args.images is not None:
            for path in os.listdir(args.images):
                path_img = os.path.join(args.images, path)
                image = proc_img(args.image)
                preds = model(image)
                preds = preds[0].permute(1, 2, 0)
                preds = preds.mul(255.0).cpu().numpy()
                preds = np.array(preds, dtype=np.int16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--image', type=str, default=None,
                        help='Super resolution for image')
    parser.add_argument('--images', type=str, default=None,
                        help='Super resolution for each image in dir')
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()
    run(args)

