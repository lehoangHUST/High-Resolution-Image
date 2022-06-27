import torch
import numpy as np
from skimage.metrics import structural_similarity


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def ssim(img1, img2) -> float:
    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        # Convert torch -> numpy
        img1, img2 = np.array(img1), np.array(img2)
    return structural_similarity(img1, img2)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count