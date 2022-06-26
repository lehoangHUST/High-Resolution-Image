from torch import nn
from torchinfo import summary
from math import sqrt
import torch
import torchvision.models as models


# Implementation  of Image Super-Resolution Using Deep Convolutional Networks (ECCV 2014)
# arxiv.org/abs/1501.00092
class SRCNN(nn.Module):
    def __init__(self, in_channels):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# Implementation of Image Super-Resolution Using Deep Convolutional Network
# https://arxiv.org/pdf/1608.00367.pdf
class FastSRCNN(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(FastSRCNN, self).__init__()
        # Feature extraction : First Part
        self.conv1 = nn.Conv2d(in_channels, 56, kernel_size=5, padding=2)
        # Mid part: Shrinking + Mapping + Expanding
        self.conv2 = nn.Conv2d(56, 12, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(12, 56, kernel_size=1, padding=0)
        # Last part
        self.deconv = nn.ConvTranspose2d(56, in_channels, 9, (upscale_factor, upscale_factor), (4, 4),
                                         (upscale_factor - 1, upscale_factor - 1), dilation=2)

        # Activation
        self.prelu = nn.PReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        for i in range(4):
            x = self.prelu(self.conv3(x))
        x = self.prelu(self.conv4(x))
        x = self.deconv(x)
        return x

    # The filter weight of each layer is a Gaussian distribution with zero mean and standard deviation initialized by
    # random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


if __name__ == '__main__':
    model = FastSRCNN(in_channels=3, upscale_factor=4)
    summary(model, input_size=(3, 94, 94))
