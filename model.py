"""
DCGAN architectures for the Generator and Discriminator.

Follows the architectural guidelines from Radford et al. (2015),
"Unsupervised Representation Learning with Deep Convolutional GANs":
  - Replace pooling with strided convolutions (Discriminator) and
    fractional-strided convolutions (Generator).
  - Use BatchNorm in both G and D (except G's output layer and D's input layer).
  - Remove fully connected hidden layers for deeper architectures.
  - ReLU activation in G for all layers except output (Tanh).
  - LeakyReLU activation in D for all layers.
"""

import torch
import torch.nn as nn


def weights_init(m):
    """
    DCGAN paper initializes weights from N(0, 0.02) for Conv/ConvTranspose/BatchNorm.
    Proper initialization measurably stabilizes early training.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Maps a latent vector z (LATENT_DIM x 1 x 1) to a synthetic image
    (CHANNELS x IMAGE_SIZE x IMAGE_SIZE) via a stack of transposed convolutions.

    Spatial map grows: 1 -> 4 -> 8 -> 16 -> 32 -> 64
    Channel depth shrinks: 8*ngf -> 4*ngf -> 2*ngf -> ngf -> CHANNELS
    """

    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super().__init__()
        ngf = feature_maps
        self.main = nn.Sequential(
            # input: (latent_dim) x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state: (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # output: (channels) x 64 x 64, pixel values in [-1, 1]
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    Binary classifier that maps an image (CHANNELS x IMAGE_SIZE x IMAGE_SIZE)
    to a single real/fake probability, via a stack of strided convolutions.

    Spatial map shrinks: 64 -> 32 -> 16 -> 8 -> 4 -> 1
    Channel depth grows: CHANNELS -> ndf -> 2*ndf -> 4*ndf -> 8*ndf -> 1
    """

    def __init__(self, feature_maps=64, channels=3):
        super().__init__()
        ndf = feature_maps
        self.main = nn.Sequential(
            # input: (channels) x 64 x 64
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf) x 32 x 32   -- no BatchNorm on the input layer (per DCGAN paper)

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
            # output: 1 x 1 x 1 -> probability the input is real
        )

    def forward(self, img):
        return self.main(img).view(-1, 1).squeeze(1)
