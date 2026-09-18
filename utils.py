"""
Helper utilities: sample-grid saving, loss-curve plotting, and a small
JSON-based training-history logger used to analyze training behavior
(mode collapse, instability) after the run finishes.
"""

import os
import json
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def save_sample_grid(images, path, nrow=8, title=None):
    """Save a grid of generated images (tensor in [-1, 1]) to disk as a PNG."""
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True)
    npimg = grid.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path, bbox_inches="tight")
    plt.close()


class TrainingHistory:
    """
    Tracks per-batch generator/discriminator losses and D's average confidence
    on real vs. fake images (D(x) and D(G(z))). These three signals are the
    standard diagnostics for spotting instability and mode collapse:

      - D(x) staying near 1.0 and D(G(z)) staying near 0.0 for many epochs
        means D has "won" and G's gradients are vanishing (stalled training).
      - G_loss oscillating wildly or spiking is a sign of instability.
      - Low sample diversity despite low G_loss suggests mode collapse
        (visually checked via the saved sample grids).
    """

    def __init__(self):
        self.g_losses = []
        self.d_losses = []
        self.d_x = []       # D's average output on real images
        self.d_g_z1 = []    # D's average output on fake images, before G's update
        self.d_g_z2 = []    # D's average output on fake images, after G's update
        self.epochs = []

    def log(self, epoch, g_loss, d_loss, d_x, d_g_z1, d_g_z2):
        self.epochs.append(epoch)
        self.g_losses.append(float(g_loss))
        self.d_losses.append(float(d_loss))
        self.d_x.append(float(d_x))
        self.d_g_z1.append(float(d_g_z1))
        self.d_g_z2.append(float(d_g_z2))

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump({
                "epochs": self.epochs,
                "g_losses": self.g_losses,
                "d_losses": self.d_losses,
                "d_x": self.d_x,
                "d_g_z1": self.d_g_z1,
                "d_g_z2": self.d_g_z2,
            }, f, indent=2)

    def plot_losses(self, path):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.g_losses, label="G", alpha=0.8)
        plt.plot(self.d_losses, label="D", alpha=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_discriminator_confidence(self, path):
        plt.figure(figsize=(10, 5))
        plt.title("Discriminator Confidence: D(x) vs D(G(z))")
        plt.plot(self.d_x, label="D(x)  [real]", alpha=0.8)
        plt.plot(self.d_g_z1, label="D(G(z)) pre-G-update  [fake]", alpha=0.8)
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Ideal equilibrium (0.5)")
        plt.xlabel("Iteration")
        plt.ylabel("Average D output")
        plt.legend()
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
