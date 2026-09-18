"""
Generate synthetic face images from a trained DCGAN checkpoint.

Usage:
    python generate.py --checkpoint ./checkpoints/dcgan_epoch_050.pt --num_images 64
    python generate.py --checkpoint ./checkpoints/dcgan_epoch_050.pt --num_images 16 --individual --out_dir ./outputs/final
    python generate.py --checkpoint ./checkpoints/dcgan_epoch_050.pt --interpolate --steps 10
"""

import os
import argparse

import torch
import torchvision.utils as vutils

from config import Config
from model import Generator
from utils import save_sample_grid


def parse_args():
    p = argparse.ArgumentParser(description="Generate faces from a trained DCGAN")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_images", type=int, default=64)
    p.add_argument("--out_dir", type=str, default="./outputs/generated")
    p.add_argument("--individual", action="store_true", help="also save each image as its own file")
    p.add_argument("--interpolate", action="store_true", help="latent-space interpolation between two random points")
    p.add_argument("--steps", type=int, default=10, help="number of interpolation steps")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_generator(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {"latent_dim": Config.LATENT_DIM, "image_size": Config.IMAGE_SIZE})
    netG = Generator(cfg["latent_dim"], Config.GEN_FEATURE_MAPS, Config.CHANNELS).to(device)
    netG.load_state_dict(ckpt["netG"])
    netG.eval()
    return netG, cfg["latent_dim"]


def generate_grid(netG, latent_dim, num_images, out_dir, device, individual=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fakes = netG(noise).detach().cpu()

    os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(out_dir, "generated_grid.png")
    nrow = max(1, int(num_images ** 0.5))
    save_sample_grid(fakes, grid_path, nrow=nrow, title="Generated Faces")
    print(f"Saved grid: {grid_path}")

    if individual:
        for idx, img in enumerate(fakes):
            path = os.path.join(out_dir, f"face_{idx:03d}.png")
            vutils.save_image(img, path, normalize=True)
        print(f"Saved {num_images} individual images to {out_dir}")


def generate_interpolation(netG, latent_dim, steps, out_dir, device, seed=None):
    """Linearly interpolate between two random latent vectors to inspect
    how smoothly the learned latent space transitions between faces —
    a meaningfully smooth interpolation is a good sign the model learned
    a structured manifold rather than memorizing/mode-collapsing."""
    if seed is not None:
        torch.manual_seed(seed)
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    alphas = torch.linspace(0, 1, steps)
    zs = torch.cat([(1 - a) * z1 + a * z2 for a in alphas], dim=0)

    with torch.no_grad():
        fakes = netG(zs).detach().cpu()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "interpolation.png")
    save_sample_grid(fakes, path, nrow=steps, title="Latent Space Interpolation")
    print(f"Saved interpolation grid: {path}")


def main():
    args = parse_args()
    device = Config.DEVICE
    netG, latent_dim = load_generator(args.checkpoint, device)
    print(f"Loaded generator from '{args.checkpoint}' (latent_dim={latent_dim}) on {device}")

    if args.interpolate:
        generate_interpolation(netG, latent_dim, args.steps, args.out_dir, device, args.seed)
    else:
        generate_grid(netG, latent_dim, args.num_images, args.out_dir, device, args.individual, args.seed)


if __name__ == "__main__":
    main()
