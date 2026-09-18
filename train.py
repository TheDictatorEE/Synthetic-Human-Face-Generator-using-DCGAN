"""
Train a DCGAN on a folder of face images.

Usage:
    python train.py
    python train.py --data_dir ./data/celeba --epochs 30 --batch_size 128

All defaults live in config.py; CLI args override them for convenience.
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from model import Generator, Discriminator, weights_init
from dataset import get_dataloader
from utils import save_sample_grid, TrainingHistory, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a DCGAN face generator")
    p.add_argument("--data_dir", type=str, default=Config.DATA_DIR)
    p.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=Config.LR)
    p.add_argument("--image_size", type=int, default=Config.IMAGE_SIZE)
    p.add_argument("--latent_dim", type=int, default=Config.LATENT_DIM)
    p.add_argument("--resume", type=str, default=None, help="path to a checkpoint .pt to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SAMPLES_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    # ---- Data ----
    dataloader, dataset = get_dataloader(
        args.data_dir, args.image_size, args.batch_size, Config.NUM_WORKERS
    )
    print(f"Loaded {len(dataset)} images from '{args.data_dir}' "
          f"({len(dataloader)} batches/epoch at batch_size={args.batch_size})")

    # ---- Models ----
    netG = Generator(args.latent_dim, Config.GEN_FEATURE_MAPS, Config.CHANNELS).to(device)
    netD = Discriminator(Config.DISC_FEATURE_MAPS, Config.CHANNELS).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from '{args.resume}' at epoch {start_epoch}")

    # ---- Loss & Optimizers ----
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(Config.BETA1, Config.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(Config.BETA1, Config.BETA2))

    real_label_val = Config.LABEL_SMOOTHING
    fake_label_val = 0.0

    # Fixed noise vector used every epoch to visualize G's progress over time
    fixed_noise = torch.randn(Config.FIXED_NOISE_SIZE, args.latent_dim, 1, 1, device=device)

    history = TrainingHistory()
    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # =========================================================
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # =========================================================
            netD.zero_grad()

            # -- real batch --
            label = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, label)
            lossD_real.backward()
            D_x = output_real.mean().item()

            # -- fake batch --
            noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            label.fill_(fake_label_val)
            output_fake = netD(fake_imgs.detach())
            lossD_fake = criterion(output_fake, label)
            lossD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # =========================================================
            # (2) Update Generator: maximize log(D(G(z)))
            #     (non-saturating trick: train G to fool D, using "real" labels)
            # =========================================================
            netG.zero_grad()
            label.fill_(1.0)  # G wants D to think fakes are real
            output = netD(fake_imgs)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            history.log(epoch, lossG.item(), lossD.item(), D_x, D_G_z1, D_G_z2)

            if i % Config.LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(f"[{elapsed:7.1f}s] Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{i}/{len(dataloader)}] "
                      f"Loss_D: {lossD.item():.4f}  Loss_G: {lossG.item():.4f}  "
                      f"D(x): {D_x:.3f}  D(G(z)): {D_G_z1:.3f} / {D_G_z2:.3f}")

        # ---- End of epoch: save sample grid & checkpoint ----
        if (epoch + 1) % Config.SAMPLE_INTERVAL == 0:
            netG.eval()
            with torch.no_grad():
                fake_samples = netG(fixed_noise).detach().cpu()
            netG.train()
            sample_path = os.path.join(Config.SAMPLES_DIR, f"epoch_{epoch+1:03d}.png")
            save_sample_grid(fake_samples, sample_path, nrow=8, title=f"Epoch {epoch+1}")
            print(f"  -> saved sample grid: {sample_path}")

        if (epoch + 1) % Config.CHECKPOINT_INTERVAL == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"dcgan_epoch_{epoch+1:03d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "config": {"latent_dim": args.latent_dim, "image_size": args.image_size},
            }, ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")

    # ---- Save training history + diagnostic plots ----
    history.save_json(os.path.join(Config.PLOTS_DIR, "history.json"))
    history.plot_losses(os.path.join(Config.PLOTS_DIR, "loss_curve.png"))
    history.plot_discriminator_confidence(os.path.join(Config.PLOTS_DIR, "d_confidence.png"))

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")
    print(f"Samples:     {Config.SAMPLES_DIR}")
    print(f"Checkpoints: {Config.CHECKPOINT_DIR}")
    print(f"Plots:       {Config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
