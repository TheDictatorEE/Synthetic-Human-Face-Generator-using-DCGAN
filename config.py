"""
Central configuration for the DCGAN Synthetic Face Generator.
Edit values here rather than passing a long list of CLI flags.
"""

import torch

class Config:
    # ---- Paths ----
    DATA_DIR = "./data"                 # expects data/<class_subdir>/*.jpg  (ImageFolder format)
    CHECKPOINT_DIR = "./checkpoints"
    SAMPLES_DIR = "./outputs/samples"
    PLOTS_DIR = "./outputs/plots"

    # ---- Data ----
    IMAGE_SIZE = 64          # DCGAN paper uses 64x64; set 128 for higher fidelity (needs deeper nets)
    CHANNELS = 3             # RGB
    BATCH_SIZE = 128
    NUM_WORKERS = 4

    # ---- Model ----
    LATENT_DIM = 100         # size of the input noise vector z
    GEN_FEATURE_MAPS = 64    # base feature map count for Generator
    DISC_FEATURE_MAPS = 64   # base feature map count for Discriminator

    # ---- Training ----
    NUM_EPOCHS = 50
    LR = 2e-4                # DCGAN paper's recommended learning rate
    BETA1 = 0.5               # Adam beta1 (paper-recommended, default 0.9 causes instability)
    BETA2 = 0.999
    LABEL_SMOOTHING = 0.9     # real labels = 0.9 instead of 1.0 (reduces D overconfidence)
    LABEL_FLIP_PROB = 0.0     # optionally flip a fraction of labels to fight mode collapse (0 = off)

    # ---- Logging / Checkpointing ----
    SAMPLE_INTERVAL = 1       # save a sample grid every N epochs
    CHECKPOINT_INTERVAL = 5   # save model weights every N epochs
    FIXED_NOISE_SIZE = 64     # number of images in the tracked sample grid (progress over epochs)
    LOG_INTERVAL = 50         # print training stats every N batches

    # ---- Device ----
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # ---- Reproducibility ----
    SEED = 42
