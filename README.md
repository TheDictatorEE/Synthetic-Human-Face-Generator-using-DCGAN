# Synthetic Human Face Generator — DCGAN

A Deep Convolutional GAN that learns the distribution of human face images
and generates new, synthetic faces from random latent noise. Built with
PyTorch, following the architectural guidelines from Radford et al. (2015),
*Unsupervised Representation Learning with Deep Convolutional GANs*.

## Project Structure

```
dcgan_face_generator/
├── config.py           # all hyperparameters in one place
├── model.py             # Generator + Discriminator (CNN architectures)
├── dataset.py           # data loading (ImageFolder or flat folder of images)
├── train.py              # training loop
├── generate.py           # generate faces / interpolate latent space from a checkpoint
├── utils.py               # sample-grid saving, loss plotting, training-history logging
├── requirements.txt
├── data/                  # put your face dataset here (see Dataset section)
├── checkpoints/            # saved model weights (created during training)
└── outputs/
    ├── samples/             # per-epoch generated sample grids
    └── plots/                 # loss curves, D-confidence plots, history.json
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU (CUDA) is strongly recommended — DCGAN on 64x64 faces is trainable on
CPU only for tiny experiments.

## 2. Dataset

Any face dataset works as long as images are roughly cropped to faces.
Common choices:

- **CelebA** (~200k celebrity faces) — https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **FFHQ** (70k high-quality faces) — https://github.com/NVlabs/ffhq-dataset
- **LFW** (Labeled Faces in the Wild) — smaller, good for quick experiments

Place images so `dataset.py` can find them. Either layout works:

```
data/
└── images/            # one subfolder is enough (ImageFolder convention)
    ├── img_0001.jpg
    ├── img_0002.jpg
    └── ...
```

or a flat folder of images directly under `data/` (auto-detected fallback).

## 3. Train

```bash
python train.py --data_dir ./data/images --epochs 50 --batch_size 128
```

Key flags (all default to `config.py` values):

| Flag | Meaning |
|---|---|
| `--epochs` | number of passes over the dataset |
| `--batch_size` | images per training step |
| `--lr` | learning rate for both G and D (paper default: 2e-4) |
| `--image_size` | output resolution (default 64x64) |
| `--latent_dim` | size of the noise vector z (default 100) |
| `--resume` | path to a `.pt` checkpoint to continue training |

During training you'll see, per logged batch:

```
Loss_D: 0.55  Loss_G: 2.10  D(x): 0.83  D(G(z)): 0.12 / 0.19
```

- **D(x)** — Discriminator's average confidence that *real* images are real (want ≈ 0.5–0.9, not pinned at 1.0)
- **D(G(z))** — its confidence that *fake* images are real, before/after G's update (want climbing toward 0.5 over training, not stuck at 0)

## 4. Generate

```bash
# Grid of 64 new random faces
python generate.py --checkpoint ./checkpoints/dcgan_epoch_050.pt --num_images 64

# Also save each face as its own file
python generate.py --checkpoint ./checkpoints/dcgan_epoch_050.pt --num_images 16 --individual

# Latent-space interpolation between two random faces (checks manifold smoothness)
python generate.py --checkpoint ./checkpoints/dcgan_epoch_050.pt --interpolate --steps 10
```

## 5. Architecture

**Generator** (z ∈ R¹⁰⁰ → 64×64×3 image): a stack of 5 transposed
convolutions that upsample a 1×1 latent vector through 4×4 → 8×8 → 16×16 →
32×32 → 64×64 feature maps, with BatchNorm + ReLU at each hidden layer and
Tanh on the output (matching the [-1, 1] normalized image range).

**Discriminator** (64×64×3 image → real/fake probability): the mirror
image — a stack of 5 strided convolutions that downsample 64×64 → 32×32 →
16×16 → 8×8 → 4×4 → 1×1, with LeakyReLU(0.2) at each hidden layer and no
BatchNorm on the very first layer (per the original paper's finding that
this destabilizes training).

Both networks are fully convolutional — no pooling layers, no fully
connected hidden layers — per the DCGAN paper's guidelines for training
stability at this scale.

## 6. Training Stabilization Techniques Used

| Technique | Why |
|---|---|
| Strided/fractional-strided convs instead of pooling | Lets the network learn its own spatial up/down-sampling instead of a fixed, lossy operator |
| BatchNorm in G and D (except G output, D input) | Keeps activations in a healthy range and prevents G from collapsing all outputs to a single point |
| Adam with β1 = 0.5 (not the default 0.9) | The default momentum causes oscillation/instability in adversarial training |
| One-sided label smoothing (real label = 0.9) | Prevents D from becoming overconfident, which would starve G of useful gradient |
| LeakyReLU(0.2) in D | Avoids the "dying ReLU" problem when gradients flow back through D into G |
| Weight init from N(0, 0.02) | The paper-recommended init; measurably reduces early-training divergence |
| `drop_last=True` in the DataLoader | Avoids a tiny, unrepresentative final batch destabilizing BatchNorm statistics |

## 7. Analyzing Training Behavior

After training, `outputs/plots/` contains:

- **`loss_curve.png`** — G and D loss over every training iteration
- **`d_confidence.png`** — D(x) vs D(G(z)) over time, with the ideal 0.5 equilibrium line
- **`history.json`** — raw numbers behind both plots, for custom analysis

### What healthy training looks like
Both losses oscillate rather than converge — GANs don't have a single loss
that monotonically decreases, because G and D are playing a minimax game
against a moving target. The useful signal is **D(x)** and **D(G(z))**
drifting toward **0.5** over time: that means D can no longer easily tell
real from fake, which is the actual goal.

### Training Instability
**Symptom:** G_loss or D_loss spikes erratically, or one loss diverges to
near-zero/near-infinity.
**Typical cause:** D has become far too strong relative to G (or vice
versa), so gradients either vanish or explode.
**What to check first:** look at `d_confidence.png` — if D(x) is pinned
near 1.0 and D(G(z)) is pinned near 0.0 for many epochs, D has "won" and
G is receiving little useful gradient. Mitigations already built in
(label smoothing, `β1=0.5`) reduce this; if it still happens, try
lowering D's learning rate relative to G's, or updating G more than once
per D step.

### Mode Collapse
**Symptom:** loss curves can look fine, but the generated sample grids
(`outputs/samples/epoch_*.png`) show the same face, or a handful of nearly
identical faces, repeated across the grid — G has found a small number of
outputs that reliably fool the current D and stopped exploring.
**How to detect it here:**
1. Visually scan consecutive sample grids for repeated faces.
2. Run `generate.py --interpolate` — a collapsed model produces
   interpolations that jump abruptly between a few fixed points rather
   than morphing smoothly.
**Mitigations to try:** reduce D's relative capacity, add noise to D's
inputs, use a minibatch-discrimination layer, or switch to a Wasserstein
loss (WGAN-GP) if collapse persists — this is the most common reported
DCGAN failure mode and one of the main motivations behind later GAN
variants.

## 8. Suggested Report Sections (for a project write-up)

1. Introduction & problem statement
2. Dataset description and preprocessing
3. DCGAN architecture (include the diagrams/tables above)
4. Training setup and hyperparameters
5. Results: sample grids across epochs (qualitative progression)
6. Loss curves and discriminator-confidence analysis
7. Discussion of instability/mode collapse observed (or avoided) in your run
8. Limitations and possible extensions (conditional DCGAN, StyleGAN comparison, FID scoring)

## References

- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation
  Learning with Deep Convolutional Generative Adversarial Networks.* arXiv:1511.06434
- Goodfellow, I. et al. (2014). *Generative Adversarial Networks.* arXiv:1406.2661
