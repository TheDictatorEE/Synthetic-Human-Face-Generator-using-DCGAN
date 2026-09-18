Synthetic Human Face Generator using DCGAN 

A Deep Convolutional Generative Adversarial Network (DCGAN) built with PyTorch to generate realistic synthetic human face images from random latent noise.

The project implements both a Generator and Discriminator using convolutional neural networks and follows the architectural principles introduced in the DCGAN paper by Radford et al.

Project Overview

The goal of this project was to understand and implement how Generative Adversarial Networks can learn the distribution of real images and generate new images that resemble the training dataset.

The model consists of two neural networks:

Generator (G): Takes a random noise vector and generates a synthetic face.
Discriminator (D): Classifies an input image as real or generated.

Both networks are trained adversarially, where the Generator continuously tries to fool the Discriminator while the Discriminator learns to distinguish real images from generated ones.

Overall Pipeline
Random Noise (z)
       ↓
   Generator
       ↓
Synthetic Face
       ↓
   Discriminator
       ↓
Real / Fake
       ↑
   Real Face
Features
DCGAN architecture implemented from scratch using PyTorch
Convolutional Generator and Discriminator
64×64 RGB face generation
Batch Normalization for training stability
ReLU activation in Generator
LeakyReLU activation in Discriminator
Tanh activation for generated images
Adam optimizer with β1 = 0.5
One-sided label smoothing
Checkpoint saving during training
Generated sample grids after training epochs
Generator and Discriminator loss visualization
Discriminator confidence analysis
Latent-space interpolation
Support for continuing training from checkpoints
Project Structure
dcgan_face_generator/
│
├── config.py              # Model and training hyperparameters
├── model.py               # Generator and Discriminator
├── dataset.py             # Dataset loading and preprocessing
├── train.py               # GAN training loop
├── generate.py            # Face generation and interpolation
├── utils.py               # Visualization and training utilities
├── requirements.txt        # Python dependencies
│
├── data/
│   └── images/             # Face dataset
│
├── checkpoints/             # Saved model checkpoints
│
└── outputs/
    ├── samples/             # Generated images
    └── plots/               # Loss and discriminator plots
Technologies Used
Python
PyTorch
Torchvision
NumPy
Matplotlib
Pillow
Dataset

The model can be trained on a face dataset such as CelebA, FFHQ, or LFW.

The images are resized to 64×64 pixels and normalized to the range [-1, 1], matching the Generator's Tanh output.

Example structure:

data/
└── images/
    ├── img_0001.jpg
    ├── img_0002.jpg
    ├── img_0003.jpg
    └── ...
Model Architecture
Generator

The Generator starts with a 100-dimensional latent noise vector.

z ∈ R¹⁰⁰
   ↓
4×4 feature map
   ↓
8×8
   ↓
16×16
   ↓
32×32
   ↓
64×64 RGB Image

The Generator uses ConvTranspose2d layers to progressively increase the spatial resolution.

Hidden layers use:

BatchNorm → ReLU

The final layer uses:

Tanh

so that the generated image values remain in the same [-1, 1] range as the normalized training images.

Discriminator

The Discriminator performs the opposite operation.

64×64×3 Image
      ↓
32×32
      ↓
16×16
      ↓
8×8
      ↓
4×4
      ↓
Real / Fake Probability

It uses strided convolutional layers to progressively reduce the spatial dimensions.

Hidden layers use:

Conv → BatchNorm → LeakyReLU(0.2)

The first discriminator layer does not use BatchNorm, following the DCGAN architecture guidelines.

Training

The Generator and Discriminator are trained together as an adversarial game.

Discriminator

The Discriminator learns to:

classify real images as real
classify generated images as fake
Generator

The Generator learns to produce images that the Discriminator classifies as real.

Conceptually:

Generator:
Noise → Fake Image → Discriminator → "Real"

Discriminator:
Real Image → "Real"
Fake Image → "Fake"

As training progresses, the Generator improves its ability to produce faces that resemble the training distribution.

Training Configuration

Example configuration:

Parameter	Value
Image Size	64×64
Latent Dimension	100
Optimizer	Adam
Learning Rate	0.0002
Adam β1	0.5
Batch Size	128
Generator Activation	ReLU + Tanh
Discriminator Activation	LeakyReLU(0.2)
Real Label	0.9

Training can be started using:

python train.py --data_dir ./data/images --epochs 50 --batch_size 128
Training Stabilization

Several techniques were incorporated to make adversarial training more stable:

1. Batch Normalization

BatchNorm is used in the hidden layers to keep activations within a stable range during training.

2. Adam with β1 = 0.5

A lower β1 value is commonly used for GAN training to reduce excessive momentum and improve training stability.

3. One-Sided Label Smoothing

Real labels are set to 0.9 instead of 1.0.

This prevents the Discriminator from becoming excessively confident and helps maintain useful gradients for the Generator.

4. LeakyReLU

The Discriminator uses LeakyReLU with a negative slope of 0.2 to allow gradients to continue flowing even when activations are negative.

5. DCGAN Initialization

Network weights are initialized using a normal distribution centered around zero with a standard deviation of 0.02.

Generated Results

During training, generated images are periodically saved to monitor how the Generator improves.

The expected progression is:

Early Training
Random / noisy images
        ↓
Basic face-like structures
        ↓
Recognizable facial features
        ↓
More realistic synthetic faces

Sample grids are saved in:

outputs/samples/
Training Analysis

I monitored the training process using:

Generator loss
Discriminator loss
Discriminator confidence on real images D(x)
Discriminator confidence on generated images D(G(z))
Generated image samples across epochs

The losses are not expected to decrease monotonically because GAN training is an adversarial optimization process rather than a conventional single-objective optimization problem.

The generated samples provide an important qualitative measure of whether the Generator is learning meaningful facial features.

Mode Collapse

One of the major challenges considered in this project is mode collapse.

Mode collapse occurs when the Generator produces very similar images for different latent vectors.

For example:

z₁ → Face A
z₂ → Face A
z₃ → Face A
z₄ → Face A

instead of generating diverse faces.

I analyzed generated sample grids and latent-space interpolation to check whether the Generator was producing diverse outputs.

Potential approaches for improving severe mode collapse include:

adjusting Generator/Discriminator learning rates
adding noise to Discriminator inputs
minibatch discrimination
using improved GAN objectives such as WGAN-GP
Latent Space Interpolation

The project also supports interpolation between two random latent vectors.

z₁ → Face A

z₁ ───────────────→ z₂

                   ↓

                Face B

Intermediate latent vectors are generated between z₁ and z₂, allowing the model's learned representation to be visualized as a smooth transition between generated faces.

Run:

python generate.py \
    --checkpoint ./checkpoints/dcgan_epoch_050.pt \
    --interpolate \
    --steps 10
Generating New Faces

Generate a grid of synthetic faces:

python generate.py \
    --checkpoint ./checkpoints/dcgan_epoch_050.pt \
    --num_images 64

Generate individual images:

python generate.py \
    --checkpoint ./checkpoints/dcgan_epoch_050.pt \
    --num_images 16 \
    --individual
What I Learned

Through this project, I gained practical understanding of:

GAN architecture and adversarial training
Generator vs. Discriminator objectives
DCGAN architecture
Transposed convolution for image generation
Convolutional feature extraction
Batch Normalization
Latent-space representations
GAN loss behavior
Training instability
Mode collapse
Checkpoint-based model training
Qualitative evaluation of generative models
Future Improvements

Possible extensions include:

Train on higher-resolution images
Conditional DCGAN for controlled face generation
Quantitative evaluation using FID
Compare DCGAN with WGAN-GP
Experiment with different latent dimensions
Improve generated image resolution
Compare against modern architectures such as StyleGAN
References

Radford, A., Metz, L., & Chintala, S. (2015).
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.

Goodfellow, I. et al. (2014).
Generative Adversarial Networks.
