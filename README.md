# BEV-VAE: A Unified BEV Representation for Generalizable Driving Scene Synthesis
[Zeming Chen](https://scholar.google.com/citations?hl=zh-CN&user=u_KlPbgAAAAJ), [Hang Zhao](https://scholar.google.com/citations?hl=zh-CN&user=DmahiOYAAAAJ).
## Abstract
<details>
<summary><b>TL; DR</b> We introduce BEV-VAE, a variational autoencoder that unifies multi-view images into a BEV representation for generalizable autonomous driving scene synthesis. </summary>
Generative modeling has shown remarkable success in vision and language, inspiring research on synthesizing driving scenes. Existing multi-view synthesis approaches typically operate in image latent spaces with cross-attention to enforce spatial consistency, but they are tightly bound to camera configurations, which limits model generalization. We propose BEV-VAE, a variational autoencoder that learns a unified Bird’s-Eye-View (BEV) representation from multi-view images, enabling encoding from arbitrary camera layouts and decoding to any desired viewpoint. Through multi-view image reconstruction and novel view synthesis, we show that BEV-VAE effectively fuses multi-view information and accurately models spatial structure. This capability allows it to generalize across camera configurations and facilitates scalable training on diverse datasets. Within the latent space of BEV-VAE, a Diffusion Transformer (DiT) generates BEV representations conditioned on 3D object layouts, enabling multi-view image synthesis with enhanced spatial consistency on nuScenes and achieving the first complete seven-view synthesis on AV2. Compared with training generative models in image latent spaces, BEV-VAE achieves superior computational efficiency. Finally, synthesized imagery significantly improves the perception performance of BEVFormer, highlighting the utility of generalizable scene synthesis for autonomous driving.
</details>

## Method
### Overall architecture of BEV-VAE with DiT for multi-view image generation.
![framework](./assets/framework.png)
In Stage 1, BEV-VAE learns to encode multi-view images into a spatially compact latent space in BEV and reconstruct them, ensuring spatial consistency. In Stage 2, DiT is trained with Classifier-Free Guidance (CFG) in this latent space to generate BEV representations from random noise, which are then decoded into multi-view images.

## Getting Started
### Environment Setup
First, create and activate a conda environment:
```bash
conda create -n bevvae python=3.10
conda activate bevvae
```
Clone the repository:
```bash
git clone https://github.com/Czm369/bev-vae.git
```
#### Dependencies
The code is tested with <b>Python 3.10.x</b> and <b>CUDA 12.x</b>.

Install PyTorch according to your CUDA version:
##### CUDA 12.1
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
##### CUDA 12.8
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```
#### Install BEV-VAE

```bash
cd ${ROOT}
pip install requirements.txt
pip install -e .
```

### Data Preparation
#### nuScenes
1. Download the nuScenes dataset from the [website](https://www.nuscenes.org/nuscenes) and and place it under `./data/`. 
2. Download the BEV-VAE preprocessed data from the [website](https://huggingface.co/datasets/czm369/nusc_bev-lidar). This repository provides additional data required by BEV-VAE, including BEV latent representations (`nusc_bev-lidar_train.tar.gz` and `nusc_bev-lidar_val.tar.gz`) encoded from nuScenes.These BEV latents can be decoded into multi-view images using nuScenes (or other compatible, such as AV2) camera configurations. The provided BEV latents can be directly used to train DiT, significantly reducing training cost by skipping the BEV-VAE encoding stage.
3. After preparation, you should have the following files:

```bash
bev-vae/data/nusc
├── maps
├── samples
├── sweeps
├── v1.0-trainval
└── nusc
    ├── scene2frame.json
    ├── scene2sensor2extrinsic.json
    ├── scene2sensor2intrinsic.json
    ├── scene2sensor2stamp2token.json
    ├── scene2stamp2annotation.json
    ├── sensor_cache.feather
    ├── synchronization_cache.feather
    ├── token2ego.json
    └── token2file.json
```

#### AV2
1. Download the AV2 dataset from the [website](https://www.argoverse.org/av2) and and place it under `./data/`. 
2. Download the BEV-VAE preprocessed data from the [website](https://huggingface.co/datasets/czm369/av2_bev-lidar). This repository provides additional data required by BEV-VAE, including BEV latent representations (`av2_bev-lidar_train.tar.gz` and `av2_bev-lidar_val.tar.gz`) encoded from AV2.These BEV latents can be decoded into multi-view images using AV2 (or other compatible, such as nuScenes) camera configurations. The provided BEV latents can be directly used to train DiT, significantly reducing training cost by skipping the BEV-VAE encoding stage.
3. After preparation, you should have the following files:

```bash
bev-vae/data/av2/sensor/
├── train
├── val
└── av2
    ├── log2sensor2extrinsic.json
    ├── log2sensor2intrinsic.json
    ├── log2stamp2annotation.json
    ├── log2stamp2ego.json
    ├── sensor_cache.feather
    └── synchronization_cache.feather
```
### Model Preparation
1. Download the pre-trained BEV-VAE `bev-vae_329089c03f0d.ckpt` from the [website](https://huggingface.co/czm369/BEV-VAE) and place it under `./ckpt/stage1`. 
2. Download the pre-trained Inception `pt_inception-2015-12-05-6726825d.pth` for FID evaluation from the [website](https://huggingface.co/czm369/BEV-VAE) and place it under `./ckpt/fid`. 
3. Download the pre-trained LoFTR `loftr_outdoor.ckpt` for MVSC evaluation from the [website](https://huggingface.co/czm369/BEV-VAE) and place it under `./ckpt/`. 

### Model Inference

Before running inference, update the dataset and checkpoint root paths in `eval_single.sh` to match your local environment:
```bash
export NUSCENES_DATA_DIR="/root/bev-vae/data/nusc/"
export ARGOVERSE_DATA_DIR="/root/bev-vae/data/av2/"
export CKPT_DIR="/root/bev-vae/ckpt/"
export OUTPUT_DIR="/root/bev-vae/logs/"
```
On <b>a single RTX 5090 GPU</b>, BEV-VAE can perform <b>multi-view image reconstruction</b> with `batch_size=4` at least.
#### nuScenes
```bash
bash eval_single.sh test-bev-vae_nusc-val_1x1x4x1_4e-5_1504
```
#### AV2
```bash
bash eval_single.sh test-bev-vae_av2-val_1x1x4x1_8e-5_5880
```

## Experiments

### Datasets
This study uses four multi-camera autonomous driving datasets that differ substantially in scale, camera configuration, annotated categories, and recording locations. Despite these differences, all datasets provide full 360° coverage of the surrounding scene.

| Dataset  | #Frames | #Cameras | #Classes |                    Recording Locations                       |
|:--------:|:-------:|:--------:|:--------:|:------------------------------------------------------------:|
|   WS101  |   17k   |    5     |    0     |               London, San Francisco Bay Area                 | 
| nuScenes |   155k  |    6     |    23    |          Boston, Pittsburgh, Las Vegas, Singapore            | 
|   AV2    |   224k  |    7     |    30    | Austin, Detroit, Miami, Pittsburgh, Palo Alto, Washington DC | 
|  nuPlan  |  3.11M  |    8     |    7     |          Boston, Pittsburgh, Las Vegas, Singapore            | 

We introduce a new hybrid autonomous driving dataset configuration, <b>PAS</b>, which combines nu<b>P</b>lan, <b>A</b>V2, and nu<b>S</b>cenes.

### Multi-view Image Reconstruction
BEV-VAE learns unified BEV representations by reconstructing multi-view images, integrating semantics from all camera views while modeling 3D spatial structure. Reconstruction metrics provide an indirect evaluation of the quality of the learned BEV representations. For reference, we compare with SD-VAE, a foundational model trained on LAION-5B, which encodes a single $256\times256$ image into a $32 \times32\times4$ latent. In contrast, BEV-VAE encodes multiple $256\times256$ views into a $32\times32\times16$ BEV latent, facing the more challenging task of modeling underlying 3D structure.
#### Reconstruction metrics on nuScenes compared with SD-VAE.
|  Model  | Training | Validation | PSNR $\uparrow$ | SSIM $\uparrow$ | MVSC $\uparrow$ | rFID $\downarrow$ |
|:-------:|:--------:|:----------:|:---------------:|:---------------:|:---------------:|:-----------------:|
| SD-VAE  | LAION-5B |  nuScenes  |   <b>29.63</b>  |  <b>0.8283</b>  |     0.9292      |    <b>2.18</b>    |
| BEV-VAE | nuScenes |  nuScenes  |      26.13      |     0.7231      |     0.9250      |       6.66        |
| BEV-VAE |   PAS    |  nuScenes  |      28.88      |     0.8028      |  <b>0.9756</b>  |       4.74        |

#### Reconstruction metrics on AV2 compared with SD-VAE.
|  Model  | Training | Validation | PSNR $\uparrow$ | SSIM $\uparrow$ | MVSC $\uparrow$ | rFID $\downarrow$ |
|:-------:|:--------:|:----------:|:---------------:|:---------------:|:---------------:|:-----------------:|
| SD-VAE  | LAION-5B |    AV2     |   <b>27.81</b>  |  <b>0.8229</b>  |     0.8962      |    <b>1.87</b>    |
| BEV-VAE |   AV2    |    AV2     |      26.02      |     0.7651      |     0.9197      |       4.15        |
| BEV-VAE |   PAS    |    AV2     |      27.29      |     0.8028      |  <b>0.9461</b>  |       2.82        |

SD-VAE focuses on per-view image fidelity, whereas PAS-trained BEV-VAE achieves superior multi-view spatial consistency (MVSC).
#### Multi-view image reconstruction on nuScenes
<b>Click the image below</b> to watch the ego view rotate 360° horizontally.
[![rec_nusc](./assets/rec_nusc_more.png)](https://www.bilibili.com/video/BV1drHNzLEi8)

#### Multi-view image reconstruction on AV2
<b>Click the image below</b> to watch the ego view rotate 360° horizontally.
[![rec_av2](./assets/rec_av2_more.png)](https://www.bilibili.com/video/BV1drHNzLEr3)

#### Multi-view image reconstruction on nuPlan
<b>Click the image below</b> to watch the ego view rotate 360° horizontally.
[![rec_nupl](./assets/rec_nupl_more.png)](https://www.bilibili.com/video/BV1irHNzjEaZ)

### Novel View Synthesis
![rec_rot](./assets/rec_rot.png)
<b>Novel view synthesis via camera pose modifications on nuScenes.</b> Row 1 shows real images from the nuScenes validation set, and Rows 2-3 show reconstructions with all cameras rotated 30° left and right, where the cement truck and tower crane truck remain consistent across views without deformation. 

![nvs_cam](./assets/nvs_cam.png)
<b>Novel view synthesis cross camera configurations.</b> Row 1 presents real images from the nuPlan validation set. Row 2 and Row 3 show reconstructions using camera parameters from AV2 and nuScenes, respectively. The model captures dataset-specific vehicle priors: AV2 reconstructions include both the front and rear of the ego vehicle, while nuScenes reconstructions mainly show the rear (with the rightmost image corresponding to the rear-view camera for alignment).

### Zero-shot BEV Representation Construction

![rec_ws101](./assets/rec_ws101.png)
<b>Zero-shot BEV representation construction on WS101.</b> Row 1 shows real images from the WS101 validation set. Rows 2 and 3 show zero-shot and fine-tuned reconstructions, respectively, with object shapes preserved in the zero-shot results and further sharpened after fine-tuning.
|  Model  |  Training   | Validation | PSNR $\uparrow$ | SSIM $\uparrow$ | MVSC $\uparrow$ | rFID $\downarrow$ |
|:-------:|:-----------:|:----------:|:---------------:|:---------------:|:---------------:|:-----------------:|
| SD-VAE  |  LAION-5B   |   WS101    |      23.38      |  <b>0.7050</b>  |     0.8580      |    <b>4.59</b>    |
| BEV-VAE |     PAS     |   WS101    |      16.6       |     0.3998      |     0.8309      |       56.7        |
| BEV-VAE | PAS + WS101 |   WS101    |   <b>23.46</b>  |     0.6844      |  <b>0.9505</b>  |       13.78       |

<b>Zero-shot and fine-tuned reconstruction metrics on WS101 compared with SD-VAE.</b>

### Autonomous Driving Scene Synthesis
#### Autonomous driving scene synthesis from AV2 to nuScenes.
![gen_av2](./assets/gen_av2.png)
BEV-VAE with DiT generates a BEV representation from 3D bounding boxes of AV2, which can then be decoded into multi-view images according to the camera configurations of nuScenes.

#### Multi-view image generation on AV2 with 3D object layout editing.
<b>Click the image below</b> to watch the ego view rotate 360° horizontally.
[![gen_drop_av2](./assets/gen_drop_av2.png)](https://www.bilibili.com/video/BV1drHNzLEr2)

#### Multi-view image generation on nuScenes with 3D object layout editing.
<b>Click the image below</b> to watch the ego view rotate 360° horizontally.
[![gen_deop_nusc](./assets/gen_drop_nusc.png)](https://www.bilibili.com/video/BV1orHNzLEto)

### Data Augmentation for Perception
BEV-VAE w/ DiT using the Historical Frame Replacement strategy (randomly replacing real frames with generated ones) improves BEVFormer’s perception by enabling the model to learn invariance of object locations relative to appearance.

| Perception Model | Generative Model |       Augmentation Strategy       | mAP$\uparrow$ | NDS$\uparrow$ |
|:----------------:|:----------------:|:---------------------------------:|:-------------:|:-------------:|
|  BEVFormer Tiny  |        -         |                  -                |     25.2      |     35.4      |
|  BEVFormer Tiny  |      BEVGen      | Training Set + 6k Synthetic Data  |  <b>27.3</b>  |     37.2      |
|  BEVFormer Tiny  |  BEV-VAE w/ DiT  |   Historical Frame Replacement    |     27.1      |  <b>37.4</b>  |


## TODO
- [x] releasing the paper
- [x] tutorial
- [x] pretrained weights for BEV-VAE
- [x] inference code 
- [ ] train code