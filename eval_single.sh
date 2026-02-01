#!/bin/bash

export CUDA_VISIBLE_DEVICES=0        
export NUSCENES_DATA_DIR="/root/bev-vae/data/nusc/"
export ARGOVERSE_DATA_DIR="/root/bev-vae/data/av2/"
export CKPT_DIR="/root/bev-vae/ckpt/"
export OUTPUT_DIR="/root/bev-vae/logs/"

python bev_vae/eval.py experiment=$1 devices=1