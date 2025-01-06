#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python sample_ddpd.py \
    --checkpoint="/home/ubuntu/simo/ddpd/checkpoints/checkpoint_iter_1000.pt" \
    --num-samples=1 \
    --temperature=0.8 \
    --output-dir="samples" \
    --device="cuda" 