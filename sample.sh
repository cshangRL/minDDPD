#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python sample_ddpd.py \
    --checkpoint="checkpoint_iter_63500.pt" \
    --num-samples=1 \
    --temperature=0.8 \
    --output-dir="samples" \
    --device="cuda" 