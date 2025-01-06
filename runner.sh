export CUDA_VISIBLE_DEVICES=2,3

torchrun --nproc_per_node=2 \
    train_ddpd.py \
    --batch-size=16 \
    --planner-lr=0.1 \
    --denoiser-lr=0.1 \
    --weight-decay=0.01 \
    --max-iters=100000 \
    --warmup-iters=50 \
    --lr-decay-iters=100000 \
    --grad-clip=1.0 \
    --grad-accumulation-steps=1 \
    --log-interval=5 \
    --save-interval=200 \
    --wandb-project=ddpd \
    --wandb-name="test" \
    --ckpt-dir="checkpoints" \
    --mnist False