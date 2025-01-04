export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --nproc_per_node=6 \
    train_ddpd.py \
    --batch-size=16 \
    --planner-lr=3e-4 \
    --denoiser-lr=3e-4 \
    --weight-decay=0.1 \
    --max-iters=100000 \
    --warmup-iters=500 \
    --lr-decay-iters=100000 \
    --grad-clip=1.0 \
    --grad-accumulation-steps=1 \
    --log-interval=5 \
    --save-interval=500 \
    --wandb-project=ddpd \
    --wandb-name="test"