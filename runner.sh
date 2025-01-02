export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/home/ubuntu/simo/ddpd:$PYTHONPATH

/home/ubuntu/py311cuda/bin/python torchrun \
    --nproc_per_node=8 \
    train_ddpd.py \
    --batch-size=8 \
    --planner-lr=1e-4 \
    --denoiser-lr=1e-4 \
    --weight-decay=0.1 \
    --max-iters=100000 \
    --warmup-iters=2000 \
    --lr-decay-iters=100000 \
    --grad-clip=1.0 \
    --grad-accumulation-steps=1 \
    --log-interval=10 \
    --save-interval=1000 \
    --wandb-project=ddpd \
    --wandb-name="test"