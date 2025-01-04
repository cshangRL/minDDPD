
# minDDPD: Minimal Implementation of Discrete Diffusion with Planned Denoising

This is an unofficial implementation of [Think While You Generate: Discrete Diffusion with Planned Denoising](https://arxiv.org/abs/2410.06264).

## Overview

DDPD introduces a novel approach to discrete diffusion models by incorporating a planning mechanism. Instead of using a fixed corruption schedule, the model learns to strategically decompose the diffusion process into two stages:

1. **Planner**: Determines which tokens should be masked for corruption
2. **Denoiser**: Reconstructs the original tokens from corrupted inputs

Naturally, training pipeline consists of training two models. These can be trained independently, but I suspect there is room to improve via joint training. Let's see...

## Installation

```bash
git clone https://github.com/fal-ai-community/minDDPD
cd minDDPD
pip install -r requirements.txt
```

## Training

Basic training command:
```bash
torchrun --nproc_per_node=NUM_GPUS train_ddpd.py \
    --batch-size 32 \
    --planner-lr 2e-4 \
    --denoiser-lr 2e-4 \
    --max-iters 2000
```

## Model Architecture

The implementation uses a transformer-based architecture with:
- Rotary positional embeddings
- RMSNorm for layer normalization
- Cross-attention for conditioning
- Efficient parallel linear layers for self, cross, and MLP.
- 2D RoPE for positional embeddings.

## Trainings

- muP for initialization and learning rate.
- 32x32 image tokens.


```python
class DDPDConfig:
    model_type: str = "ddpd"
    block_size: int = 1024  # 32x32 image tokens
    vocab_size: int = int(2**16 + 1)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 512
    timesteps: int = 1000
```

## Citation

```bibtex
@article{liu2024think,
  title={Think While You Generate: Discrete Diffusion with Planned Denoising},
  author={Liu, Sulin and Nam, Juno and Campbell, Andrew and Stärk, Hannes and Xu, Yilun and Jaakkola, Tommi and Gómez-Bombarelli, Rafael},
  journal={arXiv preprint arXiv:2410.06264},
  year={2024}
}
```

If you use this codebase, please cite this github repository as well.

```bibtex
@misc{ryu2024miniddpd,
  author = {Simo Ryu},
  title = {minDDPD},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fal-ai-community/minDDPD}},
}
```

## License

MIT

## Acknowledgments

This implementation is inspired by the original paper by Liu et al. Thanks to the authors for their innovative work in discrete diffusion models.
