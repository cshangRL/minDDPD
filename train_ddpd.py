import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
import math
import inspect
import click
import os
import wandb
import json
from safetensors.torch import safe_open


@dataclass
class DDPDConfig:
    model_type: str = "ddpd"
    block_size: int = 1024  # 32x32 image tokens
    vocab_size: int = int(2**16 + 1)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 512
    qk_layernorm: bool = True
    timesteps: int = 1000
    max_t: float = 0.98
    num_classes: int = 1000  # Number of ImageNet classes


class ImageTokenDataset(Dataset):
    def __init__(
        self,
        safetensor_path="/home/ubuntu/simo/nano-diffusion-speedrun/tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors",
        debug=False,
    ):
        print(f"Initializing ImageTokenDataset with path: {safetensor_path}")
        self.safetensor_path = safetensor_path

        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        print(f"Loading metadata from: {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
                self.total_samples = self.metadata["total_samples"]
                print(f"Total samples in metadata: {self.total_samples}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            raise

        print(f"Loading tensors from: {safetensor_path}")
        try:
            with safe_open(safetensor_path, framework="pt") as f:
                self.indices = f.get_tensor("indices").to(torch.uint16).long()
                self.labels = f.get_tensor("labels").long()
                print(
                    f"Loaded indices shape: {self.indices.shape}, labels shape: {self.labels.shape}"
                )
        except Exception as e:
            print(f"Error loading tensors: {e}")
            raise

        if debug:
            samplesze = 64
            self.indices = self.indices[:samplesze]
            self.labels = self.labels[:samplesze]
            self.total_samples = samplesze
            print(f"Debug mode: reduced to {samplesze} samples")
            print(self.labels)

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        try:
            # Get indices and reshape to 1D
            indices = self.indices[idx].reshape(-1)
            label = self.labels[idx]
            return indices, label
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            raise


class Rotary(nn.Module):
    def __init__(self, dim, base=100, h=64, w=64, var_like_order=False):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / (dim)))
        self.h = h
        self.w = w

        t_h = torch.arange(h).type_as(self.inv_freq)
        t_w = torch.arange(w).type_as(self.inv_freq)
        freqs_h = torch.outer(t_h, self.inv_freq).unsqueeze(1)
        freqs_w = torch.outer(t_w, self.inv_freq).unsqueeze(0)
        freqs_h = freqs_h.repeat(1, w, 1)
        freqs_w = freqs_w.repeat(h, 1, 1)
        freqs_hw = torch.cat([freqs_h, freqs_w], 2)

        self.register_buffer("freqs_hw_cos", freqs_hw.cos())
        self.register_buffer("freqs_hw_sin", freqs_hw.sin())
        self.cache_cos = None
        self.cache_sin = None

    def forward(
        self, x, height_width=None, extend_with_register_tokens=0, augment=False
    ):
        if self.cache_cos is not None and self.cache_sin is not None:
            return self.cache_cos, self.cache_sin

        if height_width is not None:
            this_h, this_w = height_width
        else:
            this_hw = x.shape[1]
            this_h, this_w = int(this_hw**0.5), int(this_hw**0.5)

        if augment:
            start_h = torch.randint(0, self.h - this_h + 1, (1,)).item()
            start_w = torch.randint(0, self.w - this_w + 1, (1,)).item()
        else:
            start_h = 0
            start_w = 0

        cos = self.freqs_hw_cos[start_h : start_h + this_h, start_w : start_w + this_w]
        sin = self.freqs_hw_sin[start_h : start_h + this_h, start_w : start_w + this_w]

        cos = cos.clone().reshape(this_h * this_w, -1)
        sin = sin.clone().reshape(this_h * this_w, -1)

        if extend_with_register_tokens > 0:
            cos = torch.cat(
                [
                    torch.ones(extend_with_register_tokens, cos.shape[1]).to(
                        cos.device
                    ),
                    cos,
                ],
                0,
            )
            sin = torch.cat(
                [
                    torch.zeros(extend_with_register_tokens, sin.shape[1]).to(
                        sin.device
                    ),
                    sin,
                ],
                0,
            )

        self.cache_cos = cos[None, :, None, :]
        self.cache_sin = sin[None, :, None, :]

        return self.cache_cos, self.cache_sin  # 1, T, 1, D


def apply_rotary_emb(x, cos, sin):
    cos, sin = cos[:, : x.shape[1]], sin[:, : x.shape[1]]
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Combined projections for self-attention and MLP
        self.chunked_fc = nn.Linear(config.n_embd, 8 * config.n_embd, bias=False)
        self.attn_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Cross attention
        self.cross_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.cross_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.cross_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # MLP output projection
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # init proj to zeros
        torch.nn.init.zeros_(self.attn_proj.weight)
        torch.nn.init.zeros_(self.cross_proj.weight)
        torch.nn.init.zeros_(self.mlp_proj.weight)

        # make initialization bit smaller than typical
        torch.nn.init.normal_(
            self.chunked_fc.weight, mean=0.0, std=0.02 / math.sqrt(config.n_embd)
        )

    def forward(self, x, freq=None, context=None):
        B, T, C = x.size()
        H = self.n_head

        # Combined self-attention + MLP input projection
        qkv_mlp = F.rms_norm(x, (x.size(-1),))
        chunks = self.chunked_fc(qkv_mlp).split([C, C, C, 4 * C, C], dim=-1)
        q, k, v, mlp_intermediate, cross_q = chunks

        # Self attention
        q = q.view(B, T, H, self.head_dim)
        k = k.view(B, T, H, self.head_dim)
        v = v.view(B, T, H, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_proj(attn)

        # Cross attention
        if context is not None:
            _, S, _ = context.size()

            q = cross_q.view(B, T, H, self.head_dim)
            k = self.cross_k(context).view(B, S, H, self.head_dim)
            v = self.cross_v(context).view(B, S, H, self.head_dim)

            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

            cross_attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
            )
            cross_attn = cross_attn.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.cross_proj(cross_attn)

        mlp = self.mlp_proj(F.relu(mlp_intermediate).square())
        x = x + mlp

        return x


class DDPDModel(nn.Module):
    def __init__(self, config, model_type="planner"):
        super().__init__()
        self.config = config
        self.model_type = model_type

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size + 1, config.n_embd),
                wce=nn.Embedding(
                    config.num_classes, 16 * config.n_embd
                ),  # Class embedding
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )

        if model_type == "planner":
            self.head = nn.Linear(config.n_embd, 1, bias=False)
        else:  # denoiser
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        dim = config.n_embd // (2 * config.n_head)
        print(f"Rotary half of head dim: {dim}")
        self.rotary = Rotary(dim, base=100, h=64, w=64)
        self.time_embed = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # init head zero

        torch.nn.init.zeros_(self.head.weight)

    def forward(self, idx, time, class_labels, targets=None, mask=None):
        b, t = idx.size()

        x = self.transformer.wte(idx)
        class_emb = self.transformer.wce(class_labels).reshape(b, 16, -1)

        if self.model_type == "denoiser":
            time_emb = self._get_time_embedding(time * 1000, self.config.n_embd)
            cond = torch.cat([class_emb, time_emb.unsqueeze(1)], dim=1)
        else:
            cond = class_emb

        freq = self.rotary(None, height_width=(32, 32))

        for block in self.transformer.h:
            x = block(x, freq, context=cond)

        x = F.rms_norm(x, (x.size(-1),))

        logits = self.head(x)
        if self.model_type == "planner":
            logits = logits.squeeze(-1)

        if targets is not None:
            if self.model_type == "planner":
                loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            else:  # denoiser
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
                )
                loss = (loss * mask.view(-1)).sum() / mask.sum()
            return logits, loss
        return logits

    def _get_time_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(1000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return self.time_embed(emb)

    @torch.no_grad()
    def sample(
        self,
        denoiser,
        class_labels=None,
        batch_size=1,
        sequence_length=128,
        temperature=1.0,
        top_k=None,
        device="cuda",
        dynamic=False,
    ):
        if self.model_type != "planner":
            raise ValueError("Sampling can only be done with planner model")

        x = torch.randint(
            0, self.config.vocab_size - 1, (batch_size, sequence_length), device=device
        )

        if class_labels is None:
            class_labels = torch.randint(
                0, self.config.num_classes, (batch_size,), device=device
            )

        time_steps = torch.linspace(0.99, 0.02, 50, device=device)

        for t in time_steps:
            current_t = torch.full((batch_size,), t, device=device)

            planner_logits = self(x, current_t, class_labels)
            planner_probs = torch.sigmoid(planner_logits)
            init_mask = torch.bernoulli(planner_probs).bool()

            if dynamic:
                num_tokens_to_mask = min(int(t * sequence_length), sequence_length)
                _, indices = torch.topk(planner_probs, num_tokens_to_mask, dim=1)
                mask = torch.zeros_like(planner_probs, dtype=torch.bool)
                mask.scatter_(1, indices, True)

                percent_masked = mask.sum().item() / mask.numel() * 100
                print(
                    f"Timestep {t:.3f}: Masked {percent_masked:.1f}% of tokens ({num_tokens_to_mask} tokens) instead of {init_mask.sum().item() / init_mask.numel() * 100:.1f}%"
                )
            else:
                mask = init_mask

            if mask.sum() > 0:
                x[mask] = self.config.vocab_size - 1
                denoiser_logits = denoiser(x, current_t, class_labels)

                masked_logits = denoiser_logits[mask]

                masked_logits = masked_logits / temperature

                probs = F.softmax(masked_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                x[mask] = next_tokens

        return x


def print0(s):
    if dist.get_rank() == 0:
        print(s)


def configure_optimizers(
    model, weight_decay, learning_rate, betas, device_type, custom_lrs={}
):
    param_groups = []
    param_dict = {pn: p for pn, p in model.named_parameters()}
    for name, param in model.named_parameters():

        if "weight" in name:
            fan_in = torch.nn.init._calculate_fan_in_and_fan_out(param)[0]
            lr = learning_rate / max(fan_in, 1)  # Avoid division by zero
            param_groups.append(
                {"params": [param], "weight_decay": weight_decay, "lr": lr}
            )
        else:
            if name in custom_lrs:
                lr = custom_lrs[name]
            else:
                lr = learning_rate * 0.1
            param_groups.append(
                {"params": [param], "weight_decay": weight_decay, "lr": lr}
            )

        print0(f"name: {name}, lr: {lr}")

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(param_groups, betas=betas, **extra_args)

    return optimizer


def setup_distributed():
    if dist.is_initialized():
        return

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_step(
    batch,
    planner,
    denoiser,
    planner_optimizer,
    denoiser_optimizer,
    grad_accumulation_steps,
):
    device = batch[0].device
    indices, labels = batch
    indices = indices.to(device)
    labels = labels.to(device)
    t = torch.rand(indices.shape[0], device=device)

    # Create binary mask based on timestep t
    mask = torch.bernoulli(t.unsqueeze(1).expand(-1, indices.shape[1])).bool()

    # Create corrupted version by cloning original indices
    corrupted = indices.clone()
    corrupted_as_null = indices.clone()
    corrupted_as_null[mask] = planner.module.config.vocab_size - 1

    # Only corrupt tokens where mask is True
    # Sample random tokens from vocab range for corrupted positions
    num_masked = mask.sum().item()
    if num_masked > 0:
        corrupted[mask] = torch.randint(
            0, planner.module.config.vocab_size - 1, (num_masked,), device=device
        )

    planner_logits, planner_loss = planner(corrupted, t, labels, targets=mask)
    planner_loss = planner_loss / grad_accumulation_steps
    planner_loss.backward()

    denoiser_logits, denoiser_loss = denoiser(
        corrupted_as_null, t, labels, targets=indices, mask=mask
    )
    denoiser_loss = denoiser_loss / grad_accumulation_steps
    denoiser_loss.backward()

    return {
        "planner_loss": planner_loss.item() * grad_accumulation_steps,
        "denoiser_loss": denoiser_loss.item() * grad_accumulation_steps,
    }


def get_lr_scheduler(optimizer, warmup_iters, lr_decay_iters, max_iters):
    def lr_lambda(step):
        if step < warmup_iters:
            return step / warmup_iters
        else:
            return (max_iters - step) / lr_decay_iters

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_samples(planner, denoiser, device, sequence_length, num_samples=4):
    planner.eval()
    denoiser.eval()
    with torch.no_grad():
        # Sample random class labels for visualization
        class_labels = torch.randint(
            0, planner.module.config.num_classes, (num_samples,), device=device
        )
        samples = planner.module.sample(
            denoiser.module,
            class_labels=class_labels,
            batch_size=num_samples,
            sequence_length=sequence_length,
            temperature=0.7,
            device=device,
        )
        # Convert samples to text format
        sample_texts = []
        for sample, label in zip(samples, class_labels):
            # Reshape back to 32x32 for visualization
            sample_2d = sample.reshape(32, 32)
            sample_text = f"Class {label.item()}: " + " ".join(
                map(str, sample_2d.cpu().tolist())
            )
            sample_texts.append(sample_text)
    planner.train()
    denoiser.train()
    return sample_texts


@click.command()
@click.option("--batch-size", default=32, help="Batch size per GPU")
@click.option("--planner-lr", default=2e-4, help="Planner learning rate")
@click.option("--denoiser-lr", default=2e-4, help="Denoiser learning rate")
@click.option("--weight-decay", default=0.1, help="Weight decay")
@click.option("--max-iters", default=2000, help="Maximum iterations")
@click.option("--warmup-iters", default=100, help="Warmup iterations")
@click.option("--lr-decay-iters", default=2000, help="LR decay iterations")
@click.option("--grad-clip", default=1.0, help="Gradient clipping")
@click.option(
    "--grad-accumulation-steps", default=2, help="Gradient accumulation steps"
)
@click.option("--log-interval", default=100, help="Log interval")
@click.option("--save-interval", default=1000, help="Save interval")
@click.option("--wandb-project", default="ddpd", help="Weights & Biases project name")
@click.option(
    "--wandb-entity",
    default=None,
    help="Weights & Biases entity (username or team name)",
)
@click.option("--wandb-name", default=None, help="Weights & Biases run name")
def train(
    batch_size,
    planner_lr,
    denoiser_lr,
    weight_decay,
    max_iters,
    warmup_iters,
    lr_decay_iters,
    grad_clip,
    grad_accumulation_steps,
    log_interval,
    save_interval,
    wandb_project,
    wandb_entity,
    wandb_name,
):
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize wandb only on rank 0
    if local_rank == 0:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                "batch_size": batch_size,
                "planner_lr": planner_lr,
                "denoiser_lr": denoiser_lr,
                "weight_decay": weight_decay,
                "max_iters": max_iters,
                "warmup_iters": warmup_iters,
                "lr_decay_iters": lr_decay_iters,
                "grad_clip": grad_clip,
                "grad_accumulation_steps": grad_accumulation_steps,
            },
        )

    print(f"Rank {local_rank}: Creating dataset")
    train_dataset = ImageTokenDataset(
        safetensor_path="/home/ubuntu/simo/nano-diffusion-speedrun/tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors"
    )
    print(f"Rank {local_rank}: Dataset created successfully")

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,  # Reduced num_workers
        pin_memory=True,
    )
    print(f"Rank {local_rank}: DataLoader created successfully")

    # Initialize models
    config = DDPDConfig(
        vocab_size=int(2**16) + 1,  # ImageNet tokens + 1 for mask.
        block_size=1024,  # 32x32 image tokens
        n_layer=12,
        n_head=6,
        n_embd=768,
    )

    print(f"Rank {local_rank}: Creating models with config: {config}")
    planner = DDPDModel(config, model_type="planner").to(device)
    denoiser = DDPDModel(config, model_type="denoiser").to(device)

    planner = DDP(planner, device_ids=[local_rank], find_unused_parameters=True)
    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=True)
    print(f"Rank {local_rank}: Models created successfully")

    # planner = torch.compile(planner, mode="reduce-overhead")
    # denoiser = torch.compile(denoiser, mode="reduce-overhead")

    # Setup optimizers and schedulers
    planner_optimizer = configure_optimizers(
        planner, weight_decay, planner_lr, (0.9, 0.95), "cuda"
    )
    denoiser_optimizer = configure_optimizers(
        denoiser, weight_decay, denoiser_lr, (0.9, 0.95), "cuda"
    )

    planner_scheduler = get_lr_scheduler(
        planner_optimizer, warmup_iters, lr_decay_iters, max_iters
    )
    denoiser_scheduler = get_lr_scheduler(
        denoiser_optimizer, warmup_iters, lr_decay_iters, max_iters
    )

    # Training loop
    iter_num = 0
    train_iter = iter(train_dataloader)

    while iter_num < max_iters:
        planner.train()
        denoiser.train()

        planner_optimizer.zero_grad(set_to_none=True)
        denoiser_optimizer.zero_grad(set_to_none=True)

        losses = []
        for _ in range(grad_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_sampler.set_epoch(iter_num)
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            losses.append(
                train_step(
                    batch,
                    planner,
                    denoiser,
                    planner_optimizer,
                    denoiser_optimizer,
                    grad_accumulation_steps,
                )
            )

        # Gradient clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(planner.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)

        planner_optimizer.step()
        denoiser_optimizer.step()

        planner_scheduler.step()
        denoiser_scheduler.step()

        if iter_num % log_interval == 0 and local_rank == 0:
            avg_losses = {
                k: sum(d[k] for d in losses) / len(losses) for k in losses[0].keys()
            }

            lr = planner_scheduler.get_last_lr()[0]

            print(
                f"iter {iter_num}: planner_loss {avg_losses['planner_loss']:.4f}, "
                f"denoiser_loss {avg_losses['denoiser_loss']:.4f}, "
                f"lr {lr:.2e}"
            )

            # Log metrics to wandb
            wandb.log(
                {
                    "iter": iter_num,
                    "planner_loss": avg_losses["planner_loss"],
                    "denoiser_loss": avg_losses["denoiser_loss"],
                    "learning_rate": lr,
                }
            )

        if iter_num % save_interval == 0 and local_rank == 0:
            sample_texts = log_samples(
                planner, denoiser, device, 1024
            )  # 32x32 = 1024 tokens
            wandb.log(
                {
                    "samples": wandb.Table(
                        columns=["sample_id", "text"],
                        data=[[i, text] for i, text in enumerate(sample_texts)],
                    )
                }
            )
            checkpoint = {
                "planner_state_dict": planner.module.state_dict(),
                "denoiser_state_dict": denoiser.module.state_dict(),
                "planner_optimizer": planner_optimizer.state_dict(),
                "denoiser_optimizer": denoiser_optimizer.state_dict(),
                "planner_scheduler": planner_scheduler.state_dict(),
                "denoiser_scheduler": denoiser_scheduler.state_dict(),
                "config": {
                    "planner_config": planner.module.config,
                    "denoiser_config": denoiser.module.config,
                },
            }
            save_path = f"checkpoint_iter_{iter_num}.pt"
            torch.save(checkpoint, save_path)

        iter_num += 1

    if local_rank == 0:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
