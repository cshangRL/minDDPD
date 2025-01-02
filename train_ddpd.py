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
    model_type: str = 'ddpd'
    block_size: int = 1024  # 32x32 image tokens
    vocab_size: int = 68000  # 1000 tokens + 1 mask token
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = True 
    qk_layernorm: bool = True
    timesteps: int = 1000
    max_t: float = 0.98

class ImageTokenDataset(Dataset):
    def __init__(self, safetensor_path="/home/ubuntu/simo/nano-diffusion-speedrun/tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors", debug=False):
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
                print(f"Loaded indices shape: {self.indices.shape}, labels shape: {self.labels.shape}")
        except Exception as e:
            print(f"Error loading tensors: {e}")
            raise

        if debug:
            samplesze = 1024 * 10
            self.indices = self.indices[:samplesze]
            self.labels = self.labels[:samplesze]
            self.total_samples = samplesze
            print(f"Debug mode: reduced to {samplesze} samples")

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        try:
            # Get indices and reshape to 1D
            indices = self.indices[idx].reshape(-1)
            # Replace randomly with mask token (67999)
            indices = indices.masked_fill_(torch.rand((indices.shape)) < 0.05, 67999)
            return indices
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

    def forward(self, x, height_width=None, extend_with_register_tokens=0, augment=False):
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
            cos = torch.cat([
                torch.ones(extend_with_register_tokens, cos.shape[1]).to(cos.device),
                cos,
            ], 0)
            sin = torch.cat([
                torch.zeros(extend_with_register_tokens, sin.shape[1]).to(sin.device),
                sin,
            ], 0)

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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        if config.qk_layernorm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, kv_cache=None, freq=None, v1=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)  # B, T, n_head, D
        cos, sin = freq
    
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, freq=None):
        x = x + self.attn(self.ln_1(x), freq=freq)
        x = x + self.mlp(self.ln_2(x))
        return x

class DDPDPlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.mask_predictor = nn.Linear(config.n_embd, 1)
        dim = (config.n_embd // (2 * config.n_head))
        print(f"Rotary half of head dim: {dim}")
        self.rotary = Rotary(dim, base=100, h=64, w=64)

        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, time, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        
        time_emb = self._get_time_embedding(time, self.config.n_embd)
        
        x = self.transformer.drop(tok_emb + pos_emb + time_emb.unsqueeze(1))
        freq = self.rotary(None, height_width=(32, 32))
        for block in self.transformer.h:
            x = block(x, freq)
            
        x = self.transformer.ln_f(x)
        
        mask_logits = self.mask_predictor(x).squeeze(-1)
        
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(mask_logits, targets.float())
            return mask_logits, loss
        return mask_logits

    def _get_time_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb

    @torch.no_grad()
    def sample(self, denoiser, batch_size=1, sequence_length=128, temperature=1.0, top_k=None, device='cuda'):
        x = torch.full(
            (batch_size, sequence_length), 
            self.config.vocab_size - 1, 
            dtype=torch.long, 
            device=device
        )
        
        time_steps = torch.linspace(0.98, 0.02, 50, device=device)
        
        for t in time_steps:
            current_t = torch.full((batch_size,), t, device=device)
            
            planner_logits = self(x, current_t)
            planner_probs = torch.sigmoid(planner_logits)
            
            mask = torch.bernoulli(planner_probs * t).bool()
            
            if mask.sum() > 0:
                denoiser_logits = denoiser(x, current_t)
                
                masked_logits = denoiser_logits[mask]
                
                masked_logits = masked_logits / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(masked_logits, min(top_k, masked_logits.size(-1)))
                    masked_logits[masked_logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(masked_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                x[mask] = next_tokens
        
        return x

class DDPDDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary = Rotary(config.n_embd // (2 * config.n_head))

        
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, time, targets=None, mask=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        
        time_emb = self._get_time_embedding(time, self.config.n_embd)
        
        x = self.transformer.drop(tok_emb + pos_emb + time_emb.unsqueeze(1))
        freq = self.rotary(None, height_width=(32, 32))

        for block in self.transformer.h:
            x = block(x, freq)
            
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        
        if targets is not None and mask is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none'
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()
            return logits, loss
        return logits

    def _get_time_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
                
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0
    assert len(param_dict.keys() - union_params) == 0

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    
    return optimizer

def setup_distributed():
    if dist.is_initialized():
        return
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
        
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def train_step(batch, planner, denoiser, planner_optimizer, denoiser_optimizer, grad_accumulation_steps):
    device = batch.device
    t = torch.rand(batch.shape[0], device=device)
    
    mask = torch.bernoulli(t.unsqueeze(1).expand(-1, batch.shape[1]))
    corrupted = batch.clone()
    corrupted[mask.bool()] = 67999  # Last token is mask token
    

    planner_logits, planner_loss = planner(corrupted, t, targets=mask)
    planner_loss = planner_loss / grad_accumulation_steps
    planner_loss.backward()
    
 
    denoiser_logits, denoiser_loss = denoiser(corrupted, t, targets=batch, mask=mask)
    denoiser_loss = denoiser_loss / grad_accumulation_steps
    denoiser_loss.backward()
    
    return {
        'planner_loss': planner_loss.item() * grad_accumulation_steps,
        'denoiser_loss': denoiser_loss.item() * grad_accumulation_steps,
    }

def get_lr_scheduler(optimizer, warmup_iters, lr_decay_iters, max_iters):
    def lr_lambda(step):
        if step < warmup_iters:
            return float(step) / float(max(1, warmup_iters))
        decay_ratio = float(step - warmup_iters) / float(max(1, lr_decay_iters - warmup_iters))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * decay_ratio)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def log_samples(planner, denoiser, device, sequence_length, num_samples=4):
    planner.eval()
    denoiser.eval()
    with torch.no_grad():
        samples = planner.module.sample(
            denoiser.module,
            batch_size=num_samples,
            sequence_length=sequence_length,
            temperature=0.7,
            device=device
        )
        # Convert samples to text format
        sample_texts = []
        for sample in samples:
            # Reshape back to 32x32 for visualization
            sample_2d = sample.reshape(32, 32)
            sample_text = ' '.join(map(str, sample_2d.cpu().tolist()))
            sample_texts.append(sample_text)
    planner.train()
    denoiser.train()
    return sample_texts

@click.command()
@click.option('--batch-size', default=32, help='Batch size per GPU')
@click.option('--planner-lr', default=1e-3, help='Planner learning rate')
@click.option('--denoiser-lr', default=1e-3, help='Denoiser learning rate')
@click.option('--weight-decay', default=0.1, help='Weight decay')
@click.option('--max-iters', default=2000, help='Maximum iterations')
@click.option('--warmup-iters', default=100, help='Warmup iterations')
@click.option('--lr-decay-iters', default=2000, help='LR decay iterations')
@click.option('--grad-clip', default=1.0, help='Gradient clipping')
@click.option('--grad-accumulation-steps', default=2, help='Gradient accumulation steps')
@click.option('--log-interval', default=100, help='Log interval')
@click.option('--save-interval', default=1000, help='Save interval')
@click.option('--wandb-project', default='ddpd', help='Weights & Biases project name')
@click.option('--wandb-entity', default=None, help='Weights & Biases entity (username or team name)')
@click.option('--wandb-name', default=None, help='Weights & Biases run name')
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
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f'cuda:{local_rank}')
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize wandb only on rank 0
    if local_rank == 0:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                'batch_size': batch_size,
                'planner_lr': planner_lr,
                'denoiser_lr': denoiser_lr,
                'weight_decay': weight_decay,
                'max_iters': max_iters,
                'warmup_iters': warmup_iters,
                'lr_decay_iters': lr_decay_iters,
                'grad_clip': grad_clip,
                'grad_accumulation_steps': grad_accumulation_steps,
            }
        )
    
    print(f"Rank {local_rank}: Creating dataset")
    train_dataset = ImageTokenDataset(safetensor_path="/home/ubuntu/simo/nano-diffusion-speedrun/tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors", debug=False)
    print(f"Rank {local_rank}: Dataset created successfully")
    
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,  # Reduced num_workers
        pin_memory=True
    )
    print(f"Rank {local_rank}: DataLoader created successfully")
    
    # Initialize models
    config = DDPDConfig(
        vocab_size=68000,  # ImageNet tokens
        block_size=1024,  # 32x32 image tokens
        n_layer=12,
        n_head=8,
        n_embd=768,
        dropout=0.1
    )
    
    print(f"Rank {local_rank}: Creating models with config: {config}")
    planner = DDPDPlanner(config).to(device)
    denoiser = DDPDDenoiser(config).to(device)
    
    planner = DDP(planner, device_ids=[local_rank], find_unused_parameters=True)
    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=True)
    print(f"Rank {local_rank}: Models created successfully")
    
    # Setup optimizers and schedulers
    planner_optimizer = configure_optimizers(planner, weight_decay, planner_lr, (0.9, 0.95), 'cuda')
    denoiser_optimizer = configure_optimizers(denoiser, weight_decay, denoiser_lr, (0.9, 0.95), 'cuda')
    
    planner_scheduler = get_lr_scheduler(planner_optimizer, warmup_iters, lr_decay_iters, max_iters)
    denoiser_scheduler = get_lr_scheduler(denoiser_optimizer, warmup_iters, lr_decay_iters, max_iters)
    
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
            
            batch = batch.to(device)
            losses.append(train_step(
                batch, planner, denoiser,
                planner_optimizer, denoiser_optimizer,
                grad_accumulation_steps
            ))
        
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
                k: sum(d[k] for d in losses) / len(losses)
                for k in losses[0].keys()
            }
            
            lr = planner_scheduler.get_last_lr()[0]
            
            print(f"iter {iter_num}: planner_loss {avg_losses['planner_loss']:.4f}, "
                    f"denoiser_loss {avg_losses['denoiser_loss']:.4f}, "
                    f"lr {lr:.2e}")
            
            # Log metrics to wandb
            wandb.log({
                'iter': iter_num,
                'planner_loss': avg_losses['planner_loss'],
                'denoiser_loss': avg_losses['denoiser_loss'],
                'learning_rate': lr,
            })
            
            # Generate and log samples periodically
            if iter_num % (log_interval * 5) == 0:
                sample_texts = log_samples(planner, denoiser, device, 1024)  # 32x32 = 1024 tokens
                wandb.log({
                    'samples': wandb.Table(
                        columns=['sample_id', 'text'],
                        data=[[i, text] for i, text in enumerate(sample_texts)]
                    )
                })
        
        if iter_num % save_interval == 0 and local_rank == 0:
            checkpoint = {
                'planner_state_dict': planner.module.state_dict(),
                'denoiser_state_dict': denoiser.module.state_dict(),
                'planner_optimizer': planner_optimizer.state_dict(),
                'denoiser_optimizer': denoiser_optimizer.state_dict(),
                'planner_scheduler': planner_scheduler.state_dict(),
                'denoiser_scheduler': denoiser_scheduler.state_dict(),
                'config': {
                    'planner_config': planner.module.config,
                    'denoiser_config': denoiser.module.config
                }
            }
            save_path = f'checkpoint_iter_{iter_num}.pt'
            torch.save(checkpoint, save_path)
            if local_rank == 0:
                wandb.save(save_path)
        
        iter_num += 1
    
    if local_rank == 0:
        wandb.finish()
    cleanup_distributed()
 
if __name__ == "__main__":
    train()