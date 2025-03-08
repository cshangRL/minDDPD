import json
import os

import numpy as np
import torch
from safetensors.torch import safe_open
from torch.utils.data import DataLoader, Dataset

from PIL import Image

from model import DDPDConfig, DDPDModel, configure_optimizers
import wandb

MASK_IDX = 0

class ImageTokenDataset(Dataset):
    def __init__(
        self,
        safetensor_path="./tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors",
        debug=False,
    ):
        print(f"Initializing ImageTokenDataset with path: {safetensor_path}")
        self.safetensor_path = safetensor_path

        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        print(f"Loading metadata from: {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.total_samples = self.metadata["total_samples"]
            print(f"Total samples in metadata: {self.total_samples}")

        print(f"Loading tensors from: {safetensor_path}")

        with safe_open(safetensor_path, framework="pt") as f:
            self.indices = f.get_tensor("indices").to(torch.uint16).long()
            self.labels = f.get_tensor("labels").long()
            print(
                f"Loaded indices shape: {self.indices.shape}, labels shape: {self.labels.shape}"
            )

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


class MNISTTokenDataset(Dataset):
    def __init__(self, debug=False):
        print("Initializing MNISTTokenDataset")
        from torchvision import transforms
        from torchvision.datasets import MNIST

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),  # Resize to 8x8 like the ImageNet dataset
                transforms.Lambda(self.scale_to_int),  # Scale to 16-bit range
            ]
        )

        self.mnist = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.total_samples = len(self.mnist)
        print(f"Total samples: {self.total_samples}")

    def scale_to_int(self, x):  # Named function, picklable
        return (x * 7.0).long()

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        try:
            image, label = self.mnist[idx]
            # Flatten the 8x8 image into 64 tokens
            indices = image.reshape(-1)
            return indices, label
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            raise

def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.backends.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device

def set_device(device):
    if torch.backends.mps.is_available():
        torch.device("mps")
    elif torch.backends.cuda.is_available():
        torch.cuda.set_device(device)
    else:
        torch.cpu.set_device(device)

def train_step(
    batch,
    planner,
    denoiser,
    grad_accumulation_steps,
):
    device = get_device()

    indices, labels = batch
    indices = indices.to(device)
    labels = labels.to(device)
    t = torch.rand(indices.shape[0], device=device)

    # Create binary mask based on timestep t
    mask = torch.bernoulli(t.unsqueeze(1).expand(-1, indices.shape[1])).bool()

    # Create corrupted version by cloning original indices
    input_indices = indices.clone()

    # Only corrupt tokens where mask is True
    # Sample random tokens from vocab range for corrupted positions
    num_masked = mask.sum().item()
    if num_masked > 0:
        input_indices[mask] = torch.randint(0, MASK_IDX, (num_masked,), device=device)
    corrupted_as_null = input_indices.clone()

    planner_logits, planner_loss = planner(input_indices, t, labels, targets=mask)
    planner_loss = planner_loss / grad_accumulation_steps
    planner_loss.backward()

    # print acc of planner
    with torch.no_grad():
        # since planner is binary classification, we can use accuracy
        where_one = (planner_logits > 0).float()
        where_zero = (planner_logits < 0).float()
        acc = (where_one * mask.float() + where_zero * (1 - mask.float())).mean()

    denoiser_logits, denoiser_loss = denoiser(
        corrupted_as_null, t, labels, targets=indices, mask=mask
    )
    denoiser_loss = denoiser_loss / grad_accumulation_steps
    denoiser_loss.backward()

    with torch.no_grad():
        denoiser_at_mask = denoiser_logits[mask]
        indices_at_mask = indices[mask]
        acc = (denoiser_at_mask.argmax(dim=-1) == indices_at_mask.long()).float().mean()

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


DECODER_PATH = "./tokenize_dataset/pretrained_ckpts/Cosmos-Tokenizer-DI8x8/decoder.jit"


def log_samples(planner, denoiser, device, sequence_length, num_samples=4, mnist=False):
    planner.eval()
    denoiser.eval()
    with torch.no_grad():
        # Sample random class labels for visualization
        class_labels = torch.tensor([1, 9, 94, 299], device=device)
        
        samples = planner.sample(
            denoiser,
            class_labels=class_labels,
            batch_size=num_samples,
            sequence_length=sequence_length,
            temperature=1.0,
            device=device,
        )
        # Convert samples to text format
        samples_pil = []
        
        for sample, label in zip(samples, class_labels):
            # Reshape back to 32x32 for visualization
            sample_2d = sample.reshape(32, 32)
            # to PIL image. make it to 1 channel
            image = sample_2d.cpu().float().numpy() / 8  # shape : 32, 32
            image = Image.fromarray(image, mode="L")
            samples_pil.append(image)

    planner.train()
    denoiser.train()
    return samples_pil

def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = DDPDConfig(**checkpoint["config"]["planner_config"].__dict__)

    planner = DDPDModel(config, model_type="planner").to(device)
    denoiser = DDPDModel(config, model_type="denoiser").to(device)

    planner.load_state_dict(checkpoint["planner_state_dict"])
    denoiser.load_state_dict(checkpoint["denoiser_state_dict"])

    planner.eval()
    denoiser.eval()

    return planner, denoiser

def train():
    # parameeters
    batch_size=32 #, help="Batch size per GPU")
    planner_lr=2e-4 #, help="Planner learning rate")
    denoiser_lr=2e-4 #, help="Denoiser learning rate")
    weight_decay=0.1 #, help="Weight decay")
    max_iters=2001 #, help="Maximum iterations")
    warmup_iters=100 #, help="Warmup iterations")
    lr_decay_iters=2000 #, help="LR decay iterations")
    grad_clip=1.0 #, help="Gradient clipping")
    grad_accumulation_steps=2 #, help="Gradient accumulation steps"
    log_interval=10 #, help="Log interval")
    save_interval=100 #, help="Save interval")
    wandb_project="ddpd" #, help="Weights & Biases project name")
    wandb_entity=None #,help="Weights & Biases entity (username or team name)",
    wandb_name=None #, help="Weights & Biases run name")
    mnist=True #, help="Use MNIST dataset")
    ckpt_dir="checkpoints" #, help="Checkpoint directory")

    checkpoint = "./checkpoints/checkpoint_iter_1000.pt" 

    device = get_device()
    set_device(device)

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


    # Initialize wandb only on rank 0
    train_dataset = MNISTTokenDataset(debug=True)
    print(f"Dataset created successfully")
    
    #train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
    )
    print(f"DataLoader created successfully")

    # Initialize models
    config = DDPDConfig(
        vocab_size=int(2**16) + 1,  # ImageNet tokens + 1 for mask.
        block_size=1024,  # 32x32 image tokens
        n_layer=14,
        n_head=6,
        n_embd=768,
    )

    config.vocab_size = 9
    config.block_size = 1024
    config.num_classes = 10

    global MASK_IDX
    MASK_IDX = config.vocab_size - 1
    planner = DDPDModel(config, model_type="planner").to(device)
    denoiser = DDPDModel(config, model_type="denoiser").to(device)
    print(f"Models created successfully")

    if os.path.exists(checkpoint):
        print("Loading models...")
        planner, denoiser = load_models(checkpoint, device)


    # planner = torch.compile(planner, mode="reduce-overhead")
    # denoiser = torch.compile(denoiser, mode="reduce-overhead")

    # Setup optimizers and schedulers
    planner_optimizer = configure_optimizers(
        planner, weight_decay, planner_lr, (0.95, 0.99), device
    )
    denoiser_optimizer = configure_optimizers(
        denoiser, weight_decay, denoiser_lr, (0.95, 0.99), device
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

    ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

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
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            with ctx:
                losses.append(
                    train_step(
                        batch,
                        planner,
                        denoiser,
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

        if iter_num % log_interval == 0:
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

        if iter_num % save_interval == 0:
            sample_pil_images = log_samples(
                planner, denoiser, device, 1024, mnist=mnist
            )  # 32x32 = 1024 tokens
            wandb.log(
                {
                    "images": [
                        wandb.Image(img, caption=f"Samples {i}")
                        for i, img in enumerate(sample_pil_images)
                    ]
                }
            )

        if iter_num % 1000 == 999:
            checkpoint = {
                "planner_state_dict": planner.state_dict(),
                "denoiser_state_dict": denoiser.state_dict(),
                "planner_optimizer": planner_optimizer.state_dict(),
                "denoiser_optimizer": denoiser_optimizer.state_dict(),
                "planner_scheduler": planner_scheduler.state_dict(),
                "denoiser_scheduler": denoiser_scheduler.state_dict(),
                "config": {
                    "planner_config": planner.config,
                    "denoiser_config": denoiser.config,
                },
            }
            save_path = f"{ckpt_dir}/checkpoint_iter_{iter_num}.pt"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")

        iter_num += 1
    
    wandb.finish()

if __name__ == "__main__":
    train()
