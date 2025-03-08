import os

import numpy as np
import torch
import torch.jit as jit
from PIL import Image

from train_ddpd import DDPDConfig, DDPDModel


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


def decode_tokens(tokens, decoder_path, device):
    # Load the JIT decoder model
    decoder = jit.load(decoder_path).to(device)

    # Reshape tokens to 32x32 grid
    tokens = tokens.reshape(-1, 32, 32)

    # Decode tokens to images
    with torch.no_grad():
        images = decoder(tokens)  # Should output B x 3 x H x W

    return images


def save_images(images, output_dir, class_labels=None):
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        # Convert from tensor to PIL Image
        img = (img.clamp(-1, 1) + 1) * 127.5
        img = img.permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)

        # Save with class label if provided
        if class_labels is not None:
            filename = f"sample_{i}_class_{class_labels[i].item()}.png"
        else:
            filename = f"sample_{i}.png"

        img.save(os.path.join(output_dir, filename))

def sample(
):
    checkpoint = "./checkpoints/checkpoint_iter_1000.pt" 
    decoder_path ="./tokenize_dataset/pretrained_ckpts/Cosmos-Tokenizer-DI8x8/decoder.jit" #, help="Path to the JIT decoder model"
    num_samples=1  # help="Number of samples to generate")
    temperature=0.8 # help="Sampling temperature")
    top_k = None # default=None, type=int, help="Top-k sampling parameter")
    output_dir ="samples" #, help="Directory to save generated samples" 
    class_label ="726,917,13,939" #"Specific class label to generate (optional)",
    device = "mps"

    device = torch.device(device)

    # Load models
    print("Loading models...")
    planner, denoiser = load_models(checkpoint, device)

    # Set up class labels
    if class_label is not None:
        # split the class label string by comma and convert to int
        class_labels = torch.tensor(
            [int(label) for label in class_label.split(",")], device=device
        )
        num_samples = class_labels.shape[0]
    else:
        class_labels = torch.randint(
            0, planner.config.num_classes, (num_samples,), device=device
        )

    # Generate samples
    print("Generating samples...")
    with torch.no_grad():
        tokens = planner.sample(
            denoiser,
            class_labels=class_labels,
            batch_size=num_samples,
            sequence_length=1024,  # 32x32 tokens
            temperature=temperature,
            top_k=top_k,
            device=device,
            dynamic=True,
            infer_time_from_planner=True,
        )

    # Decode tokens to images
    tokens = tokens.reshape(-1, 32, 32)
    print("Decoding tokens to images...")
    images = decode_tokens(tokens, decoder_path, device)

    # Save generated images
    print("Saving images...")
    save_images(images, output_dir, class_labels)
    print(f"Generated {num_samples} samples in {output_dir}")


if __name__ == "__main__":
    sample()
