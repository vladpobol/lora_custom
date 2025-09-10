from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import ImageCaptionDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Stable Diffusion 1.5")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, default="sks")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    return parser.parse_args()


def _safe_enable_xformers(pipe: StableDiffusionPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as _:
        # Not available in environment; proceed without xFormers
        pass


def main():
    args = parse_args()
    logger.info(f"Starting training with args: {args}")

    logs_dir = Path(args.output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    project_config = ProjectConfiguration(project_dir=str(logs_dir))
    accelerator = Accelerator(log_with="tensorboard", project_config=project_config)
    device = accelerator.device

    logger.info("Loading dataset...")
    dataset = ImageCaptionDataset(
        args.train_data_dir,
        placeholder_token=args.placeholder_token,
        resolution=args.resolution,
    )
    logger.info(f"Dataset loaded: {len(dataset)} images")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Load model & tokenizer
    logger.info("Loading Stable Diffusion model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    _safe_enable_xformers(pipe)
    tokenizer: AutoTokenizer = pipe.tokenizer

    # add placeholder token
    logger.info(f"Adding placeholder token: {args.placeholder_token}")
    with torch.no_grad():
        num_added = tokenizer.add_tokens(args.placeholder_token)
    if num_added:
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    # prepare LoRA on UNet with proper target modules for diffusers UNet
    logger.info("Setting up LoRA...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "to_k",
            "to_q", 
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2"
        ],
    )
    
    try:
        unet = get_peft_model(pipe.unet, lora_config)
        unet.train()
        logger.info("LoRA model created successfully")
    except Exception as e:
        logger.error(f"Error creating LoRA model: {e}")
        # Fallback to simpler target modules
        logger.info("Trying with simpler LoRA config...")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.1,
            bias="none",
            target_modules=["to_k", "to_q", "to_v"],
        )
        unet = get_peft_model(pipe.unet, lora_config)
        unet.train()

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)

    # Prepare for (optional) DDP / mixed precision; works fine on single GPU too
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    logger.info(f"Starting training for {args.epochs} epochs...")
    global_step = 0
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                tokenized = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(device)
                encoder_hidden_states = pipe.text_encoder(input_ids)[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                ckpt_dir = Path(args.output_dir) / f"step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved at step {global_step}")
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            global_step += 1
        logger.info("Epoch %d complete, loss %.4f", epoch + 1, loss.item())

    # final save
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(args.output_dir)
        # Save base pipe snapshot to align versions used at inference
        pipe.save_pretrained(args.output_dir, safe_serialization=True)
        logger.info(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main() 