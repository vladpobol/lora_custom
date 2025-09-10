from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from peft import get_peft_model, LoraConfig, TaskType
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

def main():
    args = parse_args()
    accelerator = Accelerator(log_with="tensorboard", project_config=None)
    device = accelerator.device

    dataset = ImageCaptionDataset(
        args.train_data_dir,
        placeholder_token=args.placeholder_token,
        resolution=args.resolution,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Load model & tokenizer
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    tokenizer: AutoTokenizer = pipe.tokenizer

    # add placeholder token
    with torch.no_grad():
        num_added = tokenizer.add_tokens(args.placeholder_token)
    if num_added:
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    # prepare LoRA
    lora_config = LoraConfig(
        task_type=TaskType.UNET_TUNING,
        r=4,
        lora_alpha=4,
        lora_dropout=0.1,
    )
    unet = get_peft_model(pipe.unet, lora_config)
    unet.train()

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)

    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                latents = pipe.vae.encode(batch["pixel_values"].to(device, dtype=torch.float16)).latent_dist.sample()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = pipe.text_encoder(tokenizer(batch["caption"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device))[
                    0
                ]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                ckpt_dir = Path(args.output_dir) / f"step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(ckpt_dir)
            global_step += 1
        logger.info("Epoch %d complete, loss %.4f", epoch, loss.item())

    # final save
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(args.output_dir)
        pipe.save_pretrained(args.output_dir, safe_serialization=True)

if __name__ == "__main__":
    main() 