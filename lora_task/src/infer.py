from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Generate images with (optionally) LoRA adapter")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--lora_path", type=str, help="Path to folder with LoRA weights (peft model)")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="outputs/gen")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pipeline(lora_path: str | None = None):
    base_model = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    if lora_path:
        config = PeftConfig.from_pretrained(lora_path)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path, config=config)
        pipe.unet.eval()
    return pipe


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    pipe = load_pipeline(args.lora_path)
    pipe.to("cuda")

    images: List[Image.Image] = []
    for _ in range(args.num_images):
        img = pipe(args.prompt, num_inference_steps=35).images[0]
        images.append(img)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(out_dir / f"img_{i}.png")

if __name__ == "__main__":
    main() 