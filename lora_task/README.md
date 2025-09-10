# LoRA Fine-Tuning Boilerplate

Minimal pipeline to fine-tune an open-source text-to-image model (Stable Diffusion 1.5) with LoRA adapters and generate images in web services.

## Features
* Pre-processing pipeline for user images
* LoRA training script (DreamBooth-style) powered by 🤗 Diffusers & PEFT
* Inference script that merges LoRA or keeps separate adapters
* Works on consumer GPUs (<=12 GB) with 512² resolution

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data preparation
Put training images in a directory, e.g. `data/rock_art/` (8-15 images). Optionally add a text file with captions; otherwise prompt template will use a single token placeholder.

## Training
```bash
python src/train_lora.py \
    --train_data_dir data/rock_art \
    --output_dir outputs/rock_art \
    --placeholder_token "rockart" \
    --resolution 512 \
    --batch_size 2 \
    --epochs 10
```

## Inference
```bash
python src/infer.py \
    --lora_path outputs/rock_art \
    --prompt "a painting in <rockart> style of a castle at sunset" \
    --num_images 4
```

See `examples/demo.ipynb` for an end-to-end walkthrough. 