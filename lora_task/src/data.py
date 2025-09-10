from __future__ import annotations

"""Data preprocessing utilities for LoRA fine-tuning.

The main class `ImageCaptionDataset` reads images from a directory.  
Captions are resolved in the following order (first match wins):
1. If a `.txt` file with the same stem as the image exists, its contents are used as caption.
2. If a global `captions.txt` file exists inside the directory and contains `filename<tab>caption` pairs, the corresponding caption is used.
3. Otherwise the `default_caption_template` is formatted with `placeholder_token` and returned.

Returned images are centre-cropped / resized to a square resolution, converted to *tensor* with values in [-1, 1] as expected by Stable Diffusion.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

__all__ = ["ImageCaptionDataset", "build_transform"]


def build_transform(resolution: int) -> T.Compose:
    """Return torchvision transform that outputs *tensor* in [-1, 1]."""

    return T.Compose(
        [
            T.Resize(resolution, interpolation=Image.BICUBIC),
            T.CenterCrop(resolution),
            T.ToTensor(),
            # convert [0,1] -> [-1,1]
            T.Normalize([0.5], [0.5]),
        ]
    )


class ImageCaptionDataset(Dataset):
    """Simple image-caption dataset for DreamBooth/LoRA fine-tuning."""

    IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}

    def __init__(
        self,
        root: str | Path,
        placeholder_token: str = "sks",
        resolution: int = 512,
        default_caption_template: str = "a photo of <{token}>",
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.placeholder_token = placeholder_token
        self.default_caption_template = default_caption_template
        self.transform = build_transform(resolution)

        self._items: List[Tuple[Path, str]] = self._gather_items()

    # ---------------------------------------------------------------------
    # Dataset protocol
    # ---------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D401
        return len(self._items)

    def __getitem__(self, idx: int):
        img_path, caption = self._items[idx]
        # load RGB consistently
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "caption": caption,
        }

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _gather_items(self) -> List[Tuple[Path, str]]:
        captions_global = self._read_global_captions()
        items: List[Tuple[Path, str]] = []
        for img_path in sorted(self.root.iterdir()):
            if img_path.suffix.lstrip(".").lower() not in self.IMAGE_EXTS:
                continue
            caption = self._resolve_caption(img_path, captions_global)
            items.append((img_path, caption))
        if not items:
            raise RuntimeError(f"No training images found in {self.root}")
        return items

    # -- caption helpers ---------------------------------------------------

    def _read_global_captions(self) -> dict[str, str]:
        file = self.root / "captions.txt"
        if not file.exists():
            return {}
        mapping: dict[str, str] = {}
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip() or "\t" not in line:
                continue
            fname, caption = line.split("\t", maxsplit=1)
            mapping[fname.strip()] = caption.strip()
        return mapping

    def _resolve_caption(self, img_path: Path, global_caps: dict[str, str]) -> str:
        # 1) sidecar <name>.txt file
        sidecar = img_path.with_suffix(".txt")
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
        # 2) global captions.txt mapping
        if img_path.name in global_caps:
            return global_caps[img_path.name]
        # 3) fallback default template
        return self.default_caption_template.format(token=self.placeholder_token) 