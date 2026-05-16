"""
Image augmentations for CXR SFT training.

Applied independently to each view (frontal, lateral) before the view-dropout step.
Each augmentation has p=0.5 unless otherwise noted.
"""

from __future__ import annotations

import random
from typing import Optional

from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF


def augment_image(img: Image.Image, p: float = 0.5) -> Image.Image:
    """Apply random augmentations to a single image."""

    # Random crop (scale 0.7–0.9, then resize back) — always applied
    w, h = img.size
    scale = random.uniform(0.7, 0.9)
    crop_w, crop_h = int(w * scale), int(h * scale)
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    img = img.crop((x, y, x + crop_w, y + crop_h)).resize((w, h), Image.BILINEAR)

    # Brightness ±35%
    if random.random() < p:
        factor = random.uniform(0.65, 1.35)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast ±35%
    if random.random() < p:
        factor = random.uniform(0.65, 1.35)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Gamma (0.6–1.5) via lookup table
    if random.random() < p:
        gamma = random.uniform(0.6, 1.5)
        img = TF.adjust_gamma(img, gamma)

    return img


def prepare_views(
    frontal_path: Optional[str],
    lateral_path: Optional[str],
    drop_prob: float = 0.2,
    augment: bool = True,
    image_size: tuple = (512, 512),
) -> list[Image.Image]:
    """
    Load, augment, and apply view dropout.

    Args:
        frontal_path: path to frontal PNG, or None
        lateral_path: path to lateral PNG, or None
        drop_prob: probability of dropping one view (never both)
        augment: whether to apply augmentations
        image_size: resize target for both views

    Returns:
        List of 1 or 2 PIL images in [frontal, lateral] order.
    """
    def load(path: Optional[str]) -> Optional[Image.Image]:
        if not path:
            return None
        try:
            img = Image.open(path).convert("RGB").resize(image_size, Image.BILINEAR)
            return augment_image(img) if augment else img
        except Exception:
            return None

    frontal = load(frontal_path)
    lateral = load(lateral_path)

    has_frontal = frontal is not None
    has_lateral = lateral is not None

    # View dropout — only when both views exist
    if has_frontal and has_lateral and random.random() < drop_prob:
        if random.random() < 0.5:
            frontal = None
        else:
            lateral = None

    views = [v for v in [frontal, lateral] if v is not None]

    # Fallback: if nothing loaded, try the other view
    if not views:
        fallback = load(frontal_path) or load(lateral_path)
        if fallback:
            views = [fallback]

    return views
