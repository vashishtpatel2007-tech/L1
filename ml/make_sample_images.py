"""Generates a synthetic 3-class image dataset so you can rehearse the ENTIRE
CNN pipeline tonight without having any real data.

    python ml/make_sample_images.py
    python ml/train_cnn.py --data data/raw/images --epochs 5

Replace data/raw/images with real photos tomorrow; the command is identical.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "images"
CLASSES = {"healthy": (60, 160, 70), "diseased": (165, 130, 45), "waterlogged": (55, 95, 165)}


def make(colour: tuple[int, int, int], blobs: int, seed: int) -> Image.Image:
    rnd = random.Random(seed)
    base = np.full((224, 224, 3), colour, dtype="uint8")
    noise = np.random.default_rng(seed).normal(0, 22, (224, 224, 3))
    im = Image.fromarray(np.clip(base + noise, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(im)
    for _ in range(blobs):
        x, y = rnd.randint(0, 200), rnd.randint(0, 200)
        r = rnd.randint(8, 34)
        shade = tuple(max(0, min(255, c + rnd.randint(-70, 70))) for c in colour)
        d.ellipse([x, y, x + r, y + r], fill=shade)
    return im.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.1)))


def main(per_class: int = 90) -> None:
    for i, (name, colour) in enumerate(CLASSES.items()):
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for j in range(per_class):
            make(colour, blobs=4 + i * 5, seed=i * 1000 + j).save(d / f"{name}_{j:03d}.jpg", quality=88)
        print(f"{name}: {per_class} images -> {d}")
    print(f"\nDone. Train with:\n  python ml/train_cnn.py --data {OUT} --epochs 5")


if __name__ == "__main__":
    main()
