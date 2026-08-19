#!/usr/bin/env python3
"""Trainiert den Icon-Klassifikator auf synthetischen Daten.

Warum synthetisch: alle Icons liegen als Game-Assets vor. Wir rendern sie per
Alpha-Compositing auf realistische Bloodweb-Hintergründe (dunkle Verläufe,
Rauschen, Tint, Blur, Skalierungs-Jitter) — so lernt das Netz Form statt
Pixelperfect, inkl. Shader-/Skin-Toleranz.

Output: assets/model.pt + assets/classes.json
Aufruf: .venv/bin/python tools/train_classifier.py [--epochs N] [--limit K]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

CROP = 96  # Input-Größe des Netzes (px, quadratisch)


def load_icons(icons_dir: str) -> dict[str, np.ndarray]:
    out = {}
    for f in sorted(os.listdir(icons_dir)):
        if not f.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(icons_dir, f), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            out[f[:-4]] = img
    return out


def make_bg(rng: np.random.Generator) -> np.ndarray:
    """Dunkler Bloodweb-artiger Hintergrund: Verlauf + leichte Textur."""
    g = np.linspace(0, 1, CROP)[None, :]
    g = np.repeat(g, CROP, axis=0)
    if rng.random() < 0.5:
        g = g.T  # mal horizontal, mal vertikal
    base = rng.integers(10, 45)
    amp = rng.integers(5, 35)
    gray = (base + amp * g).astype(np.float32)
    noise = rng.normal(0, rng.uniform(2, 10), (CROP, CROP)).astype(np.float32)
    gray = np.clip(gray + noise, 0, 255)
    # leichter Farbton (Bloodweb wirkt bräunlich/violett)
    b = np.clip(gray * rng.uniform(0.8, 1.1), 0, 255)
    r = np.clip(gray * rng.uniform(0.9, 1.3), 0, 255)
    return cv2.merge([b, gray, r]).astype(np.uint8)


def synthesize(rng: np.random.Generator, rgba: np.ndarray) -> np.ndarray:
    """Icon mit Augmentation auf Hintergrund compositen -> CROPxCROP BGR."""
    size = int(rng.integers(56, 89))  # Icon füllt 58-92% des Crops
    icon = cv2.resize(rgba, (size, size))
    bg = make_bg(rng)
    # Positions-Jitter
    maxoff = CROP - size
    y = int(rng.integers(max(1, maxoff // 2 - 6), max(2, maxoff // 2 + 6)))
    x = int(rng.integers(max(1, maxoff // 2 - 6), max(2, maxoff // 2 + 6)))
    y = min(y, maxoff)
    x = min(x, maxoff)
    bgr = icon[:, :, :3].astype(np.float32)
    a = icon[:, :, 3:4].astype(np.float32) / 255.0
    region = bg[y : y + size, x : x + size].astype(np.float32)
    bg[y : y + size, x : x + size] = (bgr * a + region * (1 - a)).astype(np.uint8)
    # Helligkeit/Blur-Augmentation
    bg = cv2.convertScaleAbs(bg, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-25, 25))
    if rng.random() < 0.4:
        bg = cv2.GaussianBlur(bg, (3, 3), 0)
    return bg


def build_model(num_classes: int):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 48
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 24
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 12
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),  # 6
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.2), nn.Linear(256, num_classes),
    )


def batch(rng, icons, names, bs):
    xs, ys = [], []
    for _ in range(bs):
        i = int(rng.integers(0, len(names)))
        img = synthesize(rng, icons[names[i]])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        xs.append(gray[None])
        ys.append(i)
    return np.stack(xs), np.array(ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batches", type=int, default=300, help="Batches pro Epoch")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="nur erste K Icons (Debug)")
    args = ap.parse_args()

    icons_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icons")
    icons = load_icons(icons_dir)
    names = sorted(icons)
    if args.limit:
        names = names[: args.limit]
        icons = {n: icons[n] for n in names}
    print(f"{len(names)} Klassen")

    import torch
    from torch import nn

    torch.manual_seed(0)
    rng = np.random.default_rng(42)
    model = build_model(len(names))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        tot, hit, n = 0.0, 0, 0
        for _ in range(args.batches):
            xs, ys = batch(rng, icons, names, args.bs)
            xt = torch.from_numpy(xs)
            yt = torch.from_numpy(ys)
            opt.zero_grad()
            out = model(xt)
            loss = lossf(out, yt)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yt)
            hit += int((out.argmax(1) == yt).sum())
            n += len(yt)
        print(f"epoch {epoch+1}/{args.epochs}: loss={tot/n:.4f} train-acc={hit/n:.3f}", flush=True)

    # Validierung: frische synthetische Samples
    model.eval()
    vrng = np.random.default_rng(123)
    xs, ys = batch(vrng, icons, names, min(2000, 20 * len(names)))
    with torch.no_grad():
        out = model(torch.from_numpy(xs))
    acc = float((out.argmax(1) == torch.from_numpy(ys)).float().mean())
    print(f"val-acc (frische Synthetik): {acc:.3f}")

    out_dir = os.path.dirname(icons_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(names, f)
    print(f"gespeichert: {out_dir}/model.pt, classes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
