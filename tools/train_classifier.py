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
    """Bloodweb-artiger Hintergrund, bewusst breit gestreut (Domain-Robustheit):
    dunkle bis mittlere Helligkeit, Verläufe, Textur, Rauschen, Farbtints."""
    g = np.linspace(0, 1, CROP)[None, :]
    g = np.repeat(g, CROP, axis=0)
    if rng.random() < 0.5:
        g = g.T
    base = rng.integers(8, 95)          # absichtlich breiter als der echte Web-BG
    amp = rng.integers(0, 40)
    gray = (base + amp * g).astype(np.float32)
    noise = rng.normal(0, rng.uniform(2, 14), (CROP, CROP)).astype(np.float32)
    gray = np.clip(gray + noise, 0, 255)
    b = np.clip(gray * rng.uniform(0.75, 1.15), 0, 255)
    gg = np.clip(gray * rng.uniform(0.75, 1.15), 0, 255)
    r = np.clip(gray * rng.uniform(0.85, 1.35), 0, 255)
    return cv2.merge([b, gg, r]).astype(np.uint8)


def _draw_web_artifacts(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Node-Ring (Raritätsfarbe) + gelegentliche Verbindungslinien wie im Bloodweb."""
    center = CROP // 2
    if rng.random() < 0.85:  # Ring fast immer sichtbar
        ring_colors = [(60, 40, 30), (30, 90, 40), (90, 60, 140), (140, 60, 160), (60, 30, 140)]
        color = ring_colors[int(rng.integers(0, len(ring_colors)))]
        radius = int(rng.integers(40, 48))
        cv2.circle(img, (center, center), radius, color, int(rng.integers(1, 3)))
    for _ in range(int(rng.integers(0, 3))):  # 0-2 Linien durchs Bild
        angle = rng.uniform(0, np.pi)
        x1 = int(center - 60 * np.cos(angle))
        y1 = int(center - 60 * np.sin(angle))
        x2 = int(center + 60 * np.cos(angle))
        y2 = int(center + 60 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (50, 40, 35), 1)
    return img


def synthesize(rng: np.random.Generator, rgba: np.ndarray) -> np.ndarray:
    """Icon mit Augmentation auf Hintergrund compositen -> CROPxCROP BGR."""
    size = int(rng.integers(54, 91))  # Icon füllt 56-95% des Crops
    icon = cv2.resize(rgba, (size, size))
    bg = make_bg(rng)
    # Positions-Jitter
    maxoff = CROP - size
    y = int(rng.integers(0, max(1, maxoff + 1)))
    x = int(rng.integers(0, max(1, maxoff + 1)))
    y = min(y, maxoff)
    x = min(x, maxoff)
    bgr = icon[:, :, :3].astype(np.float32)
    a = icon[:, :, 3:4].astype(np.float32) / 255.0
    region = bg[y : y + size, x : x + size].astype(np.float32)
    bg[y : y + size, x : x + size] = (bgr * a + region * (1 - a)).astype(np.uint8)
    bg = _draw_web_artifacts(rng, bg)
    # Helligkeit/Kontrast/Blur breit augmentieren
    bg = cv2.convertScaleAbs(bg, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-35, 35))
    if rng.random() < 0.5:
        k = 3 if rng.random() < 0.8 else 5
        bg = cv2.GaussianBlur(bg, (k, k), 0)
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


def normalize(gray: np.ndarray) -> np.ndarray:
    """Per-Bild standardisieren (Mittelwert/Std) — entkoppelt von globaler Helligkeit,
    zwingt das Netz auf Icon-Form statt Hintergrund-Statistik."""
    gray = gray.astype(np.float32) / 255.0
    return (gray - gray.mean()) / (gray.std() + 1e-5)


def batch(rng, icons, names, bs):
    xs, ys = [], []
    for _ in range(bs):
        i = int(rng.integers(0, len(names)))
        img = synthesize(rng, icons[names[i]])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xs.append(normalize(gray)[None])
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
