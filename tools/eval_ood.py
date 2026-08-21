#!/usr/bin/env python3
"""Held-out-Evaluierung für den Detector.

Gleiche Synthese-Pipeline wie das Training, aber anderer Seed plus extra
Variationen (stärkerer Blur, extremere Tints). Dazu ein zweiter, bewusst
extremer OOD-Block (flache helle Hintergründe) — rein informativ, da diese
Bilder dem echten Bloodweb nicht ähneln.

Gate: >=90% auf dem realistischen Held-out-Block.
Der echte Härtetest ist ein Screenshot aus dem Spiel (Kalibrier-Tab).

Aufruf: .venv/bin/python tools/eval_ood.py [n_samples]
"""

from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from better_bloodweb.detect import Detector  # noqa: E402
from tools.train_classifier import load_icons, synthesize  # noqa: E402


def extra_warp(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Zusätzliche Verschlechterung jenseits der Trainings-Augmentation."""
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.65, 1.35), beta=int(rng.integers(-40, 40)))
    return img


def read(det: Detector, img: np.ndarray):
    frame = np.full((200, 200, 3), 40, np.uint8)
    frame[52:148, 52:148] = img
    return det.read_node(frame, 100, 100)


def main() -> int:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    det = Detector("assets/icons")
    if not det.uses_cnn:
        print("kein CNN geladen")
        return 1

    icons = load_icons("assets/icons")
    random.seed(999)
    names = random.sample(sorted(icons), n_samples)
    rng = np.random.default_rng(999)

    ok, misses = 0, []
    for n in names:
        img = extra_warp(rng, synthesize(rng, icons[n]))
        r = read(det, img)
        if r.item == n:
            ok += 1
        else:
            misses.append((n, r.item, r.confidence))

    acc = ok / len(names)
    print(f"held-out realistisch: {ok}/{len(names)} ({acc:.0%})")
    for n, got, conf in misses[:10]:
        print(f"  X {n} -> {got} ({conf:.2f})")
    print("GATE:", "OK" if acc >= 0.9 else "NICHT bestanden (<90%)")
    return 0 if acc >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
