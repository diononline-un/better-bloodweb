"""Capture via KDE spectacle (Wayland, kein Dialog)."""

from __future__ import annotations

import subprocess
import tempfile

import cv2
import numpy as np


def grab() -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        path = f.name
        subprocess.run(
            ["spectacle", "-b", "-n", "-f", "-o", path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("spectacle lieferte kein lesbares Bild")
    return img
