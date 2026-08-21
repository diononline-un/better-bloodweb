"""Capture via mss (X11, Windows, macOS)."""

from __future__ import annotations

import cv2
import mss
import numpy as np


def grab() -> np.ndarray:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])
        return np.asarray(shot)[:, :, :3].copy()  # BGRA -> BGR (Kanäle 0-2)
