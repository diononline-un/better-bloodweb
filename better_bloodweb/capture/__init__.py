"""Screenshot-Capture mit plattformabhängigen Backends.

Autodetect:
- Wayland + KDE    -> spectacle
- Wayland + wlroots (sway/hyprland) -> grim
- sonst (X11, Windows, macOS) -> mss

Alle Backends liefern numpy.ndarray im BGR-Format (OpenCV-kompatibel).
"""

from __future__ import annotations

import os
import sys

import numpy as np


def _detect_backend() -> str:
    override = os.environ.get("BB_CAPTURE_BACKEND")
    if override:
        return override
    if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        if "kde" in desktop or "plasma" in desktop:
            return "spectacle"
        return "grim"
    return "mss"


def capture(backend: str | None = None) -> np.ndarray:
    """Fullscreen-Screenshot als BGR-Array."""
    name = backend or _detect_backend()
    if name == "spectacle":
        from . import spectacle_backend

        return spectacle_backend.grab()
    if name == "grim":
        from . import grim_backend

        return grim_backend.grab()
    if name == "mss":
        from . import mss_backend

        return mss_backend.grab()
    raise ValueError(f"unbekanntes capture backend: {name}")


def backend_name() -> str:
    return _detect_backend()
