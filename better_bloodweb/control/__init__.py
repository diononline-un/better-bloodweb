"""Maus-Input mit plattformabhängigen Backends.

Autodetect:
- Wayland -> ydotool (uinput, compositor-unabhängig)
- sonst   -> pyautogui (X11 via XTest, Windows, macOS)
"""

from __future__ import annotations

import os
import sys
import time
import random


def _detect_backend() -> str:
    override = os.environ.get("BB_INPUT_BACKEND")
    if override:
        return override
    if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
        return "ydotool"
    return "pyautogui"


def _backend():
    name = _detect_backend()
    if name == "ydotool":
        from . import ydotool_backend

        return ydotool_backend
    if name == "pyautogui":
        from . import pyautogui_backend

        return pyautogui_backend
    raise ValueError(f"unbekanntes input backend: {name}")


def click(x: int, y: int, humanize: bool = True) -> None:
    """Linksklick an absoluter Bildschirmposition, optional mit menschlichem Timing."""
    be = _backend()
    be.move(x, y)
    if humanize:
        time.sleep(random.uniform(0.08, 0.22))
    be.left_click()


def backend_name() -> str:
    return _detect_backend()
