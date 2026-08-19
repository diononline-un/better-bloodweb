"""Input via pyautogui (X11/XTest, Windows, macOS)."""

from __future__ import annotations

import pyautogui

pyautogui.FAILSAFE = False  # Failsafe würde in Fensterecken hart abbrechen


def move(x: int, y: int) -> None:
    pyautogui.moveTo(int(x), int(y))


def left_click() -> None:
    pyautogui.click(button="left")
