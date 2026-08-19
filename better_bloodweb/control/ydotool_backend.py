"""Input via ydotool (uinput-Kernel-Interface, Wayland-universal).

Braucht laufenden ydotoold (Socket /run/user/<uid>/.ydotool_socket oder
/tmp/.ydotool_socket je nach Setup) bzw. Root-Rechte.
"""

from __future__ import annotations

import subprocess


def _run(*args: str) -> None:
    subprocess.run(["ydotool", *args], check=True)


def move(x: int, y: int) -> None:
    _run("mousemove", "--absolute", "-x", str(int(x)), "-y", str(int(y)))


def left_click() -> None:
    # 0xC0 = left button press + release
    _run("click", "0xC0")
