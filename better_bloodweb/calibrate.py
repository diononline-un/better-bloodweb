"""Kalibrierung: Node-Positionen des Bloodwebs, normalisiert (0..1) zur Auflösung.

Der Kalibrier-Assistent (GUI) zeigt einen Screenshot des Bloodwebs, der Nutzer
klickt jedes Node-Zentrum einmal an. Gespeichert wird calibration.json.

Topology wird bewusst NICHT kalibriert: der Runner arbeitet nach
"click-and-verify" — nicht erreichbare Nodes lehnt das Spiel still ab,
gegessene Nodes verschwinden von selbst. Beides wird im Loop erkannt.
"""

from __future__ import annotations

import json
import os

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration.json")


def load() -> list[tuple[float, float]]:
    """Liste normalisierter Node-Positionen [(x, y), ...], leer wenn nicht kalibriert."""
    if not os.path.exists(CALIBRATION_FILE):
        return []
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(float(p["x"]), float(p["y"])) for p in data.get("nodes", [])]


def save(nodes: list[tuple[float, float]]) -> None:
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump({"nodes": [{"x": x, "y": y} for x, y in nodes]}, f, indent=2)


def to_pixels(nodes: list[tuple[float, float]], width: int, height: int) -> list[tuple[int, int]]:
    return [(int(x * width), int(y * height)) for x, y in nodes]
