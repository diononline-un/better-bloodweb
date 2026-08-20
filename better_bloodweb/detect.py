"""Icon-Erkennung auf Bloodweb-Screenshots.

Ansatz: Grayscale-Template-Matching mit CLAHE-Vornormalisierung (Helligkeits-/
Kontrastausgleich) und Multi-Scale-Fits. Grayscale ist farb-tolerant; CLAHE macht
das Matching zusätzlich robust gegen Shader, Nacht-Modi und Skin-Hintergründe.

Optional (wenn assets/models/best.pt vorhanden): YOLO-Detektor via ultralytics —
kontextbasierte Erkennung wie BloodEmporium, deutlich unempfindlicher gegen
Skins/Hintergründe. Gewichte nicht enthalten (Lizenz OG-Repo unklar), siehe README.

Rarität (Kosten) wird separat über die Ringfarbe am Node-Rand geschätzt.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

MATCH_THRESHOLD = 0.55
NODE_CROP_RADIUS = 48  # px um Node-Zentrum (bei Referenzauflösung 1920x1080)
CNN_INPUT = 96  # Input-Kantenlänge des Klassifikators (muss zu Training passen)

# Ringfarbe (BGR-Hue in HSV) -> geschätzte Node-Kosten in Bloodpoints.
# Werte sind Näherungen; DBD ändert Kosten gelegentlich pro Patch.
RARITY_COST = {
    "common": 3000,
    "uncommon": 4000,
    "rare": 5000,
    "very_rare": 6000,
    "ultra_rare": 7000,
    "unknown": 4000,
}

# Hue-Bereiche (OpenCV H 0-179) für Node-Ringfarben
_RARITY_HUES = [
    ("ultra_rare", (0, 10), (170, 179)),   # pink/rot
    ("very_rare", (125, 170), None),       # lila
    ("rare", (95, 125), None),             # grün-lila Übergang grob
    ("uncommon", (35, 95), None),          # grün/gelb
    ("common", (10, 35), None),            # braun/gelb
]


@dataclass
class NodeReading:
    item: str          # Template-Name oder "unknown"
    confidence: float  # 0..1
    rarity: str
    cost: int


class IconDB:
    """Lädt Icon-Templates aus assets/icons/<name>.png (Grayscale)."""

    def __init__(self, icons_dir: str):
        self.templates: dict[str, np.ndarray] = {}
        if not os.path.isdir(icons_dir):
            return
        for fname in sorted(os.listdir(icons_dir)):
            if not fname.lower().endswith(".png"):
                continue
            img = cv2.imread(os.path.join(icons_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # gleiche Normalisierung wie im Matching-Pfad (CLAHE)
            img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)
            self.templates[os.path.splitext(fname)[0]] = img

    def __len__(self) -> int:
        return len(self.templates)


class Detector:
    """Nutzt den trainierten CNN-Klassifikator (assets/model.pt), falls vorhanden.
    Sonst Fallback: Template Matching. Gleiche API (read_node)."""

    PREFILTER_TOP = 20  # Kandidaten aus dem groben Vorfilter fürs Feinmatching
    CNN_CONFIDENCE_FLOOR = 0.15  # Softmax-Wahrscheinlichkeit, darunter = unknown

    def __init__(self, icons_dir: str):
        self.db = IconDB(icons_dir)
        self._scaled_cache: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
        self._tiny_cache: list[tuple[str, np.ndarray]] | None = None
        self._cnn = None
        self._cnn_classes: list[str] = []
        assets_dir = os.path.dirname(icons_dir.rstrip("/"))
        model_path = os.path.join(assets_dir, "model.pt")
        classes_path = os.path.join(assets_dir, "classes.json")
        if os.path.exists(model_path) and os.path.exists(classes_path):
            try:
                self._load_cnn(model_path, classes_path)
            except Exception as exc:  # Modell kaputt -> Template-Fallback
                print(f"CNN nicht ladbar ({exc}), Fallback auf Template Matching.")

    def _load_cnn(self, model_path: str, classes_path: str) -> None:
        import json

        import torch

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tools.train_classifier import build_model

        with open(classes_path, "r", encoding="utf-8") as f:
            self._cnn_classes = json.load(f)
        model = build_model(len(self._cnn_classes))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        self._cnn = model

    @property
    def uses_cnn(self) -> bool:
        return self._cnn is not None

    def _scaled_templates(self, ch: int, cw: int) -> list[tuple[str, np.ndarray]]:
        """Pro Crop-Größe einmalig alle Templates auf die 4 Fits resizen (Cache)."""
        key = (ch, cw)
        if key in self._scaled_cache:
            return self._scaled_cache[key]
        out: list[tuple[str, np.ndarray]] = []
        for name, tmpl in self.db.templates.items():
            th, tw = tmpl.shape
            for fit in (0.5, 0.65, 0.8, 0.95):
                s = min(cw / tw, ch / th) * fit
                t = cv2.resize(tmpl, None, fx=s, fy=s)
                if t.shape[0] >= 8 and t.shape[1] >= 8:
                    out.append((name, t))
        self._scaled_cache[key] = out
        return out

    def _tiny_templates(self) -> list[tuple[str, np.ndarray]]:
        """24px-Versionen aller Templates für den groben Vorfilter (Cache)."""
        if self._tiny_cache is None:
            self._tiny_cache = [
                (name, cv2.resize(tmpl, (24, 24))) for name, tmpl in self.db.templates.items()
            ]
        return self._tiny_cache

    def read_node(self, frame: np.ndarray, x: int, y: int, scale: float = 1.0) -> NodeReading:
        """Liest Item + Rarität eines Nodes an Position (x, y) im Frame."""
        r = int(NODE_CROP_RADIUS * scale)
        h, w = frame.shape[:2]
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w, x + r), min(h, y + r)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return NodeReading("unknown", 0.0, "unknown", RARITY_COST["unknown"])

        item, conf = self._match_item(crop)
        rarity = self._read_rarity(crop)
        return NodeReading(item, conf, rarity, RARITY_COST.get(rarity, 4000))

    def _match_item(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if self.uses_cnn:
            return self._classify_cnn(crop_bgr)
        return self._match_template(crop_bgr)

    def _classify_cnn(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        import torch

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tools.train_classifier import normalize

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (CNN_INPUT, CNN_INPUT))
        norm = normalize(gray)
        with torch.no_grad():
            out = self._cnn(torch.from_numpy(norm[None, None]))
            probs = torch.softmax(out, dim=1)[0]
        conf, idx = float(probs.max()), int(probs.argmax())
        if conf < self.CNN_CONFIDENCE_FLOOR:
            return "unknown", conf
        return self._cnn_classes[idx], conf

    def _match_template(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if not self.db.templates:
            return "unknown", 0.0
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # CLAHE: lokaler Kontrastausgleich gegen Shader/Skin/Helligkeit
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        ch, cw = gray.shape

        # Stufe 1: grober Vorfilter mit 24px-Templates (billig), Top-N behalten
        small = cv2.resize(gray, (48, 48))
        coarse: list[tuple[str, float]] = []
        for name, t in self._tiny_templates():
            res = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
            coarse.append((name, float(res.max())))
        coarse.sort(key=lambda x: x[1], reverse=True)
        candidates = {name for name, _ in coarse[: self.PREFILTER_TOP]}

        # Stufe 2: Feinmatching nur auf Kandidaten, volle Fit-Range
        best_name, best_score = "unknown", 0.0
        for name, t in self._scaled_templates(ch, cw):
            if name not in candidates:
                continue
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_name, best_score = name, score
        if best_score < MATCH_THRESHOLD:
            return "unknown", best_score
        return best_name, best_score

    @staticmethod
    def _read_rarity(crop_bgr: np.ndarray) -> str:
        """Schätzt Rarität über dominante gesättigte Farbe am Crop-Rand (Node-Ring)."""
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cy, cx, r = h // 2, w // 2, min(h, w) // 2 - 2
        cv2.circle(mask, (cx, cy), r, 255, 2)
        sat = hsv[:, :, 1]
        ring = hsv[(mask > 0) & (sat > 80)]
        if ring.size == 0:
            return "unknown"
        hue = int(np.median(ring[:, 0]))
        for entry in _RARITY_HUES:
            name, lo_hi, alt = entry[0], entry[1], entry[2]
            lo, hi = lo_hi
            if lo <= hue <= hi or (alt and alt[0] <= hue <= alt[1]):
                return name
        return "unknown"
