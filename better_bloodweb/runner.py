"""Run-Loop: Screenshot -> Nodes lesen -> besten Node klicken -> verifizieren.

Click-and-verify statt Topologie-Graph:
- Node nach Klick weiterlesen. Item weiterhin erkannt -> Kauf fehlgeschlagen
  (nicht erreichbar / Entity frisst gerade) -> Node bekommt Fail-Zähler.
- Node nicht mehr erkennbar -> gekauft oder weggefressen -> erledigt.
- Fail-Schwelle erreicht -> Node wird diese Session übersprungen.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from . import calibrate, capture, control
from .detect import Detector, NodeReading
from .profiles import Profile

MAX_FAILS = 2
CLICK_DELAY = (0.35, 0.9)       # s zwischen Aktionen
AFTER_CLICK_WAIT = (0.8, 1.4)   # s bis zur Verifikation


@dataclass
class RunStats:
    bp_spent: int = 0
    purchased: list[str] = field(default_factory=list)
    skipped: int = 0


class Runner:
    def __init__(
        self,
        profile: Profile,
        detector: Detector,
        log: Callable[[str], None] = print,
        should_stop: Callable[[], bool] = lambda: False,
    ):
        self.profile = profile
        self.detector = detector
        self.log = log
        self.should_stop = should_stop
        self.stats = RunStats()
        self._fails: dict[int, int] = {}
        self._done: set[int] = set()

    def _pick_node(self, readings: dict[int, NodeReading]) -> int | None:
        best_idx, best_score = None, float("-inf")
        for idx, r in readings.items():
            if idx in self._done or self._fails.get(idx, 0) >= MAX_FAILS:
                continue
            if r.confidence < 0.3 and r.item == "unknown":
                continue  # leer/weggefressen
            score = self.profile.score(r.item, r.cost)
            if score > best_score:
                best_idx, best_score = idx, score
        return best_idx

    def step(self) -> bool:
        """Ein Loop-Durchlauf. False = Run beenden."""
        if self.should_stop():
            self.log("Stop angefordert.")
            return False
        if self.stats.bp_spent >= self.profile.bp_limit:
            self.log(f"BP-Limit erreicht ({self.stats.bp_spent}).")
            return False

        nodes = calibrate.load()
        if not nodes:
            self.log("Nicht kalibriert — Abbruch.")
            return False

        frame = capture.capture()
        h, w = frame.shape[:2]
        pixels = calibrate.to_pixels(nodes, w, h)
        scale = w / 1920.0

        readings = {
            i: self.detector.read_node(frame, x, y, scale) for i, (x, y) in enumerate(pixels)
        }

        idx = self._pick_node(readings)
        if idx is None:
            self.log("Kein kaufbaren Node mehr — Web fertig oder leer.")
            return False

        x, y = pixels[idx]
        before = readings[idx]
        self.log(
            f"Klicke Node {idx}: {before.item} ({before.rarity}, ~{before.cost} BP, conf {before.confidence:.2f})"
        )
        control.click(x, y)
        time.sleep(random.uniform(*AFTER_CLICK_WAIT))

        # Verifizieren
        frame2 = capture.capture()
        after = self.detector.read_node(frame2, x, y, scale)
        same = after.item == before.item and after.confidence > 0.3
        if same:
            self._fails[idx] = self._fails.get(idx, 0) + 1
            self.log(f"  Kauf fehlgeschlagen ({self._fails[idx]}/{MAX_FAILS}) — evtl. nicht erreichbar.")
        else:
            self._done.add(idx)
            self.stats.bp_spent += before.cost
            self.stats.purchased.append(before.item)
            self.log(f"  Gekauft. Gesamt ~{self.stats.bp_spent} BP.")

        time.sleep(random.uniform(*CLICK_DELAY))
        return True

    def run(self, max_steps: int = 200) -> RunStats:
        self.log(
            f"Run startet: Profil={self.profile.name}, Modus={self.profile.mode}, "
            f"Limit={self.profile.bp_limit} BP, Capture={capture.backend_name()}, Input={control.backend_name()}"
        )
        for _ in range(max_steps):
            if not self.step():
                break
        self.log(f"Run beendet: {len(self.stats.purchased)} Items, ~{self.stats.bp_spent} BP.")
        return self.stats
