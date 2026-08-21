"""Preference-Profile: YAML-Dateien in profiles/.

Format:
    character: claudette
    mode: balanced            # cheap-first | pref-first | balanced
    bp_limit: 500000
    tiers:
      bloody-party-streamers: 10
      map-offering-junk: -5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
MODES = ("cheap-first", "pref-first", "balanced")


@dataclass
class Profile:
    name: str
    character: str = ""
    mode: str = "balanced"
    bp_limit: int = 500_000
    tiers: dict[str, int] = field(default_factory=dict)

    def tier_of(self, item: str) -> int:
        return self.tiers.get(item, 0)

    def score(self, item: str, cost: int) -> float:
        """Kauf-Score je nach Modus. Höher = früher kaufen."""
        tier = self.tier_of(item)
        if self.mode == "cheap-first":
            # billig zuerst, Tiers als Tiebreak
            return -cost + tier * 10_000
        if self.mode == "pref-first":
            return tier * 1_000_000 - cost
        # balanced: Tier-pro-BP
        return (tier + 1) / max(cost, 1) * 1_000_000


def path_for(name: str) -> str:
    return os.path.join(PROFILES_DIR, f"{name}.yaml")


def load(name: str) -> Profile:
    with open(path_for(name), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Profile(
        name=name,
        character=data.get("character", ""),
        mode=data.get("mode", "balanced"),
        bp_limit=int(data.get("bp_limit", 500_000)),
        tiers={str(k): int(v) for k, v in (data.get("tiers") or {}).items()},
    )


def save(profile: Profile) -> None:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    data = {
        "character": profile.character,
        "mode": profile.mode,
        "bp_limit": profile.bp_limit,
        "tiers": dict(sorted(profile.tiers.items())),
    }
    with open(path_for(profile.name), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def list_profiles() -> list[str]:
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(PROFILES_DIR)
        if f.endswith(".yaml")
    )
