---
status: in-development
progress: 60
version: "0.0.1"
tags: [python, linux, wayland, opencv, pyqt6, dead-by-daylight, automation]
scope: "Linux/Wayland-nativer Bloodweb-Prioritizer für Dead by Daylight: Screenshot + CNN-Icon-Erkennung + Klick-Automation, Preference-Profile pro Charakter, kein Prozess-Zugriff"
last_updated: "2026-08-20"
next_steps:
  - ydotool installieren (sudo nötig, User-Aufgabe — siehe README)
  - Kalibrierung am echten Bloodweb + Testlauf mit kleinem BP-Limit
  - Real-Screenshot gegen Detector verifizieren (bisher nur synthetisch evaluiert)
  - PR #1 mergen nach Review
---

# Aktueller Stand
- MVP steht: PyQt6-GUI (Run/Profile/Kalibrierung), YAML-Profile, Capture-Backends (spectacle/grim/mss), Input (ydotool/pyautogui), Runner mit click-and-verify
- CNN-Klassifikator (torch CPU, 1371 Klassen): val-acc 99.6%, held-out 92% — Template Matching nur Fallback
- 1371 Icons aus BloodEmporium gespiegelt (update-assets.sh), Credits in README
- Repo: github.com/diononline-un/better-bloodweb, PR #1 offen
