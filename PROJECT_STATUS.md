---
status: in-development
progress: 10
version: "0.0.1"
tags: [python, linux, wayland, opencv, pyqt6, dead-by-daylight, automation]
scope: "Linux/Wayland-nativer Bloodweb-Prioritizer für Dead by Daylight: Screenshot + Icon-Erkennung + Klick-Automation, preference profiles pro Charakter, kein Prozess-Zugriff"
last_updated: "2026-08-20"
next_steps:
  - Core-Module fertigstellen (capture/control/detect/graph/profiles)
  - GUI (PyQt6): Profil-Editor, Tier-Slider, Run-Panel, Kalibrier-Assistent
  - Icon-Assets integrieren + update-assets.sh
  - ydotool-Setup dokumentieren (sudo, input-Gruppe)
  - Kalibrierung + Testlauf am echten Bloodweb
---

# Aktueller Stand
- Repo initialisiert, venv mit opencv/numpy/pyyaml/networkx/PyQt6/mss auf Python 3.14 verifiziert
- Konzept: pixel-basiert (spectacle/grim/mss), Input via ydotool/pyautogui — kein Memory-/Prozess-Zugriff
- Inspiriert von IIInitiationnn/BloodEmporium (Windows-only), Credits in README
