# better-bloodweb

Linux/Wayland-nativer Bloodweb-Prioritizer für Dead by Daylight. Erkennt Items per
Screenshot + Bilderkennung, klickt die laut deinem Profil besten Nodes — ohne jeglichen
Zugriff auf den Game-Prozess.

Inspiriert von [BloodEmporium](https://github.com/IIInitiationnn/BloodEmporium)
(Windows-only). Danke an IIInitiationnn für Konzept, Algorithmus-Idee und Icon-Assets.
better-bloodweb ist eine unabhängige Neuimplementierung, nicht affiliated mit BloodEmporium
oder Behaviour Interactive.

## Warnhinweis (ToS)

Automations-Tools verletzen formal die BHVR Terms of Use ("auto/macro software").
Dieses Tool liest nur Bildschirm-Pixel und synthetisiert Maus-Events auf OS-Ebene
(kein Injection, kein Memory-Zugriff, keine API-Aufrufe) — das Risikoprofil entspricht
dem von BloodEmporium, für das seit 2021 keine verifizierten Bans bekannt sind.
Nutzung auf eigene Gefahr. Kein Bann ist garantiert ausgeschlossen.

## Features

- Preference-Profile pro Charakter (YAML + GUI-Editor, Tiers -999..999)
- Modi: `cheap-first` (BP-effizient), `pref-first` (Priorität egal was kostet), `balanced`
- Wayland-nativ: Capture via spectacle (KDE) / grim (wlroots) / mss (X11, Windows)
- Input via ydotool (Wayland, universal) / pyautogui (Windows, X11)
- **CNN-Klassifikator** (torch, CPU): auf synthetischen Daten trainiert (Icons per
  Alpha-Compositing + Augmentation) — robust gegen Skins, Shader, Helligkeit.
  Template Matching (CLAHE, Multi-Scale) nur als Fallback, wenn kein Modell trainiert ist
- Kalibrier-Assistent für beliebige Auflösungen
- Bloodpoint-Limit, Start/Stop, Live-Log

### Modell trainieren

```bash
.venv/bin/python tools/train_classifier.py            # ~35 min auf CPU
.venv/bin/python tools/train_classifier.py --epochs 20 --batches 800  # besser, langsamer
```

Erzeugt `assets/model.pt` + `assets/classes.json`. Danach nutzt der Detector
automatisch das CNN.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Wayland: ydotool (Input)

```bash
sudo apt install ydotool
sudo usermod -aG input $USER   # danach neu einloggen
systemctl --user enable --now ydotoold 2>/dev/null || ydotoold &
```

### Assets

```bash
./tools/update-assets.sh   # lädt Icon-Set
```

### Start

```bash
.venv/bin/python main.py
```

1. DBD im randlosen Fenster starten, Bloodweb öffnen
2. In better-bloodweb: Kalibrierung durchführen (Assistent führt durch)
3. Profil wählen/erstellen, Tiers setzen, BP-Limit setzen
4. Start — Stop-Button beendet jederzeit (nach dem aktuellen Klick)

Shaders/Farbfilter/Colourblind-Modus im Spiel möglichst deaktivieren; das Matching
ist farb-tolerant, aber nicht farb-immun.

## Sync der Assets

Neue DBD-Chapters bringen neue Items. `./tools/update-assets.sh` zieht aktuelle Icons
(Quelle: BloodEmporium-Repo, das sie mitpflegt). Danke nochmals an deren Maintainer.
