# Changelog

## 2026-08-20 — CNN trainiert + evaluiert, Repo auf GitHub

- CNN-Klassifikator trainiert (12 Epochs, CPU, 1371 Klassen): val-acc 99.6%, held-out mit Extra-Degradation 92% (tools/eval_ood.py als Gate)
- Lektion: erste Version kollabierte out-of-distribution (0/30) — Fix: breitere Augmentation (Ringe, Linien, Helligkeit 8-95), per-Bild-Standardisierung, realistischere Held-out-Eval
- Fehlschläge degradieren sicher: niedrige Konfidenz → "unknown" → Node wird übersprungen statt falsch gekauft
- Repo live: github.com/diononline-un/better-bloodweb (private), PR #1 (feat/mvp)
- Offen: ydotool-Installation braucht sudo (User), Kalibrierung + Test am echten Bloodweb

## 2026-08-20 — MVP-Grundgerüst + CNN-Klassifikator

- Repo-Setup: venv (Python 3.14), PyQt6-GUI (Run/Profile/Kalibrierung), Profiles als YAML, Capture-Backends (spectacle/grim/mss), Input-Backends (ydotool/pyautogui)
- Icon-Set aus BloodEmporium gespiegelt (1371 Icons, tools/update-assets.sh, Credits an IIInitiationnn)
- Erkennung: Template Matching (CLAHE, Multi-Scale) als Fallback erwies sich in Tests als zu schwach (~4/15) — daher CNN-Klassifikator auf synthetischen Daten (Alpha-Compositing + Augmentation), Template Matching bleibt Fallback ohne Modell
- Runner: click-and-verify statt Topologie-Graph (nicht erreichbare Nodes lehnt das Spiel ab, weggefressene verschwinden — beides implizit erkannt)
- Offen: Training abschließen + val-acc messen, ydotool-Installation (sudo, User-Aufgabe), Kalibrierung + Testlauf am echten Bloodweb, GitHub-Push
