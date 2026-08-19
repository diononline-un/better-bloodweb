#!/usr/bin/env bash
# Laedt das Icon-Set aus dem BloodEmporium-Repo (Credits: IIInitiationnn/BloodEmporium)
# und normalisiert Dateinamen zu kebab-case Item-Keys in assets/icons/.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Sparse-Clone BloodEmporium (nur assets/)..."
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/IIInitiationnn/BloodEmporium.git "$TMP/og" -q
git -C "$TMP/og" sparse-checkout set assets

mkdir -p "$REPO_DIR/assets/icons"
python3 - "$TMP/og/assets" "$REPO_DIR/assets/icons" <<'EOF'
import os, re, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
# Verzeichnisse ohne Bloodweb-Icons ueberspringen
skip = {"backgrounds", "images", "models", "origins", "presets", "retired", "unused", "onetime"}
prefixes = ("T_UI_iconsFavors_", "T_UI_iconAddon_", "T_UI_iconPerks_", "T_UI_iconItems_",
            "iconFavors_", "iconAddon_", "iconPerks_", "iconItems_", "IconPerks_", "IconAddon_",
            "iconsFavors_", "T_UI_icon")

def key(name: str) -> str:
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break
    # camelCase/PascalCase -> kebab-case
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", name)
    return name.replace(" ", "-").replace("_", "-").lower()

count = 0
for root, dirs, files in os.walk(src):
    dirs[:] = [d for d in dirs if d not in skip]
    for f in files:
        if f.lower().endswith(".png"):
            shutil.copy2(os.path.join(root, f), os.path.join(dst, key(f)))
            count += 1
print(f"{count} Icons nach {dst}")
EOF
echo "Fertig."
