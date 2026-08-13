"""Build a reviewable roadstyle -> surface-category mapping from all worlds.

Classifies each distinct roadstyle by (name, sound, texture folder) and writes a
human-editable JSON map plus prints a review table grouped by surface. The JSON
is the authoritative source consumed by the surface extractor, so ambiguous
calls can be corrected by hand after review.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "zwift_surfaces", "style_surface_map.json")

# Texture folder (lowercased) -> surface category. Most authoritative signal.
TEXTURE_SURFACE = {
    "cobblestone": "Cobbles", "brickroad": "Cobbles", "innsbruck": "Cobbles",
    "france": "Cobbles",  # France cobble-blend styles all use BRICK sound
    "dirtroad": "Dirt", "mountaintrail": "Dirt", "desert": "Dirt",
    "gravelroad": "Gravel", "gravelmountain": "Gravel",
    "beachpath": "Sand", "japanislands": "Sand",
    "snowy": "Snow",
}
SOUND_SURFACE = {
    "BRICK": "Cobbles",
    "GRAVEL": "Gravel",
    "WWISE_ROADTYPE_WOODENBRIDGE": "Wood",
    # DIRT / SNOW / PAVEMENT deliberately excluded (unreliable on their own)
}


def classify(name: str, sound: str, texture: str) -> tuple[str, str]:
    """Return (surface, reason). Name keywords win, then texture, then sound."""
    n = name.upper()
    # Explicit name keywords (strongest, most specific)
    if "COBBLE" in n or "BRICK" in n:
        return "Cobbles", "name"
    if "WOOD" in n:
        return "Wood", "name"
    if "SNOW" in n:
        return "Snow", "name"
    if "GRAVEL" in n:
        return "Gravel", "name"
    if "SAND" in n or "BEACH" in n:
        return "Sand", "name"
    if n == "DOTHINT":
        return "Tarmac", "name (painted guide line)"
    if "DIRT" in n or "MOUNTAIN" in n or "MOUTAIN" in n or "TRAIL" in n or n == "DESERT":
        return "Dirt", "name"
    # Texture folder
    tex = texture.lower()
    if tex in TEXTURE_SURFACE:
        return TEXTURE_SURFACE[tex], f"texture:{texture}"
    # Sound (only the reliable ones)
    if sound in SOUND_SURFACE:
        return SOUND_SURFACE[sound], f"sound:{sound}"
    return "Tarmac", "default"


def main():
    styles: dict[str, dict] = {}
    for w in sorted(os.listdir(os.path.join(ZWIFT, "assets", "Worlds"))):
        p = os.path.join(ZWIFT, "assets", "Worlds", w, "data_1.wad")
        if not (w.startswith("world") and os.path.isfile(p)):
            continue
        wid = int(re.sub(r"\D", "", w) or 0)
        txt = next(v for k, v in read_wad_entries(p, keep_substrings=("roadstyle.xml",)).items()
                   if k.lower().endswith("roadstyle.xml")).decode("utf-8", "replace")
        for m in re.finditer(r"<segment\s+([^>]*?)/>", txt):
            a = m.group(1)
            nm = re.search(r'style="([^"]+)"', a)
            if not nm:
                continue
            snd = re.search(r'sound="([^"]*)"', a)
            tex = re.search(r'texture="([^",]*)', a)
            folder = ""
            if tex:
                parts = re.split(r"[\\/]", tex.group(1))
                folder = parts[-2] if len(parts) >= 2 else tex.group(1)
            e = styles.setdefault(nm.group(1), {
                "sound": snd.group(1) if snd else "",
                "texture": folder, "worlds": set()})
            e["worlds"].add(wid)

    rows = []
    for name, e in styles.items():
        surface, reason = classify(name, e["sound"], e["texture"])
        rows.append({"style": name, "surface": surface, "reason": reason,
                     "sound": e["sound"], "texture": e["texture"],
                     "worlds": sorted(e["worlds"])})

    # Print grouped by surface
    order = ["Tarmac", "Cobbles", "Wood", "Dirt", "Gravel", "Sand", "Snow"]
    rows.sort(key=lambda r: (order.index(r["surface"]) if r["surface"] in order else 9, r["style"]))
    cur = None
    for r in rows:
        if r["surface"] != cur:
            cur = r["surface"]
            print(f"\n=== {cur} ===")
        print(f"  {r['style']:22s} sound={r['sound']:28s} tex={r['texture']:20s} "
              f"[{r['reason']}] worlds={r['worlds']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({r["style"]: r["surface"] for r in
                   sorted(rows, key=lambda r: r["style"])}, f, indent=1)
    print(f"\nWrote {len(rows)} styles -> {OUT}")


if __name__ == "__main__":
    main()
