"""Generate the bundled data for the Power Breakdown Sauce mod.

Emits a compact ES module (``bike-model.mjs``) holding every frame's
per-upgrade-stage CdA bias + weight, every wheel's CdA bias (road + TT) +
weight, the physics constants, the style -> surface map, and the per-bike-type
surface Crr table.  Also emits ``road-styles.mjs`` with the authoritative
per-road resolved style sectors (reusing the same extraction that generates
``zwift_surfaces/world_*.json``) so the mod can look up the surface at the
rider's road position independently of Sauce's own road-style projection.

Run from the repo root:  python tools/build_power_breakdown_data.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bike_comparison.bike_data import get_bike_database, BASE_CDA, REF_FRONTAL_AREA  # noqa: E402
from bike_comparison.physics import (  # noqa: E402
    _REF_HEIGHT_M, _REF_WEIGHT_KG, _HEIGHT_EXPONENT, _WEIGHT_EXPONENT,
)
from shared.surface_lookup import DEFAULT_CRR, FRAME_TYPE_TO_BIKE_TYPE  # noqa: E402

MOD_DIR = os.path.join(ROOT, "mods", "sauce_mod_power_breakdown")
SURFACE_DIR = os.path.join(ROOT, "zwift_surfaces")

def load_frame_names():
    """Game-frame-id -> friendly ZwiftInsider name; only mapped (non-null) frames."""
    path = os.path.join(ROOT, "zwiftdata", "frame_zi_match.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("matches", {})


def build_frames(db):
    names = load_frame_names()
    frames = []
    for fr in db.list_frames():
        zi_name = names.get(fr["frameid"])
        if not zi_name:          # skip frames with no mapped ZI name
            continue
        frames.append({
            "id": fr["frameid"],
            "name": zi_name,
            "type": fr.get("frametype", "Standard"),
            # 6-stage (0..5) cumulative CdA bias and weight (grams).
            "cda": [round(float(c or 0.0), 5) for c in fr["framecda_bias_stages"]],
            "wt": [int(round(float(w or 0.0))) for w in fr["frameweight_g_stages"]],
        })
    frames.sort(key=lambda f: f["name"].lower())
    return frames


def build_wheels(db):
    wheels = []
    for wh in db.list_wheels():
        wheels.append({
            "id": wh["wheelid"],
            "make": wh["wheelmake"],
            "model": wh["wheelmodel"],
            "cda": round(float(wh["wheelcda_bias"] or 0.0), 5),
            "cdaTt": round(float(wh.get("wheelcda_bias_tt") or wh["wheelcda_bias"] or 0.0), 5),
            "wt": int(round(float(wh["wheelweight_g"] or 0.0))),
        })
    return wheels


def build_crr_table():
    with open(os.path.join(ROOT, "zwiftmap_surfaces", "surface_data.json"),
              encoding="utf-8") as f:
        data = json.load(f)
    table = {}
    for bike_type in ("road_bike", "mtb", "gravel_bike"):
        merged = dict(DEFAULT_CRR)
        for k, v in (data.get("crr_values", {}).get(bike_type, {}) or {}).items():
            if v is not None:
                merged[k] = v
        table[bike_type] = merged
    return table


def build_style_map():
    with open(os.path.join(SURFACE_DIR, "style_surface_map.json"), encoding="utf-8") as f:
        return json.load(f)


_ROAD_ID_RE = re.compile(r"<id>(\d+)</id>")
_DEFAULT_STYLE_RE = re.compile(r"<defaultStyle>(\d+)</defaultStyle>")


def build_road_styles(style_surface):
    """Per-road resolved style sectors, keyed by Sauce courseId (== our mapID).

    Reuses the exact authoritative extraction (``_markers`` + last-covering-wins
    resolution) that generates ``zwift_surfaces/world_*.json`` so the mod's
    surface lookup matches the in-game-validated surface map road-for-road.

    Shape: ``{courseId: {roadId: {"d": defaultStyleName, "s": [[a, b, name], ...]}}}``
    where ``a``/``b`` are normalised road-percent [0, 1] arc positions (last
    covering sector wins over the default). Pure-tarmac roads with no overrides
    are omitted; the mod treats a missing road as NORMAL (-> Tarmac).
    """
    from shared import surface_map_dev as smd  # imported lazily; needs local WADs

    exs = smd._exs
    wads = smd._world_wads()
    if not wads:
        return None
    out = {}
    for map_id in sorted(wads):
        try:
            road_txt, styles = smd._read_road_style(map_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  world {map_id}: skipped road styles ({exc})")
            continue
        roads = {}
        for block in exs._road_blocks(road_txt):
            idm = _ROAD_ID_RE.search(block)
            if not idm:
                continue
            rid = int(idm.group(1))
            dm = _DEFAULT_STYLE_RE.search(block)
            base = int(dm.group(1)) if dm else exs.UNSET_STYLE
            if base == exs.UNSET_STYLE or not (0 <= base < len(styles)):
                base = 0  # NORMAL / tarmac
            default_name = styles[base]
            sectors = []
            for a, b, s in exs._markers(block):
                if 0 <= s < len(styles):
                    sectors.append([round(a, 4), round(b, 4), styles[s]])
            # Omit pure-tarmac roads with no overrides to keep the bundle lean;
            # the mod defaults an unknown road to NORMAL (-> Tarmac).
            if not sectors and style_surface.get(default_name, "Tarmac") == "Tarmac":
                continue
            roads[rid] = {"d": default_name, "s": sectors}
        if roads:
            out[map_id] = roads
    return out


def main():
    db = get_bike_database()

    style_surface = build_style_map()

    model = {
        "constants": {
            "BASE_CDA": BASE_CDA,
            "REF_FRONTAL_AREA": REF_FRONTAL_AREA,
            "REF_HEIGHT_M": _REF_HEIGHT_M,
            "REF_WEIGHT_KG": _REF_WEIGHT_KG,
            "HEIGHT_EXPONENT": _HEIGHT_EXPONENT,
            "WEIGHT_EXPONENT": _WEIGHT_EXPONENT,
            "AIR_DENSITY": 1.225,
            "GRAVITY": 9.8067,
        },
        "frameTypeToBikeType": FRAME_TYPE_TO_BIKE_TYPE,
        "crr": build_crr_table(),
        "styleMap": style_surface,
        "frames": build_frames(db),
        "wheels": build_wheels(db),
    }

    out_mjs = os.path.join(MOD_DIR, "bike-model.mjs")
    with open(out_mjs, "w", encoding="utf-8") as f:
        f.write("// AUTO-GENERATED by tools/build_power_breakdown_data.py — do not edit.\n")
        f.write("export const MODEL = ")
        json.dump(model, f, separators=(",", ":"), ensure_ascii=False)
        f.write(";\n")
    print(f"wrote {out_mjs} ({os.path.getsize(out_mjs) // 1024} KB, "
          f"{len(model['frames'])} frames, {len(model['wheels'])} wheels)")

    road_styles = build_road_styles(style_surface)
    out_roads = os.path.join(MOD_DIR, "road-styles.mjs")
    if road_styles is None:
        print(f"skipped {out_roads}: no local Zwift WADs found "
              f"(existing file, if any, left untouched)")
    else:
        with open(out_roads, "w", encoding="utf-8") as f:
            f.write("// AUTO-GENERATED by tools/build_power_breakdown_data.py — do not edit.\n")
            f.write("export const ROAD_STYLES = ")
            json.dump(road_styles, f, separators=(",", ":"), ensure_ascii=False)
            f.write(";\n")
        n_roads = sum(len(v) for v in road_styles.values())
        print(f"wrote {out_roads} ({os.path.getsize(out_roads) // 1024} KB, "
              f"{len(road_styles)} worlds, {n_roads} roads)")


if __name__ == "__main__":
    main()
