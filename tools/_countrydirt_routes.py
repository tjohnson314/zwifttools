"""Identify Makuri (world 9) routes with the most COUNTRYDIRT surface."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).parent.parent
surf = json.loads((BASE / "zwift_surfaces" / "world_9.json").read_text(encoding="utf-8"))
routes = json.loads((BASE / "zwift_routes" / "world_9.json").read_text(encoding="utf-8"))["routes"]

# KD-tree of every road vertex, labelled by its raw style.
xs, zs, styles = [], [], []
for seg in surf["segments"]:
    s = seg.get("style")
    x = seg.get("x", [])
    z = seg.get("z", [])
    xs.extend(x)
    zs.extend(z)
    styles.extend([s] * len(x))
tree = cKDTree(np.column_stack([xs, zs]))
styles = np.array(styles)


def leg_style_dist(leg):
    """Return {style: metres} for one leg."""
    out = {}
    if not leg or not leg.get("x"):
        return out
    d = leg["d"]
    _, idx = tree.query(np.column_stack([leg["x"], leg["z"]]), k=1)
    lab = styles[idx]
    for i in range(len(d) - 1):
        seg_len = d[i + 1] - d[i]
        if seg_len <= 0:
            continue
        out[lab[i]] = out.get(lab[i], 0.0) + seg_len
    return out


rows = []
for r in routes:
    tot = {}
    for leg in (r.get("leadin"), r.get("route")):
        for st, dist in leg_style_dist(leg).items():
            tot[st] = tot.get(st, 0.0) + dist
    total_m = sum(tot.values())
    cd = tot.get("COUNTRYDIRT", 0.0)
    if total_m > 0:
        rows.append((r.get("name", ""), cd, total_m, cd / total_m * 100.0))

rows.sort(key=lambda x: -x[3])
print(f"{'Route':<45}{'CountryDirt m':>14}{'Total m':>12}{'%':>8}")
print("-" * 79)
for name, cd, total, pct in rows:
    if cd > 0:
        print(f"{name[:44]:<45}{cd:>14,.0f}{total:>12,.0f}{pct:>7.1f}%")
