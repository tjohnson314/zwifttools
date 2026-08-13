"""Dump style sequence along a named world-9 route by distance."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).parent.parent
TARGET = sys.argv[1] if len(sys.argv) > 1 else "Fine and Sandy"

surf = json.loads((BASE / "zwift_surfaces" / "world_9.json").read_text(encoding="utf-8"))
routes = json.loads((BASE / "zwift_routes" / "world_9.json").read_text(encoding="utf-8"))["routes"]

xs, zs, styles = [], [], []
for seg in surf["segments"]:
    s = seg.get("style")
    x = seg.get("x", [])
    z = seg.get("z", [])
    xs.extend(x)
    zs.extend(z)
    styles.extend([s] * len(x))
pts = np.column_stack([xs, zs])
tree = cKDTree(pts)
styles = np.array(styles)

route = next(r for r in routes if r.get("name") == TARGET)


def dump(leg, label):
    if not leg or not leg.get("x"):
        return
    d = np.asarray(leg["d"])
    dist, idx = tree.query(np.column_stack([leg["x"], leg["z"]]), k=1)
    lab = styles[idx]
    # collapse into contiguous runs of same style
    print(f"\n=== {label} ({len(d)} pts, {d[-1]-d[0]:.0f} m) ===")
    print(f"{'start_m':>9}{'end_m':>9}{'len_m':>8}  {'style':<22}{'avg_nn_m':>9}")
    i = 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        seg_len = d[min(j, len(d) - 1)] - d[i]
        nn = dist[i:j + 1].mean()
        print(f"{d[i]:>9.0f}{d[min(j,len(d)-1)]:>9.0f}{seg_len:>8.0f}  {str(lab[i]):<22}{nn:>9.1f}")
        i = j + 1


dump(route.get("leadin"), "LEADIN")
dump(route.get("route"), "MAIN")
