"""Inspect nearest segment styles for the first N metres of a world-9 route."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).parent.parent
TARGET = sys.argv[1] if len(sys.argv) > 1 else "Fine and Sandy"
MAX_D = float(sys.argv[2]) if len(sys.argv) > 2 else 260.0

surf = json.loads((BASE / "zwift_surfaces" / "world_9.json").read_text(encoding="utf-8"))
routes = json.loads((BASE / "zwift_routes" / "world_9.json").read_text(encoding="utf-8"))["routes"]

# Build per-segment KD-trees so we can see which distinct road each point sits on.
seg_pts, seg_style, seg_id = [], [], []
for si, seg in enumerate(surf["segments"]):
    x = seg.get("x", [])
    z = seg.get("z", [])
    seg_pts.extend(zip(x, z))
    seg_style.extend([seg.get("style")] * len(x))
    seg_id.extend([si] * len(x))
seg_pts = np.asarray(seg_pts)
seg_style = np.array(seg_style)
seg_id = np.array(seg_id)
tree = cKDTree(seg_pts)

route = next(r for r in routes if r.get("name") == TARGET)
main = route["route"]
d = np.asarray(main["d"])
px = np.asarray(main["x"])
pz = np.asarray(main["z"])

mask = d <= MAX_D
print(f"{TARGET}: first {MAX_D:.0f} m ({mask.sum()} pts)")
print(f"{'d_m':>6}  nearest styles within 20 m (style:dist)")
for i in np.where(mask)[0]:
    idxs = tree.query_ball_point([px[i], pz[i]], r=20.0)
    if not idxs:
        dist, j = tree.query([px[i], pz[i]], k=1)
        print(f"{d[i]:>6.0f}  <none in 20m> nearest {seg_style[j]} @ {dist:.1f}m")
        continue
    # closest vertex per style
    best = {}
    for j in idxs:
        dd = float(np.hypot(px[i] - seg_pts[j, 0], pz[i] - seg_pts[j, 1]))
        st = seg_style[j]
        if st not in best or dd < best[st]:
            best[st] = dd
    ordered = sorted(best.items(), key=lambda kv: kv[1])
    txt = "  ".join(f"{st}:{dd:.1f}" for st, dd in ordered)
    print(f"{d[i]:>6.0f}  {txt}")
