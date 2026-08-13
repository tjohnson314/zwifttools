"""Test a combined multi-route affine fit for a world vs the single-best fit."""
import glob
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from race_replay.data_cleaner import load_route_data, haversine

ROOT = Path(__file__).resolve().parent.parent
MAP_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 9


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return re.sub(r"\s+", "-", s)


routes = json.load(open(ROOT / "zwift_routes" / f"world_{MAP_ID}.json", encoding="utf-8"))["routes"]
by_slug = {norm(r.get("name", "")): r for r in routes}

# Gather (matched_localxz, latlng) samples from every cached GPS route for this world.
samples = []  # list of (mx, mz, lat, lng, slug)
for path in glob.glob(str(ROOT / "zwiftmap_surfaces" / "*_route.json")):
    slug = os.path.basename(path).replace("_route.json", "").replace("_", "-")
    r = by_slug.get(slug)
    if r is None:
        continue
    rt = r.get("route") or {}
    rx = np.asarray(rt.get("x", []), float)
    rz = np.asarray(rt.get("z", []), float)
    rd = np.asarray(rt.get("d", []), float)
    if len(rx) < 3:
        continue
    try:
        route = load_route_data(slug)
    except Exception as e:  # noqa: BLE001
        print(f"  {slug}: load error {e}")
        continue
    if route is None:
        continue
    frac = (route.distance - route.distance[0]) / (route.distance[-1] - route.distance[0])
    tgt = frac * rd[-1]
    mx = np.interp(tgt, rd, rx)
    mz = np.interp(tgt, rd, rz)
    for a, b, ll in zip(mx, mz, route.latlng):
        samples.append((a, b, ll[0], ll[1], slug))
    print(f"  matched {slug:26s} ({len(mx)} pts)")

if not samples:
    print("no samples")
    sys.exit()

arr = np.array([(a, b, la, lo) for a, b, la, lo, _ in samples])
design = np.column_stack([arr[:, 0], arr[:, 1], np.ones(len(arr))])
target = arr[:, 2:4]
coef = np.linalg.lstsq(design, target, rcond=None)[0]
f = design @ coef
err = haversine(f[:, 0], f[:, 1], target[:, 0], target[:, 1])
print(f"\nCOMBINED fit ({len(routes)} world routes, {len(samples)} pts from "
      f"{len(set(s[4] for s in samples))} GPS routes):")
print(f"  p50={np.percentile(err,50):.1f}m  p95={np.percentile(err,95):.1f}m  max={err.max():.1f}m")
# per-route residual
slugs = np.array([s[4] for s in samples])
for sl in sorted(set(slugs)):
    m = slugs == sl
    print(f"    {sl:26s} p95={np.percentile(err[m],95):7.1f}m  n={m.sum()}")
print("\ncoef:")
for row in coef.tolist():
    print("   ", row)
