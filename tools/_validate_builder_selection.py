"""Validate builder selection for world 9: leadin auto-detect + coverage tie-break."""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from race_replay.data_cleaner import load_route_data
from tools._build_world_calibration import (
    _leg_geom, _fit_geom, _is_better, P95_LIMIT, COVERAGE_TIE_M,
)

ROOT = Path(__file__).resolve().parent.parent


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", "-", re.sub(r"[^a-z0-9]+", " ", s.lower()).strip())


routes = json.load(open(ROOT / "zwift_routes" / "world_9.json", encoding="utf-8"))["routes"]
by_slug = {norm(r.get("name", "")): r for r in routes}
cands = ["makuri-madness", "neokyo-all-nighter", "red-zone-repeats", "shisa-shakedown"]

print(f"P95_LIMIT={P95_LIMIT}  COVERAGE_TIE_M={COVERAGE_TIE_M}\n")
best = None
for slug in cands:
    r = by_slug.get(slug)
    route = load_route_data(slug)
    if r is None or route is None:
        print(f"  {slug}: unavailable")
        continue
    picked = None
    for keys in (("route",), ("leadin", "route")):
        rx, rz, rd = _leg_geom(r, keys)
        out = _fit_geom(route, rx, rz, rd)
        if out is None:
            continue
        coef, p95, cov = out
        if picked is None or p95 < picked[1]:
            picked = (coef, p95, cov, keys)
    coef, p95, cov, keys = picked
    variant = "leadin+route" if len(keys) == 2 else "route"
    new = {"coef": coef.tolist(), "route": r.get("name"), "p95_m": round(p95, 1),
           "slug": slug, "coverage": round(cov, 1)}
    print(f"  {slug:22s} p95={p95:7.1f}m cov={cov:8.0f}m [{variant:12s}]  "
          f"eligible={p95 <= P95_LIMIT}")
    if _is_better(new, best):
        best = new

print(f"\nSELECTED for world 9 -> {best['route']!r}  p95={best['p95_m']}m  cov={best['coverage']}m")
