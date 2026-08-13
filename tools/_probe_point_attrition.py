"""One-off: show how telemetry points attrition down to per-style stable counts."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_surface_crr as A  # noqa: E402

route_d, route_elev, route_surf, leadin_m, style_map = A.build_route_profile()
route_surf = np.array(route_surf)
telem = A.fetch_telemetry()

t = np.asarray(telem["timeInSec"], float)
speed = np.asarray(telem["speedInCmPerSec"], float) / 100.0
dist = np.asarray(telem["distanceInCm"], float) / 100.0
alt = np.asarray(telem["altitudeInCm"], float) / 100.0
n = min(len(t), len(speed), len(dist), len(alt))
t, speed, dist, alt = t[:n], speed[:n], dist[:n], alt[:n]

delta, r2, _ = A.find_offset(dist, alt, route_d, route_elev, leadin_m)
rd = dist - delta
on_route = (rd >= route_d[0]) & (rd <= route_d[-1])
idx = np.clip(np.searchsorted(route_d, rd), 0, len(route_d) - 1)
surf_pt = np.where(on_route, route_surf[idx], None)

stable = np.zeros(n, bool)
for i in range(1, n - 1):
    s = surf_pt[i]
    if s is not None and surf_pt[i - 1] == s and surf_pt[i + 1] == s:
        stable[i] = True

print(f"total telemetry points ....... {n}")
print(f"offset delta ................. {delta:.0f} m (lead-in {leadin_m:.0f} m), R^2 {r2:.3f}")
print(f"dropped: before route start .. {int(np.sum(rd < route_d[0]))}")
print(f"dropped: past route end ...... {int(np.sum(rd > route_d[-1]))}")
print(f"on-route points .............. {int(np.sum(on_route))}")
print(f"  -> stable (surf==neighbours) {int(np.sum(stable))}")
print(f"  -> transition/edge dropped .. {int(np.sum(on_route) - np.sum(stable))}")
uniq, cnts = np.unique([s for s in surf_pt[stable]], return_counts=True)
print("stable split by style:")
for u, c in sorted(zip(uniq, cnts), key=lambda kv: -kv[1]):
    print(f"    {u:20s} {c}")
