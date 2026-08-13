"""One-off: WET/WOODEN Crr with vs without coasting (power<=0) points."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_surface_crr as A  # noqa: E402

setup = A.get_bike_stats(A.FRAME_ID, A.WHEEL_ID, A.UPGRADE_LEVEL)
frontal = A.frontal_area_from_rider(A.RIDER_HEIGHT_M, A.RIDER_WEIGHT_KG)
cda = setup.cd * frontal
mass_kg = A.RIDER_WEIGHT_KG + setup.weight_kg

route_d, route_elev, route_surf, leadin_m, style_map = A.build_route_profile()
route_surf = np.array(route_surf)
telem = A.fetch_telemetry()

t = np.asarray(telem["timeInSec"], float)
power = np.asarray(telem["powerInWatts"], float)
speed = np.asarray(telem["speedInCmPerSec"], float) / 100.0
dist = np.asarray(telem["distanceInCm"], float) / 100.0
alt = np.asarray(telem["altitudeInCm"], float) / 100.0
n = min(len(t), len(power), len(speed), len(dist), len(alt))
t, power, speed, dist, alt = t[:n], power[:n], speed[:n], dist[:n], alt[:n]

delta, r2, _ = A.find_offset(dist, alt, route_d, route_elev, leadin_m)
rd = dist - delta
on_route = (rd >= route_d[0]) & (rd <= route_d[-1])
idx = np.clip(np.searchsorted(route_d, rd), 0, len(route_d) - 1)
surf_pt = np.where(on_route, route_surf[idx], None)

with np.errstate(divide="ignore", invalid="ignore"):
    gradient = np.gradient(alt, dist)
    accel = np.gradient(speed, t)
gradient[~np.isfinite(gradient)] = 0.0
accel[~np.isfinite(accel)] = 0.0
crr = A.solve_crr(power, speed, gradient, mass_kg, cda, accel)

stable = np.zeros(n, bool)
for i in range(1, n - 1):
    s = surf_pt[i]
    if s is not None and surf_pt[i - 1] == s and surf_pt[i + 1] == s:
        stable[i] = True


def report(label, extra_mask):
    print(f"\n=== {label} ===")
    print(f"{'style':10s} {'N':>5s} {'mean':>10s} {'CI low':>10s} {'CI high':>10s}")
    for s in sorted({x for x in surf_pt[stable] if x is not None}):
        m = stable & (surf_pt == s) & np.isfinite(crr) & extra_mask
        nn, mean, lo, hi = A.ci95(crr[m])
        print(f"{s:10s} {nn:5d} {mean:10.5f} {lo:10.5f} {hi:10.5f}")


all_mask = np.ones(n, bool)
moving = power > 0
print(f"stable points: {int(stable.sum())} | coasting(power<=0) among them: "
      f"{int((stable & (power <= 0)).sum())}")
report("WITH coasting (raw, as reported)", all_mask)
report("WITHOUT coasting (power>0 only)", moving)
