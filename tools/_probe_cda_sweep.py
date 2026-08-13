"""Fix WET Crr = 0.004 and back out CdA per point; bin by speed.

Uses the activity currently configured in analyze_surface_crr (cached telemetry).
Also runs a joint least-squares fit of (Crr, CdA) using the speed variation from
rollers, to cross-check whether the ~0.0007 Crr offset is really a CdA error.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_surface_crr as A  # noqa: E402

FIXED_CRR = 0.004
TARGET_STYLE = "WET"

setup = A.get_bike_stats(A.FRAME_ID, A.WHEEL_ID, A.UPGRADE_LEVEL)
frontal = A.frontal_area_from_rider(A.RIDER_HEIGHT_M, A.RIDER_WEIGHT_KG)
cda_model = setup.cd * frontal
mass_kg = A.RIDER_WEIGHT_KG + setup.weight_kg
rho, g, eta = A.AIR_DENSITY, A.GRAVITY, A.DRIVETRAIN_LOSS

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

stable = np.zeros(n, bool)
for i in range(1, n - 1):
    s = surf_pt[i]
    if s is not None and surf_pt[i - 1] == s and surf_pt[i + 1] == s:
        stable[i] = True

sel = stable & (surf_pt == TARGET_STYLE) & (power > 0) & (speed > 0.1)

v = speed[sel]
theta = np.arctan(gradient[sel])
rhs = power[sel] * (1 - eta) / v
f_grav = mass_kg * g * gradient[sel]
f_inertia = mass_kg * accel[sel]
f_roll = FIXED_CRR * mass_kg * g * np.cos(theta)
f_aero = rhs - f_grav - f_roll - f_inertia
cda_pt = f_aero / (0.5 * rho * v * v)

print(f"activity {A.ACTIVITY_ID} | {TARGET_STYLE} stable moving pts: {sel.sum()} "
      f"| align R^2 {r2:.3f}")
print(f"model CdA (current) = {cda_model:.4f} m^2 | mass {mass_kg:.2f} kg | "
      f"fixed Crr = {FIXED_CRR}\n")

# --- CdA binned by speed --------------------------------------------------
kmh = v * 3.6
edges = np.arange(20, 48, 3.0)
print(f"{'speed bin':>12s} {'N':>4s} {'mean km/h':>10s} {'mean CdA':>9s} "
      f"{'CI low':>8s} {'CI high':>8s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (kmh >= lo) & (kmh < hi)
    if m.sum() == 0:
        continue
    nn, mean, cl, ch = A.ci95(cda_pt[m])
    print(f"  {lo:4.0f}-{hi:4.0f}   {nn:4d} {np.mean(kmh[m]):10.1f} "
          f"{mean:9.4f} {cl:8.4f} {ch:8.4f}")

# --- Joint least-squares fit of (Crr, CdA) --------------------------------
# y = Crr*(m g cos) + CdA*(0.5 rho v^2)
y = rhs - f_grav - f_inertia
X = np.column_stack([mass_kg * g * np.cos(theta), 0.5 * rho * v * v])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"\njoint fit (uses roller speed variation): "
      f"Crr = {coef[0]:.5f}, CdA = {coef[1]:.4f} m^2")
print(f"  vs model CdA {cda_model:.4f}  -> "
      f"{100*(coef[1]-cda_model)/cda_model:+.1f}%")
