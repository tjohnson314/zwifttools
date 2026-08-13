"""Fix Crr = 0.004 and back out CdA over all stable moving Tarmac points on the
currently-configured ride; also show the reverse (fix CdA to model, back out Crr)
and the joint fit for reference.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_surface_crr as A  # noqa: E402

FIXED_CRR = 0.004

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

# Tarmac-category, stable (same style on both neighbours), moving, valid speed.
cat = np.array([style_map.get(s, "Tarmac") if s is not None else None for s in surf_pt])
stable = np.zeros(n, bool)
for i in range(1, n - 1):
    s = surf_pt[i]
    if s is not None and surf_pt[i - 1] == s and surf_pt[i + 1] == s:
        stable[i] = True

sel = stable & (cat == "Tarmac") & (power > 0) & (speed > 0.1)

v = speed[sel]
theta = np.arctan(gradient[sel])
rhs = power[sel] * (1 - eta) / v
f_grav = mass_kg * g * gradient[sel]
f_inertia = mass_kg * accel[sel]
cos = np.cos(theta)

# 1) Fix Crr -> back out CdA per point.
f_roll = FIXED_CRR * mass_kg * g * cos
f_aero = rhs - f_grav - f_roll - f_inertia
cda_pt = f_aero / (0.5 * rho * v * v)

# 2) Fix CdA (model) -> back out Crr per point (for comparison).
f_aero_m = 0.5 * rho * cda_model * v * v
crr_pt = (rhs - f_grav - f_aero_m - f_inertia) / (mass_kg * g * cos)

# 3) Joint fit.
y = rhs - f_grav - f_inertia
X = np.column_stack([mass_kg * g * cos, 0.5 * rho * v * v])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)

print(f"activity {A.ACTIVITY_ID} | Tarmac stable moving pts: {sel.sum()} "
      f"| align R^2 {r2:.3f}")
print(f"model: CdA {cda_model:.4f} m^2 | mass {mass_kg:.2f} kg | "
      f"frontal {frontal:.4f} | cd {setup.cd:.4f}\n")

nn, mc, cl, ch = A.ci95(cda_pt)
print(f"[fix Crr = {FIXED_CRR:.4f}]  -> CdA = {mc:.4f} m^2  "
      f"(95% CI {cl:.4f}..{ch:.4f}, n={nn})")
print(f"    vs model CdA {cda_model:.4f}  ->  {100*(mc-cda_model)/cda_model:+.1f}%")

nn2, mcr, cl2, ch2 = A.ci95(crr_pt)
print(f"[fix CdA = {cda_model:.4f}] -> Crr = {mcr:.5f}  "
      f"(95% CI {cl2:.5f}..{ch2:.5f}, n={nn2})")

print(f"[joint fit]              -> Crr = {coef[0]:.5f}, CdA = {coef[1]:.4f} m^2")
