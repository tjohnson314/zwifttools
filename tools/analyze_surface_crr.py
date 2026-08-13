"""Estimate per-surface rolling resistance (Crr) from a Zwift ride's telemetry.

Pipeline:
  1. Fetch 1 Hz telemetry for the activity (Zwift API, password grant).
  2. Build the route's surface profile via the exact road-id join against the
     game WAD (authoritative style -> surface map).
  3. Align telemetry distance to route distance (elevation-shape search) and
     assign a surface to every second.
  4. Keep only points whose surface equals both neighbours' (drop transitions).
  5. Invert the project physics model (no draft -- TT bike) with Crr the only
     unknown, per point.
  6. Report per-surface: N points, mean Crr, 95% CI.

Raw Crr values are reported as-is (no clamping to "plausible" ranges).

Usage:
  python tools/analyze_surface_crr.py                # fetch + analyse
  python tools/analyze_surface_crr.py --dry-run      # build route+bike only
Credentials: set ZWIFT_USERNAME / ZWIFT_PASSWORD env vars, or you will be
prompted (password read via getpass, never echoed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_zwift_routes import read_wad_entries, load_multiroot  # noqa: E402
from extract_zwift_surfaces import parse_roadstyles, UNSET_STYLE  # noqa: E402
from _probe_route_surface_exact import parse_roads  # noqa: E402
from bike_comparison.bike_data import get_bike_stats  # noqa: E402
from bike_comparison.physics import frontal_area_from_rider  # noqa: E402

# --- Physics constants (match bike_comparison/physics.py) --------------------
AIR_DENSITY = 1.225
GRAVITY = 9.8067
DRIVETRAIN_LOSS = 0.025

# --- Ride / rider / route configuration -------------------------------------
ACTIVITY_ID = "2202922917312921632"
WORLD = "world1"
NAME_HASH = "3680493479"          # Southern Coast Cruise
RIDER_HEIGHT_M = 1.85
RIDER_WEIGHT_KG = 68.0
FRAME_ID = "Zwift_TT"
WHEEL_ID = "dtswissarc1100dicut85disc"
UPGRADE_LEVEL = 5

ZWIFT_DIR = r"C:\Program Files (x86)\Zwift"
ZWIFT_TOKEN_URL = "https://secure.zwift.com/auth/realms/zwift/protocol/openid-connect/token"
CLIENT_ID = "Zwift_Mobile_Link"
STYLE_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "zwift_surfaces", "style_surface_map.json")
TELEM_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_telem_{ACTIVITY_ID}.json")


# ---------------------------------------------------------------------------
# 1. Telemetry
# ---------------------------------------------------------------------------
def get_token():
    import getpass
    import requests
    user = os.environ.get("ZWIFT_USERNAME") or input("Zwift email: ").strip()
    pw = os.environ.get("ZWIFT_PASSWORD") or getpass.getpass("Zwift password: ")
    resp = requests.post(ZWIFT_TOKEN_URL, data={
        "client_id": CLIENT_ID, "grant_type": "password",
        "username": user, "password": pw,
    }, timeout=20)
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_telemetry():
    """Return the raw telemetry dict, using an on-disk cache when present."""
    if os.path.exists(TELEM_CACHE):
        print(f"Using cached telemetry: {TELEM_CACHE}")
        with open(TELEM_CACHE, encoding="utf-8") as f:
            return json.load(f)
    from shared.data_fetcher import fetch_rider_telemetry
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    telem, _activity, err = fetch_rider_telemetry(ACTIVITY_ID, headers)
    if err:
        raise RuntimeError(f"Telemetry fetch failed: {err}")
    with open(TELEM_CACHE, "w", encoding="utf-8") as f:
        json.dump(telem, f)
    print(f"Fetched + cached telemetry -> {TELEM_CACHE}")
    return telem


# ---------------------------------------------------------------------------
# 2. Route surface profile (exact road-id join, authoritative style map)
# ---------------------------------------------------------------------------
def load_style_surface_map():
    with open(STYLE_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_route_profile():
    """Return route arrays (d_m, elev_raw, style[]) and lead-in distance.

    Points are labelled by their raw roadstyle NAME (e.g. PACKEDSAND, WOODEN,
    ANCIENTBRICK) rather than the collapsed surface category, so each distinct
    style's Crr can be measured independently.
    """
    style_map = load_style_surface_map()  # kept for reference/reporting
    wad = os.path.join(ZWIFT_DIR, "assets", "Worlds", WORLD, "data_1.wad")
    entries = read_wad_entries(wad, keep_substrings=("/routes/", "road.xml", "roadstyle.xml"))
    road_xml = next(v for k, v in entries.items()
                    if k.lower().endswith("road.xml") and "roadstyle" not in k.lower()
                    ).decode("utf-8", "replace")
    styles = parse_roadstyles(next(v for k, v in entries.items()
                                   if k.lower().endswith("roadstyle.xml")))
    roads = parse_roads(road_xml)

    def style_at(rid, t):
        base, markers = roads.get(rid, (UNSET_STYLE, []))
        style = base
        for a, b, s in markers:
            if a <= t <= b:
                style = s
        if style == UNSET_STYLE or not (0 <= style < len(styles)):
            style = 0
        return styles[style].upper()

    route = None
    for nm, data in entries.items():
        if "/routes/" not in nm or not nm.endswith(".xml"):
            continue
        root = load_multiroot(data)
        r = root.find("route")
        if r is not None and r.get("nameHash") == NAME_HASH:
            route = (r, root)
            break
    if route is None:
        raise RuntimeError(f"Route {NAME_HASH} not found in {WORLD}")
    r, root = route
    leadin_m = float(r.get("leadinDistanceInMeters") or 0.0)

    cp = root.find("highrescheckpoint")
    pts = cp.findall("entry")
    d_m = np.zeros(len(pts))
    elev = np.zeros(len(pts))
    surf = []
    prev = None
    for i, e in enumerate(pts):
        x, z, y = float(e.get("x")), float(e.get("z")), float(e.get("y"))
        if prev is not None:
            d_m[i] = d_m[i - 1] + math.hypot(x - prev[0], z - prev[1]) / 100.0
        elev[i] = y
        surf.append(style_at(int(e.get("road")), float(e.get("time"))))
        prev = (x, z)
    return d_m, elev, surf, leadin_m, style_map


# ---------------------------------------------------------------------------
# 3. Alignment: find distance offset so telemetry altitude best matches the
#    route elevation shape (route y has an unknown linear scale/datum per world,
#    so fit a*y+b at each candidate offset and keep the best R^2).
# ---------------------------------------------------------------------------
def find_offset(tel_d, tel_alt, route_d, route_elev, leadin_guess):
    lo = max(-300.0, leadin_guess - 800.0)
    hi = leadin_guess + 800.0
    best = (None, -np.inf, None)
    for delta in np.arange(lo, hi + 1e-9, 5.0):
        rd = tel_d - delta
        m = (rd >= route_d[0]) & (rd <= route_d[-1])
        if m.sum() < 30:
            continue
        ry = np.interp(rd[m], route_d, route_elev)
        a = tel_alt[m]
        if np.std(ry) < 1e-6:
            continue
        # linear fit a ~ p*ry + q ; R^2
        p = np.polyfit(ry, a, 1)
        pred = np.polyval(p, ry)
        ss_res = np.sum((a - pred) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best[1]:
            best = (delta, r2, p)
    return best  # (delta_m, r2, linear_fit)


# ---------------------------------------------------------------------------
# 5. Physics inversion: solve Crr per point (no draft).
# ---------------------------------------------------------------------------
def solve_crr(power_w, speed_mps, gradient, mass_kg, cda, accel_mps2):
    v = speed_mps
    crr = np.full_like(v, np.nan, dtype=float)
    ok = v > 0.1
    theta = np.arctan(gradient)
    f_grav = mass_kg * GRAVITY * gradient
    f_aero = 0.5 * AIR_DENSITY * cda * v * np.abs(v)
    f_inertia = mass_kg * accel_mps2  # power into changing kinetic energy
    rhs = np.where(ok, power_w * (1 - DRIVETRAIN_LOSS) / np.where(ok, v, 1.0), np.nan)
    f_roll = rhs - f_grav - f_aero - f_inertia
    denom = mass_kg * GRAVITY * np.cos(theta)
    crr = np.where(ok, f_roll / denom, np.nan)
    return crr


# ---------------------------------------------------------------------------
# 6. Reporting
# ---------------------------------------------------------------------------
def ci95(vals):
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    mean = float(np.mean(vals))
    if n < 2:
        return (n, mean, np.nan, np.nan)
    sd = float(np.std(vals, ddof=1))
    sem = sd / math.sqrt(n)
    try:
        from scipy.stats import t as _t
        tcrit = float(_t.ppf(0.975, n - 1))
    except Exception:
        tcrit = 1.96
    half = tcrit * sem
    return (n, mean, mean - half, mean + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Build route + bike setup only (no telemetry fetch).")
    args = ap.parse_args()

    # Bike setup ------------------------------------------------------------
    setup = get_bike_stats(FRAME_ID, WHEEL_ID, UPGRADE_LEVEL)
    if setup is None:
        raise RuntimeError("Bike setup not found")
    frontal_area = frontal_area_from_rider(RIDER_HEIGHT_M, RIDER_WEIGHT_KG)
    cda = setup.cd * frontal_area
    bike_kg = setup.weight_kg
    mass_kg = RIDER_WEIGHT_KG + bike_kg
    print("=== Bike / rider ===")
    print(f"  {setup}")
    print(f"  bike weight {bike_kg:.3f} kg | total mass {mass_kg:.3f} kg")
    print(f"  frontal area {frontal_area:.4f} m^2 | CdA {cda:.4f} m^2 (no draft)")

    # Route profile ---------------------------------------------------------
    route_d, route_elev, route_surf, leadin_m, style_map = build_route_profile()
    route_surf = np.array(route_surf)
    print("\n=== Route ===")
    print(f"  {WORLD} nameHash {NAME_HASH} | {route_d[-1]/1000:.2f} km, "
          f"{len(route_d)} checkpoints, lead-in {leadin_m:.0f} m")
    uniq, cnts = np.unique(route_surf, return_counts=True)
    print("  checkpoint style mix: " +
          ", ".join(f"{u}->{style_map.get(u, 'Tarmac')} {c}"
                    for u, c in sorted(zip(uniq, cnts), key=lambda kv: -kv[1])))

    if args.dry_run:
        print("\n(dry run: skipping telemetry fetch)")
        return

    # Telemetry -------------------------------------------------------------
    telem = fetch_telemetry()
    print("\n=== Telemetry ===")
    print(f"  raw keys: {sorted(telem.keys())}")
    t = np.asarray(telem.get("timeInSec", []), dtype=float)
    power = np.asarray(telem.get("powerInWatts", []), dtype=float)
    speed = np.asarray(telem.get("speedInCmPerSec", []), dtype=float) / 100.0  # m/s
    dist = np.asarray(telem.get("distanceInCm", []), dtype=float) / 100.0       # m
    alt = np.asarray(telem.get("altitudeInCm", []), dtype=float) / 100.0        # m
    n = min(len(t), len(power), len(speed), len(dist), len(alt))
    t, power, speed, dist, alt = t[:n], power[:n], speed[:n], dist[:n], alt[:n]
    print(f"  {n} points | dist {dist[-1]/1000:.2f} km "
          f"(route+leadin expected {(route_d[-1]+leadin_m)/1000:.2f} km) | "
          f"avg power {np.mean(power):.0f} W")

    # Alignment -------------------------------------------------------------
    delta, r2, fit = find_offset(dist, alt, route_d, route_elev, leadin_m)
    print(f"  best distance offset {delta:.0f} m (lead-in {leadin_m:.0f} m), "
          f"elevation-shape R^2 {r2:.3f}")

    # Assign surface per telemetry point ------------------------------------
    rd = dist - delta
    idx = np.searchsorted(route_d, rd)
    idx = np.clip(idx, 0, len(route_d) - 1)
    surf_pt = np.where((rd >= route_d[0]) & (rd <= route_d[-1]), route_surf[idx], None)

    # Gradient from telemetry altitude vs distance, acceleration from speed --
    # (guard against stationary points where distance/time do not advance)
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient = np.gradient(alt, dist)
        accel = np.gradient(speed, t)
    gradient[~np.isfinite(gradient)] = 0.0
    accel[~np.isfinite(accel)] = 0.0

    # Crr per point ---------------------------------------------------------
    crr = solve_crr(power, speed, gradient, mass_kg, cda, accel)

    # Stability filter: surface equals both neighbours ----------------------
    stable = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        s = surf_pt[i]
        if s is not None and surf_pt[i - 1] == s and surf_pt[i + 1] == s:
            stable[i] = True

    print("\n=== Per-style Crr (stable points only, raw / unclamped) ===")
    print(f"{'style':22s} {'category':9s} {'N':>6s} {'mean Crr':>12s} "
          f"{'95% CI low':>12s} {'95% CI high':>12s}")
    # Coasting (power<=0) makes the inversion ill-conditioned (Crr absorbs all
    # CdA/gradient error), so exclude those points from the Crr solve.
    moving = power > 0
    surfaces = sorted({s for s in surf_pt[stable] if s is not None})
    total_used = 0
    for s in surfaces:
        m = stable & moving & (surf_pt == s) & np.isfinite(crr)
        n_s, mean_s, lo_s, hi_s = ci95(crr[m])
        total_used += n_s
        print(f"{s:22s} {style_map.get(s, 'Tarmac'):9s} {n_s:6d} "
              f"{mean_s:12.5f} {lo_s:12.5f} {hi_s:12.5f}")
    print(f"\nTotal stable points used: {total_used}")
    n_coast = int(np.sum(stable & (power <= 0)))
    if n_coast:
        print(f"(note: excluded {n_coast} stable coasting points (power<=0); "
              f"Crr is ill-defined there)")


if __name__ == "__main__":
    main()
