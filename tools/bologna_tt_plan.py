"""Compare NP-optimal TT plans on the Bologna Time Trial for four bike strategies.

Uses the same physics and Lagrangian NP water-filling as
``bike_comparison.pacing_planner.plan_tt_pacing``, generalised so CdA and mass
may change along the course (for the mid-race bike swap). The NP budget is
optimised jointly over the whole ride, so the swap plan redistributes effort
across both legs.

Run:  python tools/bologna_tt_plan.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bike_comparison.bike_data import get_bike_stats
from bike_comparison.physics import (
    AIR_DENSITY, DRIVETRAIN_LOSS, GRAVITY, rider_cda, speed_from_power,
)
from bike_comparison.pacing_planner import (
    V_FLOOR, _build_chunks, _normalized_power, _traverse, load_route_profile,
    plan_tt_pacing,
)

ROUTE_ID = "2843604888"
ROUTE_NAME = "Bologna Time Trial"
WORLD = "BOLOGNATT"

RIDER_WEIGHT_KG = 66.0
RIDER_HEIGHT_M = 1.85
NP_TARGET_W = 350.0
UPGRADE_LEVEL = 5

CRR = 0.004
MAX_CHUNK_M = 10.0
MAX_POWER_MULT = 2.5
BISECTION_ITERS = 50

SWAP_AT_M = 6000.0
SWAP_PENALTY_S = 10.0

CADEX = "CadexTri2022"
AETHOS = "SpecializedAethos2021"
DT_DISC = "dtswissarc1100dicut85disc"
WAKE = "princetoncarbonworkswake6560"


def bike(frame_id, wheel_id):
    setup = get_bike_stats(frame_id, wheel_id, UPGRADE_LEVEL)
    if setup is None:
        raise SystemExit(f"Unknown bike combo {frame_id}/{wheel_id}")
    return setup


def plan_legs(route, legs):
    """NP-optimal plan where ``legs`` is [(start_m, BikeSetup), ...] sorted by start."""
    length, grad, mid_dist, end_dist = _build_chunks(route, MAX_CHUNK_M, CRR)
    n = len(length)

    starts = np.array([s for s, _ in legs], dtype=float)
    leg_idx = np.clip(np.searchsorted(starts, mid_dist, side="right") - 1, 0, len(legs) - 1)
    base_cda = rider_cda(RIDER_HEIGHT_M, RIDER_WEIGHT_KG)
    leg_mass = np.array([RIDER_WEIGHT_KG + b.weight_kg for _, b in legs])
    leg_cda = np.array([base_cda + b.cda_bias for _, b in legs])

    mass = leg_mass[leg_idx]
    cda = leg_cda[leg_idx]
    aero_k = 0.5 * AIR_DENSITY * cda
    inv_mass = 1.0 / mass
    f_grav = mass * GRAVITY * grad
    f_roll = CRR * mass * GRAVITY * np.cos(np.arctan(grad))
    one_minus_eta = 1.0 - DRIVETRAIN_LOSS
    p_max = MAX_POWER_MULT * NP_TARGET_W

    v_grid = np.linspace(V_FLOOR, 35.0, 400)
    p_raw = ((f_grav + f_roll)[:, None] + aero_k[:, None] * v_grid[None, :] ** 2) \
        * v_grid[None, :] / one_minus_eta
    feasible = p_raw <= p_max
    p_eff = np.clip(p_raw, 0.0, None)
    p_eff4 = p_eff ** 4
    inv_v = (1.0 / v_grid)[None, :]
    chunk_idx = np.arange(n)

    v_enter = np.empty(n)
    time_arr = np.empty(n)

    def optimal_power(mu):
        cost = np.where(feasible, inv_v * (1.0 + mu * p_eff4), np.inf)
        return p_eff[chunk_idx, np.argmin(cost, axis=1)]

    def forward_sim(pw):
        b0 = legs[leg_idx[0]][1]
        v = max(speed_from_power(float(pw[0]), float(grad[0]), RIDER_WEIGHT_KG,
                                 b0.weight_kg, float(cda[0]), CRR), V_FLOOR)
        for c in range(n):
            v_enter[c] = v
            v, dt = _traverse(v, float(pw[c]) * one_minus_eta, float(f_grav[c]),
                              float(f_roll[c]), float(aero_k[c]), float(inv_mass[c]),
                              float(length[c]))
            time_arr[c] = dt

    def np_for(mu):
        pw = optimal_power(mu)
        forward_sim(pw)
        return _normalized_power(pw, time_arr)

    power = optimal_power(0.0)
    forward_sim(power)
    if _normalized_power(power, time_arr) > NP_TARGET_W:
        mu_lo, mu_hi = 0.0, 1e-12
        expand = 0
        while np_for(mu_hi) > NP_TARGET_W and expand < 80:
            mu_hi *= 4.0
            expand += 1
        for _ in range(BISECTION_ITERS):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            if np_for(mu_mid) > NP_TARGET_W:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
        power = optimal_power(0.5 * (mu_lo + mu_hi))
        forward_sim(power)

    times = time_arr.copy()
    return {
        "length": length, "grad": grad, "end": end_dist, "power": power,
        "times": times, "leg_idx": leg_idx, "cda": leg_cda, "mass": leg_mass,
        "ride_time": float(times.sum()),
        "np": _normalized_power(power, times),
        "avg_power": float(np.sum(power * times) / times.sum()),
    }


def fmt(t):
    m, s = divmod(t, 60.0)
    return f"{int(m)}:{s:05.2f}"


def print_km_splits(res):
    print(f"  {'km':>9} {'grad%':>6} {'avgW':>5} {'kph':>5} {'split':>8} {'elapsed':>9}")
    km_idx = np.floor((res["end"] - 1e-6) / 1000.0).astype(int)
    elapsed = 0.0
    for k in range(km_idx.max() + 1):
        m = km_idx == k
        L = res["length"][m].sum()
        t = res["times"][m].sum()
        elapsed += t
        g = np.sum(res["grad"][m] * res["length"][m]) / L * 100.0
        w = np.sum(res["power"][m] * res["times"][m]) / t
        print(f"  {k:>4}-{k + L / 1000:<4.2f} {g:>6.1f} {w:>5.0f} {L / t * 3.6:>5.1f} "
              f"{fmt(t):>8} {fmt(elapsed):>9}")


def main():
    route = load_route_profile(ROUTE_ID, ROUTE_NAME, world=WORLD)
    print(f"{route.name}: {route.display_distance_km:.2f} km, "
          f"{route.display_ascent_m:.0f} m ascent")
    print(f"Rider {RIDER_WEIGHT_KG:.0f} kg / {RIDER_HEIGHT_M * 100:.0f} cm, "
          f"NP target {NP_TARGET_W:.0f} W, upgrade level {UPGRADE_LEVEL}\n")

    cadex_disc = bike(CADEX, DT_DISC)
    cadex_wake = bike(CADEX, WAKE)
    aethos_wake = bike(AETHOS, WAKE)

    options = [
        ("Cadex Tri + DT Swiss disc", [(0.0, cadex_disc)], 0.0),
        ("Cadex Tri + Princeton Wake", [(0.0, cadex_wake)], 0.0),
        ("Aethos + Princeton Wake", [(0.0, aethos_wake)], 0.0),
        (f"Cadex/disc -> Aethos/Wake @ {SWAP_AT_M / 1000:g} km (+{SWAP_PENALTY_S:g}s)",
         [(0.0, cadex_disc), (SWAP_AT_M, aethos_wake)], SWAP_PENALTY_S),
    ]

    results = []
    for name, legs, penalty in options:
        res = plan_legs(route, legs)
        res["total"] = res["ride_time"] + penalty
        results.append((name, legs, res))

    print(f"{'Option':<48} {'CdA':>13} {'kg':>11} {'NP':>5} {'avgW':>5} {'Total':>9}")
    for name, legs, res in results:
        cda = "/".join(f"{c:.4f}" for c in res["cda"])
        kg = "/".join(f"{m - RIDER_WEIGHT_KG:.2f}" for m in res["mass"])
        print(f"{name:<48} {cda:>13} {kg:>11} {res['np']:>5.0f} "
              f"{res['avg_power']:>5.0f} {fmt(res['total']):>9}")

    best = min(results, key=lambda r: r[2]["total"])
    print(f"\nFastest: {best[0]} in {fmt(best[2]['total'])}")
    for name, _, res in results:
        if res is not best[2]:
            print(f"  {name}: +{res['total'] - best[2]['total']:.1f} s")

    for name, legs, res in results:
        print(f"\n{name}")
        if len(legs) > 1:
            t1 = res["times"][res["leg_idx"] == 0].sum()
            print(f"  Leg 1 (to {SWAP_AT_M / 1000:g} km): {fmt(t1)}, "
                  f"leg 2: {fmt(res['ride_time'] - t1)}, swap: +{SWAP_PENALTY_S:g}s")
        print_km_splits(res)

    # Cross-check single-bike options against the production planner.
    print("\nValidation vs plan_tt_pacing (single-bike options):")
    for name, legs, res in results:
        if len(legs) != 1:
            continue
        b = legs[0][1]
        ref = plan_tt_pacing(
            route, RIDER_WEIGHT_KG, RIDER_HEIGHT_M, b.weight_kg,
            rider_cda(RIDER_HEIGHT_M, RIDER_WEIGHT_KG) + b.cda_bias, NP_TARGET_W,
            crr=CRR, max_chunk_m=MAX_CHUNK_M, max_power_mult=MAX_POWER_MULT,
        )
        print(f"  {name:<48} script {fmt(res['ride_time'])}  "
              f"planner {fmt(ref.total_time_seconds)}")


if __name__ == "__main__":
    main()
