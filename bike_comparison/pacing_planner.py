"""
Optimal TT pacing planner for Zwift routes.

Given a rider (height, weight, normalized-power target), a bike configuration
(weight, CdA effect), and a route, this computes a pacing plan: the power output
the rider should hold at each point of the course to minimise total time while
keeping the *normalized* power (NP) equal to the target.

Why normalized power
--------------------
A pure *average*-power budget lets the optimiser "bank" energy: it dumps power
capped hard on every climb and coasts every descent at zero, because that
bang-bang profile still averages out to the target.  The result is long,
physiologically infeasible stretches pinned at the power ceiling.  Normalized
power — NP = (mean over 30 s rolling windows of P⁴)^(1/4) — penalises that
variability heavily (the 4th power weights surges), so holding the *same* NP
yields a far smoother, more sustainable plan that still rewards pushing uphill.

Method
------
The route is divided into small chunks (each at most ``max_chunk_m`` metres) so
every chunk has an essentially constant gradient.  Because the marginal NP cost
of a watt grows like P³, minimising time at a fixed NP is a Lagrangian
water-filling problem: for a price ``μ`` on NP, each chunk independently picks
the steady speed ``v`` that minimises ``(1/v)·(1 + μ·P(v)⁴)``, where

    P(v) = clip( (F_grav + F_roll + ½·ρ·CdA·v²)·v / (1-η),  0,  p_max )

is the steady power to hold ``v`` on that chunk.  Small ``μ`` lets power sit at
the cap (fast, spiky); large ``μ`` flattens it toward constant power.  ``μ`` is
found by bisection so the momentum simulation's realised NP equals the target.
The result shifts effort off the climbs and onto the flats/descents until the
time saved per unit of NP-cost is equal everywhere — a smooth, feasible plan.

The final per-chunk times and speeds come from a *momentum* integrator: the
surplus speed carried out of one chunk becomes the entry speed of the next, so
the plan reflects real coasting/acceleration rather than isolated steady states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bike_comparison.physics import (
    speed_from_power,
    AIR_DENSITY,
    GRAVITY,
    DRIVETRAIN_LOSS,
)

V_FLOOR = 0.3          # m/s — numerical floor so the rider never fully stalls
V_FLOOR2 = V_FLOOR * V_FLOOR


def _traverse(v0, drive, f_grav, f_roll, aero_k, inv_mass, length):
    """Integrate one chunk, returning (exit_speed, time_seconds).

    Steps in *distance* with the kinematic update ``v² = v₀² + 2·a·dl``, using
    the exact physics-model force balance

        F_net = F_drive - F_grav - F_roll - ½·ρ·CdA·v² ,   F_drive = P·(1-η)/v

    (``drive`` is ``P·(1-η)``; ``f_grav``/``f_roll`` are the speed-independent
    gravity/rolling forces; ``aero_k = ½·ρ·CdA``).  The step length is refined
    where the rider is slow: because ``F_drive = P·(1-η)/v`` grows as ``v→0``,
    a single long step would overshoot the speed instead of letting the force
    self-limit as ``v`` rises.  Limiting each sub-step so ``v²`` changes by at
    most ~50 % keeps the constant-acceleration assumption valid, so no
    artificial force cap is needed — the force is always the true physics value.
    """
    v = v0 if v0 > V_FLOOR else V_FLOOR
    t = 0.0
    dist = 0.0
    while dist < length - 1e-9:
        remaining = length - dist
        f_drive = drive / v
        a = (f_drive - f_grav - f_roll - aero_k * v * v) * inv_mass
        if a > 0.0:
            dl = 0.5 * v * v / a          # so 2·a·dl ≤ ½·v²
            if dl < 0.5:
                dl = 0.5
            if dl > remaining:
                dl = remaining
        else:
            dl = remaining
        v2 = v * v + 2.0 * a * dl
        v_new = math.sqrt(v2) if v2 > V_FLOOR2 else V_FLOOR
        v_avg = 0.5 * (v + v_new)
        t += dl / v_avg
        v = v_new
        dist += dl
    return v, t


def _normalized_power(power, times, window_s: float = 30.0) -> float:
    """Normalized power (W) for a per-chunk power/time profile.

    Resamples the distance-chunk profile onto a 1-second time grid, takes the
    30-second rolling average, then NP = (mean(rolling_avg⁴))^(1/4).
    """
    total_t = float(np.sum(times))
    if total_t <= 0.0:
        return 0.0
    n_sec = max(1, int(math.ceil(total_t)))
    cum_t = np.cumsum(times)
    # Sample each 1-second slot at its midpoint and map it to the chunk it falls in.
    sample_t = np.arange(n_sec) + 0.5
    idx = np.clip(np.searchsorted(cum_t, sample_t, side="right"), 0, len(power) - 1)
    p_sec = np.asarray(power, dtype=float)[idx]
    w = int(window_s)
    if n_sec >= w > 0:
        ravg = np.convolve(p_sec, np.ones(w) / w, mode="valid")
    else:
        ravg = np.array([float(np.mean(p_sec))])
    return float(np.mean(ravg ** 4) ** 0.25)


@dataclass
class PacingPlanResult:
    """Result of a pacing optimisation."""
    route_name: str
    total_time_seconds: float
    total_distance_km: float
    total_ascent_m: float
    avg_speed_kph: float
    avg_power_w: float
    normalized_power_w: float
    max_power_w: float
    min_power_w: float

    # Per-point series (downsampled for the API response / chart).
    distance_km: list = field(default_factory=list)
    altitude_m: list = field(default_factory=list)
    power_w: list = field(default_factory=list)
    speed_kph: list = field(default_factory=list)
    gradient_pct: list = field(default_factory=list)

    # Aggregated pacing table (one row per display section).
    sections: list = field(default_factory=list)

    @property
    def total_time_formatted(self) -> str:
        s = int(round(self.total_time_seconds))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"


def _build_chunks(route, max_chunk_m, crr):
    """Split the route into <=``max_chunk_m`` chunks with constant gradient.

    Returns arrays ``(length, gradient, mid_dist, end_dist)`` where ``mid_dist``
    and ``end_dist`` are cumulative distances (m) used for charting/altitude.
    """
    dist = np.asarray(route.distance_m, dtype=float)
    alt = np.asarray(route.altitude_m, dtype=float)
    seg_len = np.diff(dist)
    seg_alt = np.diff(alt)
    with np.errstate(divide="ignore", invalid="ignore"):
        seg_grad = np.where(seg_len > 0, seg_alt / seg_len, 0.0)
    seg_grad = np.clip(seg_grad, -0.40, 0.40)

    lengths: list[float] = []
    grads: list[float] = []
    ends: list[float] = []
    cum = 0.0
    for i in range(len(seg_len)):
        L = float(seg_len[i])
        if L <= 0:
            continue
        nsub = max(1, int(math.ceil(L / max_chunk_m)))
        clen = L / nsub
        g = float(seg_grad[i])
        for _ in range(nsub):
            cum += clen
            lengths.append(clen)
            grads.append(g)
            ends.append(cum)

    length_arr = np.asarray(lengths, dtype=float)
    grad_arr = np.asarray(grads, dtype=float)
    end_arr = np.asarray(ends, dtype=float)
    mid_arr = end_arr - 0.5 * length_arr
    return length_arr, grad_arr, mid_arr, end_arr


def plan_tt_pacing(
    route,
    rider_weight_kg: float,
    rider_height_m: float,
    bike_weight_kg: float,
    cda: float,
    power_target_w: float,
    *,
    crr: float = 0.004,
    max_chunk_m: float = 10.0,
    max_power_mult: float = 2.5,
    downsample_points: int = 400,
    n_sections: int = 40,
) -> PacingPlanResult:
    """Compute the optimal pacing plan for a route.

    Args:
        route: A ``RouteProfile`` (distance_m / altitude_m arrays).
        rider_weight_kg: Rider mass (kg).
        rider_height_m: Rider height (m) — used only for display consistency;
            the caller supplies the already-computed ``cda``.
        bike_weight_kg: Bike mass (kg).
        cda: Absolute CdA (m²) for this rider + bike.
        power_target_w: Target *normalized* power (NP, W) — the effort budget the
            plan is optimised against (30 s rolling, 4th-power weighted).
        crr: Rolling-resistance coefficient.
        max_chunk_m: Maximum chunk length (m).
        max_power_mult: Cap on each chunk's power as a multiple of the target,
            so the rider pushes hard on climbs but not beyond a realistic
            ceiling (e.g. 2.5 × 250 W = 625 W).
        downsample_points: Number of points in the returned chart series.
        n_sections: Number of rows in the aggregated pacing table.
    """
    length, grad, mid_dist, end_dist = _build_chunks(route, max_chunk_m, crr)
    n = len(length)
    if n < 2:
        raise ValueError("Route is too short to plan.")

    total_mass = rider_weight_kg + bike_weight_kg
    inv_mass = 1.0 / total_mass
    aero_k = 0.5 * AIR_DENSITY * cda
    cos_slope = np.cos(np.arctan(grad))
    f_grav = total_mass * GRAVITY * grad
    f_roll = crr * total_mass * GRAVITY * cos_slope

    one_minus_eta = 1.0 - DRIVETRAIN_LOSS
    p_max = max_power_mult * power_target_w
    f_sum = f_grav + f_roll  # speed-independent resistive force per chunk (N)

    def step(c, v_in, p):
        """Integrate chunk ``c`` at power ``p`` W entering at ``v_in`` m/s."""
        return _traverse(v_in, p * one_minus_eta, float(f_grav[c]),
                         float(f_roll[c]), aero_k, inv_mass, float(length[c]))

    # ── NP-optimal power allocation (Lagrangian water-filling) ─────────────
    # Under a *normalized*-power budget the marginal cost of a watt scales as P³
    # (because NP⁴ = mean(P⁴)): a watt added to a 0 W section is almost free, while
    # a watt added to a 600 W section is hugely expensive.  The optimum therefore
    # shifts effort *off* the climbs and *onto* the flats/descents until the time
    # saved per unit of NP-cost is equal on every chunk — a far smoother profile
    # than the constant-speed (energy-optimal) plan.
    #
    # Minimising total time at a fixed NP is, via a Lagrange multiplier μ ≥ 0,
    # separable: each chunk independently minimises
    #     h_c(v) = (1/v)·(1 + μ·P_c(v)⁴),   P_c(v) = clip(φ_c(v), 0, p_max)
    # where φ_c(v) = (f_grav_c + f_roll_c + aero_k·v²)·v/(1-η) is the steady power to
    # hold speed v on chunk c.  μ prices NP against time and is found by bisection
    # so the momentum simulation's realised NP equals the target.  Each chunk's
    # optimum v is read off a shared speed grid (h_c is unimodal in v).
    v_grid = np.linspace(V_FLOOR, 35.0, 400)
    aero_term = aero_k * v_grid * v_grid                     # ½·ρ·CdA·v²   (G,)
    # Steady power to hold each grid speed on each chunk (n, G), *unclipped*.  A
    # (chunk, speed) pair is infeasible when that power exceeds the cap — you
    # physically cannot hold that speed within the power ceiling — so it is
    # excluded.  Clipping it to p_max instead would let the optimiser "buy" a
    # high speed while paying only the (now constant) capped power, whose 1/v
    # cost keeps falling with v; that spurious branch produces scattered
    # full-power spikes on chunks that aren't actually steep enough to need them.
    p_raw = (f_sum[:, None] + aero_term[None, :]) * v_grid[None, :] / one_minus_eta
    feasible = p_raw <= p_max
    p_eff = np.clip(p_raw, 0.0, None)                        # coast (P=0) on descents
    p_eff4 = p_eff ** 4
    inv_v = (1.0 / v_grid)[None, :]                          # time weight per speed
    chunk_idx = np.arange(n)

    v_enter = np.empty(n)
    time_arr = np.empty(n)

    def optimal_power(mu):
        """NP-optimal per-chunk power for Lagrange price ``mu`` (≥ 0)."""
        cost = np.where(feasible, inv_v * (1.0 + mu * p_eff4), np.inf)
        gi = np.argmin(cost, axis=1)
        return p_eff[chunk_idx, gi]

    def forward_sim(pw):
        """Momentum forward pass for power profile ``pw``; fills v_enter/time_arr."""
        v = max(speed_from_power(float(pw[0]), float(grad[0]),
                                 rider_weight_kg, bike_weight_kg, cda, crr), V_FLOOR)
        for c in range(n):
            v_enter[c] = v
            v, dt = step(c, v, float(pw[c]))
            time_arr[c] = dt

    def np_for(mu):
        """Realised NP (and power profile) for Lagrange price ``mu``."""
        pw = optimal_power(mu)
        forward_sim(pw)
        return _normalized_power(pw, time_arr), pw

    # μ = 0 ignores NP entirely → power pinned at the cap → maximum NP.  Raising μ
    # lowers every chunk's power, so NP decreases monotonically in μ; bracket then
    # bisect μ to land on the target NP.
    np_lo, power = np_for(0.0)              # μ = 0 → highest NP the plan allows
    if np_lo > power_target_w:              # else target unreachable — ride flat out
        mu_lo, mu_hi = 0.0, 1e-12
        np_hi, _ = np_for(mu_hi)
        expand = 0
        while np_hi > power_target_w and expand < 80:
            mu_hi *= 4.0
            np_hi, _ = np_for(mu_hi)
            expand += 1
        for _ in range(50):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            np_mid, _ = np_for(mu_mid)
            if np_mid > power_target_w:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
        power = optimal_power(0.5 * (mu_lo + mu_hi))
        forward_sim(power)

    times = time_arr
    total_time = float(np.sum(times))
    speed_mps = np.where(times > 0, length / times, 0.0)
    sum_t = float(np.sum(times))
    sum_pt = float(np.sum(power * times))
    avg_power_time = (sum_pt / sum_t) if sum_t > 0.0 else 0.0
    normalized_power = _normalized_power(power, times)


    display_dist_km = route.display_distance_km
    avg_speed_kph = (display_dist_km / (total_time / 3600.0)) if total_time > 0 else 0.0

    # Altitude at each chunk midpoint for the chart.
    altitude = np.interp(mid_dist, np.asarray(route.distance_m, dtype=float),
                         np.asarray(route.altitude_m, dtype=float))

    # ── Downsample the per-chunk series for the response ──────────────────
    if n > downsample_points:
        idx = np.unique(np.round(np.linspace(0, n - 1, downsample_points)).astype(int))
    else:
        idx = np.arange(n)

    result = PacingPlanResult(
        route_name=route.name,
        total_time_seconds=total_time,
        total_distance_km=display_dist_km,
        total_ascent_m=route.display_ascent_m,
        avg_speed_kph=round(avg_speed_kph, 1),
        avg_power_w=round(avg_power_time, 1),
        normalized_power_w=round(normalized_power, 1),
        max_power_w=round(float(np.max(power)), 1),
        min_power_w=round(float(np.min(power)), 1),
        distance_km=[round(float(mid_dist[i]) / 1000.0, 3) for i in idx],
        altitude_m=[round(float(altitude[i]), 1) for i in idx],
        power_w=[round(float(power[i])) for i in idx],
        speed_kph=[round(float(speed_mps[i]) * 3.6, 1) for i in idx],
        gradient_pct=[round(float(grad[i]) * 100.0, 1) for i in idx],
    )

    # ── Aggregated pacing table (contiguous sections of ~equal distance) ──
    result.sections = _build_sections(
        length, grad, power, times, end_dist, n_sections
    )
    return result


def _build_sections(length, grad, power, times, end_dist, n_sections):
    """Group chunks into up to ``n_sections`` contiguous distance-equal rows."""
    n = len(length)
    n_sections = max(1, min(n_sections, n))
    total_dist = float(end_dist[-1])
    bounds = np.linspace(0.0, total_dist, n_sections + 1)

    sections = []
    c = 0
    for s in range(n_sections):
        seg_end = bounds[s + 1]
        start_c = c
        while c < n and end_dist[c] <= seg_end + 1e-6:
            c += 1
        if s == n_sections - 1:
            c = n  # absorb any rounding remainder into the last section
        if c <= start_c:
            continue
        sl = length[start_c:c]
        seg_len = float(np.sum(sl))
        if seg_len <= 0:
            continue
        seg_time = float(np.sum(times[start_c:c]))
        # Distance-weighted average power / gradient over the section.
        avg_pow = float(np.sum(power[start_c:c] * sl) / seg_len)
        avg_grad = float(np.sum(grad[start_c:c] * sl) / seg_len)
        start_dist = float(end_dist[start_c] - sl[0])
        sections.append({
            "start_km": round(start_dist / 1000.0, 2),
            "end_km": round(float(end_dist[c - 1]) / 1000.0, 2),
            "distance_m": round(seg_len, 1),
            "avg_gradient_pct": round(avg_grad * 100.0, 1),
            "power_w": round(avg_pow),
            "time_seconds": round(seg_time, 1),
            "avg_speed_kph": round((seg_len / seg_time) * 3.6, 1) if seg_time > 0 else 0.0,
        })
    return sections
