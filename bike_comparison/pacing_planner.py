"""
Optimal TT pacing planner for Zwift routes.

Given a rider (height, weight, overall average power target), a bike
configuration (weight, CdA effect), and a route, this computes a pacing plan:
the power output the rider should hold at each point of the course to minimise
total time while keeping the *time*-average power (total energy / total time,
i.e. the figure a bike computer reports) equal to the target.

Method
------
The route is divided into small chunks (each at most ``max_chunk_m`` metres) so
every chunk has an essentially constant gradient.  Minimising time for a fixed
time-average power is a Lagrangian problem whose optimum has a simple form: on
all terrain that is neither power-capped nor coasting the rider holds a single
common steady-state speed ``v*`` — i.e. *more* power uphill and *less* downhill
(the classic "hold your speed" result).  The steady power needed to hold ``v*``
on a chunk is

    P = (F_grav + F_roll + ½·ρ·CdA·v*²)·v* / (1-η)

clamped to ``[0, p_max]``: chunks that would need more than the per-chunk cap
``p_max`` are climb-limited (ridden slower at ``p_max``) and chunks whose steady
power would be negative are coasted at zero power (ridden faster).  The single
free parameter ``v*`` is found by bisection so that the momentum simulation's
realised time-average power matches the target.

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


@dataclass
class PacingPlanResult:
    """Result of a pacing optimisation."""
    route_name: str
    total_time_seconds: float
    total_distance_km: float
    total_ascent_m: float
    avg_speed_kph: float
    avg_power_w: float
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
    avg_power_target_w: float,
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
        avg_power_target_w: Target *time-average* power (W) — total energy divided
            by total time, i.e. the average a bike computer reports.
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
    p_max = max_power_mult * avg_power_target_w
    f_sum = f_grav + f_roll  # speed-independent resistive force per chunk (N)

    def step(c, v_in, p):
        """Integrate chunk ``c`` at power ``p`` W entering at ``v_in`` m/s."""
        return _traverse(v_in, p * one_minus_eta, float(f_grav[c]),
                         float(f_roll[c]), aero_k, inv_mass, float(length[c]))

    # ── Direct (water-filling) allocation under a time-average power budget ──
    # At the optimum every chunk that is neither power-capped nor coasting is
    # ridden at one common steady-state speed v* — the classic result that on
    # moderate terrain you hold a constant speed, which means *more* power uphill
    # and *less* downhill.  The steady power that holds v* on chunk c is
    #     P_c = (f_grav_c + f_roll_c + aero_k·v*²)·v* / (1-η)
    # clamped to [0, p_max]: chunks needing more than the cap are climb-limited
    # (ridden slower at p_max) and chunks whose steady power would be negative are
    # coasted at P=0 (ridden faster).  v* is found by bisection so the momentum
    # simulation's time-average power ΣP·t / Σt equals the target.  This lands on
    # the same optimum the greedy converged toward, in O(n) per bisection step
    # instead of hundreds of thousands of increments.
    v_enter = np.empty(n)
    time_arr = np.empty(n)

    def powers_for_vstar(vstar):
        """Steady-state power per chunk to hold ``vstar``, clamped to [0, p_max]."""
        p_int = (f_sum + aero_k * vstar * vstar) * vstar / one_minus_eta
        return np.clip(p_int, 0.0, p_max)

    def forward_sim(pw):
        """Momentum forward pass for power profile ``pw``; fills v_enter/time_arr."""
        v = max(speed_from_power(float(pw[0]), float(grad[0]),
                                 rider_weight_kg, bike_weight_kg, cda, crr), V_FLOOR)
        for c in range(n):
            v_enter[c] = v
            v, dt = step(c, v, float(pw[c]))
            time_arr[c] = dt

    def avg_power_for(vstar):
        """Realised time-average power (and power profile) for target speed ``vstar``."""
        pw = powers_for_vstar(vstar)
        forward_sim(pw)
        st = float(np.sum(time_arr))
        return ((float(np.sum(pw * time_arr)) / st) if st > 0.0 else 0.0), pw

    # Bisect the common target speed v* so the realised time-average power hits
    # the target.  Higher v* ⇒ more interior power ⇒ higher average, so the map
    # v* → average power is monotonic and a root exists below the cap.
    lo, hi = V_FLOOR, 30.0
    avg_hi, _ = avg_power_for(hi)
    if avg_hi <= avg_power_target_w:
        vstar = hi  # target unreachable even flat-out — ride as hard as possible
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            avg_mid, _ = avg_power_for(mid)
            if avg_mid < avg_power_target_w:
                lo = mid
            else:
                hi = mid
        vstar = 0.5 * (lo + hi)

    _, power = avg_power_for(vstar)  # leaves v_enter/time_arr set to final profile
    sum_t = float(np.sum(time_arr))
    sum_pt = float(np.sum(power * time_arr))

    times = time_arr
    total_time = float(np.sum(times))
    speed_mps = np.where(times > 0, length / times, 0.0)
    avg_power_time = (sum_pt / sum_t) if sum_t > 0.0 else 0.0


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
