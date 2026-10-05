"""
Team time trial (TTT) pacing planner for a heterogeneous group of riders.

The group shares one speed and rotates in a fixed order; each rider pulls for
their own duration, then drops to the back.  Fatigue and recovery follow the
two-parameter critical-power model with the differential W' balance
(Froncioni, Skiba and Clarke). Drafting reduces aerodynamic drag only, with
configurable reductions defaulting to 25 % for the second rider and 40 % behind.

Optimisation
------------
* Pull durations come from a closed-form flat-road periodic analysis: with the
  deficit ``x = W' - B``, each pull is the affine map ``x -> x + (P - CP)·t``
  and each recovery ``x -> x·exp(-(CP - P)·t/W')``, so one rotation cycle has a
  fixed point and a worst-case deficit that must stay within ``W'``.  A rider
  who only has to last ``T`` seconds (until dropped or finishing) can average
  ``CP + W'/T``, so that effective CP shapes their pulls.
* The control is lead power as a fraction of the current leader's CP, not a
    target speed. A forward momentum pass carries speed between route chunks
    and rotation changes. Dual prices penalise W' spending; power proposals use
    entering speed and a downstream momentum-value approximation.
* No rider may exceed ``max_power_cp_mult × CP``. Power drives acceleration;
    insufficient drive produces physical coasting, never a forced speed drop.
    Followers' speed-matching dissipation is recorded separately from pedaling.
    W' uses conservative substep peak power so intermediate depletion is not
    hidden by chunk averaging. Recovery is simulated but not rewarded in pricing.
* WTRL scores the 3rd finisher for teams of 4 and the 4th for teams of 5-8, so
  the outer search also chooses when the weakest riders are spent and dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from bike_comparison.pacing_planner import V_FLOOR, _build_chunks, _traverse
from bike_comparison.physics import AIR_DENSITY, GRAVITY, rider_cda, speed_from_power

DRAFT_FACTORS = (1.0, 0.75, 0.60)
MIN_TEAM_SIZE = 4
MAX_TEAM_SIZE = 8
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def scoring_rider_count(team_size: int) -> int:
    """WTRL: the 3rd finisher scores for teams of 4, the 4th for teams of 5-8."""
    if not MIN_TEAM_SIZE <= team_size <= MAX_TEAM_SIZE:
        raise ValueError(
            f"WTRL teams have {MIN_TEAM_SIZE}-{MAX_TEAM_SIZE} riders, got {team_size}."
        )
    return 3 if team_size == 4 else 4


@dataclass
class TTTRider:
    """One team member with a two-parameter critical-power model."""
    name: str
    weight_kg: float
    height_m: float
    cp_w: float
    w_prime_j: float
    bike_weight_kg: float = 8.0
    # Absolute rider+bike CdA; defaults to the rider-only CdA on a zero-bias bike.
    cda: Optional[float] = None
    # Scales W' recovery speed (>1 recovers faster than the generic model).
    recovery_scale: float = 1.0

    def __post_init__(self):
        for attr in ("weight_kg", "height_m", "cp_w", "w_prime_j",
                     "bike_weight_kg", "recovery_scale"):
            value = float(getattr(self, attr))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{self.name}: {attr} must be a positive number.")
        if self.cda is not None and not (math.isfinite(self.cda) and self.cda > 0.0):
            raise ValueError(f"{self.name}: cda must be a positive number.")

    @property
    def total_mass_kg(self) -> float:
        return self.weight_kg + self.bike_weight_kg

    @property
    def effective_cda(self) -> float:
        return self.cda if self.cda is not None else rider_cda(self.height_m, self.weight_kg)


@dataclass
class FlatRotation:
    """Sustainable flat-road rotation: group speed and per-rider pull durations."""
    speed_mps: float
    pulls_s: list
    lead_power_w: list
    draft_power_w: list

    @property
    def speed_kph(self) -> float:
        return self.speed_mps * 3.6


@dataclass
class TTTPlanResult:
    """Result of a TTT pacing optimisation."""
    route_name: str
    team_size: int
    scoring_rider_count: int
    feasible: bool
    total_time_seconds: float
    total_distance_km: float
    total_ascent_m: float
    avg_speed_kph: float
    riders: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    flat_rotation: Optional[dict] = None

    distance_km: list = field(default_factory=list)
    altitude_m: list = field(default_factory=list)
    gradient_pct: list = field(default_factory=list)
    speed_kph: list = field(default_factory=list)
    leader: list = field(default_factory=list)
    power_w: dict = field(default_factory=dict)
    wbal_j: dict = field(default_factory=dict)
    entry_speed_kph: list = field(default_factory=list)
    exit_speed_kph: list = field(default_factory=list)
    peak_power_w: dict = field(default_factory=dict)
    braking_w: dict = field(default_factory=dict)
    pulls: list = field(default_factory=list)

    @property
    def total_time_formatted(self) -> str:
        s = int(round(self.total_time_seconds))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class _Team:
    def __init__(self, riders: Sequence[TTTRider], crr: float, *,
                 draft_second_pct: float = 25.0, draft_rest_pct: float = 40.0):
        for label, value in (("draft_second_pct", draft_second_pct),
                             ("draft_rest_pct", draft_rest_pct)):
            if not 0.0 <= value < 100.0:
                raise ValueError(f"{label} must be >= 0 and < 100.")
        self.draft_factors = (1.0, 1.0 - draft_second_pct / 100.0,
                              1.0 - draft_rest_pct / 100.0)
        self.riders = list(riders)
        self.n = len(self.riders)
        self.names = [r.name for r in self.riders]
        self.mass = np.array([r.total_mass_kg for r in self.riders], dtype=float)
        self.cp = np.array([r.cp_w for r in self.riders], dtype=float)
        self.wp = np.array([r.w_prime_j for r in self.riders], dtype=float)
        self.rec = np.array([r.recovery_scale for r in self.riders], dtype=float)
        self.aero_k = 0.5 * AIR_DENSITY * np.array(
            [r.effective_cda for r in self.riders], dtype=float)
        self.crr = crr


def _next_puller(active, pulls, leader: int) -> int:
    n = len(active)
    for step in range(1, n + 1):
        i = (leader + step) % n
        if active[i] and pulls[i] > 0.0:
            return i
    raise ValueError("At least one active rider must pull.")


def _line_order(active, pulls, leader: int) -> list:
    order = [i for i in range(len(active)) if active[i]]
    pullers = [i for i in order if pulls[i] > 0.0]
    k = pullers.index(leader)
    return pullers[k:] + pullers[:k] + [i for i in order if pulls[i] <= 0.0]


def _line_factors(active, pulls, leader: int, draft_factors=DRAFT_FACTORS) -> np.ndarray:
    """Draft factor per rider: leader, next puller, then the rest (non-pullers last)."""
    factors = np.zeros(len(active))
    for pos, i in enumerate(_line_order(active, pulls, leader)):
        factors[i] = draft_factors[min(pos, len(draft_factors) - 1)]
    return factors


def _wbal_step(B, P, seg, cp, wp, rec):
    """Advance the differential W' balance over ``seg`` seconds at power ``P``."""
    over = P > cp
    decay = np.exp(-np.maximum(cp - P, 0.0) * seg * rec / wp)
    return np.where(over, B - (P - cp) * seg, wp - (wp - B) * decay)


def _flat_feasible(team, active, pulls, v, reserve_fraction, max_power_cp_mult,
                   cp=None) -> bool:
    mask = np.array(active, dtype=bool)
    pullers = [i for i in range(team.n) if active[i] and pulls[i] > 0.0]
    if not pullers:
        return False
    base = team.crr * team.mass * GRAVITY * v
    aero = team.aero_k * v ** 3
    cp = team.cp if cp is None else cp
    wp, rec = team.wp, team.rec
    maps = []
    for leader in pullers:
        P = base + aero * _line_factors(active, pulls, leader, team.draft_factors)
        if np.any(P[mask] > max_power_cp_mult * team.cp[mask]):
            return False
        t = pulls[leader]
        over = P > cp
        a = np.where(over, 1.0, np.exp(-np.maximum(cp - P, 0.0) * t * rec / wp))
        b = np.where(over, (P - cp) * t, 0.0)
        maps.append((a, b))
    A = np.ones(team.n)
    Bc = np.zeros(team.n)
    for a, b in maps:
        A, Bc = a * A, a * Bc + b
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(A < 1.0 - 1e-12, Bc / (1.0 - A), np.where(Bc > 0.0, np.inf, 0.0))
    worst = x.copy()
    for a, b in maps:
        x = a * x + b
        worst = np.maximum(worst, x)
    limit = (1.0 - reserve_fraction) * wp
    return bool(np.all(worst[mask] <= limit[mask] + 1e-9))


def _max_flat_speed(team, active, pulls, reserve_fraction, max_power_cp_mult,
                    lo=1.0, hi=30.0, iterations=24, cp=None) -> float:
    def ok(v):
        return _flat_feasible(team, active, pulls, v, reserve_fraction, max_power_cp_mult, cp)
    if not ok(lo):
        return 0.0
    if ok(hi):
        return hi
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _solve_flat(team, active, *, reserve_fraction, max_power_cp_mult,
                min_pull_s, max_pull_s, pull_step_s, cp=None) -> FlatRotation:
    candidates = [0.0] + [float(t) for t in np.arange(min_pull_s, max_pull_s + 1e-9, pull_step_s)]
    start = min(max(30.0, min_pull_s), max_pull_s)
    pulls = np.array([start if a else 0.0 for a in active])
    best = _max_flat_speed(team, active, pulls, reserve_fraction, max_power_cp_mult, cp=cp)
    for _ in range(6):
        improved = False
        for i in range(team.n):
            if not active[i]:
                continue
            for cand in candidates:
                if cand == pulls[i]:
                    continue
                trial = pulls.copy()
                trial[i] = cand
                if not np.any(trial > 0.0):
                    continue
                # Only a strictly faster schedule is accepted, so test that first.
                if not _flat_feasible(team, active, trial, best + 1e-6,
                                      reserve_fraction, max_power_cp_mult, cp):
                    continue
                best = _max_flat_speed(team, active, trial, reserve_fraction,
                                       max_power_cp_mult, lo=best + 1e-6, cp=cp)
                pulls = trial
                improved = True
        if not improved:
            break
    v = best
    mask = np.array(active, dtype=bool)
    base = team.crr * team.mass * GRAVITY * v
    lead = np.where(mask, base + team.aero_k * v ** 3, 0.0)
    draft = np.where(mask, base + team.aero_k * team.draft_factors[-1] * v ** 3, 0.0)
    return FlatRotation(float(v), pulls.tolist(), lead.tolist(), draft.tolist())


def solve_flat_rotation(
    riders: Sequence[TTTRider],
    *,
    crr: float = 0.004,
    reserve_fraction: float = 0.0,
    max_power_cp_mult: float = 2.0,
    min_pull_s: float = 20.0,
    max_pull_s: float = 60.0,
    pull_step_s: float = 10.0,
    draft_second_pct: float = 25.0,
    draft_rest_pct: float = 40.0,
) -> FlatRotation:
    """Fastest indefinitely sustainable flat-road rotation for a group of riders.

    Pulls are restricted to 0 (never pulls) or ``[min_pull_s, max_pull_s]``:
    the model has no changeover cost, so ever-shorter pulls would always look
    better.
    """
    if not riders:
        raise ValueError("At least one rider is required.")
    team = _Team(riders, crr, draft_second_pct=draft_second_pct, draft_rest_pct=draft_rest_pct)
    return _solve_flat(team, tuple([True] * team.n), reserve_fraction=reserve_fraction,
                       max_power_cp_mult=max_power_cp_mult, min_pull_s=min_pull_s,
                       max_pull_s=max_pull_s, pull_step_s=pull_step_s)


@dataclass
class _Sim:
    total_time: float
    speeds: np.ndarray
    chunk_time: np.ndarray
    power: np.ndarray
    min_b: np.ndarray
    wbal: np.ndarray
    front: np.ndarray
    lead_energy: np.ndarray
    pulls_start: np.ndarray
    enter_speeds: np.ndarray
    exit_speeds: np.ndarray
    peak_power: np.ndarray
    braking_power: np.ndarray
    efforts: np.ndarray
    motion_feasible: bool = True
    pulls: list = field(default_factory=list)


@dataclass
class _Plan:
    efforts: np.ndarray
    sim: _Sim
    feasible: bool
    worst_slack: float
    lam: np.ndarray
    pull_overrides: tuple = ()

    @property
    def objective(self) -> float:
        return self.sim.total_time if self.feasible else math.inf


class _CoursePlanner:
    def __init__(self, route, team: _Team, *, crr, max_chunk_m, reserve_fraction,
                 reserve_release_m, max_power_cp_mult, min_pull_s, max_pull_s,
                 pull_step_s, speed_step_mps):
        self.team = team
        length, grad, mid, end = _build_chunks(route, max_chunk_m, crr)
        if len(length) < 2:
            raise ValueError("Route is too short to plan.")
        self.length, self.grad, self.mid, self.end = length, grad, mid, end
        self.C = len(length)
        cos = np.cos(np.arctan(grad))
        self.f_static = GRAVITY * (grad[:, None] + crr * cos[:, None]) * team.mass[None, :]
        self.effort_grid = np.unique(np.r_[np.arange(0.0, max_power_cp_mult, 0.025),
                          max_power_cp_mult])
        self.reserve_fraction = reserve_fraction
        self.reserve_release_m = reserve_release_m
        self.max_power_cp_mult = max_power_cp_mult
        self.flat_kwargs = dict(max_power_cp_mult=max_power_cp_mult, min_pull_s=min_pull_s,
                                max_pull_s=max_pull_s, pull_step_s=pull_step_s)
        self.min_pull_s, self.max_pull_s = min_pull_s, max_pull_s
        self._flat_cache: dict = {}
        self._cp_eff_cache: dict = {}
        self._plan_cache: dict = {}
        self._mask_cache: dict = {}
        self.lam = 1.0 / (10.0 * team.cp)
        self.warm_efforts: Optional[np.ndarray] = None
        self.progress_callback = None
        self.progress_stage = "Team together"
        self.max_step_m = 5.0

    def flat(self, active, cp_eff=None) -> FlatRotation:
        key = (active, cp_eff)
        if key not in self._flat_cache:
            cp = None if cp_eff is None else np.array(cp_eff, dtype=float)
            self._flat_cache[key] = _solve_flat(self.team, active, reserve_fraction=0.0,
                                                cp=cp, **self.flat_kwargs)
        return self._flat_cache[key]

    def cp_eff(self, drop_chunk) -> tuple:
        """CP + usable W' spread over each rider's estimated time until exit."""
        key = tuple(int(c) for c in drop_chunk)
        if key not in self._cp_eff_cache:
            v_ref = self.flat(tuple([True] * self.team.n)).speed_mps or 10.0
            exit_time = self.end[np.array(key) - 1] / v_ref
            usable = (1.0 - self.reserve_fraction) * self.team.wp
            self._cp_eff_cache[key] = tuple(
                float(round(x)) for x in self.team.cp + usable / exit_time)
        return self._cp_eff_cache[key]

    def pulls_for(self, active, mult, cp_eff=None) -> np.ndarray:
        base = np.array(self.flat(active, cp_eff).pulls_s, dtype=float)
        return np.where(base > 0.0,
                        np.clip(base * mult, self.min_pull_s, self.max_pull_s), 0.0)

    def masks(self, drop_chunk):
        key = tuple(int(c) for c in drop_chunk)
        if key not in self._mask_cache:
            dc = np.array(key)
            active = np.arange(self.C)[:, None] < dc[None, :]
            exit_end = self.end[np.clip(dc - 1, 0, self.C - 1)]
            held = (exit_end[None, :] - self.end[:, None]) > self.reserve_release_m
            reserve = np.where(held, self.reserve_fraction * self.team.wp[None, :], 0.0)
            self._mask_cache[key] = (active, reserve)
        return self._mask_cache[key]

    def advance(self, chunk, speed, effort, leader, factors, mask, distance, *, enforce_caps=True):
        team = self.team
        ratio = team.mass / team.mass[leader]
        aero = team.aero_k * factors
        delta = aero - ratio * aero[leader]
        cap = self.max_power_cp_mult * team.cp
        requested = min(max(float(effort), 0.0), self.max_power_cp_mult) * team.cp[leader]

        def traverse(drive):
            exit_speed, duration = _traverse(
                speed, drive, float(self.f_static[chunk, leader]), 0.0,
                float(aero[leader]), 1.0 / team.mass[leader], distance)
            peak = np.maximum(ratio * drive + delta * speed ** 3,
                              ratio * drive + delta * exit_speed ** 3)
            return exit_speed, duration, peak

        drive = requested
        exit_speed, duration, peak = traverse(drive)
        if enforce_caps:
            coast_speed, _, _ = traverse(0.0)
            minimum_speed = min(speed, coast_speed)
            maximum_speed = max(speed, exit_speed)
            worst_delta = np.maximum(delta * minimum_speed ** 3, delta * maximum_speed ** 3)
            drive_limit = float(np.min(((cap - worst_delta) / ratio)[mask]))
            drive = min(requested, max(drive_limit, 0.0))
            exit_speed, duration, peak = traverse(drive)
            if np.any(peak[mask] > cap[mask] + 1e-7):
                lo, hi = 0.0, drive
                for _ in range(16):
                    mid = 0.5 * (lo + hi)
                    trial_speed, trial_time, trial_peak = traverse(mid)
                    if np.any(trial_peak[mask] > cap[mask]):
                        hi = mid
                    else:
                        lo = mid
                drive = lo
                exit_speed, duration, peak = traverse(drive)
        mean_speed = 0.5 * (speed + exit_speed)
        required = ratio * drive + delta * mean_speed ** 3
        power = np.where(mask, np.maximum(required, 0.0), 0.0)
        braking = np.where(mask, np.maximum(-required, 0.0), 0.0)
        return exit_speed, duration, power, braking, np.where(mask, np.maximum(peak, 0.0), 0.0)

    def simulate(self, efforts, drop_chunk, mult, initial_speed_mps=V_FLOOR,
                 pull_overrides=()) -> _Sim:
        """Hold the lead-power/CP effort from each pull's start, carrying momentum."""
        team, C, n = self.team, self.C, self.team.n
        cp, wp, rec = team.cp, team.wp, team.rec
        B = wp.copy()
        shape = (C, n)
        min_b = np.full(shape, np.nan)
        wbal = np.full(shape, np.nan)
        power = np.zeros(shape)
        front = np.zeros(shape)
        pulls_start = np.zeros(shape)
        lead_energy = np.zeros(n)
        actual = np.empty(C)
        chunk_time = np.empty(C)
        enter_speeds = np.empty(C)
        exit_speeds = np.empty(C)
        peak_power = np.zeros(shape)
        braking_power = np.zeros(shape)
        cp_eff = self.cp_eff(drop_chunk)
        active = None
        previous_overrides = None
        pulls = None
        leader = n - 1
        remaining = 0.0
        pull_effort = None
        pull_log = []
        total_elapsed = 0.0
        factor_cache: dict = {}
        speed = max(float(initial_speed_mps), V_FLOOR)
        motion_feasible = True
        for c in range(C):
            act = tuple(bool(c < drop_chunk[i]) for i in range(n))
            overrides = tuple((i, duration) for i, start, duration in pull_overrides
                              if c >= start and act[i])
            if act != active or overrides != previous_overrides:
                was_leading = active is not None and act[leader]
                active = act
                previous_overrides = overrides
                pulls = self.pulls_for(active, mult, cp_eff).copy()
                for i, duration in overrides:
                    pulls[i] = duration
                mask = np.array(active)
                factor_cache = {}
                if was_leading and pulls[leader] > 0.0:
                    remaining = min(remaining, pulls[leader])
                else:
                    leader = _next_puller(active, pulls, leader)
                    remaining = pulls[leader]
                    pull_effort = None
            length = float(self.length[c])
            enter_speeds[c] = speed
            pulls_start[c] = pulls
            seg_min = B.copy()
            distance_left = length
            elapsed = 0.0
            while distance_left > 1e-8:
                if pull_effort is None:
                    pull_effort = min(max(float(efforts[c]), 0.0), self.max_power_cp_mult)
                    pull_log.append({
                        "rider": team.names[leader],
                        "start_time_s": total_elapsed,
                        "duration_s": 0.0,
                        "start_km": float(self.end[c] - distance_left) / 1000.0,
                        "end_km": float(self.end[c] - distance_left) / 1000.0,
                        "power_w": pull_effort * cp[leader],
                        "power_wkg": pull_effort * cp[leader] / team.riders[leader].weight_kg,
                    })
                if leader not in factor_cache:
                    factor_cache[leader] = _line_factors(active, pulls, leader, team.draft_factors)
                requested = pull_effort * cp[leader]
                acceleration = (requested / speed - self.f_static[c, leader] -
                                team.aero_k[leader] * speed ** 2) / team.mass[leader]
                distance = min(self.max_step_m, distance_left, speed * remaining)
                if abs(acceleration) > 1e-9 and (
                    acceleration > 0.0 or speed > V_FLOOR + 1e-8):
                    distance = min(distance, 0.05 * speed ** 2 / abs(acceleration))
                for _ in range(8):
                    exit_speed, dt, P, braking, peak = self.advance(
                        c, speed, pull_effort, leader, factor_cache[leader], mask, distance,
                        enforce_caps=False)
                    if dt <= remaining + 1e-9:
                        break
                    distance *= remaining / dt * (1.0 - 1e-9)
                if speed <= V_FLOOR + 1e-8 and exit_speed <= V_FLOOR + 1e-8 and (
                        P[leader] < self.f_static[c, leader] * V_FLOOR +
                        team.aero_k[leader] * V_FLOOR ** 3):
                    motion_feasible = False
                B = _wbal_step(B, peak, dt, cp, wp, rec)
                seg_min = np.minimum(seg_min, B)
                power[c] += P * dt
                braking_power[c] += braking * dt
                peak_power[c] = np.maximum(peak_power[c], peak)
                front[c, leader] += dt
                lead_energy[leader] += P[leader] * dt
                pull_log[-1]["duration_s"] += dt
                pull_log[-1]["end_km"] = float(self.end[c] - distance_left + distance) / 1000.0
                total_elapsed += dt
                remaining -= dt
                elapsed += dt
                distance_left -= distance
                speed = exit_speed
                if remaining <= 1e-8:
                    leader = _next_puller(active, pulls, leader)
                    remaining = pulls[leader]
                    pull_effort = None
            chunk_time[c] = elapsed
            actual[c] = length / elapsed
            exit_speeds[c] = speed
            power[c] /= elapsed
            braking_power[c] /= elapsed
            min_b[c, mask] = seg_min[mask]
            wbal[c, mask] = B[mask]
        return _Sim(float(np.sum(chunk_time)), actual, chunk_time, power, min_b, wbal, front,
                    lead_energy, pulls_start, enter_speeds, exit_speeds, peak_power,
                    braking_power, np.asarray(efforts).copy(), motion_feasible, pull_log)

    def slack(self, sim: _Sim, drop_chunk) -> np.ndarray:
        active, reserve = self.masks(drop_chunk)
        margin = np.where(active, (sim.min_b - reserve) / self.team.wp[None, :], np.inf)
        return np.min(margin, axis=0)

    def price_efforts(self, lam, sim: _Sim, drop_chunk) -> np.ndarray:
        team = self.team
        active, _ = self.masks(drop_chunk)
        cp = team.cp
        n_pos = len(team.draft_factors)
        share = np.zeros((self.C, team.n, n_pos))
        mean_drive = np.zeros(self.C)
        mean_aero = np.zeros(self.C)
        groups: dict = {}
        for c in range(self.C):
            key = (tuple(bool(a) for a in active[c]), tuple(sim.pulls_start[c]))
            groups.setdefault(key, []).append(c)
        # Price each rider's position mix over the whole rotation, so the speed
        # depends on terrain rather than on last pass's rotation phase.
        for (act, pulls), rows in groups.items():
            pulls = np.array(pulls)
            cycle = float(np.sum(pulls))
            mix = np.zeros((team.n, n_pos))
            for leader in np.nonzero(pulls > 0.0)[0]:
                for pos, rider_index in enumerate(_line_order(act, pulls, int(leader))):
                    mix[rider_index, min(pos, n_pos - 1)] += pulls[leader] / cycle
            share[rows] = mix
            lead_share = mix[:, 0]
            mean_drive[rows] = np.sum(lead_share * team.cp / team.mass)
            mean_aero[rows] = np.sum(lead_share * team.aero_k / team.mass)
        entering = sim.enter_speeds[:, None]
        effort = self.effort_grid[None, :]
        force = self.f_static[:, :1] / team.mass[0]
        lo = np.full((self.C, len(self.effort_grid)), V_FLOOR)
        hi = np.sqrt(entering ** 2 + 2.0 * self.length[:, None] *
                     (effort * mean_drive[:, None] / V_FLOOR + np.maximum(-force, 0.0)))
        hi = np.maximum(hi, V_FLOOR)
        for _ in range(20):
            exiting = 0.5 * (lo + hi)
            mean_speed = 0.5 * (entering + exiting)
            acceleration = (effort * mean_drive[:, None] / mean_speed - force -
                            mean_aero[:, None] * mean_speed ** 2)
            residual = exiting ** 2 - entering ** 2 - 2.0 * self.length[:, None] * acceleration
            hi = np.where(residual > 0.0, exiting, hi)
            lo = np.where(residual > 0.0, lo, exiting)
        exiting = 0.5 * (lo + hi)
        mean_speed = 0.5 * (entering + exiting)
        duration = self.length[:, None] / mean_speed
        drive_power = team.mass[None, None, :] * mean_drive[:, None, None] * effort[:, :, None]
        rate = np.zeros((self.C, len(self.effort_grid), team.n))
        for pos, d in enumerate(team.draft_factors):
            delta = team.aero_k[None, :] * d - team.mass[None, :] * mean_aero[:, None]
            P = drive_power + delta[:, None, :] * mean_speed[:, :, None] ** 3
            rate += share[:, None, :, pos] * np.maximum(P - cp[None, None, :], 0.0)
        rate = np.where(active[:, None, :], rate, 0.0)
        cost = (1.0 + rate @ lam) * duration
        downstream = np.zeros(self.C)
        for chunk in range(self.C - 2, -1, -1):
            speed = sim.enter_speeds[chunk + 1]
            drag = 2.0 * mean_aero[chunk + 1] * speed
            transmission = math.exp(-drag * sim.chunk_time[chunk + 1])
            downstream[chunk] = transmission * downstream[chunk + 1] - (
                sim.chunk_time[chunk + 1] / speed) * transmission
        cost += downstream[:, None] * (exiting - sim.exit_speeds[:, None])
        idx = np.argmin(cost, axis=1)
        return self.effort_grid[idx]

    def _check(self, efforts, drop_chunk, mult, pull_overrides=()):
        sim = self.simulate(efforts, drop_chunk, mult, pull_overrides=pull_overrides)
        slack = self.slack(sim, drop_chunk)
        capped = np.all(sim.peak_power <= self.max_power_cp_mult * self.team.cp[None, :] + 1e-6)
        return sim, slack, bool(np.min(slack) > 0.0 and capped and sim.motion_feasible)

    def _publish(self, plan, drop_chunk, mult, iteration, iterations, step):
        if self.progress_callback is not None:
            self.progress_callback(plan, drop_chunk, mult, iteration, iterations, step)

    def solve(self, drop_chunk, mult, iterations: int, pull_overrides=()) -> _Plan:
        key = (tuple(int(c) for c in drop_chunk), tuple(np.round(mult, 4)), pull_overrides)
        if key in self._plan_cache:
            return self._plan_cache[key]
        lam = self.lam.copy()
        if self.warm_efforts is not None:
            efforts = self.warm_efforts.copy()
        else:
            efforts = np.ones(self.C)
        sim = self.simulate(efforts, drop_chunk, mult, pull_overrides=pull_overrides)
        best: Optional[_Plan] = None
        near: Optional[_Plan] = None
        fallback: Optional[_Plan] = None
        step = 1.0
        for iteration in range(1, iterations + 1):
            efforts = 0.5 * efforts + 0.5 * self.price_efforts(lam, sim, drop_chunk)
            sim, slack, feasible = self._check(efforts, drop_chunk, mult, pull_overrides)
            worst = float(np.min(slack))
            plan = _Plan(efforts, sim, feasible, worst, lam.copy(), pull_overrides)
            if feasible and (best is None or sim.total_time < best.sim.total_time):
                best = plan
            if worst > -0.1 and (near is None or sim.total_time < near.sim.total_time):
                near = plan
            if fallback is None or worst > fallback.worst_slack:
                fallback = plan
            self._publish(plan, drop_chunk, mult, iteration, iterations, "power search")
            lam = np.clip(lam * np.exp(-step * np.clip(slack, -1.0, 1.0)), 1e-7, 0.05)
            step *= 0.95
        tightened = [p for p in (self._tighten(c, drop_chunk, mult)
                                 for c in (best, near) if c is not None) if p is not None]
        if not tightened and fallback is not None:
            rescued = self._tighten(fallback, drop_chunk, mult)
            if rescued is not None:
                tightened = [rescued]
        if tightened:
            result = min(tightened, key=lambda p: p.sim.total_time)
            self.lam = result.lam
            self.warm_efforts = result.efforts
        else:
            result = fallback
        self._plan_cache[key] = result
        return result

    def _tighten(self, plan: _Plan, drop_chunk, mult) -> Optional[_Plan]:
        """Scale power efforts uniformly to sit on the W' feasibility boundary."""
        lo, hi = (1.0, 1.06) if plan.feasible else (0.3, 1.0)
        sim, slack, feasible = self._check(plan.efforts * lo, drop_chunk, mult, plan.pull_overrides)
        if not feasible:
            return None
        best = _Plan(plan.efforts * lo, sim, True, float(np.min(slack)), plan.lam, plan.pull_overrides)
        iterations = 8 if plan.feasible else 12
        self._publish(best, drop_chunk, mult, 0, iterations, "reserve check")
        for iteration in range(1, iterations + 1):
            mid = 0.5 * (lo + hi)
            efforts = plan.efforts * mid
            sim, slack, feasible = self._check(efforts, drop_chunk, mult, plan.pull_overrides)
            if feasible:
                lo = mid
                best = _Plan(efforts, sim, True, float(np.min(slack)), plan.lam, plan.pull_overrides)
            else:
                hi = mid
            candidate = _Plan(efforts, sim, feasible, float(np.min(slack)), plan.lam, plan.pull_overrides)
            self._publish(candidate, drop_chunk, mult, iteration, iterations, "reserve check")
        return best


    def finish_effort(self, plan: _Plan, drop_chunk, mult) -> _Plan:
        if not plan.feasible:
            return plan
        best = plan
        self.progress_stage = "Finish effort"
        for fraction in (0.5, 0.75, 0.875, 0.9375):
            start = int(np.searchsorted(self.end, fraction * self.end[-1], side="right"))
            base = best.efforts.copy()
            lo, hi = 0.0, 1.0
            iterations = 12
            step = f"closing {(1.0 - fraction) * 100.0:g}%"
            for iteration in range(1, iterations + 1):
                fraction_used = hi if iteration == 1 else 0.5 * (lo + hi)
                efforts = base.copy()
                efforts[start:] += fraction_used * np.maximum(
                    self.max_power_cp_mult - efforts[start:], 0.0)
                sim, slack, feasible = self._check(efforts, drop_chunk, mult, best.pull_overrides)
                candidate = _Plan(efforts, sim, feasible, float(np.min(slack)), best.lam,
                                  best.pull_overrides)
                if feasible:
                    lo = fraction_used
                    if sim.total_time < best.sim.total_time:
                        best = candidate
                else:
                    hi = fraction_used
                self._publish(candidate, drop_chunk, mult, iteration, iterations, step)
                if iteration == 1 and feasible:
                    break
        return best


    def reconsider_non_pullers(self, plan, drop_chunk, mult, iterations):
        if not plan.feasible:
            return plan
        best = plan
        candidates = [rider_index for rider_index in range(self.team.n)
                      if drop_chunk[rider_index] == self.C and
                      not np.any(plan.sim.front[:, rider_index] > 0.0)]
        candidates.sort(key=lambda rider_index: float(
            plan.sim.wbal[-1, rider_index] / self.team.wp[rider_index]), reverse=True)
        starts = [0]
        starts.extend(int(np.searchsorted(self.end, fraction * self.end[-1], side="right"))
                      for fraction in (0.5, 0.75))
        for rider_index in candidates:
            for start in sorted(set(starts)):
                overrides = tuple(entry for entry in best.pull_overrides if entry[0] != rider_index)
                overrides += ((rider_index, start, self.min_pull_s),)
                start_km = (self.end[start] - self.length[start]) / 1000.0
                self.progress_stage = f"Reconsider pulls: {self.team.names[rider_index]} from {start_km:.1f} km"
                trial = self.solve(drop_chunk, mult, iterations, pull_overrides=overrides)
                if trial.feasible and trial.sim.total_time < best.sim.total_time:
                    trial = self.finish_effort(trial, drop_chunk, mult)
                    if trial.sim.total_time < best.sim.total_time:
                        best = trial
        return best


def _golden_min(f, lo: float, hi: float, evals: int):
    a, b = lo, hi
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max(0, evals - 2)):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = f(d)
    return (c, fc) if fc <= fd else (d, fd)


def _weakness_ranking(planner: _CoursePlanner, crr: float) -> list:
    """Riders ordered weakest first: mean required lead power relative to CP."""
    team = planner.team
    flat = planner.flat(tuple([True] * team.n))
    p_ref = float(np.mean(flat.lead_power_w)) or float(np.mean(team.cp))
    mean_weight = float(np.mean([r.weight_kg for r in team.riders]))
    mean_bike = float(np.mean([r.bike_weight_kg for r in team.riders]))
    mean_cda = float(np.mean([r.effective_cda for r in team.riders]))
    cache: dict = {}
    ratio = np.zeros(team.n)
    total_t = 0.0
    for c in range(planner.C):
        g = round(float(planner.grad[c]), 3)
        if g not in cache:
            cache[g] = speed_from_power(p_ref, g, mean_weight, mean_bike, mean_cda, crr)
        v = max(cache[g], 1.0)
        t = planner.length[c] / v
        lead = np.maximum(planner.f_static[c] * v + team.aero_k * v ** 3, 0.0)
        ratio += t * lead / team.cp
        total_t += t
    return list(np.argsort(-ratio / total_t))


def plan_ttt_pacing(
    route,
    riders: Sequence[TTTRider],
    *,
    crr: float = 0.004,
    reserve_fraction: float = 0.1,
    reserve_release_m: float = 2000.0,
    max_power_cp_mult: float = 1.5,
    min_pull_s: float = 20.0,
    max_pull_s: float = 60.0,
    pull_step_s: float = 10.0,
    max_chunk_m: float = 100.0,
    speed_step_mps: float = 0.1,
    allow_drops: bool = True,
    drop_candidates: Optional[Sequence[str]] = None,
    drop_search_evals: int = 6,
    drop_search_passes: int = 1,
    search_iterations: int = 12,
    final_iterations: int = 40,
    refine_pulls: bool = True,
    downsample_points: int = 400,
    progress_callback: Optional[Callable[[dict], None]] = None,
    draft_second_pct: float = 25.0,
    draft_rest_pct: float = 40.0,
) -> TTTPlanResult:
    """Optimise a WTRL team time trial for riders listed in rotation order.

    Args:
        route: A ``RouteProfile``.
        riders: 4-8 riders in their fixed rotation order.
        reserve_fraction: W' fraction each rider keeps in hand, as a margin for
            CP/W' estimation error, until ``reserve_release_m`` before they
            finish or are dropped.
        max_power_cp_mult: Ceiling on any rider's power, including accelerations,
            as a multiple of CP.
        draft_second_pct, draft_rest_pct: Aerodynamic drag reductions in percent
            for the second rider and third-and-later riders, each in [0, 100).
        min_pull_s, max_pull_s, pull_step_s: Allowed pull durations (0 = never
            pulls), defaulting to 20-60 seconds in 10-second steps; the model has
            no changeover cost, so a minimum is required.
        allow_drops: Let riders beyond the scoring count be spent and dropped.
        drop_candidates: Rider names, in the order they may be dropped;
            defaults to weakest first for this course.
        progress_callback: Called after each iteration with progress metadata
            and the best feasible plan so far, or None until one is found.
    """
    team_size = len(riders)
    scoring = scoring_rider_count(team_size)
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in [0, 1).")
    if not max_power_cp_mult > 1.0:
        raise ValueError("max_power_cp_mult must be greater than 1.")
    if not 0.0 < min_pull_s <= max_pull_s:
        raise ValueError("Pull duration limits must satisfy 0 < min_pull_s <= max_pull_s.")
    names = [r.name for r in riders]
    if len(set(names)) != len(names):
        raise ValueError("Rider names must be unique.")

    team = _Team(riders, crr, draft_second_pct=draft_second_pct, draft_rest_pct=draft_rest_pct)
    planner = _CoursePlanner(
        route, team, crr=crr, max_chunk_m=max_chunk_m, reserve_fraction=reserve_fraction,
        reserve_release_m=reserve_release_m, max_power_cp_mult=max_power_cp_mult,
        min_pull_s=min_pull_s, max_pull_s=max_pull_s, pull_step_s=pull_step_s,
        speed_step_mps=speed_step_mps)
    C, n = planner.C, team.n
    incumbent = None
    incumbent_result = None
    progress_iteration = 0

    def report(candidate, drop_chunk, multipliers, iteration, iterations, step):
        nonlocal incumbent, incumbent_result, progress_iteration
        improved = candidate.feasible and (
            incumbent is None or candidate.sim.total_time < incumbent[0].sim.total_time)
        if improved:
            incumbent = (candidate, np.array(drop_chunk).copy(), multipliers.copy())
            if progress_callback is not None:
                incumbent_result = _build_result(
                    route, planner, candidate, drop_chunk, multipliers, scoring, downsample_points)
        if progress_callback is not None:
            progress_iteration += 1
            progress_callback({
                "iteration": progress_iteration,
                "stage": planner.progress_stage,
                "step": step,
                "candidate_iteration": iteration,
                "candidate_iterations": iterations,
                "candidate_feasible": candidate.feasible,
                "improved": bool(improved),
                "plan": incumbent_result,
            })

    planner.progress_callback = report

    if drop_candidates is not None:
        unknown = set(drop_candidates) - set(names)
        if unknown:
            raise ValueError(f"Unknown drop candidates: {sorted(unknown)}")
        ranking = [names.index(name) for name in drop_candidates]
    else:
        ranking = _weakness_ranking(planner, crr)
    max_drops = min(team_size - scoring, len(ranking)) if allow_drops else 0

    mult = np.ones(n)

    def drop_chunks_for(dropped, fracs):
        dc = np.full(n, C, dtype=int)
        for i, f in zip(dropped, fracs):
            dc[i] = int(np.clip(round(f * C), 1, C - 1))
        return dc

    best_dc = np.full(n, C, dtype=int)
    best_val = planner.solve(best_dc, mult, search_iterations).objective
    for m in range(1, max_drops + 1):
        planner.progress_stage = f"Drop search ({m} rider{'s' if m != 1 else ''})"
        dropped = ranking[:m]
        fracs = list(np.linspace(0.45, 0.8, m)) if m > 1 else [0.6]

        def objective(fr, dropped=dropped):
            return planner.solve(drop_chunks_for(dropped, fr), mult, search_iterations).objective

        current = objective(fracs)
        for _ in range(drop_search_passes):
            for j in range(m):
                def along(x, j=j):
                    trial = list(fracs)
                    trial[j] = x
                    return objective(trial)
                x, val = _golden_min(along, 0.02, 0.98, drop_search_evals)
                if val < current:
                    fracs[j], current = x, val
        if current < best_val:
            best_val, best_dc = current, drop_chunks_for(dropped, fracs)
        elif current > best_val + 1.0:
            break

    if refine_pulls:
        for i in range(n):
            planner.progress_stage = f"Pull refinement: {names[i]}"
            for factor in (0.7, 1.4):
                trial = mult.copy()
                trial[i] *= factor
                val = planner.solve(best_dc, trial, search_iterations).objective
                if val < best_val:
                    best_val, mult = val, trial

    planner._plan_cache.clear()
    planner.progress_stage = "Final refinement"
    plan = planner.solve(best_dc, mult, final_iterations)
    if incumbent is not None and (
            not plan.feasible or incumbent[0].sim.total_time < plan.sim.total_time):
        plan, best_dc, mult = incumbent
    plan = planner.finish_effort(plan, best_dc, mult)
    if refine_pulls:
        plan = planner.reconsider_non_pullers(plan, best_dc, mult, max(2, search_iterations // 2))
    if incumbent is not None and incumbent[0].sim.total_time < plan.sim.total_time:
        plan, best_dc, mult = incumbent
    return _build_result(route, planner, plan, best_dc, mult, scoring, downsample_points)


def _build_result(route, planner: _CoursePlanner, plan: _Plan, drop_chunk, mult,
                  scoring: int, downsample_points: int) -> TTTPlanResult:
    team, sim, C = planner.team, plan.sim, planner.C
    active, _ = planner.masks(drop_chunk)
    total_time = sim.total_time
    dist_km = route.display_distance_km

    rider_rows = []
    for i, r in enumerate(team.riders):
        act = active[:, i]
        active_time = float(np.sum(sim.chunk_time[act]))
        energy = float(np.sum(sim.power[act, i] * sim.chunk_time[act]))
        front_time = float(np.sum(sim.front[:, i]))
        dropped = drop_chunk[i] < C
        min_wbal = float(np.nanmin(sim.min_b[:, i]))
        rider_rows.append({
            "name": r.name,
            "cp_w": r.cp_w,
            "w_prime_j": r.w_prime_j,
            "finishes": not dropped,
            "drop_km": round(float(planner.end[drop_chunk[i] - 1]) / 1000.0, 2) if dropped else None,
            "drop_time_s": round(float(np.sum(sim.chunk_time[:drop_chunk[i]])), 1) if dropped else None,
            "min_wbal_j": round(min_wbal),
            "min_wbal_pct": round(100.0 * min_wbal / r.w_prime_j, 1),
            "avg_power_w": round(energy / active_time, 1) if active_time > 0 else 0.0,
            "time_on_front_s": round(front_time, 1),
            "avg_lead_power_w": round(float(sim.lead_energy[i]) / front_time, 1) if front_time > 0 else None,
        })

    phases = []
    start = 0
    cp_eff = planner.cp_eff(drop_chunk)
    for c in range(1, C + 1):
        if c == C or not np.array_equal(active[c], active[start]) or not np.array_equal(
                sim.pulls_start[c], sim.pulls_start[start]):
            act = tuple(bool(a) for a in active[start])
            pulls = sim.pulls_start[start]
            phases.append({
                "start_km": round(float(planner.end[start] - planner.length[start]) / 1000.0, 2),
                "end_km": round(float(planner.end[c - 1]) / 1000.0, 2),
                "riders": [team.names[i] for i in range(team.n) if act[i]],
                "pulls_s": {team.names[i]: round(float(pulls[i]), 1)
                            for i in range(team.n) if act[i]},
                "flat_speed_kph": round(planner.flat(act, cp_eff).speed_kph, 1),
            })
            start = c

    full_flat = planner.flat(tuple([True] * team.n))
    flat_summary = {
        "speed_kph": round(full_flat.speed_kph, 1),
        "pulls_s": {team.names[i]: full_flat.pulls_s[i] for i in range(team.n)},
        "lead_power_w": {team.names[i]: round(full_flat.lead_power_w[i]) for i in range(team.n)},
        "draft_power_w": {team.names[i]: round(full_flat.draft_power_w[i]) for i in range(team.n)},
    }

    if C > downsample_points:
        idx = np.unique(np.round(np.linspace(0, C - 1, downsample_points)).astype(int))
    else:
        idx = np.arange(C)
    altitude = np.interp(planner.mid, np.asarray(route.distance_m, dtype=float),
                         np.asarray(route.altitude_m, dtype=float))
    leader_idx = np.argmax(sim.front, axis=1)

    def series(values, i):
        return [round(float(values[c, i]), 1) if active[c, i] else None for c in idx]

    return TTTPlanResult(
        route_name=route.name,
        team_size=team.n,
        scoring_rider_count=scoring,
        feasible=plan.feasible,
        total_time_seconds=total_time,
        total_distance_km=dist_km,
        total_ascent_m=route.display_ascent_m,
        avg_speed_kph=round(dist_km / (total_time / 3600.0), 1) if total_time > 0 else 0.0,
        riders=rider_rows,
        phases=phases,
        pulls=sim.pulls,
        flat_rotation=flat_summary,
        distance_km=[round(float(planner.mid[c]) / 1000.0, 3) for c in idx],
        altitude_m=[round(float(altitude[c]), 1) for c in idx],
        gradient_pct=[round(float(planner.grad[c]) * 100.0, 1) for c in idx],
        speed_kph=[round(float(sim.speeds[c]) * 3.6, 1) for c in idx],
        leader=[team.names[int(leader_idx[c])] for c in idx],
        power_w={name: series(sim.power, i) for i, name in enumerate(team.names)},
        wbal_j={name: series(sim.wbal, i) for i, name in enumerate(team.names)},
        entry_speed_kph=[round(float(sim.enter_speeds[c]) * 3.6, 3) for c in idx],
        exit_speed_kph=[round(float(sim.exit_speeds[c]) * 3.6, 3) for c in idx],
        peak_power_w={name: series(sim.peak_power, i) for i, name in enumerate(team.names)},
        braking_w={name: series(sim.braking_power, i) for i, name in enumerate(team.names)},
    )
