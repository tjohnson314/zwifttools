"""
Race analyzer: find where a rider lost contact with their group and why.

Pipeline (all analytical — physics + statistics, no learned model yet):

1. Put every rider on a shared 1 Hz race clock (``RaceGrid``).
2. Group riders each second with a time-gap rule: sort by distance and start a
   new group wherever the gap to the rider ahead exceeds ``GROUP_GAP_SEC``.
   Gaps are measured in *seconds* (when the rider ahead passed the same spot),
   which is robust to speed and less sensitive to distance-alignment error.
3. For the selected rider, detect separations: the rider's group splits and
   the larger part ends up ahead.  A split where a few riders ride away from a
   bigger group is a flier and is ignored for the riders left behind.  If the
   rider regains the group within ``DROP_CONFIRM_SEC`` it is a near miss,
   otherwise a drop.
4. For each event, estimate what it took to stay: convert every survivor's
   power into the selected rider's body (mass scales gravity/rolling/kinetic
   terms, CdA scales the aero term — so the survivor's draft carries over),
   and compare against what the rider actually did.
5. Look at the minutes before: excess power vs. the riders around you, draft and
   position, and W' balance (Skiba differential model) with a counterfactual
   of what W' would have been left without the excess.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bike_comparison.physics import AIR_DENSITY, GRAVITY, DRIVETRAIN_LOSS, rider_cda

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Gap (seconds) to the rider ahead that splits two groups.  Can be overridden
# per request with ?gap_sec= on the analysis API.
GROUP_GAP_SEC = 1.5

# Distance glitches (route alignment snapping to the wrong road): a 1-second
# distance change this far from the rider's speed is impossible...
DIST_JUMP_TOL_M = 50.0
# After a jump, the trace is rebuilt from speed until the original distance
# agrees with the rebuilt path (within this many metres) for this long.
DIST_REJOIN_TOL_M = 15.0
DIST_RECOVER_SEC = 10
DIST_REPAIR_MAX_SEC = 600       # give up (and leave it) if it never recovers

# A split counts when the part that went ahead has more than FLIER_MAX_RIDERS
# riders and at least this share of the rider's group (smaller = breakaway).
SPLIT_MIN_FRAC = 0.3

# "Good moves": getting away from your group, or bridging up to the group ahead.
GETAWAY_MIN_GAP_SEC = 10.0     # the gap you open has to reach this...
GETAWAY_CONFIRM_SEC = 180      # ...within this long, and they never get back to you
GETAWAY_LEADUP_SEC = 30        # the attack is taken to start this long before the split
BRIDGE_MIN_GAP_SEC = 5.0       # a bridge closes at least this much
BRIDGE_LOOKBACK_SEC = 900      # how far back to look for the biggest gap you closed
MAX_GOOD_MOVES = 8

# Smallest group (including the rider) whose splits we analyse.
MIN_GROUP_SIZE = 3

# Off the group for longer than this = dropped; back on sooner = near miss.
DROP_CONFIRM_SEC = 120

# Separations shorter than this are treated as gap-measurement flicker.
MIN_SEPARATION_SEC = 5

# Separations this close to the finish are the sprint, not getting dropped.
FINISH_EXCLUDE_M = 1000.0

# How far before the selection to look for wasted effort.
WASTE_LOOKBACK_SEC = 300

# Selection window search limits around the separation.  The hardest part of a
# selection often comes before the gap actually opens, so look back this far
# for the pack's peak effort.
SELECTION_LOOKBACK_SEC = 90
SELECTION_MAX_BEFORE_SEC = 600
SELECTION_MAX_AFTER_SEC = 180

# Draft loss: still in the group by the gap rule, but getting much less draft
# than group-mates (hanging off the back, on the front, or in the wind).
DRAFT_SMOOTH_SEC = 10        # smoothing for per-second draft estimates
DRAFT_LOSS_PCT = 15.0        # draft saving this many points below group-mates
DRAFT_LOSS_MIN_SEC = 10      # sustained at least this long
DRAFT_LOSS_MERGE_SEC = 3     # bridge brief blips inside one episode
# How far before a separation a draft-loss episode counts as part of it.
DRAFT_LOSS_LINK_SEC = 120

# Sitting up: power well below what the group needed, right as contact goes.
SAT_UP_FRAC = 0.5            # below this fraction of the group's power
SAT_UP_MIN_SEC = 6           # for at least this long
SAT_UP_SEARCH = (10, 30)     # seconds before/after the separation to search
SAT_UP_MIN_PACK_FRAC_CP = 0.6  # ignore when the group itself is soft-pedalling

# Gap-closing simulation.  Draft vs. distance to the wheel ahead, measured
# from pooled race data: roughly full draft to ~10 m, fading to none by ~25 m.
DRAFT_FULL_M = 10.0
DRAFT_ZERO_M = 25.0
CLOSE_TARGET_M = 3.0         # "back on the wheel"
CLOSE_LIMITS_SEC = (15, 30)  # how quickly the rider gets back on
CLOSE_STEP_SEC = 6           # spacing of the "react at" rows
CLOSE_MAX_ROWS = 6
CLOSE_LOOKBACK_SEC = 90      # how far before the split to look for the onset
HOLD_EXTRA_SEC = 10          # hold the wheel this long past the split
MAX_SIM_POWER_W = 1500.0
# Skip cost-to-close when replaying actual power misses the real gap by more.
GAP_MODEL_MAX_ERROR_M = 10.0

# "Position earlier": move up before the selection, then ride the surge from a
# real survivor's slot (their power, converted to the rider, includes the
# draft at that position).
MOVE_UP_SEC = (30, 60)           # move-up durations, ending as the selection starts
MOVE_UP_TARGETS = (0.5, 0.25)    # target slot as a fraction of the group (0 = front)
MOVE_UP_MIN_GROUP = 6            # positioning is moot in smaller groups
MOVE_UP_MIN_PLACES = 3           # skip moves that gain fewer places
# Draft while moving up through the bunch, as a fraction of full draft.  Race
# data shows riders moving up keep ~the group's draft (median 1.0, p25 0.9);
# 0.9 keeps estimates conservative.
MOVE_UP_DRAFT_FRAC = 0.9
MOVE_UP_SLOT_SPAN = 3            # survivors within this many places of the slot
# Report sliding back through the group by at least this many places.
POSITION_DRIFT_MIN_PLACES = 5
POSITION_DRIFT_LOOKBACK_SEC = 180   # look for the rider's best spot this far back
POSITION_SMOOTH_SEC = 9             # positions jitter in a bunch; smooth before comparing
# A move-up has to gain at least this much road: in a tight bunch, places are
# riders side by side and moving past them costs next to nothing.
MOVE_UP_MIN_GAIN_M = 10.0
# Call a W' low point "borderline" within this fraction of the W' available.
BORDERLINE_W_PRIME_FRAC = 0.1
# Energy differences smaller than this are reported as "about the same".
SAME_EFFORT_KJ = 0.5
# Targets above this multiple of CP are reported as "a full sprint — not realistic".
UNREALISTIC_POWER_FRAC_CP = 2.5
# Riding within this fraction of what the surge took counts as "no margin".
NO_MARGIN_FRAC = 0.02
WELL_UNDER_FRAC = 0.10       # more than this short = "well under"
QUICK_RECOVERY_SEC = 10      # back on within this = "closed it straight away"
SIM_DT = 0.1

# Power comparisons use this many group-mates nearest on the road, not the
# whole group (whose average is skewed by riders pulling on the front).
NEARBY_RIDERS = 5

# How many of the race's worst draft-loss episodes to list, and the smallest
# group they're listed for (taking turns in a tiny group isn't a mistake).
TOP_EXPOSURES = 8
TOP_EXPOSURE_MIN_GROUP = 5

# "Where you spent extra": efforts that cost the rider more than the riders
# around them — chasing fliers, attacks, pulls, riding out of the draft.
EFFORT_MIN_RESERVE_J = 1500.0  # reserve drawn down by at least this much...
EFFORT_MIN_EXTRA_KJ = 3.0      # ...or this much more than the riders around you
EFFORT_MIN_SEC = 10
EFFORT_MERGE_SEC = 3
MOVE_ORIGIN_SEC = 300          # a "move" = riders who left your group within this long
EVERYONE_PAID_FRAC = 0.9       # riders around you spent >= 90% of what you did: not extra
FLIER_MAX_RIDERS = 3           # a group this small up the road is a "move", not the race
CHASE_LOOKAHEAD_SEC = 180      # how long to watch whether a move gets caught
ATTACK_POWER_FRAC_CP = 1.2     # front + this much over CP = attack, otherwise a pull
ATTACK_GAP_SEC = 0.5           # gap to the rider behind that counts as getting away
ON_FRONT_MIN_FRAC = 0.3        # share of the effort spent leading to count as "you led it"
EFFORT_MATTERED_J = 1000.0     # reserve it cost you at the next close call / drop
MAX_EXTRA_EFFORTS = 10

# "Hang on": after a drop, when did the group's pace (trailing 30 s, converted
# to the rider) fall back under the rider's CP, and could they have stayed
# with it until then?  Surges often end soon after the split.
HANG_ON_MAX_SEC = 300          # stop looking after this long
HANG_ON_SHORT_SEC = 60         # "eased just N s later" when within this
# Long, moderate leaks: spending more than the riders around you for minutes
# at a time, mostly below CP (barely touches the reserve, but it's energy).
SUSTAINED_WINDOW_SEC = 300     # judged on a rolling 5-minute average...
SUSTAINED_MIN_W = 15.0         # ...at least this many watts over the riders around you
SUSTAINED_MIN_SEC = 300        # for at least this long
SUSTAINED_MERGE_SEC = 60
# Window used when the pack wasn't going hard (likely positioning/reaction).
LOW_INTENSITY_WINDOW = (60, 15)

# "What it took": this quantile of survivors' rider-equivalent power is the
# best-positioned survivor; the median is a typical one.
REQUIRED_QUANTILE = 0.25

DEFAULT_CRR = 0.004
DEFAULT_W_PRIME_J = 20000.0
# Plausible W' range used when CP/W' is estimated from the race.
W_PRIME_RANGE_J = (10000.0, 30000.0)

# Durations used to estimate CP/W' from the race itself.
CP_ESTIMATE_DURATIONS = (180, 300, 720)


# ---------------------------------------------------------------------------
# Race grid
# ---------------------------------------------------------------------------

@dataclass
class RaceGrid:
    """All riders on a shared integer-second race clock, plus grouping."""
    times: np.ndarray          # (T,) race-clock seconds
    ranks: List[int]
    names: List[str]
    weight_kg: np.ndarray      # (N,)
    mass_kg: np.ndarray        # (N,) rider + bike
    cda: np.ndarray            # (N,) absolute CdA
    dist_m: np.ndarray         # (N, T) NaN when the rider isn't racing
    power: np.ndarray          # (N, T)
    speed: np.ndarray          # (N, T) m/s
    mass_power: np.ndarray     # (N, T) wheel watts into gravity+rolling+KE
    aero_power: np.ndarray     # (N, T) wheel watts actually spent on aero
    solo_aero: np.ndarray      # (N, T) aero watts if riding solo
    draft_pct: np.ndarray      # (N, T) smoothed % of solo aero power saved
    gap_ahead: np.ndarray      # (N, T) seconds behind the rider directly ahead
    group: np.ndarray          # (N, T) group label (0 = front), -1 absent
    group_size: np.ndarray     # (N, T)
    pos_in_group: np.ndarray   # (N, T) 0 = front of own group
    inv_dist: List[np.ndarray] # per rider: monotone distance for inversion
    inv_time: List[np.ndarray] # per rider: matching times
    gap_sec: float
    finish_m: float
    # Distance glitches fixed while building the grid: [{rank, start_k, end_k, jump_m}]
    repairs: List[Dict] = None

    def time_at(self, i: int, d):
        """Race-clock time rider ``i`` passed distance ``d`` (metres)."""
        dd, tt = self.inv_dist[i], self.inv_time[i]
        if len(dd) == 0:
            return np.full(np.shape(d), np.nan)
        return np.interp(d, dd, tt)

    def gap_to(self, i: int, j: int, k):
        """Seconds rider ``j`` is ahead of rider ``i`` at grid index ``k``."""
        return self.times[k] - self.time_at(j, self.dist_m[i, k])


def _runs(mask: np.ndarray, merge_gap: int = 0, min_len: int = 1) -> List[tuple]:
    """Inclusive (start, end) index pairs of True runs in ``mask``.

    Runs separated by at most ``merge_gap`` False samples are joined; runs
    shorter than ``min_len`` are dropped.
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > merge_gap + 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


def _rolling_mean(arr: np.ndarray, window: int, center: bool = True) -> np.ndarray:
    """Row-wise NaN-tolerant rolling mean for (N, T) or (T,) arrays."""
    if arr.ndim == 1:
        return pd.Series(arr).rolling(window, center=center, min_periods=1).mean().to_numpy()
    return pd.DataFrame(arr.T).rolling(window, center=center, min_periods=1).mean().to_numpy().T


def _repair_distance(d: np.ndarray, v: np.ndarray) -> List[Dict]:
    """Fix impossible jumps in one rider's distance trace, in place.

    Route alignment occasionally snaps a rider to the wrong stretch of road:
    distance leaps ahead (or back) far faster than they were riding, then
    freezes or creeps until it drifts back to reality — which can take
    minutes.  Speed is unaffected, and integrating it tracks the true
    distance to within metres over that span, so after each jump the trace is
    rebuilt from speed until the original agrees with it again.
    Returns what was repaired.
    """
    T = len(d)
    fin = np.isfinite(d) & np.isfinite(v)
    out = []
    k = 1
    while k < T:
        if not (fin[k] and fin[k - 1]) or abs(d[k] - d[k - 1] - v[k]) <= DIST_JUMP_TOL_M:
            k += 1
            continue
        a = k - 1                                   # last good point, just before the jump
        hi = min(T, a + 1 + DIST_REPAIR_MAX_SEC)
        integ = d[a] + np.cumsum(np.nan_to_num(v[a + 1:hi], nan=0.0))
        agrees = fin[a + 1:hi] & (np.abs(d[a + 1:hi] - integ) <= DIST_REJOIN_TOL_M)
        # Recovered once the original agrees with the rebuilt path for a while.
        b = None
        for c in range(len(agrees) - DIST_RECOVER_SEC):
            if agrees[c:c + DIST_RECOVER_SEC].all():
                b = a + 1 + c
                break
        if b is None:
            out.append({'start_k': int(k), 'end_k': None, 'jump_m': float(d[k] - d[k - 1])})
            k += 1
            continue
        ride = integ[:b - a]                            # positions a+1..b
        ride = ride + (d[b] - ride[-1]) * np.arange(1, b - a + 1) / (b - a)
        d[a + 1:b] = ride[:-1]
        out.append({'start_k': int(a + 1), 'end_k': int(b - 1), 'jump_m': float(d[k] - d[k - 1])})
        k = b + 1
    return out


def build_race_grid(race_data, bike_weight_kg: float, bike_cda_bias: float,
                    gap_sec: float = GROUP_GAP_SEC) -> RaceGrid:
    """Resample every rider onto one clock, then group riders each second."""
    t_min = int(np.floor(race_data.min_time))
    t_max = int(np.ceil(race_data.max_time))
    times = np.arange(t_min, t_max + 1)
    T = len(times)
    riders = race_data.riders
    N = len(riders)

    dist = np.full((N, T), np.nan)
    power = np.full((N, T), np.nan)
    speed = np.full((N, T), np.nan)
    alt = np.full((N, T), np.nan)
    crr = np.full((N, T), DEFAULT_CRR)
    weight = np.zeros(N)
    cda = np.zeros(N)

    for i, r in enumerate(riders):
        df = r.data
        k = np.round(np.asarray(df.index, dtype=float)).astype(int) - t_min
        ok = (k >= 0) & (k < T)
        k = k[ok]
        dist[i, k] = df['distance_km'].to_numpy(dtype=float)[ok] * 1000.0
        if 'power_watts' in df.columns:
            power[i, k] = df['power_watts'].to_numpy(dtype=float)[ok]
        if 'speed_kmh' in df.columns:
            speed[i, k] = df['speed_kmh'].to_numpy(dtype=float)[ok] / 3.6
        if 'altitude_m' in df.columns:
            alt[i, k] = df['altitude_m'].to_numpy(dtype=float)[ok]
        if 'crr' in df.columns:
            c = df['crr'].to_numpy(dtype=float)[ok]
            crr[i, k] = np.where(np.isfinite(c), c, DEFAULT_CRR)
        weight[i] = float(r.weight_kg or 75.0)
        height_m = (r.height_cm or 175.0) / 100.0
        cda[i] = rider_cda(height_m, weight[i]) + bike_cda_bias

    mass = weight + bike_weight_kg

    repairs = []
    for i in range(N):
        repairs += [dict(r, rank=int(riders[i].rank)) for r in _repair_distance(dist[i], speed[i])]

    # Inverse distance -> time lookup per rider (cumulative max keeps it monotone).
    inv_dist, inv_time = [], []
    for i in range(N):
        valid = np.isfinite(dist[i])
        inv_dist.append(np.maximum.accumulate(dist[i, valid]))
        inv_time.append(times[valid].astype(float))

    grid = RaceGrid(
        times=times, ranks=[int(r.rank) for r in riders], names=[r.name for r in riders],
        weight_kg=weight, mass_kg=mass, cda=cda, dist_m=dist, power=power, speed=speed,
        mass_power=None, aero_power=None, solo_aero=None, draft_pct=None,
        gap_ahead=np.full((N, T), np.nan), group=np.full((N, T), -1, dtype=int),
        group_size=np.zeros((N, T), dtype=int), pos_in_group=np.zeros((N, T), dtype=int),
        inv_dist=inv_dist, inv_time=inv_time, gap_sec=gap_sec,
        finish_m=float(race_data.finish_line_km) * 1000.0,
        repairs=repairs,
    )

    _compute_physics(grid, alt, crr)
    _compute_groups(grid)
    return grid


def _compute_physics(grid: RaceGrid, alt: np.ndarray, crr: np.ndarray) -> None:
    """Split each rider's wheel power into mass-proportional and aero parts.

    Energy balance at the wheel:  P(1-η) = d(KE+PE)/dt + F_roll·v + P_aero.
    Everything but P_aero scales with system mass; P_aero scales with CdA and
    already includes whatever draft the rider was getting.
    """
    m = grid.mass_kg[:, None]
    v = grid.speed
    energy = 0.5 * m * v ** 2 + m * GRAVITY * alt
    d_energy = np.gradient(energy, axis=1)
    d_energy = _rolling_mean(d_energy, 5)
    rolling = crr * m * GRAVITY * v
    grid.mass_power = d_energy + rolling
    grid.aero_power = grid.power * (1 - DRIVETRAIN_LOSS) - grid.mass_power
    grid.solo_aero = 0.5 * AIR_DENSITY * grid.cda[:, None] * v ** 3

    # Ratio of smoothed sums is far steadier than smoothing per-second ratios.
    aero_s = _rolling_mean(grid.aero_power, DRAFT_SMOOTH_SEC)
    solo_s = _rolling_mean(grid.solo_aero, DRAFT_SMOOTH_SEC)
    with np.errstate(invalid='ignore', divide='ignore'):
        grid.draft_pct = 100.0 * (1.0 - aero_s / np.where(solo_s > 1.0, solo_s, np.nan))


def _compute_groups(grid: RaceGrid) -> None:
    """Time-gap grouping of the whole field at every second."""
    dist, times = grid.dist_m, grid.times
    T = len(times)
    orders = []
    ahead_l, behind_l, k_l = [], [], []
    for k in range(T):
        col = dist[:, k]
        idx = np.flatnonzero(np.isfinite(col))
        o = idx[np.argsort(-col[idx], kind='stable')]
        orders.append(o)
        if len(o) > 1:
            ahead_l.append(o[:-1])
            behind_l.append(o[1:])
            k_l.append(np.full(len(o) - 1, k))
    if ahead_l:
        ahead = np.concatenate(ahead_l)
        behind = np.concatenate(behind_l)
        ks = np.concatenate(k_l)
        gaps = np.empty(len(ahead))
        # Batch the inverse lookups per leading rider.
        for j in np.unique(ahead):
            sel = ahead == j
            gaps[sel] = times[ks[sel]] - grid.time_at(j, dist[behind[sel], ks[sel]])
        grid.gap_ahead[behind, ks] = gaps

    for k, o in enumerate(orders):
        if len(o) == 0:
            continue
        g = np.nan_to_num(grid.gap_ahead[o[1:], k], nan=0.0)
        breaks = g > grid.gap_sec
        labels = np.concatenate([[0], np.cumsum(breaks)])
        sizes = np.bincount(labels)
        first = np.concatenate([[0], np.flatnonzero(breaks) + 1])
        grid.group[o, k] = labels
        grid.group_size[o, k] = sizes[labels]
        grid.pos_in_group[o, k] = np.arange(len(o)) - first[labels]


# ---------------------------------------------------------------------------
# Rider-level models
# ---------------------------------------------------------------------------

def equivalent_power(grid: RaceGrid, target: int, others) -> np.ndarray:
    """Power ``others`` produced, converted into ``target``'s body.

    Mass-proportional terms (gravity, rolling, acceleration) scale by system
    mass; the aero term scales by CdA.  Because the aero term is what the other
    rider actually spent, their draft/position carries over.
    """
    others = np.atleast_1d(others)
    m_ratio = grid.mass_kg[target] / grid.mass_kg[others]
    c_ratio = grid.cda[target] / grid.cda[others]
    wheel = grid.mass_power[others] * m_ratio[:, None] + grid.aero_power[others] * c_ratio[:, None]
    return wheel / (1 - DRIVETRAIN_LOSS)


def mean_max_power(power: np.ndarray, duration: int) -> Optional[float]:
    p = np.nan_to_num(power, nan=0.0)
    if len(p) < duration:
        return None
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(np.max(c[duration:] - c[:-duration]) / duration)


def estimate_cp(power: np.ndarray) -> Dict:
    """CP/W' from the race's own best efforts (work = CP·t + W').

    Same linear work-duration model as the Critical Power page.  A race isn't
    a maximal test, so this is a lower bound.
    """
    pts = [(d, mean_max_power(power, d)) for d in CP_ESTIMATE_DURATIONS]
    pts = [(d, p) for d, p in pts if p]
    if len(pts) >= 2:
        t = np.array([d for d, _ in pts], dtype=float)
        work = np.array([d * p for d, p in pts])
        cp, w_prime = np.polyfit(t, work, 1)
        if cp > 0 and w_prime > 0:
            return {'cp': float(cp), 'w_prime': float(w_prime), 'source': 'race_estimate'}
    fallback = mean_max_power(power, 1200) or float(np.nanmean(power))
    return {'cp': float(fallback), 'w_prime': DEFAULT_W_PRIME_J, 'source': 'race_fallback'}


def fit_cp_to_race(power: np.ndarray, cp: float, w_prime: float):
    """Make a race-estimated CP/W' consistent with what the rider actually did.

    Best efforts in a race underestimate CP (riders rarely go flat out for
    12 min), which makes W' balance plunge below zero.  Keep W' in a
    physiological range and raise CP until the race never empties W' — the
    rider demonstrably sustained that.
    """
    w_prime = float(np.clip(w_prime, *W_PRIME_RANGE_J))
    if np.min(w_prime_balance(power, cp, w_prime)) >= 0:
        return cp, w_prime
    lo, hi = cp, cp * 1.6
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if np.min(w_prime_balance(power, mid, w_prime)) >= 0:
            hi = mid
        else:
            lo = mid
    return hi, w_prime


def w_prime_balance(power: np.ndarray, cp: float, w_prime: float,
                    start: Optional[float] = None) -> np.ndarray:
    """Skiba differential W' balance (J), 1 s steps; NaN power counts as 0 W."""
    p = np.nan_to_num(power, nan=0.0)
    out = np.empty(len(p))
    w = w_prime if start is None else start
    for i, pi in enumerate(p):
        if pi > cp:
            w -= pi - cp
        else:
            w += (cp - pi) * (w_prime - w) / w_prime
        out[i] = w
    return out


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def _majority_group(grid: RaceGrid, members: np.ndarray, k: int) -> Optional[int]:
    labels = grid.group[members, k]
    labels = labels[labels >= 0]
    if len(labels) == 0:
        return None
    return int(np.bincount(labels).argmax())


def _group_tail(grid: RaceGrid, label: int, k: int) -> Optional[int]:
    members = np.flatnonzero(grid.group[:, k] == label)
    if len(members) == 0:
        return None
    return int(members[np.argmin(grid.dist_m[members, k])])


def _split_matters(n_ahead: int, group_size: int) -> bool:
    """A split counts as losing contact once the part that went ahead is a real group.

    A few riders riding off the front is a breakaway (ignored); but if a sizeable
    share of the group goes ahead — even if it's not the bigger part — being left
    in the group behind means you missed the split.
    """
    return n_ahead > FLIER_MAX_RIDERS and n_ahead >= SPLIT_MIN_FRAC * group_size


def detect_separations(grid: RaceGrid, ri: int) -> List[Dict]:
    """Moments the rider's group split with the bigger part ahead of them."""
    present = np.flatnonzero(np.isfinite(grid.dist_m[ri]))
    if len(present) < 2:
        return []
    k, last_k = int(present[0]) + 1, int(present[-1])
    events = []
    while k <= last_k:
        if grid.group[ri, k] < 0 or grid.group[ri, k - 1] < 0:
            k += 1
            continue
        prev = np.flatnonzero(grid.group[:, k - 1] == grid.group[ri, k - 1])
        prev = prev[prev != ri]
        if len(prev) + 1 < MIN_GROUP_SIZE:
            k += 1
            continue
        cur_label = grid.group[ri, k]
        stay = prev[grid.group[prev, k] == cur_label]
        left = prev[(grid.group[prev, k] != cur_label) & np.isfinite(grid.dist_m[prev, k])]
        ahead = left[grid.dist_m[left, k] > grid.dist_m[ri, k]]
        # A real split, not a breakaway: enough riders went ahead to matter.
        if len(ahead) == 0 or not _split_matters(len(ahead), len(prev) + 1):
            k += 1
            continue

        # Follow the group that went ahead until the rider regains it or the
        # confirmation window runs out.
        k_sep = k
        rejoin_k = None
        max_gap = 0.0
        end_k = min(last_k, k_sep + DROP_CONFIRM_SEC)
        kk = k_sep
        for kk in range(k_sep, end_k + 1):
            alive = ahead[np.isfinite(grid.dist_m[ahead, kk])]
            if len(alive) == 0:
                break
            pg = _majority_group(grid, alive, kk)
            if pg is None:
                break
            if grid.group[ri, kk] == pg:
                rejoin_k = kk
                break
            tail = _group_tail(grid, pg, kk)
            if tail is not None:
                gap = float(grid.gap_to(ri, tail, kk))
                if np.isfinite(gap):
                    max_gap = max(max_gap, gap)
        out_k = rejoin_k if rejoin_k is not None else kk

        if rejoin_k is not None and rejoin_k - k_sep < MIN_SEPARATION_SEC:
            k = rejoin_k + 1
            continue

        if grid.dist_m[ri, k_sep] > grid.finish_m - FINISH_EXCLUDE_M:
            kind = 'finish'
        elif rejoin_k is not None:
            kind = 'near_miss'
        else:
            kind = 'drop'

        events.append({
            'kind': kind,
            'k_sep': k_sep,
            'k_end': out_k,
            'rejoin_k': rejoin_k,
            'prev_group': prev,
            'went_ahead': ahead,
            'max_gap_sec': max_gap,
            'group_size_before': len(prev) + 1,
            'pos_before': int(grid.pos_in_group[ri, k_sep - 1]),
        })
        k = out_k + 1
    return events


# ---------------------------------------------------------------------------
# Event analysis
# ---------------------------------------------------------------------------

def _quiet(fn, *args, **kwargs):
    """Call a nan-reduction without all-NaN-slice warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return fn(*args, **kwargs)


def _nanmean(a) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    if a.size == 0 or not np.isfinite(a).any():
        return None
    return float(np.nanmean(a))


def _selection_window(pack_req: np.ndarray, k_sep: int, cp: float, T: int):
    """Stretch where the pack was riding above the rider's CP.

    Anchored on the pack's hardest 30 s in the lookback before the split, since
    the damage is often done before the gap actually opens.
    """
    roll = _rolling_mean(pack_req, 30)
    lo_s = max(0, k_sep - SELECTION_LOOKBACK_SEC)
    seg = roll[lo_s:k_sep + 1]
    if not np.isfinite(seg).any() or np.nanmax(seg) < cp:
        before, after = LOW_INTENSITY_WINDOW
        return max(0, k_sep - before), min(T - 1, k_sep + after), False
    k_peak = lo_s + int(np.nanargmax(seg))
    thresh = 0.95 * cp
    ws = k_peak
    lo = max(0, k_sep - SELECTION_MAX_BEFORE_SEC)
    while ws - 1 >= lo and np.isfinite(roll[ws - 1]) and roll[ws - 1] >= thresh:
        ws -= 1
    we = k_peak
    hi = min(T - 1, k_sep + SELECTION_MAX_AFTER_SEC)
    while we + 1 <= hi and np.isfinite(roll[we + 1]) and roll[we + 1] >= thresh:
        we += 1
    return ws, max(we, k_sep), True


def detect_exposures(grid: RaceGrid, ri: int, mates_draft: np.ndarray,
                     nearby_eq: np.ndarray) -> List[Dict]:
    """Episodes where the rider was in a group but getting far less draft.

    ``mates_draft`` is the median draft % of the rider's group-mates each
    second; ``nearby_eq`` the mean power of the nearest group-mates,
    converted to the rider.
    """
    in_group = grid.group_size[ri] >= MIN_GROUP_SIZE
    deficit = mates_draft - grid.draft_pct[ri]
    flag = in_group & np.isfinite(deficit) & (deficit >= DRAFT_LOSS_PCT)
    out = []
    for a, b in _runs(flag, DRAFT_LOSS_MERGE_SEC, DRAFT_LOSS_MIN_SEC):
        sl = slice(a, b + 1)
        sizes = grid.group_size[ri, sl]
        pos = grid.pos_in_group[ri, sl]
        rel = pos / np.maximum(sizes - 1, 1)
        if np.mean(pos == 0) > 0.5:
            where = 'front'
        elif np.mean(rel) > 2 / 3:
            where = 'back'
        else:
            where = 'middle'
        extra = grid.power[ri, sl] - nearby_eq[sl]
        out.append({
            'k_start': a, 'k_end': b,
            'start_sec': float(grid.times[a]), 'end_sec': float(grid.times[b]),
            'duration_sec': b - a + 1,
            'distance_km': float(grid.dist_m[ri, a] / 1000.0),
            'where': where,
            'pos': int(round(np.median(pos))) + 1,
            'group_size': int(round(np.median(sizes))),
            'gap_ahead_sec': _nanmean(grid.gap_ahead[ri, sl]),
            'draft_pct': _nanmean(grid.draft_pct[ri, sl]),
            'mates_draft_pct': _nanmean(mates_draft[sl]),
            'power_w': _nanmean(grid.power[ri, sl]),
            'nearby_w': _nanmean(nearby_eq[sl]),
            'extra_kj': float(np.nansum(extra) / 1000.0),
        })
    return out


def _detect_sat_up(grid: RaceGrid, ri: int, ev: Dict, pack_req: np.ndarray,
                   wbal: np.ndarray, cp: float) -> Optional[Dict]:
    """First stretch near the split where the rider eased right off."""
    T = len(grid.times)
    k_sep = ev['k_sep']
    lo, hi = max(0, k_sep - SAT_UP_SEARCH[0]), min(T - 1, k_sep + SAT_UP_SEARCH[1])
    p3 = _rolling_mean(grid.power[ri], 3)
    req3 = _rolling_mean(pack_req, 3)
    mask = np.zeros(T, dtype=bool)
    with np.errstate(invalid='ignore'):
        mask[lo:hi + 1] = ((p3[lo:hi + 1] < SAT_UP_FRAC * req3[lo:hi + 1])
                           & (req3[lo:hi + 1] > SAT_UP_MIN_PACK_FRAC_CP * cp))
    runs = _runs(mask, 1, SAT_UP_MIN_SEC)
    if not runs:
        return None
    a, b = runs[0]
    # Gap to the back of the group that went ahead when the rider eased off.
    gap = None
    alive = ev['went_ahead'][np.isfinite(grid.dist_m[ev['went_ahead'], a])]
    pg = _majority_group(grid, alive, a) if len(alive) else None
    if pg is not None and grid.group[ri, a] != pg:
        tail = _group_tail(grid, pg, a)
        if tail is not None:
            gap = float(grid.gap_to(ri, tail, a))
    return {
        'start_sec': float(grid.times[a]),
        'duration_sec': b - a + 1,
        'after_split_sec': int(a - k_sep),
        'power_w': _nanmean(grid.power[ri, a:b + 1]),
        'group_w': _nanmean(pack_req[a:b + 1]),
        'w_prime_j': float(wbal[a]),
        'gap_sec': gap,
    }


def _avg_gradient(grid: RaceGrid, ri: int, ws: int, we: int, elevation) -> Optional[float]:
    d0, d1 = grid.dist_m[ri, ws], grid.dist_m[ri, we]
    if not (np.isfinite(d0) and np.isfinite(d1)) or d1 - d0 < 50:
        return None
    ed = elevation['distance_km'].to_numpy() * 1000.0
    ea = elevation['altitude_m'].to_numpy()
    return float((np.interp(d1, ed, ea) - np.interp(d0, ed, ea)) / (d1 - d0) * 100.0)


# ---------------------------------------------------------------------------
# Gap-closing simulation
# ---------------------------------------------------------------------------

def _draft_at(gap_m: float, d0: float) -> float:
    """Draft saving (%) at ``gap_m`` behind the wheel, given full draft ``d0``."""
    if gap_m <= DRAFT_FULL_M:
        return d0
    if gap_m >= DRAFT_ZERO_M:
        return 0.0
    return d0 * (1.0 - (gap_m - DRAFT_FULL_M) / (DRAFT_ZERO_M - DRAFT_FULL_M))


def simulate_chase(grid: RaceGrid, ri: int, tail: int, k0: int, n_sec: int, power,
                   d0: float, grade_at, fixed_draft: Optional[float] = None) -> np.ndarray:
    """Gap (m) each second as the rider chases ``tail``'s recorded trajectory.

    ``power`` is a constant (W) or a per-second array.  The rider starts from
    their real position and speed; draft fades with distance to the tail,
    unless ``fixed_draft`` (%) is given — e.g. moving up inside the bunch.
    """
    m, cda = grid.mass_kg[ri], grid.cda[ri]
    x, v = grid.dist_m[ri, k0], grid.speed[ri, k0]
    steps = int(round(1 / SIM_DT))
    gaps = np.empty(n_sec)
    for sec in range(n_sec):
        p = power[sec] if np.ndim(power) else power
        p = 0.0 if not np.isfinite(p) else p
        xt0, xt1 = grid.dist_m[tail, k0 + sec], grid.dist_m[tail, k0 + sec + 1]
        for sub in range(steps):
            gap = xt0 + (xt1 - xt0) * sub / steps - x
            f = (p * (1 - DRIVETRAIN_LOSS) / max(v, 1.0)
                 - m * GRAVITY * (grade_at(x) + DEFAULT_CRR)
                 - 0.5 * AIR_DENSITY * cda * v * v
                 * (1 - (fixed_draft if fixed_draft is not None else _draft_at(gap, d0)) / 100.0))
            v = max(v + f / m * SIM_DT, 0.5)
            x += v * SIM_DT
        gaps[sec] = xt1 - x
    return gaps


def _min_power(ok) -> Optional[float]:
    """Smallest constant power (5 W steps) for which ``ok(power)`` holds."""
    if not ok(MAX_SIM_POWER_W):
        return None
    lo, hi = 0.0, MAX_SIM_POWER_W
    while hi - lo > 5.0:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return float(np.ceil(hi / 5.0) * 5.0)


def _gap_closing_costs(grid: RaceGrid, ri: int, ev: Dict, wbal: np.ndarray, cp: float,
                       w_prime: float, elevation, pack_req: np.ndarray,
                       k_follow: int) -> Optional[Dict]:
    """What it would have cost to hold the wheel, or to close the gap later.

    Chases the real back of the group that went ahead, with draft fading by
    distance.  The model is first checked by replaying the rider's actual
    power and comparing the simulated gap with the real one.

    Getting back on only counts if you can then stay on: every option is
    costed through to ``k_follow`` (when the group eased), riding at the
    group's pace converted to the rider once back on the wheel.
    """
    k_sep = ev['k_sep']
    alive = ev['went_ahead'][np.isfinite(grid.dist_m[ev['went_ahead'], k_sep])]
    pg = _majority_group(grid, alive, k_sep) if len(alive) else None
    if pg is None:
        return None
    members = np.flatnonzero(grid.group[:, k_sep] == pg)
    tail = int(members[np.argmin(grid.dist_m[members, k_sep])])

    # How long the tail's trajectory is available to chase.
    tail_ok = np.isfinite(grid.dist_m[tail]) & np.isfinite(grid.dist_m[ri])
    lo = max(0, k_sep - CLOSE_LOOKBACK_SEC)
    gap_m = grid.dist_m[tail] - grid.dist_m[ri]
    back_on = np.flatnonzero(tail_ok[lo:k_sep + 1] & (gap_m[lo:k_sep + 1] <= CLOSE_TARGET_M))
    k_on = lo + int(back_on[-1]) if len(back_on) else lo

    def avail(k0, n):
        """Seconds of simulation available from ``k0`` (capped at ``n``)."""
        run = 0
        while run < n and k0 + run + 1 < len(grid.times) and tail_ok[k0 + run + 1]:
            run += 1
        return run

    ed = elevation['distance_km'].to_numpy() * 1000.0
    ea = elevation['altitude_m'].to_numpy()

    def grade_at(x):
        return (np.interp(x + 10, ed, ea) - np.interp(x - 10, ed, ea)) / 20.0

    pre = grid.draft_pct[ri, max(0, k_on - 60):k_on + 1]
    d0 = float(np.clip(np.nanmedian(pre), 15, 50)) if np.isfinite(pre).any() else 30.0

    # Model check: replay actual power, compare simulated and real gap.
    n_chk = avail(k_on, 30)
    model_error = None
    if n_chk >= 10:
        sim = simulate_chase(grid, ri, tail, k_on, n_chk, grid.power[ri, k_on:k_on + n_chk], d0, grade_at)
        model_error = float(abs(sim[-1] - gap_m[k_on + n_chk]))

    def ftime(k):
        return float(grid.times[k])

    def through(k0, n_first, p):
        """(lowest W', total cost): ``p`` for ``n_first`` s, then the group's pace to ``k_follow``."""
        rest = pack_req[k0 + n_first:k_follow] if k_follow > k0 + n_first else np.array([])
        rest = np.where(np.isfinite(rest), rest, cp)
        path = w_prime_balance(np.concatenate([np.full(n_first, p), rest]), cp, w_prime, start=wbal[k0])
        return float(path.min()), max(0.0, float(wbal[k0] - path.min()))

    # Hold the wheel from the onset until a little past the split.
    hold = None
    n_hold = avail(k_on, k_sep - k_on + HOLD_EXTRA_SEC)
    if n_hold >= 5:
        p = _min_power(lambda P: simulate_chase(grid, ri, tail, k_on, n_hold, P, d0, grade_at).max()
                       <= DRAFT_FULL_M)
        if p is not None:
            low, cost = through(k_on, n_hold, p)
            hold = {'start_sec': ftime(k_on), 'duration_sec': n_hold, 'power_w': p,
                    'w_prime_cost_j': cost, 'w_prime_low_j': low, 'w_prime_left_j': float(wbal[k_on]),
                    'affordable': bool(low >= 0),
                    'actual_w': _nanmean(grid.power[ri, k_on:k_on + n_hold])}

    # React later: rows from when the gap first opens past the target, to the split.
    opened = np.flatnonzero(gap_m[k_on:k_sep + 1] > CLOSE_TARGET_M + 1)
    rows = []
    if len(opened):
        ks = list(range(k_on + int(opened[0]), k_sep + 1, CLOSE_STEP_SEC))
        if ks[-1] != k_sep:
            ks.append(k_sep)
        if len(ks) > CLOSE_MAX_ROWS:
            ks = [ks[int(round(i))] for i in np.linspace(0, len(ks) - 1, CLOSE_MAX_ROWS)]
        for k_r in ks:
            row = {'react_sec': ftime(k_r), 'gap_m': float(gap_m[k_r]),
                   'w_prime_left_j': float(wbal[k_r])}
            for limit in CLOSE_LIMITS_SEC:
                n = avail(k_r, limit)
                cell = None
                if n >= limit:
                    def closes(P, k_r=k_r, n=n):
                        return simulate_chase(grid, ri, tail, k_r, n, P, d0, grade_at).min() <= CLOSE_TARGET_M
                    p = _min_power(closes)
                    if p is not None:
                        g = simulate_chase(grid, ri, tail, k_r, n, p, d0, grade_at)
                        t_hit = int(np.argmax(g <= CLOSE_TARGET_M)) + 1
                        low, cost = through(k_r, t_hit, p)
                        cell = {'power_w': p, 'time_sec': t_hit, 'w_prime_cost_j': cost,
                                'w_prime_low_j': low, 'affordable': bool(low >= 0)}
                row[f'close_{limit}'] = cell
            rows.append(row)

    return {
        'tail_name': grid.names[tail],
        'onset_sec': ftime(k_on),
        'full_draft_pct': d0,
        'model_error_m': model_error,
        'hold': hold,
        'rows': rows,
        # Set when the costs include staying on after the split until the group eased.
        'follow_until_sec': ftime(k_follow) if k_follow > k_sep + HOLD_EXTRA_SEC else None,
    }


def _position_earlier(grid: RaceGrid, ri: int, survivors: np.ndarray, surv_eq: np.ndarray,
                      ws: int, we: int, wbal: np.ndarray, cp: float, w_prime: float,
                      elevation) -> List[Dict]:
    """What if the rider had spent a little to move up before the selection?

    For each move duration and target slot: simulate moving up to the survivor
    nearest that slot (keeping most of the draft, as riders moving up in a
    bunch do), then ride the selection on the power of the survivors around
    that slot, converted to the rider.  Using several survivors keeps one
    rider's quirks from deciding the answer.  W' is tracked throughout.
    """
    ed = elevation['distance_km'].to_numpy() * 1000.0
    ea = elevation['altitude_m'].to_numpy()

    def grade_at(x):
        return (np.interp(x + 10, ed, ea) - np.interp(x - 10, ed, ea)) / 20.0

    rows = []
    for move_sec in MOVE_UP_SEC:
        k_ms = ws - move_sec
        if k_ms < 0 or not np.isfinite(grid.dist_m[ri, k_ms]):
            continue
        size = grid.group_size[ri, k_ms]
        if size < MOVE_UP_MIN_GROUP:
            continue
        members = np.flatnonzero(grid.group[:, k_ms] == grid.group[ri, k_ms])
        order = list(members[np.argsort(-grid.dist_m[members, k_ms])])   # front first
        rider_pos = order.index(ri)
        surv_slots = [(order.index(j), si) for si, j in enumerate(survivors) if j in order]
        if not surv_slots:
            continue
        pre = grid.draft_pct[ri, max(0, k_ms - 60):k_ms + 1]
        d0 = float(np.clip(np.nanmedian(pre), 15, 50)) if np.isfinite(pre).any() else 30.0
        n_avail = min(move_sec, len(grid.times) - k_ms - 2)

        for frac in MOVE_UP_TARGETS:
            want = int(round(frac * (size - 1)))
            target_pos, si = min(surv_slots, key=lambda ps: abs(ps[0] - want))
            if rider_pos - target_pos < MOVE_UP_MIN_PLACES:
                continue
            target = int(survivors[si])
            if grid.dist_m[target, k_ms] - grid.dist_m[ri, k_ms] < MOVE_UP_MIN_GAIN_M:
                continue
            # The rider actually made this move: nothing counterfactual about it.
            if np.min(grid.pos_in_group[ri, k_ms:ws + 1]) <= target_pos:
                continue

            def sim(P, n=n_avail, target=target, d0=d0):
                return simulate_chase(grid, ri, target, k_ms, n, P, d0, grade_at,
                                      fixed_draft=MOVE_UP_DRAFT_FRAC * d0)

            # Model check: with the rider's actual power (in full draft, as
            # they were), does the simulated gap to the target match reality?
            real_gap = grid.dist_m[target, k_ms + n_avail] - grid.dist_m[ri, k_ms + n_avail]
            replay = simulate_chase(grid, ri, target, k_ms, n_avail,
                                    grid.power[ri, k_ms:k_ms + n_avail], d0, grade_at, fixed_draft=d0)
            model_error = float(abs(replay[-1] - real_gap))

            p_move = _min_power(lambda P: sim(P).min() <= 1.0)
            if p_move is None:
                continue
            t_hit = int(np.argmax(sim(p_move) <= 1.0)) + 1

            # Ride the selection like each survivor within a few places of the slot.
            slot_rows = [s_i for pos, s_i in surv_slots if abs(pos - target_pos) <= MOVE_UP_SLOT_SPAN]
            mins, at_sel, sel_w = [], [], []
            for s_i in slot_rows:
                follow = surv_eq[s_i, k_ms:we + 1].copy()
                if not np.isfinite(follow).any():
                    continue
                follow = np.where(np.isfinite(follow), follow, np.nanmean(follow))
                plan = follow.copy()
                plan[:t_hit] = p_move
                path = w_prime_balance(plan, cp, w_prime, start=wbal[k_ms])
                mins.append(path.min())
                at_sel.append(path[move_sec])
                sel_w.append(np.mean(follow[move_sec:]))
            if not mins:
                continue
            actual = grid.power[ri, k_ms:k_ms + move_sec]
            move_plan = np.full(move_sec, np.median(sel_w))   # placeholder after the move
            move_plan[:t_hit] = p_move
            share = float(np.mean(np.array(mins) >= 0))
            rows.append({
                'move_start_sec': float(grid.times[k_ms]),
                'move_window_sec': move_sec,
                'move_time_sec': t_hit,
                'move_power_w': p_move,
                'from_pos': rider_pos + 1,
                'to_pos': target_pos + 1,
                'group_size': int(size),
                'slot_riders': len(mins),
                'extra_kj': float(np.nansum(p_move - actual[:t_hit]) / 1000.0),
                'w_prime_at_start_j': float(wbal[k_ms]),
                'w_prime_at_selection_j': float(np.median(at_sel)),
                'selection_w': float(np.median(sel_w)),
                'min_w_prime_j': float(np.median(mins)),
                'feasible_share': share,
                'feasible': bool(np.median(mins) >= 0),
                'model_error_m': model_error,
            })
    return rows


def _hold_position(grid: RaceGrid, ri: int, survivors: np.ndarray, surv_eq: np.ndarray,
                   drift: Dict, ws: int, we: int, k_sep: int, wbal: np.ndarray, cp: float,
                   w_prime: float) -> Optional[Dict]:
    """What if the rider had held their spot instead of sliding back?

    Purely empirical: survivors who started within a few places of the rider
    rode from there through the selection, so their power converted to the
    rider (draft at that spot included) is what holding it cost.
    """
    k_a = int(drift['from_sec'] - grid.times[0])
    members = np.flatnonzero(grid.group[:, k_a] == grid.group[ri, k_a])
    order = list(members[np.argsort(-grid.dist_m[members, k_a])])   # front first
    pos = order.index(ri)
    near = [si for si, j in enumerate(survivors)
            if j in order and abs(order.index(j) - pos) <= MOVE_UP_SLOT_SPAN]
    if not near:
        return None
    # Selection effort from the selection start (or the slide start if later);
    # extra spend only up to the split — after it the rider was already off.
    split = max(ws, k_a) - k_a
    n_cmp = k_sep - k_a
    actual = grid.power[ri, k_a:k_sep]
    mins, extra, sel_w = [], [], []
    for si in near:
        plan = surv_eq[si, k_a:we + 1].copy()
        if np.isfinite(plan).mean() < 0.8:
            continue
        plan = np.where(np.isfinite(plan), plan, np.nanmean(plan))
        path = w_prime_balance(plan, cp, w_prime, start=wbal[k_a])
        mins.append(path.min())
        extra.append(np.nansum(plan[:n_cmp] - actual) / 1000.0)
        sel_w.append(np.mean(plan[split:]))
    if not mins:
        return None
    return {
        'from_sec': drift['from_sec'],
        'pos': pos + 1,
        'group_size': len(order),
        'riders': len(mins),
        'extra_kj': float(np.median(extra)),
        'selection_w': float(np.median(sel_w)),
        'w_prime_at_start_j': float(wbal[k_a]),
        'min_w_prime_j': float(np.median(mins)),
        'feasible_share': float(np.mean(np.array(mins) >= 0)),
        'feasible': bool(np.median(mins) >= 0),
    }


def _position_drift(grid: RaceGrid, ri: int, k_a: int, k_b: int) -> Optional[Dict]:
    """Places lost between the rider's best spot in [k_a, k_b] and k_b.

    Positions jitter in a bunch, so they're smoothed first; the slide is
    measured from the last time the rider held their best (smoothed) spot.
    """
    k_a = max(k_a, 0)
    if grid.group[ri, k_b] < 0:
        return None
    sizes = grid.group_size[ri, k_a:k_b + 1].astype(float)
    rel = np.where(sizes >= MOVE_UP_MIN_GROUP,
                   grid.pos_in_group[ri, k_a:k_b + 1] / np.maximum(sizes - 1, 1), np.nan)
    if not np.isfinite(rel).any():
        return None
    rel = pd.Series(rel).rolling(POSITION_SMOOTH_SEC, center=True, min_periods=1).median().to_numpy()
    best_i = int(np.flatnonzero(rel <= np.nanmin(rel) + 1e-9)[-1])
    k_best = k_a + best_i
    a, b = int(grid.pos_in_group[ri, k_best]) + 1, int(grid.pos_in_group[ri, k_b]) + 1
    if b - a < POSITION_DRIFT_MIN_PLACES:
        return None
    out = {'from_pos': a, 'from_size': int(grid.group_size[ri, k_best]),
           'to_pos': b, 'to_size': int(grid.group_size[ri, k_b]),
           'from_sec': float(grid.times[k_best]), 'to_sec': float(grid.times[k_b]),
           'moved_up_from': None}
    # Had the rider just paid to move up to that spot?  (Then gave it back.)
    before = rel[:best_i + 1]
    if np.isfinite(before).any():
        worst_i = int(np.flatnonzero(before >= np.nanmax(before) - 1e-9)[-1])
        k_w = k_a + worst_i
        w_pos = int(grid.pos_in_group[ri, k_w]) + 1
        if w_pos - a >= POSITION_DRIFT_MIN_PLACES:
            out['moved_up_from'] = {'pos': w_pos, 'size': int(grid.group_size[ri, k_w]),
                                    'sec': float(grid.times[k_w])}
    return out


def _group_ahead(grid: RaceGrid, ri: int, k: int):
    """Members and gap (s) of the group just ahead of the rider's group, if any."""
    my = grid.group[ri, k]
    if my <= 0:
        return None, None
    ahead = np.flatnonzero(grid.group[:, k] == my - 1)
    mine = np.flatnonzero(grid.group[:, k] == my)
    tail = ahead[np.argmin(grid.dist_m[ahead, k])]
    lead = mine[np.argmax(grid.dist_m[mine, k])]
    return ahead, float(grid.gap_to(lead, tail, k))


def _gap_behind(grid: RaceGrid, ri: int, k: int, among: np.ndarray) -> Optional[float]:
    """Seconds between the rider and the nearest of ``among`` behind them."""
    d = grid.dist_m[:, k]
    behind = among[np.isfinite(d[among]) & (d[among] < d[ri])]
    if len(behind) == 0:
        return None
    j = behind[np.argmax(d[behind])]
    return float(grid.gap_to(j, ri, k))


def detect_extra_efforts(grid: RaceGrid, ri: int, cp: float, w_prime: float, wbal: np.ndarray,
                         nearby_eq: np.ndarray, exposures: List[Dict],
                         events: List[Dict]) -> List[Dict]:
    """Efforts where the rider spent more than the riders around them, and why.

    Candidates are stretches above CP that drain the reserve, plus draft-loss
    episodes.  Each is labelled by context (chasing a move, attacking, pulling,
    riding out of the draft), checked for what it bought (was the move caught?
    did the attack get away?), and replayed without the extra to see whether
    it cost the rider at their next close call or drop.
    """
    T = len(grid.times)
    p5 = _rolling_mean(grid.power[ri], 5)
    with np.errstate(invalid='ignore'):
        above = np.isfinite(p5) & (p5 > cp)
    # Two sources, kept separate so a long hard stretch doesn't swallow a
    # short, specific effort inside it; overlaps are resolved after scoring.
    merged = _runs(above, EFFORT_MERGE_SEC, EFFORT_MIN_SEC)
    merged += [(e['k_start'], e['k_end']) for e in exposures]

    later_events = [(int(e['time_sec'] - grid.times[0]), e) for e in events]
    out = []
    for a, b in merged:
        sl = slice(a, b + 1)
        sizes = grid.group_size[ri, sl]
        if np.median(sizes) < MIN_GROUP_SIZE:
            continue          # alone or in a tiny group: no one to share with
        power = _nanmean(grid.power[ri, sl])
        near = _nanmean(nearby_eq[sl])
        if power is None or near is None:
            continue
        if near >= EVERYONE_PAID_FRAC * power:
            continue          # the group surged and you paid what everyone paid
        extra_kj = float(np.nansum(grid.power[ri, sl] - nearby_eq[sl]) / 1000.0)
        cost = float(wbal[a] - np.min(wbal[sl]))
        if cost < EFFORT_MIN_RESERVE_J and extra_kj < EFFORT_MIN_EXTRA_KJ:
            continue
        on_front = float(np.mean(grid.pos_in_group[ri, sl] == 0))
        # 1st or 2nd wheel: at the front of the group, sharing the wind.
        front2 = float(np.mean(grid.pos_in_group[ri, sl] <= 1))
        size = int(np.median(sizes))

        # Context: a small move up the road?
        flier, gap0 = _group_ahead(grid, ri, a)
        if flier is None or len(flier) > FLIER_MAX_RIDERS:
            mid_flier, mid_gap = _group_ahead(grid, ri, (a + b) // 2)
            if mid_flier is not None and len(mid_flier) <= FLIER_MAX_RIDERS:
                flier, gap0 = mid_flier, mid_gap
        is_move = flier is not None and len(flier) <= FLIER_MAX_RIDERS and size >= 2 * len(flier)
        if is_move:
            # Only a move if they rode away from the rider's own group recently.
            lo = max(0, a - MOVE_ORIGIN_SEC)
            was_with = [np.mean(grid.group[flier, k] == grid.group[ri, k]) > 0.5 for k in range(lo, a + 1)]
            is_move = any(was_with)

        caught_after, gap_max = None, None
        if is_move and front2 >= ON_FRONT_MIN_FRAC:
            kind = 'chase'
            for k in range(a, min(T, b + CHASE_LOOKAHEAD_SEC)):
                alive = flier[np.isfinite(grid.dist_m[flier, k])]
                if len(alive) and np.mean(grid.group[alive, k] == grid.group[ri, k]) > 0.5:
                    caught_after = int(k - b)
                    break
        elif on_front >= 0.5 and power >= ATTACK_POWER_FRAC_CP * cp:
            kind = 'attack'
            mates0 = np.flatnonzero(grid.group[:, a] == grid.group[ri, a])
            mates0 = mates0[mates0 != ri]
            gaps = [g for g in (_gap_behind(grid, ri, k, mates0) for k in range(a, b + 1)) if g is not None]
            gap_max = max(gaps) if gaps else 0.0
        elif front2 >= 0.5:
            kind = 'pull'
        else:
            kind = 'out_of_draft'

        # Did it cost you at the next close call or drop?
        consequence = None
        nxt = next(((k, e) for k, e in later_events if k > b), None)
        if nxt is not None:
            k_n, e_n = nxt
            cf = grid.power[ri].copy()
            mates = nearby_eq[sl]
            cf[sl] = np.where(np.isfinite(mates), np.minimum(grid.power[ri, sl], mates), cf[sl])
            cf_bal = w_prime_balance(cf[: k_n + 1], cp, w_prime)
            consequence = {'event_kind': e_n['kind'], 'event_sec': e_n['time_sec'],
                           'reserve_lost_j': float(cf_bal[k_n] - wbal[k_n])}

        out.append({
            'kind': kind,
            'start_sec': float(grid.times[a]), 'end_sec': float(grid.times[b]),
            'duration_sec': int(b - a + 1),
            'distance_km': float(grid.dist_m[ri, a] / 1000.0),
            'power_w': power, 'nearby_w': near,
            'extra_kj': extra_kj, 'reserve_cost_j': cost,
            'on_front_pct': on_front * 100.0,
            'front_pct': front2 * 100.0,
            'pos': int(np.median(grid.pos_in_group[ri, sl])) + 1, 'group_size': size,
            'move_riders': int(len(flier)) if is_move else None,
            'move_gap_sec': gap0 if is_move else None,
            'caught_after_sec': caught_after,
            'attack_gap_sec': gap_max,
            'consequence': consequence,
            'surge': bool(kind == 'out_of_draft' and power >= ATTACK_POWER_FRAC_CP * cp),
        })
    # Overlapping candidates: keep the costlier one.  Then the costliest
    # efforts, shown in race order.
    def weight(e):
        return max(e['reserve_cost_j'], e['extra_kj'] * 1000)
    kept = []
    for e in sorted(out, key=weight, reverse=True):
        if all(e['end_sec'] < k['start_sec'] or e['start_sec'] > k['end_sec'] for k in kept):
            kept.append(e)
    out = sorted(kept[:MAX_EXTRA_EFFORTS], key=lambda e: e['start_sec'])
    for e in out:
        e.update(_effort_text(e, w_prime, grid.weight_kg[ri], float(grid.times[0])))
    return out


def _terrain(grid: RaceGrid, ri: int, a: int, b: int, elevation) -> tuple:
    """(average gradient %, words) over the rider's distance from index a to b."""
    d0, d1 = grid.dist_m[ri, a], grid.dist_m[ri, b]
    if not (np.isfinite(d0) and np.isfinite(d1)) or d1 - d0 < 50:
        return None, 'the road'
    ed = elevation['distance_km'].to_numpy() * 1000.0
    ea = elevation['altitude_m'].to_numpy()
    g = float((np.interp(d1, ed, ea) - np.interp(d0, ed, ea)) / (d1 - d0) * 100.0)
    if g >= 2:
        return g, f"a climb ({g:+.1f}%)"
    if g <= -2:
        return g, f"a descent ({g:+.1f}%)"
    return g, 'the flat'


def detect_good_moves(grid: RaceGrid, ri: int, wbal: np.ndarray, elevation) -> List[Dict]:
    """Moves that worked: getting away from your group, or bridging to the group ahead."""
    T = len(grid.times)
    present = np.flatnonzero(np.isfinite(grid.dist_m[ri]))
    if len(present) < 2:
        return []
    last = int(present[-1])
    weight = grid.weight_kg[ri]
    t0 = float(grid.times[0])
    moves = []

    def members(k):
        m = np.flatnonzero(grid.group[:, k] == grid.group[ri, k])
        return m[m != ri]

    def closest_gap(others, k, rider_ahead):
        """Seconds between the rider and the nearest of ``others`` (rider ahead if rider_ahead)."""
        alive = others[np.isfinite(grid.dist_m[others, k])]
        if len(alive) == 0:
            return None
        if rider_ahead:
            j = alive[np.argmax(grid.dist_m[alive, k])]
            return float(grid.gap_to(j, ri, k))
        j = alive[np.argmin(grid.dist_m[alive, k])]
        return float(grid.gap_to(ri, j, k))

    def effort(a, b):
        sl = slice(a, b + 1)
        grade, words = _terrain(grid, ri, a, b, elevation)
        return {'power_w': _nanmean(grid.power[ri, sl]), 'reserve_cost_j': float(max(0.0, wbal[a] - np.min(wbal[sl]))),
                'grade_pct': grade, 'terrain': words, 'duration_sec': int(b - a + 1)}

    k = int(present[0]) + 1
    last_end = int(present[0])       # moves don't overlap: each starts after the previous one ends
    while k <= last:
        if grid.group[ri, k] < 0 or grid.group[ri, k - 1] < 0:
            k += 1
            continue
        prev = np.flatnonzero(grid.group[:, k - 1] == grid.group[ri, k - 1])
        prev = prev[prev != ri]
        cur = members(k)

        # --- Got away: most of your group is now behind you, in another group.
        behind = prev[np.isin(prev, cur, invert=True) & np.isfinite(grid.dist_m[prev, k])]
        behind = behind[grid.dist_m[behind, k] < grid.dist_m[ri, k]]
        if len(prev) >= 1 and len(behind) >= max(1, 0.5 * len(prev)):
            reached = None
            for kk in range(k, min(last, k + GETAWAY_CONFIRM_SEC) + 1):
                g = closest_gap(behind, kk, rider_ahead=True)
                if g is not None and g >= GETAWAY_MIN_GAP_SEC:
                    reached = kk
                    break
            # ...and they never get back to you.
            stuck = reached is not None and all(
                np.mean(grid.group[behind, kk] == grid.group[ri, kk]) < 0.5
                for kk in range(k, last + 1, 5) if np.isfinite(grid.dist_m[behind, kk]).any())
            if stuck:
                a = max(last_end, k - GETAWAY_LEADUP_SEC)
                e = effort(a, reached)
                final = closest_gap(behind, last, rider_ahead=True)
                moves.append(dict(e, kind='getaway', start_sec=float(grid.times[a]),
                                  end_sec=float(grid.times[reached]), riders=int(len(behind)),
                                  gap_reached_sec=float(closest_gap(behind, reached, True)),
                                  final_gap_sec=final,
                                  sprint=bool(grid.dist_m[ri, k] > grid.finish_m - FINISH_EXCLUDE_M)))
                last_end = reached
                k = reached + 60
                continue

        # --- Bridged: riders who were in a group ahead are now in yours.
        joined = cur[(grid.group[cur, k - 1] != grid.group[ri, k - 1])
                     & (grid.dist_m[cur, k - 1] > grid.dist_m[ri, k - 1])]
        if len(joined):
            lo = max(last_end, k - BRIDGE_LOOKBACK_SEC)
            gaps = np.array([closest_gap(joined, kk, rider_ahead=False) or 0.0 for kk in range(lo, k)])
            if len(gaps) and gaps.max() >= BRIDGE_MIN_GAP_SEC:
                k_max = lo + int(np.argmax(gaps))
                sizes = grid.group_size[ri, k_max:k]
                front = np.mean(grid.pos_in_group[ri, k_max:k] == 0)
                if np.median(sizes) <= 2 or front >= 0.5:      # you did the work
                    e = effort(k_max, k)
                    moves.append(dict(e, kind='bridge', start_sec=float(grid.times[k_max]),
                                      end_sec=float(grid.times[k]), riders=int(len(joined)),
                                      gap_closed_sec=float(gaps.max()), solo=bool(np.median(sizes) <= 1)))
                    last_end = k
                    k += 60
                    continue
        k += 1

    moves = sorted(moves, key=lambda m: m['start_sec'])[:MAX_GOOD_MOVES]
    for m in moves:
        m.update(_good_move_text(m, weight, t0))
    return moves


def _good_move_text(m: Dict, weight: float, t0: float) -> Dict:
    """Title and explanation for a move that worked."""
    def at(t):
        return _fmt_dur(t - t0)

    wkg = (m['power_w'] or 0) / weight
    cost = (f" It used {m['reserve_cost_j'] / 1000:.1f} kJ of your reserve." if m['reserve_cost_j'] >= 500
            else " It barely touched your reserve.")
    climb = m['grade_pct'] is not None and m['grade_pct'] >= 2
    if m['kind'] == 'getaway':
        who = 'the rider you were with' if m['riders'] == 1 else f"the {m['riders']} riders you were with"
        title = 'Sprinted clear' if m['sprint'] else 'Got away'
        text = (f"At {at(m['start_sec'])} on {m['terrain']} you went clear of {who}: {m['power_w']:.0f} W "
                f"({wkg:.1f} W/kg) for {_dur(m['duration_sec'])} got you {m['gap_reached_sec']:.0f}s clear")
        if m['final_gap_sec'] is not None and m['final_gap_sec'] > m['gap_reached_sec'] + 2:
            text += f", and the gap kept growing — {m['final_gap_sec']:.0f}s by the end. They never got back to you."
        else:
            text += ", and they never got back to you."
        why = (" Going where the road goes up, where draft matters least, made it stick." if climb else '')
    else:
        title = 'Bridged up solo' if m['solo'] else 'Bridged up'
        text = (f"From {at(m['start_sec'])} to {at(m['end_sec'])} on {m['terrain']} you closed a "
                f"{m['gap_closed_sec']:.0f}s gap to the {'rider' if m['riders'] == 1 else str(m['riders']) + ' riders'} "
                f"ahead at {m['power_w']:.0f} W ({wkg:.1f} W/kg) over {_dur(m['duration_sec'])}.")
        why = (" Climbs are the cheapest place to bridge — there's little draft to lose." if climb else '')
    return {'title': title, 'text': text + cost + why, 'status': 'ok'}


def detect_sustained_extra(grid: RaceGrid, ri: int, cp: float, w_prime: float, wbal: np.ndarray,
                           nearby_eq: np.ndarray, mates_draft: np.ndarray,
                           events: List[Dict]) -> List[Dict]:
    """Long stretches of spending more than the riders around you.

    Short efforts are caught by ``detect_extra_efforts``; this catches the
    slow leak — e.g. 15 minutes at +30 W from riding near the front with half
    the group's draft — which barely touches W' but is real energy.
    """
    diff = grid.power[ri] - nearby_eq
    in_group = grid.group_size[ri] >= MIN_GROUP_SIZE
    diff = np.where(in_group, diff, np.nan)
    roll = pd.Series(diff).rolling(SUSTAINED_WINDOW_SEC, center=True,
                                   min_periods=SUSTAINED_WINDOW_SEC // 2).mean().to_numpy()
    with np.errstate(invalid='ignore'):
        mask = np.isfinite(roll) & (roll >= SUSTAINED_MIN_W)
    later_events = [(int(e['time_sec'] - grid.times[0]), e) for e in events]
    out = []
    for a, b in _runs(mask, SUSTAINED_MERGE_SEC, SUSTAINED_MIN_SEC):
        sl = slice(a, b + 1)
        power, near = _nanmean(grid.power[ri, sl]), _nanmean(nearby_eq[sl])
        if power is None or near is None:
            continue
        sizes = grid.group_size[ri, sl]
        rel = grid.pos_in_group[ri, sl] / np.maximum(sizes - 1, 1)
        consequence = None
        nxt = next(((k, e) for k, e in later_events if k > b), None)
        if nxt is not None:
            k_n, e_n = nxt
            cf = grid.power[ri].copy()
            mates = nearby_eq[sl]
            cf[sl] = np.where(np.isfinite(mates), np.minimum(grid.power[ri, sl], mates), cf[sl])
            cf_bal = w_prime_balance(cf[: k_n + 1], cp, w_prime)
            consequence = {'event_kind': e_n['kind'], 'event_sec': e_n['time_sec'],
                           'reserve_lost_j': float(cf_bal[k_n] - wbal[k_n])}
        out.append({
            'kind': 'sustained',
            'start_sec': float(grid.times[a]), 'end_sec': float(grid.times[b]),
            'duration_sec': int(b - a + 1),
            'distance_km': float(grid.dist_m[ri, a] / 1000.0),
            'power_w': power, 'nearby_w': near,
            'extra_kj': float(np.nansum(grid.power[ri, sl] - nearby_eq[sl]) / 1000.0),
            'reserve_cost_j': float(wbal[a] - np.min(wbal[sl])),
            'draft_pct': _nanmean(grid.draft_pct[ri, sl]),
            'mates_draft_pct': _nanmean(mates_draft[sl]),
            'front_pct': float(np.mean(grid.pos_in_group[ri, sl] <= 1) * 100.0),
            'front_third_pct': float(np.mean(rel < 1 / 3) * 100.0),
            'group_size': int(np.median(sizes)),
            'consequence': consequence,
        })
    for e in out:
        e.update(_effort_text(e, w_prime, grid.weight_kg[ri], float(grid.times[0])))
    return out


def _effort_text(e: Dict, w_prime: float, weight: float, t0: float) -> Dict:
    """Verdict, one-line explanation and advice for an extra effort."""
    def at(t):
        return _fmt_dur(t - t0)

    wkg = e['power_w'] / weight
    if e['kind'] == 'sustained':
        mins = round(e['duration_sec'] / 60)
        title = f"Spent extra for {mins} min"
        text = (f"From {at(e['start_sec'])} to {at(e['end_sec'])} you averaged {wkg:.2f} W/kg vs "
                f"{e['nearby_w'] / weight:.2f} W/kg for the riders around you (their effort converted to you) — "
                f"{(e['power_w'] - e['nearby_w']) / weight:.2f} W/kg ({e['power_w'] - e['nearby_w']:.0f} W) "
                f"more the whole way, {e['extra_kj']:.0f} kJ in total. ")
        if e['draft_pct'] is not None and e['mates_draft_pct'] is not None:
            text += (f"Wind protection {e['draft_pct']:.0f}% vs {e['mates_draft_pct']:.0f}% for the group, "
                     f"with {e['front_pct']:.0f}% of the time at 1st–2nd wheel and {e['front_third_pct']:.0f}% "
                     f"in the front third. ")
        near_front = e['front_pct'] >= 25 or e['front_third_pct'] >= 60
        if near_front:
            text += "You were riding near the front, in the wind, far more than you needed to."
            advice = ("Sit further back in the bunch when nothing's happening — near the front you pay "
                      "for the draft everyone behind you gets.")
        else:
            text += "You were often out of the draft inside the group."
            advice = "Stay tucked on a wheel inside the group — gaps in the line cost you the draft."
        text += (" Most of it was below CP, so it barely touched your reserve, but it's energy you "
                 "couldn't spend later.")
        c = e['consequence']
        if c is not None and c['reserve_lost_j'] >= EFFORT_MATTERED_J:
            what = 'drop' if c['event_kind'] == 'drop' else 'close call'
            text += (f" Without it you'd have had {c['reserve_lost_j'] / 1000:.1f} kJ more reserve at the "
                     f"{at(c['event_sec'])} {what}.")
            status = 'no'
        else:
            status = 'borderline'
        return {'title': title, 'text': text, 'advice': advice, 'status': status}
    if e['kind'] == 'chase':
        who = 'a solo rider' if e['move_riders'] == 1 else f"a {e['move_riders']}-rider move"
        title = 'Chased a move'
        text = (f"Chased {who} {'who' if e['move_riders'] == 1 else 'that'} was {e['move_gap_sec']:.0f}s up "
                f"the road — you were at the front of the "
                f"chase {e['front_pct']:.0f}% of the time, at {e['power_w']:.0f} W ({wkg:.1f} W/kg) for "
                f"{e['duration_sec']}s. ")
        if e['caught_after_sec'] is None:
            text += "They stayed away anyway."
        elif e['caught_after_sec'] <= 0:
            text += "They were caught while you were chasing."
        else:
            text += f"They were caught {e['caught_after_sec']}s after you eased."
        advice = (f"With {e['group_size'] - 1} riders behind you, let someone else start the chase — "
                  f"small moves usually get brought back.")
        useful = e['caught_after_sec'] is not None and e['front_pct'] < 50
    elif e['kind'] == 'attack':
        title = 'Attack'
        text = f"Went hard on the front at {e['power_w']:.0f} W ({wkg:.1f} W/kg) for {e['duration_sec']}s"
        if e['attack_gap_sec'] is not None and e['attack_gap_sec'] >= ATTACK_GAP_SEC:
            text += f" and opened a {e['attack_gap_sec']:.1f}s gap, but the group brought you back."
        else:
            text += " — the group stayed on your wheel, so it never became a gap."
        advice = "An attack that doesn't open a gap is pure cost — save it for when the group is tired or the road helps."
        useful = False
    elif e['kind'] == 'pull':
        title = 'Rode on the front'
        text = (f"Rode at the front of the group (1st–2nd wheel) for {e['duration_sec']}s at "
                f"{e['power_w']:.0f} W ({wkg:.1f} W/kg) with nobody up the road.")
        advice = "Sit in unless you have a reason to pull — that reserve is what you need for the next surge."
        useful = False
    elif e.get('surge'):
        title = 'Surged in the group'
        text = (f"Went hard from {_ordinal(e['pos'])} of {e['group_size']} for {e['duration_sec']}s at "
                f"{e['power_w']:.0f} W ({wkg:.1f} W/kg) — often a sign of closing a gap or moving up late.")
        advice = ("Hard surges inside the group often mean reacting late — stay closer to the wheel in front "
                  "so you don't have to.")
        useful = False
    else:
        title = 'Out of the draft'
        text = (f"Rode out of the draft ({_ordinal(e['pos'])} of {e['group_size']}) for {e['duration_sec']}s at "
                f"{e['power_w']:.0f} W ({wkg:.1f} W/kg).")
        advice = "Stay glued to a wheel — gaps in the line cost you the draft."
        useful = False
    reserve = (f"used {e['reserve_cost_j'] / 1000:.1f} kJ of your reserve" if e['reserve_cost_j'] >= 500
               else "barely touched your reserve")
    text += f" That's {_work_vs_you(e['extra_kj'])}, and it {reserve}."

    c = e['consequence']
    if c is not None and c['reserve_lost_j'] >= EFFORT_MATTERED_J:
        what = 'drop' if c['event_kind'] == 'drop' else 'close call'
        text += (f" Without it you'd have had {c['reserve_lost_j'] / 1000:.1f} kJ more reserve at the "
                 f"{at(c['event_sec'])} {what}.")
        status = 'no'
    else:
        if c is not None:
            what = 'drop' if c['event_kind'] == 'drop' else 'close call'
            text += f" Your reserve had refilled before the {at(c['event_sec'])} {what}."
        status = 'ok' if useful else 'borderline'
    return {'title': title, 'text': text, 'advice': advice, 'status': status}


def _hang_on(grid: RaceGrid, ri: int, k_sep: int, pack_req: np.ndarray, wbal: np.ndarray,
             cp: float, w_prime: float) -> Optional[Dict]:
    """How long after the split the group eased, and what staying with it would have cost."""
    T = len(grid.times)
    roll = _rolling_mean(pack_req, 30, center=False)
    hi = min(T, k_sep + HANG_ON_MAX_SEC)
    under = np.flatnonzero(np.isfinite(roll[k_sep:hi]) & (roll[k_sep:hi] < cp))
    if len(under) == 0:
        return {'eased': False, 'hang_sec': HANG_ON_MAX_SEC}
    ease = k_sep + int(under[0])
    plan = pack_req[k_sep:max(ease, k_sep + 1)]
    plan = np.where(np.isfinite(plan), plan, cp)
    path = w_prime_balance(plan, cp, w_prime, start=wbal[k_sep])
    after = pack_req[ease:min(T, ease + 60)]
    return {
        'eased': True,
        'ease_sec': float(grid.times[ease]),
        'hang_sec': int(ease - k_sep),
        'peak_w': float(np.nanmax(roll[k_sep:ease + 1])),
        'after_w': _nanmean(after),
        'reserve_at_split_j': float(wbal[k_sep]),
        'cost_j': max(0.0, float(wbal[k_sep] - path.min())),
        'low_j': float(path.min()),
    }


def _linked_exposure(exposures: List[Dict], k_sep: int) -> Optional[Dict]:
    """The draft-loss episode leading into a separation, if any."""
    linked = [e for e in exposures
              if e['k_start'] <= k_sep and e['k_end'] >= k_sep - DRAFT_LOSS_LINK_SEC]
    return max(linked, key=lambda e: e['k_end']) if linked else None


def _analyse_event(grid: RaceGrid, ri: int, ev: Dict, cp: float, w_prime: float,
                   wbal: np.ndarray, nearby_eq: np.ndarray, exposures: List[Dict],
                   elevation) -> Dict:
    T = len(grid.times)
    k_sep, k_end = ev['k_sep'], ev['k_end']
    weight = grid.weight_kg[ri]

    # Survivors: riders who were with the rider before the split and are
    # still in the group that went ahead at the end of the window.
    alive = ev['went_ahead'][np.isfinite(grid.dist_m[ev['went_ahead'], k_end])]
    pg = _majority_group(grid, alive, k_end) if len(alive) else None
    survivors = np.array([], dtype=int)
    if pg is not None:
        survivors = ev['prev_group'][grid.group[ev['prev_group'], k_end] == pg]
    if len(survivors) == 0:
        survivors = ev['went_ahead']

    surv_eq = equivalent_power(grid, ri, survivors)          # (S, T)
    pack_req = _quiet(np.nanmedian, surv_eq, axis=0)
    ws, we, intense = _selection_window(pack_req, k_sep, cp, T)
    # A draft-loss episode running into the window is part of the selection:
    # that's where the rider started paying more than the group.
    exposure = _linked_exposure(exposures, k_sep)
    if exposure is not None and exposure['k_end'] >= ws - DRAFT_LOSS_MERGE_SEC:
        ws = min(ws, exposure['k_start'])
    first_k = int(np.flatnonzero(np.isfinite(grid.dist_m[ri]))[0])
    ws = max(ws, first_k)
    we = max(we, k_sep)
    dur = we - ws + 1
    is_drop = ev['kind'] == 'drop'
    drift = (_position_drift(grid, ri, k_sep - POSITION_DRIFT_LOOKBACK_SEC, k_sep - 1)
             if ev['kind'] in ('drop', 'near_miss') else None)
    sat_up = _detect_sat_up(grid, ri, ev, pack_req, wbal, cp) if is_drop else None
    hang_on = _hang_on(grid, ri, k_sep, pack_req, wbal, cp, w_prime) if is_drop else None
    # Staying on means staying with the group until it eased, not just until the split.
    if hang_on and hang_on['eased']:
        k_follow = int(hang_on['ease_sec'] - grid.times[0])
    else:
        k_follow = min(T - 1, k_sep + HANG_ON_MAX_SEC)

    # Near miss: what getting back on actually cost (measured, not simulated).
    recovery = None
    if ev['kind'] == 'near_miss' and ev['rejoin_k'] is not None:
        rk = ev['rejoin_k']
        recovery = {
            'duration_sec': int(rk - k_sep),
            'power_w': _nanmean(grid.power[ri, k_sep:rk + 1]),
            'group_w': _nanmean(pack_req[k_sep:rk + 1]),
            'w_prime_cost_j': max(0.0, float(wbal[k_sep] - np.min(wbal[k_sep:rk + 1]))),
            'w_prime_left_j': float(np.min(wbal[k_sep:rk + 1])),
        }

    rider_p = grid.power[ri]
    actual = _nanmean(rider_p[ws:we + 1])
    surv_means = np.array([_nanmean(row[ws:we + 1]) or np.nan for row in surv_eq])
    surv_means = surv_means[np.isfinite(surv_means)]
    needed = float(np.quantile(surv_means, REQUIRED_QUANTILE)) if len(surv_means) else None
    typical = float(np.median(surv_means)) if len(surv_means) else None

    # The run-up: excess vs the riders around you, draft, position.
    ps = max(first_k, ws - WASTE_LOOKBACK_SEC)
    pre = slice(ps, ws)
    pre_actual = _nanmean(rider_p[pre])
    pre_mates = _nanmean(nearby_eq[pre])
    pre_len = ws - ps

    def draft_pct(rows, sl):
        aero = np.nansum(grid.aero_power[rows, sl])
        solo = np.nansum(grid.solo_aero[rows, sl])
        return float(100.0 * (1 - aero / solo)) if solo > 0 else None

    rider_draft = draft_pct([ri], pre)
    surv_draft = draft_pct(survivors, pre)
    sizes = grid.group_size[ri, pre]
    rel_pos = np.where(sizes > 1, grid.pos_in_group[ri, pre] / np.maximum(sizes - 1, 1), np.nan)
    in_group = sizes >= MIN_GROUP_SIZE
    front_third = float(np.mean(rel_pos[in_group] < 1 / 3) * 100) if in_group.any() else None
    back_third = float(np.mean(rel_pos[in_group] > 2 / 3) * 100) if in_group.any() else None
    on_front = int(np.sum((grid.pos_in_group[ri, pre] == 0) & in_group))

    # W' counterfactual: ride the run-up the way a typical survivor did (their
    # power converted to your body, surges and all) instead of how you did.
    # Using whole individual traces keeps normal variability in the baseline.
    cf_at_start = float(wbal[ws])
    if pre_len > 0:
        cf_ends = []
        for row in surv_eq[:, pre]:
            if np.isfinite(row).mean() > 0.8:
                filled = np.where(np.isfinite(row), row, rider_p[pre])
                cf_ends.append(w_prime_balance(filled, cp, w_prime, start=wbal[ps])[-1])
        if cf_ends:
            cf_at_start = float(np.median(cf_ends))
    saved = cf_at_start - float(wbal[ws])
    shortfall = max(0.0, (needed or 0) - (actual or 0)) * dur

    def wkg(w):
        return None if w is None else w / weight

    def ftime(k):
        return float(grid.times[k])

    result = {
        'kind': ev['kind'],
        'time_sec': ftime(k_sep),
        'distance_km': float(grid.dist_m[ri, k_sep] / 1000.0),
        'rejoin_time_sec': ftime(ev['rejoin_k']) if ev['rejoin_k'] is not None else None,
        'max_gap_sec': ev['max_gap_sec'],
        'group_size_before': ev['group_size_before'],
        'n_ahead': int(len(ev['went_ahead'])),
        'pos_before': ev['pos_before'] + 1,
        'survivor_count': int(len(survivors)),
        'selection': {
            'start_sec': ftime(ws), 'end_sec': ftime(we), 'duration_sec': int(dur),
            'intense': intense,
            'avg_gradient_pct': _avg_gradient(grid, ri, ws, we, elevation),
            'actual_w': actual, 'actual_wkg': wkg(actual),
            'needed_w': needed, 'needed_wkg': wkg(needed),
            'typical_w': typical, 'typical_wkg': wkg(typical),
        },
        'runup': {
            'start_sec': ftime(ps), 'duration_sec': int(pre_len),
            'actual_w': pre_actual, 'actual_wkg': wkg(pre_actual),
            'nearby_w': pre_mates, 'nearby_wkg': wkg(pre_mates),
            'excess_kj': ((pre_actual - pre_mates) * pre_len / 1000.0
                          if pre_actual is not None and pre_mates is not None else None),
            'draft_pct': rider_draft, 'survivor_draft_pct': surv_draft,
            'front_third_pct': front_third, 'back_third_pct': back_third,
            'on_front_sec': on_front,
        },
        'w_prime': {
            'at_selection_start_j': float(wbal[ws]),
            'at_separation_j': float(wbal[k_sep]),
            'min_in_selection_j': float(np.min(wbal[ws:we + 1])),
            'counterfactual_at_start_j': cf_at_start,
            'saved_j': saved,
            'shortfall_j': shortfall,
        },
        'draft_loss': _public_exposure(exposure),
        'sat_up': sat_up,
        'hang_on': hang_on,
        'cp_w': cp,
        'recovery': recovery,
        # Only for real drops: for a near miss the rider did get back on, and
        # the measured recovery says more than a simulated one.
        'gap_close': (_gap_closing_costs(grid, ri, ev, wbal, cp, w_prime, elevation, pack_req, k_follow)
                      if is_drop else None),
        'position_earlier': (_position_earlier(grid, ri, survivors, surv_eq, ws, we, wbal, cp,
                                               w_prime, elevation) if is_drop else []),
        'position_drift': drift,
        'hold_position': (_hold_position(grid, ri, survivors, surv_eq, drift, ws, we, k_sep,
                                         wbal, cp, w_prime)
                          if drift is not None else None),
    }
    result['findings'] = _findings(result, w_prime, weight, float(grid.times[0]))
    result.update(_narrative(result, w_prime, weight, float(grid.times[0])))
    return result


def _public_exposure(e: Optional[Dict]) -> Optional[Dict]:
    """Drop grid indices from an exposure before it goes out over the API."""
    if e is None:
        return None
    return {k: v for k, v in e.items() if not k.startswith('k_')}


def _fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _spent_vs_you(extra_kj: float) -> str:
    """'spent X kJ more/less than you', or 'spent about the same as you'."""
    if abs(extra_kj) < SAME_EFFORT_KJ:
        return 'spent about the same as you'
    return f"spent {abs(extra_kj):.1f} kJ {'more' if extra_kj > 0 else 'less'} than you"


def _reserve_outcome(low_j: float) -> str:
    """What a W' low point means for the rider, in plain words."""
    if low_j >= 0:
        return f"That leaves you {low_j / 1000:.1f} kJ of reserve at the lowest point"
    return f"That's {-low_j / 1000:.1f} kJ more than your reserve could cover"


def _fit_count(d: Dict, short: bool = False) -> str:
    """'none/k/all of their efforts would have fit (your reserve)' (riders named just before)."""
    n = int(d['riders'])
    k = int(round(d['feasible_share'] * n))
    tail = '' if short else ' your reserve'
    if n == 1:
        return f"their effort {'would' if k else 'wouldn' + chr(39) + 't'} have fit{tail}"
    if k == 0:
        return f"{'neither' if n == 2 else 'none'} of their efforts would have fit{tail}"
    if k == n:
        return f"{'both' if n == 2 else f'all {n}'} of their efforts would have fit{tail}"
    return f"{k} of their {n} efforts would have fit{tail}"


def _vs_actual(extra_kj: float) -> str:
    """'X kJ more/less than you', or 'about what you spent' when it's noise."""
    if abs(extra_kj) < SAME_EFFORT_KJ:
        return 'about what you actually spent'
    if extra_kj > 0:
        return f"{extra_kj:.1f} kJ more than you spent"
    return f"{-extra_kj:.1f} kJ less than you spent"


def _unrealistic(power_w: float, cp: float) -> bool:
    """A sustained target this far above CP is a sprint, not a plan."""
    return power_w > UNREALISTIC_POWER_FRAC_CP * cp


def _work_vs_you(extra_kj: float, who: str = 'the riders around you') -> str:
    """'X kJ more/less work than <who>', or 'about the same work as <who>'."""
    if abs(extra_kj) < SAME_EFFORT_KJ:
        return f"about the same work as {who}"
    return f"{abs(extra_kj):.1f} kJ {'more' if extra_kj > 0 else 'less'} work than {who}"


def _reserve_low_text(low_j: float) -> str:
    """Plain words for where the reserve would have bottomed out."""
    if low_j >= 0:
        return f"your reserve would have bottomed out at {low_j / 1000:.1f} kJ"
    return f"your reserve would have run out, {-low_j / 1000:.1f} kJ short"


def _position_findings(rows: List[Dict], drift: Optional[Dict], hold: Optional[Dict],
                       weight: float, at) -> List[str]:
    """Detail lines on position: sliding back, holding your spot, moving up earlier."""
    out = []
    if drift is not None:
        out.append(
            f"Position: you slid from {_ordinal(drift['from_pos'])} of {drift['from_size']} at "
            f"{at(drift['from_sec'])} to {_ordinal(drift['to_pos'])} of {drift['to_size']} by "
            f"{at(drift['to_sec'])}."
        )
    if hold is not None:
        out.append(
            f"Holding your spot: the {hold['riders']} riders who stayed on from within "
            f"{MOVE_UP_SLOT_SPAN} places of you at {at(hold['from_sec'])} did "
            f"{_work_vs_you(hold['extra_kj'], 'you')} before you were dropped, then rode the surge at "
            f"{hold['selection_w'] / weight:.2f} W/kg for you. Doing the same, "
            f"{_reserve_low_text(hold['min_w_prime_j'])} ({_fit_count(hold)})."
        )
    rows = [r for r in rows if r['model_error_m'] <= GAP_MODEL_MAX_ERROR_M]
    if not rows:
        return out
    ok = [r for r in rows if r['feasible']]
    if not ok:
        r = max(rows, key=lambda r: r['min_w_prime_j'])
        out.append(
            f"Moving up earlier wouldn't have saved it on its own: even from {_ordinal(r['to_pos'])} of "
            f"{r['group_size']}, {_reserve_low_text(r['min_w_prime_j'])}."
        )
        return out
    # Cheapest move that works, preferring more reserve left on a tie.
    r = min(ok, key=lambda r: (r['extra_kj'], -r['min_w_prime_j']))
    fit = _fit_count({'riders': r['slot_riders'], 'feasible_share': r['feasible_share']})
    out.append(
        f"Moving up earlier: from {_ordinal(r['from_pos'])} to {_ordinal(r['to_pos'])} of "
        f"{r['group_size']} starting {at(r['move_start_sec'])} takes about {r['move_power_w']:.0f} W "
        f"({r['move_power_w'] / weight:.1f} W/kg) for {r['move_time_sec']}s "
        f"({_work_vs_you(r['extra_kj'], 'you did')}). Riding the surge from there like the "
        f"{r['slot_riders']} riders who stayed on around that spot costs {r['selection_w'] / weight:.2f} W/kg "
        f"for you, and {_reserve_low_text(r['min_w_prime_j'])} ({fit})."
    )
    return out


def _gap_close_findings(gc: Optional[Dict], weight: float, cp: float, at) -> List[str]:
    """Detail lines from the gap-closing simulation."""
    if not gc:
        return []
    if gc['model_error_m'] is not None and gc['model_error_m'] > GAP_MODEL_MAX_ERROR_M:
        return [f"No cost-to-close estimate here: replaying your actual power, the model misses the real "
                f"gap by {gc['model_error_m']:.0f} m — usually because the group was strung out."]
    out = []
    if gc.get('follow_until_sec'):
        out.append(f"The costs below include staying with the group until it eased at "
                   f"{at(gc['follow_until_sec'])} — getting back on only helps if you can stay on.")
    hold = gc['hold']
    if hold is not None:
        if _unrealistic(hold['power_w'], cp):
            out.append(f"Staying on the last wheel from {at(hold['start_sec'])} would have taken a full sprint "
                       f"({hold['power_w']:.0f} W for {hold['duration_sec']}s) — not realistic.")
        else:
            verdict = 'affordable' if hold['affordable'] else 'more than you had'
            out.append(
                f"Staying on the last wheel from {at(hold['start_sec'])} needed {hold['power_w']:.0f} W "
                f"({hold['power_w'] / weight:.1f} W/kg) for {hold['duration_sec']}s — you did "
                f"{hold['actual_w']:.0f} W. That's {hold['w_prime_cost_j'] / 1000:.1f} kJ of the "
                f"{hold['w_prime_left_j'] / 1000:.1f} kJ reserve you had ({verdict})."
            )
    cells = [(r, r['close_30']) for r in gc['rows'] if r.get('close_30')]
    ok = [(r, c) for r, c in cells if c['affordable']]
    if ok:
        r, c = ok[0]
        line = (f"Closing early was affordable: at {at(r['react_sec'])} (gap {r['gap_m']:.0f} m), "
                f"{c['power_w']:.0f} W ({c['power_w'] / weight:.1f} W/kg) gets you back on in "
                f"{c['time_sec']}s — {c['w_prime_cost_j'] / 1000:.1f} kJ of reserve in all.")
        later = [(r2, c2) for r2, c2 in cells if r2['react_sec'] > ok[-1][0]['react_sec']]
        if later:
            r2, c2 = later[0]
            line += (f" By {at(r2['react_sec'])} (gap {r2['gap_m']:.0f} m) it took "
                     f"{c2['power_w']:.0f} W ({c2['power_w'] / weight:.1f} W/kg), more than you had left.")
        out.append(line)
    elif cells:
        r, c = cells[0]
        out.append(
            f"Closing the gap was never affordable: even at {at(r['react_sec'])} (gap {r['gap_m']:.0f} m) it "
            f"took {c['power_w']:.0f} W ({c['power_w'] / weight:.1f} W/kg) — {c['w_prime_cost_j'] / 1000:.1f} kJ "
            f"against the {r['w_prime_left_j'] / 1000:.1f} kJ reserve you had."
        )
    return out


def _findings(ev: Dict, w_prime: float, weight: float, t0: float) -> List[str]:
    """The numbers behind an event card ("Details"), in the same language as the card.

    ``t0`` is the race-clock start, so times read like the replay clock.
    """
    out = []
    s, r, w = ev['selection'], ev['runup'], ev['w_prime']
    x, su = ev['draft_loss'], ev['sat_up']
    cp = ev['cp_w']

    def at(t):
        return _fmt_dur(t - t0)

    near_miss = ev['kind'] == 'near_miss'
    if near_miss and ev['recovery'] is not None:
        out.append(
            f"A gap opened at {at(ev['time_sec'])} (it grew to {ev['max_gap_sec']:.1f}s); you were back on "
            f"{ev['recovery']['duration_sec']}s later."
        )

    if s['intense'] and s['needed_wkg'] is not None and s['actual_wkg'] is not None:
        grad = (f" at {s['avg_gradient_pct']:+.1f}% average gradient"
                if s['avg_gradient_pct'] is not None else '')
        if s['actual_w'] >= s['needed_w']:
            out.append(
                f"Surge: {_fmt_dur(s['duration_sec'])}{grad}. You did {s['actual_wkg']:.2f} W/kg — more than "
                f"the best-placed riders who stayed on ({s['needed_wkg']:.2f} W/kg for you). Power wasn't the "
                f"problem."
            )
        else:
            out.append(
                f"Surge: {_fmt_dur(s['duration_sec'])}{grad}. The riders who stayed on did "
                f"{s['needed_wkg']:.2f} W/kg for you (best-placed) to {s['typical_wkg']:.2f} (typical); you did "
                f"{s['actual_wkg']:.2f} ({s['actual_w'] - s['needed_w']:+.0f} W)."
            )
    elif not s['intense'] and not near_miss:
        out.append("The group wasn't riding above your CP when you lost contact — this points to position or "
                   "a slow reaction, not fitness.")

    if x is not None:
        if x['where'] == 'back':
            where = f"hanging off the back ({_ordinal(x['pos'])} of {x['group_size']}"
            if x['gap_ahead_sec'] is not None:
                where += f", {x['gap_ahead_sec']:.1f}s off the wheel"
            where += ')'
        elif x['where'] == 'front':
            where = f"on the front of a {x['group_size']}-rider group"
        else:
            where = f"mid-group ({_ordinal(x['pos'])} of {x['group_size']})"
        if x['extra_kj'] <= -SAME_EFFORT_KJ:
            work = (f"You did {-x['extra_kj']:.1f} kJ less work than the riders around you — you drifted off "
                    f"the wheel instead of closing it.")
        else:
            work = f"That's {_work_vs_you(x['extra_kj'])}."
        out.append(
            f"Out of the draft from {at(x['start_sec'])} for {x['duration_sec']}s, {where}: "
            f"{x['draft_pct']:.0f}% wind protection vs {x['mates_draft_pct']:.0f}% for the group. {work}"
        )

    out.append(f"You were {_ordinal(ev['pos_before'])} of {ev['group_size_before']} when the gap opened.")
    if r['excess_kj'] is not None and r['duration_sec'] > 0:
        draft = ''
        if r['draft_pct'] is not None and r['survivor_draft_pct'] is not None:
            draft = (f"; wind protection {r['draft_pct']:.0f}% vs {r['survivor_draft_pct']:.0f}% for the "
                     f"riders who stayed on")
        out.append(
            f"In the {_fmt_dur(r['duration_sec'])} before the surge: you averaged {r['actual_wkg']:.2f} W/kg vs "
            f"{r['nearby_wkg']:.2f} for the riders around you ({_work_vs_you(r['excess_kj'], 'them')}){draft}."
        )

    if su is not None:
        when = ('as the gap opened' if su['after_split_sec'] <= 0
                else f"{su['after_split_sec']}s after the gap opened")
        gap = (f", with the group only {su['gap_sec']:.1f}s away"
               if su['gap_sec'] is not None and 0 < su['gap_sec'] < 5 else '')
        out.append(
            f"Sat up {when}: {su['duration_sec']}s at {su['power_w'] / weight:.1f} W/kg while the group rode "
            f"{su['group_w'] / weight:.1f}, with {su['w_prime_j'] / 1000:.1f} kJ of reserve left{gap}."
        )

    out.extend(_gap_close_findings(ev['gap_close'], weight, cp, at))
    out.extend(_position_findings(ev['position_earlier'], ev['position_drift'],
                                  ev['hold_position'], weight, at))

    if near_miss:
        rec = ev['recovery']
        if rec is not None and rec['power_w'] is not None:
            group = (f" (the group was at {rec['group_w'] / weight:.1f})" if rec['group_w'] is not None else '')
            out.append(
                f"Getting back on: {rec['duration_sec']}s at {rec['power_w'] / weight:.1f} W/kg{group}, using "
                f"{rec['w_prime_cost_j'] / 1000:.1f} kJ of reserve and leaving {rec['w_prime_left_j'] / 1000:.1f} "
                f"of {w_prime / 1000:.1f} kJ."
            )
            if rec['w_prime_left_j'] < 0.25 * w_prime:
                out.append("That nearly emptied you — a close call like this can cost you the next surge.")
        return out

    if w_prime > 0:
        out.append(
            f"Reserve: {w['at_selection_start_j'] / 1000:.1f} of {w_prime / 1000:.1f} kJ when the surge "
            f"started; {w['at_separation_j'] / 1000:.1f} kJ when you were dropped."
        )
        cause = 'losing the draft' if x is not None else 'reacting late or not committing'
        short, left, saved = w['shortfall_j'], max(0.0, w['at_separation_j']), max(0.0, w['saved_j'])
        if short > 0 and s['intense']:
            if left >= short:
                out.append(
                    f"You still had {left / 1000:.1f} kJ of reserve when you were dropped — more than the "
                    f"{short / 1000:.1f} kJ of extra work staying on needed. You had the legs; {cause} cost "
                    f"you the group, not a lack of power."
                )
            elif left + saved >= short:
                out.append(
                    f"Riding the 5 minutes before like the riders who stayed on would have saved "
                    f"{saved / 1000:.1f} kJ of reserve — with the {left / 1000:.1f} kJ you had left, enough for "
                    f"the {short / 1000:.1f} kJ of extra work staying on needed. The effort beforehand likely "
                    f"cost you the group."
                )
            else:
                out.append(
                    f"Staying on needed {short / 1000:.1f} kJ more work and you had only {left / 1000:.1f} kJ of "
                    f"reserve left"
                    + (f" (riding the build-up better saves just {saved / 1000:.1f} kJ)" if saved > 0 else '')
                    + " — this surge needed more power or a better position going in."
                )
        elif w['at_separation_j'] > 0.3 * w_prime:
            out.append(f"You still had reserve left when you were dropped — {cause}, not fitness, cost you "
                       f"the group.")
    return out


def _slide_judgement(drift: Optional[Dict], hold: Optional[Dict]) -> Optional[Dict]:
    """Was sliding back a mistake, or the energy-saving choice?

    Judged by what holding the spot cost the riders who did it, converted to
    the rider: if it would have emptied their reserve, letting the group go
    past was the right call.
    """
    if drift is None:
        return None
    pos = _ordinal(drift['from_pos'])
    if hold is None:
        return {'kind': 'unknown', 'pos': pos, 'sentence': None}
    low, ref = hold['min_w_prime_j'], max(hold['w_prime_at_start_j'], 1.0)
    held = f"the riders who held it did {_work_vs_you(hold['extra_kj'], 'you')}"
    if abs(low) < BORDERLINE_W_PRIME_FRAC * ref:
        return {'kind': 'edge', 'pos': pos,
                'sentence': f"Holding {pos} would have taken you right to your limit ({held}) — "
                            f"letting some riders go was reasonable."}
    if low < 0:
        return {'kind': 'forced', 'pos': pos,
                'sentence': f"Holding {pos} would have emptied your reserve ({held}) — letting them go "
                            f"past was the right call."}
    return {'kind': 'choice', 'pos': pos,
            'sentence': f"Holding {pos} was affordable: {held} and it would have left you "
                        f"{low / 1000:.1f} kJ of reserve."}


def _margin(sel: Dict) -> Optional[str]:
    """'none' / 'under' / 'over': how the rider's surge effort compared to what it took."""
    if not sel['intense'] or sel['needed_w'] is None or sel['actual_w'] is None:
        return None
    ratio = sel['actual_w'] / sel['needed_w']
    if abs(ratio - 1) <= NO_MARGIN_FRAC:
        return 'none'
    if ratio > 1:
        return 'over'
    return 'well_under' if ratio < 1 - WELL_UNDER_FRAC else 'under'


def _dur(sec: float) -> str:
    """'45s' for short spans, 'm:ss' from 90 s up."""
    return f"{int(round(sec))}s" if sec < 90 else _fmt_dur(sec)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _narrative(ev: Dict, w_prime: float, weight: float, t0: float) -> Dict:
    """Plain-language verdict, timeline and options for one event.

    The detailed findings stay available; this is the version meant to be
    read first: one-line verdict, what happened in order, and what would have
    kept the rider on, ranked.
    """
    def at(t):
        return _fmt_dur(t - t0)

    def wkg(w):
        return f"{w / weight:.1f} W/kg"

    def kj(j):
        return f"{j / 1000:.1f} kJ"

    def reserve_left(j):
        return f"leaving {kj(j)} of reserve" if j >= 0 else f"running out of reserve by {kj(-j)}"

    s, w = ev['selection'], ev['w_prime']
    story, options = [], []

    if ev['kind'] == 'near_miss':
        rec = ev['recovery']
        summary = (f"Close call, not a drop: you lost contact at {at(ev['time_sec'])} and got back on "
                   f"{rec['duration_sec']}s later" if rec else
                   f"Close call, not a drop: you lost contact at {at(ev['time_sec'])} and got back on")
        if rec:
            summary += f", using {kj(rec['w_prime_cost_j'])} of your reserve."
        drift = ev['position_drift']
        margin = _margin(s)
        slide = _slide_judgement(drift, ev['hold_position'])
        advice = []
        if slide is not None and slide['kind'] == 'forced':
            summary += (f" You were at your limit: letting the group go past "
                        f"({_ordinal(drift['from_pos'])} → {_ordinal(drift['to_pos'])}) saved energy you "
                        f"needed, since holding your spot would have emptied you. But it left you near the "
                        f"back with no margin.")
            advice.append("Be further up before the hard part starts, so you can let riders past "
                          "without ending up at the back.")
        else:
            why = []
            if slide is not None:
                why.append(f"you slid from {_ordinal(drift['from_pos'])} to {_ordinal(drift['to_pos'])}"
                           + (" even though you could have held your spot" if slide['kind'] == 'choice' else ''))
                if slide['kind'] == 'choice':
                    advice.append("Hold your spot when the group accelerates — you could afford it here.")
            if margin == 'none':
                why.append("you were riding right at what it took, with no margin")
            elif margin == 'under':
                why.append("you were riding a little under what it took")
            elif margin == 'well_under':
                why.append("you were riding well under what it took")
            if why:
                summary += " It happened because " + ' and '.join(why) + "."
        if rec and rec['w_prime_left_j'] < 0.25 * w_prime:
            summary += f" You came out of it with only {kj(rec['w_prime_left_j'])} of {kj(w_prime)} in reserve."
        if rec and rec['duration_sec'] <= QUICK_RECOVERY_SEC:
            advice.append("Keep closing gaps immediately, like you did here — it kept this cheap.")
        if drift is not None:
            mu = drift['moved_up_from']
            if mu:
                story.append({'time_sec': mu['sec'],
                              'text': f"You were {_ordinal(mu['pos'])} of {mu['size']}."})
                story.append({'time_sec': drift['from_sec'],
                              'text': f"You moved up to {_ordinal(drift['from_pos'])} of {drift['from_size']}."})
            else:
                story.append({'time_sec': drift['from_sec'],
                              'text': f"You were {_ordinal(drift['from_pos'])} of {drift['from_size']}."})
            mins = max(1, round((drift['to_sec'] - drift['from_sec']) / 60))
            split = (f" (the group had split down to {drift['to_size']})"
                     if drift['to_size'] < 0.7 * drift['from_size'] else '')
            story.append({'time_sec': drift['from_sec'] + 1,
                          'text': f"Over the next {mins} min the group rode past you — you slid back to "
                                  f"{_ordinal(drift['to_pos'])} of {drift['to_size']}{split}."
                                  + (f" {slide['sentence']}" if slide and slide['sentence'] else '')})
        if s['intense'] and s['needed_wkg'] is not None and s['actual_wkg'] is not None:
            tail = {'none': ' — right at the limit, no margin',
                    'under': ' — a little under',
                    'well_under': ' — well under',
                    'over': ' — with some margin'}[margin]
            story.append({'time_sec': s['start_sec'],
                          'text': f"The group went hard: a {_fmt_dur(s['duration_sec'])} surge. Staying on took "
                                  f"about {s['needed_wkg']:.1f} W/kg for you; you did {s['actual_wkg']:.1f}{tail}."})
        left_with_you = ev['group_size_before'] - 1 - ev['n_ahead']
        if ev['n_ahead'] > FLIER_MAX_RIDERS and left_with_you >= 1:
            what = (f"the group split — {ev['n_ahead']} riders went clear and {left_with_you} were left with "
                    f"you; the front part got up to {ev['max_gap_sec']:.1f}s ahead of you")
        else:
            what = f"a gap opened to the rider in front (it grew to {ev['max_gap_sec']:.1f}s)"
        story.append({'time_sec': ev['time_sec'],
                      'text': f"You were {_ordinal(ev['pos_before'])} of {ev['group_size_before']} when {what}."})
        if rec and rec['power_w'] is not None:
            story.append({'time_sec': ev['rejoin_time_sec'],
                          'text': f"Back on after {rec['duration_sec']}s at {wkg(rec['power_w'])}"
                                  + (" — closing it straight away kept it cheap."
                                     if rec['duration_sec'] <= QUICK_RECOVERY_SEC else ".")})
        story.sort(key=lambda it: it['time_sec'])
        return {'summary': summary, 'story': story, 'options': options, 'advice': advice}

    if ev['kind'] == 'finish':
        return {'summary': f"Lost contact in the finale at {at(ev['time_sec'])} — that's the sprint, "
                           f"not getting dropped.", 'story': [], 'options': [], 'advice': []}

    gc = ev['gap_close']
    gc_ok = bool(gc) and (gc['model_error_m'] is None or gc['model_error_m'] <= GAP_MODEL_MAX_ERROR_M)
    rows30 = [(r, r['close_30']) for r in gc['rows'] if r.get('close_30')] if gc_ok else []
    drift, hold_pos, x, su = ev['position_drift'], ev['hold_position'], ev['draft_loss'], ev['sat_up']
    left, short = max(0.0, w['at_separation_j']), w['shortfall_j']

    # --- Verdict -----------------------------------------------------------
    if s['intense'] and short > 0 and left < short:
        lead = (f"You were short on power: staying on took about {s['needed_wkg']:.1f} W/kg through the "
                f"surge and you only had {kj(left)} of reserve left.")
    elif left > 0:
        lead = f"You had the legs — {kj(left)} of reserve was still unused when you were dropped."
    else:
        lead = "You were out of reserve when you were dropped."
    causes, advice = [], []
    slide = _slide_judgement(drift, hold_pos)
    if drift is not None:
        to = 'the back' if drift['to_pos'] >= drift['to_size'] - 1 else _ordinal(drift['to_pos'])
        if slide['kind'] == 'forced':
            causes.append("being stuck at the back with no places left to give")
            advice.append("Be further up before the hard part starts, so you can let riders past "
                          "without ending up at the back.")
        elif slide['kind'] == 'choice':
            causes.append(f"sliding from {_ordinal(drift['from_pos'])} to {to} when you could have held "
                          f"your spot")
            advice.append("Hold your spot when the group accelerates — you could afford it here.")
        else:
            causes.append(f"sliding from {_ordinal(drift['from_pos'])} to {to}")
    elif ev['pos_before'] >= 0.75 * ev['group_size_before']:
        causes.append("being at the back when the group surged")
    if any(c['affordable'] for _, c in rows30):
        causes.append("not closing the gap as soon as it opened")
        advice.append("Close a gap in the first few seconds — the cost climbs fast, especially at speed.")
    elif x is not None:
        causes.append("losing the draft")
    if su is not None:
        causes.append("easing off right after the gap opened")
        advice.append("Don't ease off as a gap opens — the group was still within reach.")
    if not s['intense'] and not causes:
        causes.append("position or a slow reaction — the group wasn't even riding hard for you")
    summary = lead
    if causes:
        joined = causes[0] if len(causes) == 1 else ', '.join(causes[:-1]) + ' and ' + causes[-1]
        summary += f" You lost the group by {joined}."
    ho = ev['hang_on']
    ho_ok = bool(ho and ho['eased'] and ho['low_j'] >= -BORDERLINE_W_PRIME_FRAC * max(ho['reserve_at_split_j'], 1.0))
    if ho_ok and ho['hang_sec'] <= HANG_ON_SHORT_SEC:
        when = ('right as you were dropped' if ho['hang_sec'] <= 2
                else f"just {ho['hang_sec']}s after you were dropped")
        summary += f" The group eased {when} — hanging on was within reach."
        advice.append("When you're on the limit in a surge, hang on a little longer — this one eased "
                      + ('right as you went.' if ho['hang_sec'] <= 2 else f"{ho['hang_sec']}s after you went."))

    # --- What happened, in order -------------------------------------------
    if drift is not None:
        to = 'last' if drift['to_pos'] >= drift['to_size'] - 1 else _ordinal(drift['to_pos'])
        mins = max(1, round((drift['to_sec'] - drift['from_sec']) / 60))
        story.append({'time_sec': drift['from_sec'],
                      'text': f"You were {_ordinal(drift['from_pos'])} of {drift['from_size']}. Over the next "
                              f"{mins} min you slid back to {to}."
                              + (f" {slide['sentence']}" if slide and slide['sentence'] else '')})
    if s['intense'] and s['needed_wkg'] is not None:
        g = s['avg_gradient_pct']
        terrain = ('a descent' if g is not None and g < -1 else
                   'a climb' if g is not None and g > 1 else 'the flat')
        story.append({'time_sec': s['start_sec'],
                      'text': f"The group went hard: a {_fmt_dur(s['duration_sec'])} surge on {terrain}. "
                              f"Staying on took about {s['needed_wkg']:.1f} W/kg for you."})
    gap_open = rows30[0][0] if rows30 else None
    if gap_open is not None:
        story.append({'time_sec': gap_open['react_sec'],
                      'text': f"A gap opened to the rider in front ({gap_open['gap_m']:.0f} m)."})
    if x is not None:
        if x['extra_kj'] < -SAME_EFFORT_KJ:
            effort = "You were putting out less than the riders around you, so the gap kept growing."
        elif x['extra_kj'] > SAME_EFFORT_KJ:
            effort = "You were working harder than the riders around you just to hang on."
        else:
            effort = "You were matching the riders around you — not enough to close it."
        story.append({'time_sec': x['start_sec'],
                      'text': f"Out of the draft: {x['draft_pct']:.0f}% wind protection vs "
                              f"{x['mates_draft_pct']:.0f}% for the group. {effort}"})
    if s['actual_wkg'] is not None:
        if s['needed_wkg'] and s['actual_wkg'] >= s['needed_wkg']:
            vs = f" — more than the {s['needed_wkg']:.1f} needed, so power wasn't the problem"
        elif s['needed_wkg']:
            vs = f" vs about {s['needed_wkg']:.1f} needed"
        else:
            vs = ''
        story.append({'time_sec': ev['time_sec'],
                      'text': f"Dropped. Through the surge you averaged {s['actual_wkg']:.1f} W/kg{vs}, "
                              f"with {kj(left)} of reserve unused."})
    if su is not None:
        gap = f" with the group only {su['gap_sec']:.1f}s away" if su['gap_sec'] and su['gap_sec'] < 5 else ''
        story.append({'time_sec': su['start_sec'],
                      'text': f"You eased off to {wkg(su['power_w'])} for {su['duration_sec']}s{gap}."})
    if ho is not None:
        if not ho['eased']:
            story.append({'time_sec': ev['time_sec'] + 1,
                          'text': f"The group kept the pressure on for at least {ho['hang_sec'] // 60} min after "
                                  f"you were dropped."})
        else:
            after = f" (down to {wkg(ho['after_w'])})" if ho['after_w'] else ''
            if ho['hang_sec'] <= 2:
                eased = 'The group eased right as you were dropped'
            elif ho['hang_sec'] <= HANG_ON_SHORT_SEC:
                eased = f"The group eased {ho['hang_sec']}s after you were dropped"
            else:
                eased = f"The group kept going, and only eased {_dur(ho['hang_sec'])} after you were dropped"
            story.append({'time_sec': ho['ease_sec'],
                          'text': f"{eased} — back under your CP{after}."})
    story.sort(key=lambda it: it['time_sec'])

    # --- What would have kept you on ---------------------------------------
    def status(low, ref):
        if abs(low) < BORDERLINE_W_PRIME_FRAC * max(ref, 1.0):
            return 'borderline'
        return 'ok' if low >= 0 else 'no'

    stay = (f", then stay with the group until it eased ({at(gc['follow_until_sec'])})"
            if gc_ok and gc.get('follow_until_sec') else '')

    def uses(cost, low, left):
        return (f"uses {kj(cost)} of the {kj(left)} reserve you had" if low >= 0
                else f"that's {kj(-low)} more than your reserve could cover")

    cp = ev['cp_w']
    if rows30 and _unrealistic(rows30[0][1]['power_w'], cp):
        r, c = rows30[0]
        options.append({'status': 'no', 'cost_j': c['w_prime_cost_j'],
                        'text': f"Close the gap right away ({at(r['react_sec'])}): it would have taken a full "
                                f"sprint ({c['power_w']:.0f} W for {c['time_sec']}s) — not realistic."})
    elif rows30:
        r, c = rows30[0]
        text = (f"Close the gap right away ({at(r['react_sec'])}): {c['power_w']:.0f} W ({wkg(c['power_w'])}) "
                f"for about {c['time_sec']}s{stay} — "
                f"{uses(c['w_prime_cost_j'], c['w_prime_low_j'], r['w_prime_left_j'])}.")
        later = next((rc for rc in rows30 if not rc[1]['affordable']), None)
        if later is not None and later[0] is not r:
            text += (f" Waiting until {at(later[0]['react_sec'])} pushes it to {later[1]['power_w']:.0f} W "
                     f"({wkg(later[1]['power_w'])}), "
                     f"more than you had.")
        options.append({'status': status(c['w_prime_low_j'], r['w_prime_left_j']),
                        'cost_j': c['w_prime_cost_j'], 'text': text})
    if hold_pos is not None:
        options.append({
            'status': status(hold_pos['min_w_prime_j'], hold_pos['w_prime_at_start_j']),
            'cost_j': max(0.0, hold_pos['extra_kj'] * 1000),
            'text': f"Hold your spot near {_ordinal(hold_pos['pos'])} instead of sliding back. "
                    f"The {hold_pos['riders']} riders who stayed on from there did "
                    f"{_work_vs_you(hold_pos['extra_kj'], 'you')} before the gap opened, then rode the surge at the equivalent of "
                    f"{hold_pos['selection_w'] / weight:.1f} W/kg for you. "
                    f"{_reserve_outcome(hold_pos['min_w_prime_j'])} — {_fit_count(hold_pos, short=True)}."})
    if gc_ok and gc['hold'] is not None and _unrealistic(gc['hold']['power_w'], cp):
        h = gc['hold']
        options.append({'status': 'no', 'cost_j': h['w_prime_cost_j'],
                        'text': f"Stay glued to the last wheel (from {at(h['start_sec'])}): it would have taken a "
                                f"full sprint ({h['power_w']:.0f} W for {h['duration_sec']}s) — not realistic."})
    elif gc_ok and gc['hold'] is not None:
        h = gc['hold']
        options.append({
            'status': status(h['w_prime_low_j'], h['w_prime_left_j']),
            'cost_j': h['w_prime_cost_j'],
            'text': f"Stay glued to the last wheel through the surge (from {at(h['start_sec'])}): "
                    f"{h['power_w']:.0f} W ({wkg(h['power_w'])}) for {h['duration_sec']}s{stay} — "
                    f"{uses(h['w_prime_cost_j'], h['w_prime_low_j'], h['w_prime_left_j'])}."})
    if ho is not None and ho['eased'] and ho['hang_sec'] > 0:
        low = ho['low_j']
        if low < 0:
            outcome = f"that's {kj(-low)} more than your reserve could cover"
        elif ho['cost_j'] < 500:
            outcome = "it would have cost next to nothing, since they weren't above your CP for long"
        else:
            outcome = f"uses {kj(ho['cost_j'])} of the {kj(ho['reserve_at_split_j'])} reserve you had"
        options.append({
            'status': status(low, ho['reserve_at_split_j']),
            'cost_j': ho['cost_j'],
            'text': f"Hang on {_dur(ho['hang_sec'])} longer: stay with the group until it eased at "
                    f"{at(ho['ease_sec'])} (their pace peaked at the equivalent of {wkg(ho['peak_w'])} for you) "
                    f"— {outcome}."})
    trusted = [r for r in ev['position_earlier'] if r['model_error_m'] <= GAP_MODEL_MAX_ERROR_M]
    if trusted:
        r = max(trusted, key=lambda r: r['min_w_prime_j'])
        options.append({
            'status': status(r['min_w_prime_j'], r['w_prime_at_start_j']),
            'cost_j': max(0.0, r['extra_kj'] * 1000),
            'text': f"Move up to {_ordinal(r['to_pos'])} before the surge ({at(r['move_start_sec'])}): "
                    f"{r['move_power_w']:.0f} W ({wkg(r['move_power_w'])}) for {r['move_time_sec']}s, then ride it like the riders "
                    f"there, {reserve_left(r['min_w_prime_j'])}."})
    rank = {'ok': 0, 'borderline': 1, 'no': 2}
    options.sort(key=lambda o: (rank[o['status']], o['cost_j']))
    for o in options:
        del o['cost_j']
    return {'summary': summary, 'story': story, 'options': options, 'advice': advice}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _clean(obj):
    """Make numpy/NaN values JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _repair_summary(grid: RaceGrid) -> List[Dict]:
    """Distance glitches fixed in this race, merged into time windows for display."""
    fixed = sorted((r for r in (grid.repairs or []) if r['end_k'] is not None), key=lambda r: r['start_k'])
    windows = []
    for r in fixed:
        if windows and r['start_k'] <= windows[-1]['end_k'] + 60:
            w = windows[-1]
            w['end_k'] = max(w['end_k'], r['end_k'])
            w['riders'].add(r['rank'])
        else:
            windows.append({'start_k': r['start_k'], 'end_k': r['end_k'], 'riders': {r['rank']}})
    return [{'start_sec': float(grid.times[w['start_k']]), 'end_sec': float(grid.times[w['end_k']]),
             'riders': len(w['riders'])} for w in windows]


def _nearby_mean(grid: RaceGrid, ri: int, values: np.ndarray, same: np.ndarray) -> np.ndarray:
    """Per-second mean of ``values`` over the group-mates nearest on the road.

    The whole group's average is skewed by whoever is pulling on the front;
    the riders right around you are the fair comparison.
    """
    out = np.full(values.shape[1], np.nan)
    d_ri = grid.dist_m[ri]
    for k in np.flatnonzero(same.any(axis=0)):
        members = np.flatnonzero(same[:, k])
        near = members[np.argsort(np.abs(grid.dist_m[members, k] - d_ri[k]))[:NEARBY_RIDERS]]
        out[k] = _quiet(np.nanmean, values[near, k])
    return out


def analyze_rider(grid: RaceGrid, race_data, rank: int,
                  cp: Optional[float] = None, w_prime: Optional[float] = None) -> Dict:
    """Full analysis for one rider: events, findings and a timeline."""
    if rank not in grid.ranks:
        raise ValueError(f"Rider rank {rank} not in race")
    ri = grid.ranks.index(rank)
    present = np.isfinite(grid.dist_m[ri])

    if cp and w_prime:
        cp_model = {'cp': float(cp), 'w_prime': float(w_prime), 'source': 'user'}
    else:
        cp_model = estimate_cp(grid.power[ri, present])
        cp_model['cp'], cp_model['w_prime'] = fit_cp_to_race(
            grid.power[ri], cp_model['cp'], cp_model['w_prime'])
    cp_v, wp_v = cp_model['cp'], cp_model['w_prime']

    wbal = w_prime_balance(grid.power[ri], cp_v, wp_v)

    # Group-mates each second, their power in the rider's body, and the draft
    # the group as a whole was getting.
    N, T = grid.dist_m.shape
    same = (grid.group == grid.group[ri][None, :]) & (grid.group >= 0)
    same[ri] = False
    eq_all = equivalent_power(grid, ri, np.arange(N))
    nearby_eq = _nearby_mean(grid, ri, eq_all, same)
    mates_draft = _quiet(np.nanmedian, np.where(same, grid.draft_pct, np.nan), axis=0)

    exposures = detect_exposures(grid, ri, mates_draft, nearby_eq)
    events = [
        _analyse_event(grid, ri, ev, cp_v, wp_v, wbal, nearby_eq, exposures,
                       race_data.elevation_profile)
        for ev in detect_separations(grid, ri)
    ]
    efforts = detect_extra_efforts(grid, ri, cp_v, wp_v, wbal, nearby_eq, exposures, events)
    sustained = detect_sustained_extra(grid, ri, cp_v, wp_v, wbal, nearby_eq, mates_draft, events)
    good_moves = detect_good_moves(grid, ri, wbal, race_data.elevation_profile)
    # Totals over the union of all flagged time, so overlaps aren't counted twice.
    flagged = np.zeros(len(grid.times), dtype=bool)
    for e in efforts + sustained:
        flagged[int(e['start_sec'] - grid.times[0]):int(e['end_sec'] - grid.times[0]) + 1] = True
    flagged_extra_kj = float(np.nansum((grid.power[ri] - nearby_eq)[flagged]) / 1000.0)
    efforts = sorted(efforts + sustained, key=lambda e: e['start_sec'])

    # Short-on-power drops: point at where the energy went beforehand.
    for ev in events:
        if ev['kind'] != 'drop' or not ev['summary'].startswith('You were short on power'):
            continue
        k_ev = int(ev['time_sec'] - grid.times[0])
        before = flagged.copy()
        before[k_ev:] = False
        spent = float(np.nansum((grid.power[ri] - nearby_eq)[before]) / 1000.0)
        if spent >= EFFORT_MIN_EXTRA_KJ:
            ev['advice'].append(
                f"Save it for when it matters: before this you spent about {spent:.0f} kJ more than the "
                f"riders around you (see \"Where you spent extra\"). The reserve model says you recovered "
                f"from each effort, but that much extra work is real fatigue it doesn't capture — cutting it "
                f"is your best bet for having more left at the end.")
    worst_exposures = sorted(
        (e for e in exposures if e['group_size'] >= TOP_EXPOSURE_MIN_GROUP),
        key=lambda e: e['extra_kj'], reverse=True)[:TOP_EXPOSURES]
    worst_exposures.sort(key=lambda e: e['start_sec'])
    key = next((i for i, e in enumerate(events) if e['kind'] == 'drop'), None)

    # Timeline (only while the rider is racing).
    idx = np.flatnonzero(present)
    power_30 = _rolling_mean(grid.power[ri], 30, center=False)
    gap_ahead = grid.gap_ahead[ri].copy()
    timeline = {
        'time_sec': grid.times[idx].tolist(),
        'group_size': grid.group_size[ri, idx].tolist(),
        'pos_in_group': grid.pos_in_group[ri, idx].tolist(),
        'gap_ahead_sec': np.round(gap_ahead[idx], 2).tolist(),
        'w_prime_bal_j': np.round(wbal[idx]).tolist(),
        'power_30s': np.round(power_30[idx], 1).tolist(),
        'nearby_eq_30s': np.round(_rolling_mean(nearby_eq, 30, center=False)[idx], 1).tolist(),
        'draft_pct': np.round(grid.draft_pct[ri, idx], 1).tolist(),
        'groupmate_draft_pct': np.round(mates_draft[idx], 1).tolist(),
    }

    return _clean({
        'rank': rank,
        'name': grid.names[ri],
        'weight_kg': grid.weight_kg[ri],
        'gap_sec': grid.gap_sec,
        'cp_model': cp_model,
        'events': events,
        'key_event_index': key,
        'draft_losses': [_public_exposure(e) for e in worst_exposures],
        'data_repairs': _repair_summary(grid),
        'extra_efforts': efforts,
        'good_moves': good_moves,
        'extra_summary': {
            'count': sum(1 for e in efforts if e['kind'] != 'sustained'),
            'sustained_count': len(sustained),
            'extra_kj': flagged_extra_kj,
            'reserve_cost_j': float(sum(e['reserve_cost_j'] for e in efforts if e['kind'] != 'sustained')),
        },
        'timeline': timeline,
    })
