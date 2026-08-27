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

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from bike_comparison.physics import (
    speed_from_power,
    AIR_DENSITY,
    GRAVITY,
    DRIVETRAIN_LOSS,
)
from shared import surface_map
from shared.world_config import MAP_TO_NAME, MAP_TO_WORLD_ID
from shared.route_lookup import load_route_cache
from race_replay.data_cleaner import fetch_route_from_zwiftmap, ROUTE_STRAVA_SEGMENTS

ROUTE_DIR = Path(__file__).parent.parent / "zwiftmap_surfaces"
ZWIFT_ROUTES_DIR = Path(__file__).parent.parent / "zwift_routes"

V_FLOOR = 0.3          # m/s — numerical floor so the rider never fully stalls
V_FLOOR2 = V_FLOOR * V_FLOOR


# ===========================================================================
# Route loading — real WAD / ZwiftMap geometry (route selector + profiles)
# ===========================================================================

def _route_name_to_slug(name: str) -> str:
    """Convert a route name to its ZwiftMap slug (matches race-replay convention)."""
    return name.lower().replace(" ", "-").replace("'", "")


def _cached_route_file(slug: str) -> Path:
    """Path to the locally-cached geometry file for a route slug."""
    return ROUTE_DIR / f'{slug.replace("-", "_")}_route.json'


def route_has_profile(
    route_name: str, world: Optional[str] = None, route_id: Optional[str] = None
) -> bool:
    """True if real elevation geometry is available.

    Prefers the WAD ``zwift_routes`` geometry (same source as the surface map),
    and falls back to ZwiftMap geometry (cached or fetchable).
    """
    if _find_wad_route(route_name, world, route_id) is not None:
        return True
    slug = _route_name_to_slug(route_name)
    return _cached_route_file(slug).exists() or slug in ROUTE_STRAVA_SEGMENTS


def _normalize_world(world: str) -> str:
    """Collapse a world name to lowercase alphanumerics for loose matching."""
    return re.sub(r"[^a-z0-9]", "", (world or "").lower())


def _world_to_map_id(world: Optional[str]) -> Optional[int]:
    """Resolve a routes_cache world name to a surface_map mapID."""
    if not world:
        return None
    key = _normalize_world(world)
    for map_name, map_id in MAP_TO_WORLD_ID.items():
        if _normalize_world(map_name) == key:
            return map_id
    return None


@lru_cache(maxsize=1)
def _wad_route_index() -> dict:
    """Map a casefolded route name to its ``zwift_routes/index.json`` entries."""
    path = ZWIFT_ROUTES_DIR / "index.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    by_name: dict[str, list] = {}
    for entry in entries:
        name = str(entry.get("name", "")).strip().casefold()
        if name:
            by_name.setdefault(name, []).append(entry)
    return by_name


@lru_cache(maxsize=1)
def _wad_route_by_hash() -> dict:
    """Map a route's ``nameHash`` (as str) to its ``index.json`` entry.

    Resolves routes whose display name differs from the WAD name
    (e.g. "Watopia Hilly Route" -> "Hilly Route").
    """
    path = ZWIFT_ROUTES_DIR / "index.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(e["nameHash"]): e for e in entries if e.get("nameHash") is not None}


def _find_wad_route(
    route_name: str, world: Optional[str] = None, route_id: Optional[str] = None
) -> Optional[dict]:
    """Find the WAD route index entry, by nameHash (route_id) then by name."""
    if route_id:
        entry = _wad_route_by_hash().get(str(route_id).strip())
        if entry is not None:
            return entry
    entries = _wad_route_index().get((route_name or "").strip().casefold())
    if not entries:
        return None
    if len(entries) > 1 and world:
        map_id = _world_to_map_id(world)
        if map_id is not None:
            for entry in entries:
                if entry.get("mapID") == map_id:
                    return entry
    return entries[0]


def route_is_loop(
    route_name: str, world: Optional[str] = None, route_id: Optional[str] = None
) -> bool:
    """True if the route is a loop (start/end coincide) and can be lapped.

    Only WAD ``zwift_routes`` geometry carries the coordinates needed to detect
    a loop; routes without WAD data are treated as non-loops.
    """
    entry = _find_wad_route(route_name, world, route_id)
    if entry is None:
        return False
    try:
        return surface_map.route_is_loop(entry["mapID"], entry["nameHash"])
    except (KeyError, OSError, ValueError):
        return False


def _load_wad_profile(
    route_name: str, world: Optional[str], include_leadin: bool,
    route_id: Optional[str] = None, laps: int = 1,
) -> Optional[dict]:
    """Build a route profile from WAD ``zwift_routes`` geometry.

    Returns distance/altitude/surface arrays identical to what the surface map
    page renders, optionally prepending the lead-in leg. Returns ``None`` when
    no WAD geometry is available for the route.
    """
    entry = _find_wad_route(route_name, world, route_id)
    if entry is None:
        return None
    data = surface_map.get_route(entry["mapID"], entry["nameHash"])
    if data is None:
        return None
    main = data.get("route")
    if not main or not main.get("d"):
        return None

    distance = np.asarray(main["d"], dtype=float)
    altitude = np.asarray(main["alt"], dtype=float)
    surfaces = np.asarray(main["surface"], dtype=object)
    source_ascent_m = float(data.get("ascent_m") or 0.0)

    # Repeat the main (non-lead-in) leg for multi-lap plans on looped routes.
    n_laps = max(1, int(laps or 1))
    if n_laps > 1 and len(distance) >= 2:
        lap_len = float(distance[-1] - distance[0])
        rel = distance - distance[0]  # 0 .. lap_len
        d_parts = [distance]
        alt_parts = [altitude]
        surf_parts = [surfaces]
        for k in range(1, n_laps):
            # Drop each lap's first point (a duplicate of the previous lap's
            # end) so the concatenated axis is strictly increasing.
            d_parts.append(distance[0] + rel[1:] + lap_len * k)
            alt_parts.append(altitude[1:])
            surf_parts.append(surfaces[1:])
        distance = np.concatenate(d_parts)
        altitude = np.concatenate(alt_parts)
        surfaces = np.concatenate(surf_parts)
        source_ascent_m *= n_laps

    leadin = data.get("leadin")
    if include_leadin and leadin and leadin.get("d"):
        # Offset by the lead-in's own geometry length (its last d), so the join
        # is seamless now that d is summed from geometry rather than the header.
        leadin_len = float(leadin["d"][-1])
        # Offset the route leg so it follows the lead-in on a single axis
        # (mirrors the surface map, which draws the route at +leadin_distance_m).
        distance = np.concatenate([
            np.asarray(leadin["d"], dtype=float),
            distance + leadin_len,
        ])
        altitude = np.concatenate([
            np.asarray(leadin["alt"], dtype=float),
            altitude,
        ])
        surfaces = np.concatenate([
            np.asarray(leadin["surface"], dtype=object),
            surfaces,
        ])
        source_ascent_m += float(data.get("leadin_ascent_m") or 0.0)

    # WAD vertical geometry is not to physical scale (e.g. Watopia altitudes read
    # ~2x true metres); horizontal distance is. Anchor the altitude to Zwift's
    # authoritative ascent by scaling about the start point so gradients — and
    # thus simulated times — are physical. World-agnostic: a no-op where the
    # geometry already matches the header ascent.
    if source_ascent_m and len(altitude) > 1:
        dalt = np.diff(altitude)
        raw_ascent = float(np.sum(dalt[dalt > 0]))
        if raw_ascent > 0:
            altitude = altitude[0] + (altitude - altitude[0]) * (source_ascent_m / raw_ascent)

    # Display distance follows the physical geometry (matches the in-game
    # odometer), not Zwift's inflated header figure.
    source_distance_m = float(distance[-1]) if len(distance) else None

    return {
        "distance_m": distance,
        "altitude_m": altitude,
        "surfaces": surfaces,
        "source_distance_m": source_distance_m or None,
        "source_ascent_m": source_ascent_m or None,
    }


def _load_route_geometry(slug: str) -> Optional[dict]:
    """
    Return raw route geometry (latlng/distance/altitude) for a slug.

    Loads from the local cache when present, otherwise fetches from ZwiftMap
    and caches the result. Returns None when no geometry is available.
    """
    path = _cached_route_file(slug)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    strava_id = ROUTE_STRAVA_SEGMENTS.get(slug)
    if not strava_id:
        return None

    data = fetch_route_from_zwiftmap(strava_id, slug)
    if data:
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Caching is best-effort; still return the fetched data
    return data


@dataclass
class RouteProfile:
    """Elevation and distance profile for a route (real ZwiftMap geometry)."""
    name: str
    distance_m: np.ndarray      # Cumulative distance in metres
    altitude_m: np.ndarray      # Altitude in metres
    lats: Optional[np.ndarray] = None
    lngs: Optional[np.ndarray] = None
    # Per-point surface tags (from WAD geometry) used for surface-aware CRR.
    surfaces: Optional[np.ndarray] = None
    world: Optional[str] = None
    # Authoritative totals from the WAD route index (Zwift's own figures).
    # Used for display so the reported stats match Zwift Insider exactly,
    # rather than re-summing the sampled ZwiftMap geometry.
    source_distance_m: Optional[float] = None
    source_ascent_m: Optional[float] = None

    @property
    def total_distance_km(self) -> float:
        return float(self.distance_m[-1]) / 1000.0

    @property
    def total_ascent_m(self) -> float:
        dalt = np.diff(self.altitude_m)
        return float(np.sum(dalt[dalt > 0]))

    @property
    def display_distance_km(self) -> float:
        """Authoritative distance if known, else computed from geometry."""
        if self.source_distance_m is not None:
            return self.source_distance_m / 1000.0
        return self.total_distance_km

    @property
    def display_ascent_m(self) -> float:
        """Authoritative ascent if known, else computed from geometry."""
        if self.source_ascent_m is not None:
            return self.source_ascent_m
        return self.total_ascent_m


def _load_routes_cache() -> dict:
    return load_route_cache() or {}


def list_routes() -> list[dict]:
    """
    Return rideable routes that have real elevation geometry available.

    Routes without ZwiftMap geometry (no cached file and no known Strava
    segment) are excluded entirely — the planner never fabricates a profile.
    """
    cache = _load_routes_cache()
    routes = []
    for route_id, info in cache.items():
        name = info.get("name", "")
        dist_m = info.get("distanceInMeters", 0)
        ascent_m = info.get("ascentInMeters", 0)
        world = info.get("map", "")

        # Skip unnamed, implausibly-huge, or zero-distance routes. Event-only
        # routes are included so they can be planned for event time estimates.
        if not name or not world:
            continue
        if dist_m <= 0 or dist_m > 200_000:
            continue

        # Only include routes with real elevation data available
        wad = _find_wad_route(name, world, route_id)
        if wad is None and not route_has_profile(name, world, route_id):
            continue
        # The WAD index is the authoritative name source; prefer it when the
        # routes_cache name differs (e.g. "Watopia Hilly Route" -> "Hilly Route").
        if wad is not None and wad.get("name"):
            name = wad["name"]

        leadin_dist_m = info.get("leadinDistanceInMeters", 0) or 0
        leadin_ascent_m = info.get("leadinAscentInMeters", 0) or 0
        routes.append({
            "id": route_id,
            "name": name,
            "world": world,
            "world_name": MAP_TO_NAME.get(world, world),
            "distance_km": round(dist_m / 1000, 1),
            "ascent_m": round(ascent_m),
            "leadin_distance_km": round(leadin_dist_m / 1000, 1),
            "leadin_ascent_m": round(leadin_ascent_m),
            "is_loop": route_is_loop(name, world, route_id),
        })

    return sorted(routes, key=lambda r: (r["world"], r["name"]))


def load_route_profile(
    route_id: str,
    route_name: str,
    world: Optional[str] = None,
    include_leadin: bool = True,
    laps: int = 1,
) -> RouteProfile:
    """
    Load the real elevation profile for a route.

    Prefers the WAD ``zwift_routes`` geometry (identical to the surface map
    page, including the lead-in), and falls back to ZwiftMap geometry when a
    route has no WAD data. Raises ValueError when no real geometry is available
    — the planner never synthesises an approximate profile.

    Args:
        route_id: Route ID, used to look up authoritative totals; also kept for
            API symmetry/logging.
        route_name: Route name, used for WAD/ZwiftMap geometry lookup.
        world: World name (e.g. 'WATOPIA') used for surface-aware CRR lookup and
            to disambiguate WAD routes that share a name.
        include_leadin: Include the route's lead-in leg in the profile (WAD
            geometry only).
        laps: Number of laps to ride. For looped routes the main (non-lead-in)
            leg is repeated this many times; the lead-in is ridden once.
    """
    wad = _load_wad_profile(route_name, world, include_leadin, route_id, laps)
    if wad is not None:
        return RouteProfile(
            name=route_name,
            distance_m=wad["distance_m"],
            altitude_m=wad["altitude_m"],
            surfaces=wad["surfaces"],
            world=world,
            source_distance_m=wad["source_distance_m"],
            source_ascent_m=wad["source_ascent_m"],
        )

    slug = _route_name_to_slug(route_name)
    data = _load_route_geometry(slug)
    if data is None:
        raise ValueError(f"No elevation data available for route '{route_name}'")

    distance_arr = np.array(data["distance"], dtype=float)
    altitude_arr = np.array(data["altitude"], dtype=float)
    latlng = data.get("latlng", [])
    lats = np.array([p[0] for p in latlng]) if latlng else None
    lngs = np.array([p[1] for p in latlng]) if latlng else None

    # Authoritative distance/ascent from the WAD route index (Zwift's figures),
    # looked up by route_id, then by name as a fallback.
    source_distance_m = None
    source_ascent_m = None
    cache = _load_routes_cache()
    info = cache.get(route_id)
    if info is None:
        info = next((v for v in cache.values() if v.get("name") == route_name), None)
    if info is not None:
        if info.get("distanceInMeters"):
            source_distance_m = float(info["distanceInMeters"])
        if info.get("ascentInMeters"):
            source_ascent_m = float(info["ascentInMeters"])

    return RouteProfile(
        name=route_name,
        distance_m=distance_arr,
        altitude_m=altitude_arr,
        lats=lats,
        lngs=lngs,
        world=world,
        source_distance_m=source_distance_m,
        source_ascent_m=source_ascent_m,
    )


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

    # Bucketing: the largest useful bucket count and the chosen divider distances.
    max_buckets: int = 0
    dividers_km: list = field(default_factory=list)

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


MAX_BUCKETS_CAP = 40       # largest bucket count the slider ever offers


def _optimal_segmentation(values, weights, k):
    """Best ``k``-segment piecewise-constant fit of a 1-D series.

    Returns the internal split indices (``k-1`` of them) that partition
    ``values`` into ``k`` contiguous segments minimising the total
    weight-weighted squared error ``Σ_seg Σ_i w_i·(v_i − v̄_seg)²`` — i.e. the
    best step-function approximation, where each step is the weighted mean of the
    points under it.  Solved exactly by dynamic programming in ``O(k·n²)``:
    prefix sums make every segment's error ``O(1)`` and the inner search over the
    previous cut is vectorised over candidate positions.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(v)
    k = max(1, min(k, n))
    if k <= 1:
        return []

    # Prefix sums for constant-time weighted mean / SSE of any segment [a, b).
    W = np.concatenate(([0.0], np.cumsum(w)))
    WV = np.concatenate(([0.0], np.cumsum(w * v)))
    WV2 = np.concatenate(([0.0], np.cumsum(w * v * v)))

    def sse(a, b):
        sw = W[b] - W[a]
        swv = WV[b] - WV[a]
        return (WV2[b] - WV2[a]) - (swv * swv / sw if sw > 0 else 0.0)

    # dp_prev[i] = min cost to split the first i points into (s-1) segments.
    dp_prev = np.array([0.0 if i == 0 else sse(0, i) for i in range(n + 1)])
    back = np.zeros((k + 1, n + 1), dtype=int)

    for s in range(2, k + 1):
        dp_cur = np.full(n + 1, np.inf)
        for i in range(s, n + 1):
            j = np.arange(s - 1, i)
            sw = W[i] - W[j]
            swv = WV[i] - WV[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                seg = (WV2[i] - WV2[j]) - np.where(sw > 0, swv * swv / sw, 0.0)
            tot = dp_prev[j] + seg
            m = int(np.argmin(tot))
            dp_cur[i] = tot[m]
            back[s][i] = int(j[m])
        dp_prev = dp_cur

    bounds = []
    i = n
    for s in range(k, 1, -1):
        j = int(back[s][i])
        bounds.append(j)
        i = j
    bounds.sort()
    return bounds


def _optimal_bucket_edges(end_dist, power, time_arr, k, max_groups=240):
    """Divider distances (m) for the best ``k``-bucket step fit of ``power``.

    Aggregates the (possibly thousands of) chunks into ``≤max_groups`` contiguous
    groups so the segmentation DP stays fast, fits the optimal power series with a
    ``k``-step function (time-weighted, since NP is time-based), then maps the
    chosen group boundaries back to course distances.
    """
    n = len(power)
    k = max(1, min(k, n))
    if k <= 1:
        return []

    m_groups = min(n, max_groups)
    cut = np.linspace(0, n, m_groups + 1).astype(int)
    vals, wts, g_end = [], [], []
    for g in range(m_groups):
        a, b = int(cut[g]), int(cut[g + 1])
        if b <= a:
            continue
        tw = float(np.sum(time_arr[a:b]))
        val = (float(np.sum(power[a:b] * time_arr[a:b]) / tw)
               if tw > 0 else float(np.mean(power[a:b])))
        vals.append(val)
        wts.append(max(tw, 1e-9))
        g_end.append(float(end_dist[b - 1]))

    g_end = np.asarray(g_end)
    k = min(k, len(vals))
    if k <= 1:
        return []
    splits = _optimal_segmentation(np.asarray(vals), np.asarray(wts), k)
    return [float(g_end[j - 1]) for j in splits]


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
    bucket_edges_m: list | None = None,
    num_buckets: int | None = None,
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
        bucket_edges_m: Optional sorted internal divider distances (m).  When
            given, the finely-varying optimal plan is collapsed into constant-
            power "buckets" between consecutive dividers — a coarse, simpler
            plan — while still matching the normalized-power target overall.
        num_buckets: Optional number of "smart" buckets.  The dividers are placed
            automatically (DP-optimal step-fit of the optimal power curve); takes
            precedence over ``bucket_edges_m``.
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

    # ── Optional: collapse the plan into constant-power buckets ───────────
    # A coarse alternative to the finely-varying optimal plan: hold *one* constant
    # power per bucket.  "Smart buckets" place the dividers automatically — the
    # best k-step approximation of the optimal power curve (segmentation DP) — or
    # explicit dividers may be supplied.  Each bucket is seeded with the time-
    # weighted mean of the optimal power over its chunks (so climbing buckets keep
    # more effort), then every bucket is scaled by a single factor — tuned by
    # bisection — so the realised NP still equals the target.
    bucket_sections = None
    dividers_km_out: list = []
    max_buckets = int(min(MAX_BUCKETS_CAP, n))

    if num_buckets is not None:
        k = max(1, min(int(num_buckets), max_buckets))
        edges_source = _optimal_bucket_edges(end_dist, power, time_arr, k)
        do_bucket = True
    elif bucket_edges_m is not None:
        edges_source = list(bucket_edges_m)
        do_bucket = True
    else:
        edges_source = []
        do_bucket = False

    if do_bucket:
        total_m = float(end_dist[-1])
        edges = sorted({round(float(e), 3) for e in edges_source
                        if 0.0 < float(e) < total_m})
        boundaries = np.array([0.0] + edges + [total_m], dtype=float)
        b_idx = np.clip(np.searchsorted(boundaries, mid_dist, side="right") - 1,
                        0, len(boundaries) - 2)

        bucket_power = np.empty_like(power)
        for b in range(len(boundaries) - 1):
            m = b_idx == b
            if not np.any(m):
                continue
            tw = float(np.sum(time_arr[m]))
            bucket_power[m] = (float(np.sum(power[m] * time_arr[m])) / tw
                               if tw > 0 else float(np.mean(power[m])))

        def np_for_scale(s):
            pw = bucket_power * s
            forward_sim(pw)
            return _normalized_power(pw, time_arr), pw

        np_base, _ = np_for_scale(1.0)
        if np_base > 0.0:
            # NP rises monotonically with the scale, so bracket then bisect.
            s_lo, s_hi = 0.1, 1.0
            np_hi, _ = np_for_scale(s_hi)
            expand = 0
            while np_hi < power_target_w and expand < 40:
                s_hi *= 1.5
                np_hi, _ = np_for_scale(s_hi)
                expand += 1
            np_low, _ = np_for_scale(s_lo)
            while np_low > power_target_w and expand < 80:
                s_lo *= 0.5
                np_low, _ = np_for_scale(s_lo)
                expand += 1
            for _ in range(50):
                s_mid = 0.5 * (s_lo + s_hi)
                np_mid, _ = np_for_scale(s_mid)
                if np_mid < power_target_w:
                    s_lo = s_mid
                else:
                    s_hi = s_mid
            power = bucket_power * (0.5 * (s_lo + s_hi))
        else:
            power = bucket_power
        forward_sim(power)

        # One table row per bucket (power is constant within each).
        bucket_sections = []
        for b in range(len(boundaries) - 1):
            m = b_idx == b
            if not np.any(m):
                continue
            idxs = np.nonzero(m)[0]
            sl = length[m]
            seg_len = float(np.sum(sl))
            if seg_len <= 0:
                continue
            seg_time = float(np.sum(time_arr[m]))
            avg_grad = float(np.sum(grad[m] * sl) / seg_len)
            start_dist = float(end_dist[idxs[0]] - length[idxs[0]])
            bucket_sections.append({
                "start_km": round(start_dist / 1000.0, 2),
                "end_km": round(float(end_dist[idxs[-1]]) / 1000.0, 2),
                "distance_m": round(seg_len, 1),
                "avg_gradient_pct": round(avg_grad * 100.0, 1),
                "power_w": round(float(power[idxs[0]])),
                "time_seconds": round(seg_time, 1),
                "avg_speed_kph": round((seg_len / seg_time) * 3.6, 1) if seg_time > 0 else 0.0,
            })
        dividers_km_out = [round(e / 1000.0, 3) for e in edges]

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
    if bucket_sections is not None:
        result.sections = bucket_sections
    else:
        result.sections = _build_sections(
            length, grad, power, times, end_dist, n_sections
        )
    result.max_buckets = max_buckets
    result.dividers_km = dividers_km_out
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
