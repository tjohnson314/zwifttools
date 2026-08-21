"""
Ride Simulator for Zwift routes.

Given a rider's biometric data, bike setup, and constant power output,
computes the simulated time to complete a route using the existing physics engine.

Routes use real per-point altitude/distance geometry from ZwiftMap. Any route
whose slug is present in ``route_strava_segments.json`` can be fetched on demand
(and is cached to zwiftmap_surfaces/*_route.json), reusing the same fetch/cache
pipeline as the race-replay tool. Routes without real geometry are excluded
entirely — the simulator never fabricates an approximate profile.
"""

from __future__ import annotations

import json
import re
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from bike_comparison.bike_data import BikeSetup
from bike_comparison.physics import (
    speed_from_power,
    frontal_area_from_rider,
    AIR_DENSITY,
    GRAVITY,
    DRIVETRAIN_LOSS,
)
from shared.surface_lookup import (
    compute_crr_array,
    get_bike_type_for_frame,
    surface_types_to_crr,
)
from shared import surface_map
from race_replay.data_cleaner import fetch_route_from_zwiftmap, ROUTE_STRAVA_SEGMENTS

ROUTE_DIR = Path(__file__).parent.parent / "zwiftmap_surfaces"
ROUTES_CACHE = Path(__file__).parent.parent / "routes_cache.json"
ZWIFT_ROUTES_DIR = Path(__file__).parent.parent / "zwift_routes"

# routes_cache.json world names that don't normalise cleanly to surface_map's
# WORLD_NAMES (e.g. the Bologna TT world).
_WORLD_ALIASES = {"bolognatt": 6}


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
    if key in _WORLD_ALIASES:
        return _WORLD_ALIASES[key]
    for map_id, name in surface_map.WORLD_NAMES.items():
        if _normalize_world(name) == key:
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

    The routes_cache.json key is the same nameHash, so this resolves routes
    whose cache name differs from the WAD name (e.g. "Watopia Hilly Route").
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
    source_distance_m = float(data.get("distance_m") or 0.0)
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
        source_distance_m *= n_laps
        source_ascent_m *= n_laps

    leadin = data.get("leadin")
    if include_leadin and leadin and leadin.get("d"):
        leadin_len = float(data.get("leadin_distance_m") or leadin["d"][-1])
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
        source_distance_m += float(data.get("leadin_distance_m") or 0.0)
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
    # Authoritative totals from routes_cache.json (Zwift's own figures).
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


@dataclass
class SimulationResult:
    """Results of a simulated ride."""
    route_name: str
    total_time_seconds: float
    total_distance_km: float
    total_ascent_m: float
    avg_speed_kph: float

    # Per-segment time series (downsampled for the API response)
    distance_km: list[float] = field(default_factory=list)
    altitude_m: list[float] = field(default_factory=list)
    speed_kph: list[float] = field(default_factory=list)
    gradient_pct: list[float] = field(default_factory=list)
    surfaces: list[str] = field(default_factory=list)
    surface_breakdown: list[dict] = field(default_factory=list)

    @property
    def total_time_formatted(self) -> str:
        t = int(round(self.total_time_seconds))
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


def _load_routes_cache() -> dict:
    if ROUTES_CACHE.exists():
        with open(ROUTES_CACHE) as f:
            return json.load(f)
    return {}


def list_routes() -> list[dict]:
    """
    Return rideable routes that have real elevation geometry available.

    Routes without ZwiftMap geometry (no cached file and no known Strava
    segment) are excluded entirely — the simulator never fabricates a profile.
    """
    cache = _load_routes_cache()
    routes = []
    for route_id, info in cache.items():
        name = info.get("name", "")
        dist_m = info.get("distanceInMeters", 0)
        ascent_m = info.get("ascentInMeters", 0)
        world = info.get("map", "")

        # Skip event-only, unnamed, implausibly-huge, or zero-distance routes
        if info.get("eventOnly", False):
            continue
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
    — the simulator never synthesises an approximate profile.

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

    # Authoritative distance/ascent from routes_cache.json (Zwift's figures),
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


def simulate_ride(
    route: RouteProfile,
    rider_weight_kg: float,
    rider_height_m: float,
    power_w: float,
    bike_setup: BikeSetup,
    default_crr: float = 0.004,
    downsample_points: int = 500,
) -> SimulationResult:
    """
    Simulate a complete Zwift ride and return timing and speed data.

    Integrates the rider's equation of motion forward in time using small
    fixed time steps, so the speed at any moment depends on the *previous*
    speed (momentum) rather than being the instantaneous steady-state speed
    for the local gradient.  This correctly captures a rider carrying speed
    over the crest of a hill, accelerating on descents, and bleeding that
    speed off on the following climb.

    At each time step the net force is

        F_net = F_drive - F_gravity - F_rolling - F_aero

    where ``F_drive = P·(1-η)/v`` is the propulsive force delivered to the
    wheel.  Acceleration ``a = F_net / m`` updates the speed
    (``v += a·dt``) and the speed advances the position (``x += v·dt``).
    The gradient and rolling resistance are looked up from the route at the
    rider's current position.

    Args:
        route: RouteProfile with distance and altitude arrays.
        rider_weight_kg: Rider mass in kg.
        rider_height_m: Rider height in metres (used for frontal area).
        power_w: Constant power output in watts.
        bike_setup: BikeSetup with Cd and weight from the bike database.
        default_crr: Rolling resistance coefficient (overridden per-point when
                     surface data is available).
        downsample_points: Max number of points returned in the profile arrays.

    Returns:
        SimulationResult with total time and per-segment data.
    """
    dist = route.distance_m
    alt = route.altitude_m
    n = len(dist)

    if n < 2:
        raise ValueError("Route profile must have at least 2 points")

    # Rider frontal area and CdA
    frontal_area = frontal_area_from_rider(rider_height_m, rider_weight_kg)
    cda = bike_setup.cd * frontal_area
    total_mass = rider_weight_kg + bike_setup.weight_kg

    # Per-point CRR: prefer WAD per-point surface tags (same surfaces the
    # surface map shows), then GPS surface polygons, else a constant default.
    bike_type = get_bike_type_for_frame(bike_setup.frame_type)
    if route.surfaces is not None and len(route.surfaces) == n:
        crr_arr = surface_types_to_crr(route.surfaces, bike_type)
    elif route.lats is not None and route.lngs is not None and route.world:
        crr_arr = compute_crr_array(route.lats, route.lngs, route.world, bike_type)
    else:
        crr_arr = np.full(n, default_crr)

    # Per-segment gradient and CRR (segment i spans node i -> node i+1)
    seg_dist = np.diff(dist)      # metres
    seg_alt  = np.diff(alt)       # metres elevation change
    seg_crr  = crr_arr[:-1]       # CRR at the start of each segment

    # Gradient (clamped to ±40% to avoid physics blow-ups on noisy data)
    with np.errstate(divide='ignore', invalid='ignore'):
        gradient = np.where(seg_dist > 0, seg_alt / seg_dist, 0.0)
    gradient = np.clip(gradient, -0.40, 0.40)

    total_dist_m = float(dist[-1])
    n_seg = len(gradient)

    # ── Forward time-stepping integration ─────────────────────────────────
    dt = 0.1                      # seconds per step
    v_floor = 0.5                 # m/s — floor so the rider never fully stalls
    drive_const = power_w * (1.0 - DRIVETRAIN_LOSS)
    max_time = 6 * 3600.0         # 6 h guard against runaway loops

    # Start at the steady-state speed for the first segment so the rider isn't
    # accelerating unrealistically from a dead stop; momentum evolves from there.
    v = speed_from_power(
        power_w, gradient[0], rider_weight_kg, bike_setup.weight_kg, cda, seg_crr[0]
    )
    v = max(v, v_floor)

    x = 0.0
    t = 0.0
    seg_idx = 0

    # Record speed at each route node (index into dist/alt) as the rider passes it.
    node_speed = np.empty(n)
    node_speed[0] = v
    next_node = 1

    while x < total_dist_m and t < max_time:
        # Advance the active segment pointer to match the current position.
        while seg_idx < n_seg - 1 and x >= dist[seg_idx + 1]:
            seg_idx += 1

        grad = gradient[seg_idx]
        crr = seg_crr[seg_idx]
        cos_slope = np.cos(np.arctan(grad))

        f_drive = drive_const / max(v, v_floor)
        f_gravity = total_mass * GRAVITY * grad
        f_rolling = crr * total_mass * GRAVITY * cos_slope
        f_aero = 0.5 * AIR_DENSITY * cda * v * v
        accel = (f_drive - f_gravity - f_rolling - f_aero) / total_mass

        v = max(v + accel * dt, v_floor)
        x += v * dt
        t += dt

        # Capture speed at every route node the rider has just passed.
        while next_node < n and x >= dist[next_node]:
            node_speed[next_node] = v
            next_node += 1

    # Fill any nodes not reached (e.g. hit the time guard) with the last speed.
    if next_node < n:
        node_speed[next_node:] = v

    # Back out the exact finish time by removing the final-step overshoot.
    overshoot = x - total_dist_m
    if v > 0 and overshoot > 0:
        t -= overshoot / v
    total_time = t

    # Report Zwift's authoritative totals for display; fall back to the
    # geometry-derived values when the route isn't in routes_cache.json.
    # Average speed is computed from the same displayed distance so the
    # results card stays internally consistent.
    display_dist_km = route.display_distance_km
    avg_speed_kph = (display_dist_km / (total_time / 3600)) if total_time > 0 else 0.0

    # Downsample for the response payload
    if n > downsample_points:
        idx = np.round(np.linspace(0, n - 1, downsample_points)).astype(int)
        idx = np.unique(idx)
    else:
        idx = np.arange(n)

    # Gradient is per-segment (length n-1); reuse the last segment value for the
    # final node so the array lines up with the node-indexed distance/altitude.
    grad_nodes = np.append(gradient, gradient[-1])

    # Per-surface distance breakdown + downsampled per-point surface tags, so
    # the elevation chart can colour each segment like the surface map page.
    surfaces_ds: list[str] = []
    breakdown_list: list[dict] = []
    if route.surfaces is not None and len(route.surfaces) == n:
        surfaces_ds = [str(route.surfaces[i]) for i in idx]
        seg_len_full = np.diff(dist)
        totals: dict[str, float] = {}
        for i in range(n - 1):
            seg_len = float(seg_len_full[i])
            if seg_len <= 0:
                continue
            surf = str(route.surfaces[i])
            totals[surf] = totals.get(surf, 0.0) + seg_len
        breakdown_list = sorted(
            ({"surface": s, "distance_m": round(d, 1),
              "color": surface_map.SURFACE_COLORS.get(
                  s, surface_map.SURFACE_COLORS["Unknown"])}
             for s, d in totals.items()),
            key=lambda e: -e["distance_m"],
        )

    return SimulationResult(
        route_name=route.name,
        total_time_seconds=total_time,
        total_distance_km=display_dist_km,
        total_ascent_m=route.display_ascent_m,
        avg_speed_kph=round(avg_speed_kph, 1),
        distance_km=[round(float(dist[i]) / 1000, 3) for i in idx],
        altitude_m=[round(float(alt[i]), 1) for i in idx],
        speed_kph=[round(float(node_speed[i]) * 3.6, 1) for i in idx],
        gradient_pct=[round(float(grad_nodes[i]) * 100, 1) for i in idx],
        surfaces=surfaces_ds,
        surface_breakdown=breakdown_list,
    )
