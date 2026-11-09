"""
Surface Map Module

Serves data for the Surface Map / Route Explorer page.

Both the road-surface geometry (``zwift_surfaces/world_*.json``) and the route
geometry (``zwift_routes/world_*.json``) are expressed in the same
``zwift_local_m`` coordinate frame (x/z metres extracted from the game WAD
files).

For worlds that have a stored GPS calibration (``world_gps_calibration.json``),
the local x/z frame is projected onto the world's map PNG so the network and
routes align with the background image (this also fixes map orientation). Worlds
without a calibration fall back to drawing the raw local frame with no image.

Output uses a single 2D "plot space" (y increasing downward) so the frontend can
render every world identically.
"""

import json
import math
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from shared.world_config import (
    WORLD_CONFIG,
    get_world_altitude_scale,
    WORLD_NAMES,
    MAPID_TO_CONFIG,
)

_BASE = Path(__file__).parent.parent
SURFACE_DIR = _BASE / "zwift_surfaces"
ROUTE_DIR = _BASE / "zwift_routes"
MAPS_DIR = _BASE / "static" / "maps"
CALIBRATION_FILE = SURFACE_DIR / "world_gps_calibration.json"
STYLE_MAP_FILE = SURFACE_DIR / "style_surface_map.json"
WAD_SEGMENTS_FILE = _BASE / "zwift_route_segments_wad.json"
ZRL_SEGMENTS_FILE = _BASE / "zrl_route_segments.json"

VIDEO_ROUTES_MAP_ID = 1000
_WORLD_NAMES = {**WORLD_NAMES, VIDEO_ROUTES_MAP_ID: "Video Routes"}

# mapID -> human readable world name (WORLD_NAMES) and mapID -> WORLD_CONFIG key
# (MAPID_TO_CONFIG) are imported from shared.world_config.

SURFACE_COLORS = {
    "Tarmac": "#7d828b",
    "Cobbles": "#9c7a63",
    "Brick": "#b5503c",
    "Dirt": "#9a6634",
    "Gravel": "#cbb079",
    "Wood": "#c07f3c",
    "Sand": "#ddcd91",
    "Grass": "#5f9a4d",
    "Unknown": "#555a63",
}

SURFACE_ORDER = [
    "Tarmac", "Cobbles", "Brick", "Dirt", "Gravel",
    "Wood", "Sand", "Grass", "Unknown",
]


@lru_cache(maxsize=1)
def _style_map() -> dict:
    """Authoritative style -> surface mapping (e.g. SNOW -> Tarmac)."""
    with open(STYLE_MAP_FILE, encoding="utf-8") as f:
        return json.load(f)


def _seg_surface(seg: dict) -> str:
    """Resolve a segment's display surface from its style via the style map."""
    return _style_map().get(seg.get("style"), "Unknown")


def _surface_path(map_id: int) -> Path:
    if map_id == VIDEO_ROUTES_MAP_ID:
        return SURFACE_DIR / "video_routes.json"
    return SURFACE_DIR / f"world_{map_id}.json"


def _route_path(map_id: int) -> Path:
    if map_id == VIDEO_ROUTES_MAP_ID:
        return ROUTE_DIR / "video_routes.json"
    return ROUTE_DIR / f"world_{map_id}.json"


@lru_cache(maxsize=1)
def _load_calibration() -> dict:
    if not CALIBRATION_FILE.exists():
        return {}
    with open(CALIBRATION_FILE, encoding="utf-8") as f:
        return json.load(f)


def _png_size(path: Path):
    """Return (width, height) of a PNG by reading its IHDR header."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


@lru_cache(maxsize=None)
def _projection(map_id: int):
    """Build the plot-space projection for a world.

    Returns a dict with:
      mode:       'image' or 'local'
      project:    callable(xs, zs) -> (np.ndarray X, np.ndarray Y) in plot space
      background: {image, width, height} or None
      bounds:     default plot-space bounds to fit
    """
    calib = _load_calibration().get(str(map_id))
    config = WORLD_CONFIG.get(MAPID_TO_CONFIG.get(map_id, ""))

    if calib and config:
        size = _png_size(MAPS_DIR / f"{config['slug']}.png")
        if size:
            width, height = size
            coef = np.asarray(calib["coef"], dtype=float)  # (3, 2): [lat, lng]
            lng_min, lng_max = config["lng_min"], config["lng_max"]
            lat_min, lat_max = config["lat_min"], config["lat_max"]

            def project(xs, zs):
                xs = np.asarray(xs, dtype=float)
                zs = np.asarray(zs, dtype=float)
                if len(xs) == 0:
                    return np.empty(0), np.empty(0)
                latlng = np.column_stack([xs, zs, np.ones(len(xs))]) @ coef
                lat, lng = latlng[:, 0], latlng[:, 1]
                px = (lng - lng_min) / (lng_max - lng_min) * width
                py = (lat_max - lat) / (lat_max - lat_min) * height
                return px, py

            return {
                "mode": "image",
                "project": project,
                "background": {"image": config["image"], "width": width, "height": height},
                "bounds": {"min_x": 0.0, "max_x": float(width),
                           "min_y": 0.0, "max_y": float(height)},
            }

    # Fallback: raw local frame with z flipped so north is up (y increases down).
    world = _load_surface_world(map_id)
    min_x, max_x, min_z, max_z = world["bounds"]

    def project(xs, zs):
        return np.asarray(xs, dtype=float), -np.asarray(zs, dtype=float)

    return {
        "mode": "local",
        "project": project,
        "background": None,
        "bounds": {"min_x": min_x, "max_x": max_x, "min_y": -max_z, "max_y": -min_z},
    }


@lru_cache(maxsize=None)
def _load_surface_world(map_id: int):
    """Load a world's surface segments and build a KD-tree for surface lookups."""
    with open(_surface_path(map_id), encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    all_x: list[float] = []
    all_z: list[float] = []
    labels: list[str] = []
    for seg in segments:
        xs = seg.get("x", [])
        zs = seg.get("z", [])
        surface = _seg_surface(seg)
        all_x.extend(xs)
        all_z.extend(zs)
        labels.extend([surface] * len(xs))

    if all_x:
        tree = cKDTree(np.column_stack([all_x, all_z]))
        bounds = (float(min(all_x)), float(max(all_x)),
                  float(min(all_z)), float(max(all_z)))
    else:
        tree = None
        bounds = (0.0, 0.0, 0.0, 0.0)

    return {"segments": segments, "bounds": bounds,
            "tree": tree, "labels": np.array(labels)}


@lru_cache(maxsize=None)
def _load_route_world(map_id: int):
    with open(_route_path(map_id), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_wad_segments() -> dict:
    if not WAD_SEGMENTS_FILE.exists():
        return {}
    with open(WAD_SEGMENTS_FILE, encoding="utf-8") as f:
        return json.load(f).get("routes", {})


@lru_cache(maxsize=1)
def _load_segment_names() -> dict[int, str]:
    if not ZRL_SEGMENTS_FILE.exists():
        return {}
    with open(ZRL_SEGMENTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for route in data.values():
        for segment in route.get("segments", []):
            names[int(segment["wad_hash"])] = segment["name"]
    return names


def _path_slice(points, start_fraction, end_fraction, positions=None):
    """Return the route polyline between normalized WAD checkpoint positions."""
    if len(points) < 2:
        return []
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions if positions is not None else np.arange(len(points)), dtype=float)
    start = start_fraction * positions[-1]
    end = end_fraction * positions[-1]
    interior = points[(positions > start) & (positions < end)]
    samples = [
        [float(np.interp(position, positions, points[:, i])) for i in range(2)]
        for position in (start, end)
    ]
    return [samples[0], *interior.tolist(), samples[1]]


def _route_path_slice(route, leadin, main, start_fraction, end_fraction):
    """Slice packed route legs using the original WAD checkpoint weighting."""
    segment_path = route.get("segment_path")
    if segment_path and segment_path.get("x") and segment_path.get("z"):
        points = list(zip(segment_path["x"], segment_path["z"]))
        return _path_slice(points, start_fraction, end_fraction, segment_path.get("d"))
    lead_points = list(zip(leadin.get("local_x", []), leadin.get("local_z", []))) if leadin else []
    main_points = list(zip(main.get("local_x", []), main.get("local_z", []))) if main else []
    lead_count = max(int(route.get("leadin_checkpoint_count", len(lead_points))), 1)
    main_count = max(int(route.get("route_checkpoint_count", len(main_points))), 1)
    total_steps = max(lead_count + main_count - 1, 1)
    start = start_fraction * total_steps
    end = end_fraction * total_steps
    lead_last = lead_count - 1
    main_start = lead_count
    chunks = []
    if lead_points and start <= lead_last:
        chunks.append(_path_slice(
            lead_points,
            max(start, 0) / max(lead_last, 1),
            min(end, lead_last) / max(lead_last, 1),
        ))
    if main_points and end >= main_start:
        chunks.append(_path_slice(
            main_points,
            max(start - main_start, 0) / max(main_count - 1, 1),
            min(end - main_start, main_count - 1) / max(main_count - 1, 1),
        ))
    path = []
    for chunk in chunks:
        if path and chunk and path[-1] == chunk[0]:
            path.extend(chunk[1:])
        else:
            path.extend(chunk)
    return path


def _route_segment_markers(map_id: int, route: dict, leadin: dict | None, main: dict | None) -> list[dict]:
    """Project timing-line checkpoint crossings, with legacy percentage fallback."""
    segment_entries = route.get("segments")
    if segment_entries is None:
        segment_entries = _load_wad_segments().get(route.get("name", "").strip(), {}).get("segments", [])
    if not segment_entries or not (leadin or main):
        return []
    projection = _projection(map_id)
    names = _load_segment_names()
    markers = []
    pass_numbers = {}
    segment_path = route.get("segment_path") or {}
    positions = segment_path.get("d")
    checkpoint_last = positions[-1] if positions else max(
        route.get("leadin_checkpoint_count", 0) + route.get("route_checkpoint_count", 0) - 1, 1)
    for index, segment in enumerate(segment_entries, start=1):
        distance = segment.get("distance_m")
        maximum = segment.get("max_distance_m")
        if distance is not None and maximum is not None and round(maximum - distance, 2) > 25:
            continue
        start_fraction = (segment["start_checkpoint"] / checkpoint_last
                          if "start_checkpoint" in segment else segment["percent_start"])
        end_fraction = (segment["end_checkpoint"] / checkpoint_last
                        if "end_checkpoint" in segment else segment.get("resolved_percent_end", segment["percent_end"]))
        local_path = _route_path_slice(route, leadin, main, start_fraction, end_fraction)
        if segment.get("boundary_source") == "closed_route_lap":
            arch_fraction = segment["timing_arch_checkpoint"] / checkpoint_last
            local_path = (
                _route_path_slice(route, leadin, main, arch_fraction, end_fraction)
                + _route_path_slice(route, leadin, main, start_fraction, arch_fraction)[1:]
            )
        if len(local_path) < 2:
            continue
        x, y = projection["project"](
            [point[0] for point in local_path],
            [point[1] for point in local_path],
        )
        name = segment.get("name") or names.get(int(segment["hash"]), f"Segment {index}")
        segment_hash = int(segment["hash"])
        pass_numbers[segment_hash] = pass_numbers.get(segment_hash, 0) + 1
        markers.append({
            "hash": segment_hash,
            "pass": pass_numbers[segment_hash],
            "name": name,
            "type": "kom" if "kom" in name.casefold() else "sprint" if "sprint" in name.casefold() else "segment",
            "direction": segment.get("direction"),
            "start_distance_m": segment.get("start_distance_m"),
            "end_distance_m": segment.get("end_distance_m"),
            "latlng": local_to_latlng(
                map_id,
                [point[0] for point in local_path],
                [point[1] for point in local_path],
            ),
            "path": [
                {"x": round(float(px), 1), "y": round(float(py), 1)}
                for px, py in zip(x, y)
            ],
        })
    return markers


def _match_surfaces(map_id: int, xs, zs) -> list[str]:
    """Tag each (x, z) point with the surface of the nearest road vertex."""
    world = _load_surface_world(map_id)
    tree = world["tree"]
    if tree is None or len(xs) == 0:
        return ["Unknown"] * len(xs)
    _, idx = tree.query(np.column_stack([xs, zs]), k=1)
    return world["labels"][idx].tolist()


def list_worlds() -> list[dict]:
    """List worlds that have both surface and route data available."""
    worlds = []
    for map_id, name in _WORLD_NAMES.items():
        if not _surface_path(map_id).exists() or not _route_path(map_id).exists():
            continue
        surf = _load_surface_world(map_id)
        routes = _load_route_world(map_id).get("routes", [])
        if not surf["segments"] or not routes:
            continue
        surfaces = sorted(
            {_seg_surface(s) for s in surf["segments"]},
            key=lambda s: SURFACE_ORDER.index(s) if s in SURFACE_ORDER else 99,
        )
        worlds.append({
            "mapID": map_id,
            "name": name,
            "route_count": len(routes),
            "segment_count": len(surf["segments"]),
            "surfaces": surfaces,
            "has_map": _projection(map_id)["mode"] == "image",
        })
    return worlds


def get_world_background(map_id: int) -> dict:
    """Return just the map background image + full-world plot bounds for a world.

    Lightweight companion to :func:`get_world_surfaces` for callers that only
    need the base image to draw a single route over (no road network).
    """
    proj = _projection(map_id)
    return {
        "mode": proj["mode"],
        "background": proj["background"],
        "bounds": proj["bounds"],
    }


def get_world_surfaces(map_id: int) -> dict:
    """Return the road network for a world plus its route list (metadata only)."""
    world = _load_surface_world(map_id)
    proj = _projection(map_id)

    segments = []
    for seg in world["segments"]:
        X, Y = proj["project"](seg.get("x", []), seg.get("z", []))
        segments.append({
            "surface": _seg_surface(seg),
            "road_id": seg.get("road_id"),
            "x": [round(float(v), 1) for v in X],
            "y": [round(float(v), 1) for v in Y],
        })

    routes = []
    def projected_paths(paths):
        result = {}
        for road_id, pieces in paths.items():
            result[str(road_id)] = []
            for piece in pieces:
                X, Y = proj["project"](piece["x"], piece["z"])
                result[str(road_id)].append({"x": [round(float(value), 3) for value in X],
                                             "y": [round(float(value), 3) for value in Y]})
        return result

    for r in _load_route_world(map_id).get("routes", []):
        routes.append({
            "nameHash": r.get("nameHash"),
            "name": r.get("name", ""),
            "distance_m": r.get("distance_m", 0.0),
            "ascent_m": r.get("ascent_m", 0.0),
            "leadin_distance_m": r.get("leadin_distance_m", 0.0),
            "leadin_ascent_m": r.get("leadin_ascent_m", 0.0),
            "sport_type": r.get("sport_type", 0),
            "event_only": r.get("event_only", False),
            "free_ride_leadin_distance_m": r.get("start_locations", {}).get("free_ride", {}).get("leadin_distance_m"),
            "free_ride_leadin_ascent_m": r.get("start_locations", {}).get("free_ride", {}).get("leadin_ascent_m"),
            "available_modes": [mode for mode, version in r.get("route_versions", {"race": {"available": True}}).items()
                                if version.get("available")],
            "road_ids": {mode: version.get("road_ids", [road["road_id"] for road in r.get("road_nodes", [])])
                         for mode, version in r.get("route_versions", {"race": {"available": True}}).items()
                         if version.get("available")},
            "road_paths": {mode: projected_paths(version.get("road_paths", {
                str(road["road_id"]): [leg for leg in (r.get("leadin"), r.get("route")) if leg]
                for road in r.get("road_nodes", [])}))
                for mode, version in r.get("route_versions", {"race": {"available": True}}).items()
                if version.get("available")},
        })
    routes.sort(key=lambda r: r["name"].lower())

    surfaces_present = sorted(
        {s["surface"] for s in segments},
        key=lambda s: SURFACE_ORDER.index(s) if s in SURFACE_ORDER else 99,
    )

    tolerance_x, tolerance_y = proj["project"]([0, 5, 0], [0, 0, 5])
    basis_x, basis_y = proj["project"]([0, 1, 0], [0, 0, 1])
    basis = np.array([
        [basis_x[1] - basis_x[0], basis_x[2] - basis_x[0]],
        [basis_y[1] - basis_y[0], basis_y[2] - basis_y[0]],
    ], dtype=float)
    meters_per_plot_x = float(np.linalg.norm(np.linalg.solve(basis, [1.0, 0.0])))
    return {
        "mapID": map_id,
        "name": _WORLD_NAMES.get(map_id, f"World {map_id}"),
        "projection": proj["mode"],
        "disconnected_routes": map_id == VIDEO_ROUTES_MAP_ID,
        "background": proj["background"],
        "bounds": proj["bounds"],
        "meters_per_plot_x": meters_per_plot_x,
        "segments": segments,
        "routes": routes,
        "road_match_tolerance": math.sqrt(sum((float(tolerance_x[index]) - float(tolerance_x[0])) ** 2 +
            (float(tolerance_y[index]) - float(tolerance_y[0])) ** 2 for index in (1, 2))),
        "colors": {s: SURFACE_COLORS.get(s, SURFACE_COLORS["Unknown"])
                   for s in surfaces_present},
    }


def _pack_leg(map_id: int, leg: dict | None) -> dict | None:
    """Build a serialisable leg (leadin or main) with per-point surface tags."""
    if not leg:
        return None
    xs = leg.get("x", [])
    zs = leg.get("z", [])
    if not xs:
        return None

    # Prefer the authoritative per-point surface precomputed at extraction time
    # (road-id + time join, robust to parallel roads). Fall back to spatial
    # nearest-vertex matching for legacy route files without stored surfaces.
    stored = leg.get("surface")
    if stored is not None and len(stored) == len(xs):
        surfaces = list(stored)
    else:
        surfaces = _match_surfaces(map_id, np.asarray(xs, dtype=float),
                                   np.asarray(zs, dtype=float))
    X, Y = _projection(map_id)["project"](xs, zs)
    return {
        "d": [round(v, 1) for v in leg.get("d", [])],
        "alt": [round(v, 2) for v in leg.get("alt", [])],
        "x": [round(float(v), 1) for v in X],
        "y": [round(float(v), 1) for v in Y],
        "local_x": [round(float(v), 1) for v in xs],
        "local_z": [round(float(v), 1) for v in zs],
        "surface": surfaces,
    }


def _anchor_leg_alt(leg: dict | None, scale: float, anchor: float) -> None:
    """Scale a packed leg's altitude to physical metres by a per-world factor.

    WAD vertical geometry is not always to physical scale (Watopia altitudes
    read ~2x true metres, New York ~1.32x) while horizontal distance is. The
    factor is the stored per-world WAD->physical scale
    (:func:`get_world_altitude_scale`). ``anchor`` must be the SAME raw
    altitude value for both the lead-in and main legs (e.g. the lead-in's
    first point) so the two legs stay joined at their shared boundary; using
    each leg's own first point independently (as before) desyncs the two legs
    whenever the lead-in has significant net elevation change (e.g. Watopia's
    Canopies and Coastlines climbs ~140m raw over its lead-in), producing a
    large spurious jump where the lead-in meets the route. This is a no-op for
    worlds already at physical scale (``scale == 1``).
    """
    if not leg or not scale or scale == 1.0:
        return
    alt = leg.get("alt")
    if not alt:
        return
    leg["alt"] = [round(anchor + (v - anchor) * scale, 2) for v in alt]


def _surface_breakdown(leg: dict | None) -> dict[str, float]:
    """Distance (metres) spent on each surface across a packed leg."""
    totals: dict[str, float] = {}
    if not leg:
        return totals
    ds = leg["d"]
    surfaces = leg["surface"]
    for i in range(len(ds) - 1):
        seg_len = ds[i + 1] - ds[i]
        if seg_len <= 0:
            continue
        surf = surfaces[i]
        totals[surf] = totals.get(surf, 0.0) + seg_len
    return totals


def route_is_loop(map_id: int, name_hash: int, threshold_m: float = 100.0) -> bool:
    """True if the main route leg starts and ends at (nearly) the same point.

    Uses the raw ``x``/``z`` game-local coordinates (metres) of the route leg,
    matching the loop test used by the race-replay tool (start/end < 100 m).
    Note: Zwift's own ``supported_laps`` flag marks more routes as lappable
    (e.g. Hilly Route, whose start banner is offset ~350 m from the lap seam),
    but we only enable multi-lap where the geometry actually closes so the
    repeated profile joins without an inferred connector.
    """
    routes = _load_route_world(map_id).get("routes", [])
    route = next((r for r in routes if r.get("nameHash") == name_hash), None)
    if route is None:
        return False
    leg = route.get("route") or {}
    xs = leg.get("x", [])
    zs = leg.get("z", [])
    if len(xs) < 2 or len(zs) < 2:
        return False
    dx = float(xs[0]) - float(xs[-1])
    dz = float(zs[0]) - float(zs[-1])
    return (dx * dx + dz * dz) ** 0.5 < threshold_m


def _route_profile_offset_m(route: dict) -> float:
    """Measured main-leg start on the displayed axis, including its connector."""
    leadin = route.get("leadin")
    if not leadin or not leadin.get("d"):
        return 0.0
    offset = leadin["d"][-1]
    main = route.get("route")
    if main and main.get("x") and main.get("z"):
        offset += float(np.hypot(
            main["x"][0] - leadin["x"][-1],
            main["z"][0] - leadin["z"][-1],
        ))
    if route.get("leadin_view"):
        offset -= route["leadin_view"]["paddock_distance_m"]
    return round(offset, 2)


def get_route(map_id: int, name_hash: int, mode: str | None = None) -> dict | None:
    """Return full geometry + elevation + per-point surface for one route."""
    routes = _load_route_world(map_id).get("routes", [])
    route = next((r for r in routes if r.get("nameHash") == name_hash), None)
    if route is None:
        return None
    if mode not in (None, "race", "free_ride"):
        raise ValueError("Unknown route mode")
    versions = route.get("route_versions", {"race": {"available": True}})
    available_modes = [key for key, value in versions.items() if value.get("available")]
    if mode == "free_ride" and mode not in available_modes:
        mode = "race"
    version = versions.get(mode, {})

    raw_leadin = _pack_leg(map_id, route.get("leadin"))
    leadin_view = route.get("leadin_view") if mode is None else None
    lead_profile = version.get("leadin") if mode == "free_ride" else route.get("leadin")
    leadin = _pack_leg(map_id, leadin_view["profile"] if leadin_view else lead_profile)
    main = _pack_leg(map_id, route.get("route"))
    mode_offset = (_route_profile_offset_m(route) if mode is None else
                   _route_profile_offset_m({"leadin": lead_profile, "route": route.get("route")}))

    # Scale each leg's altitude to physical metres by this world's stored
    # WAD->physical factor so the elevation profile and gradients are correct
    # (WAD vertical geometry is not always physical — see helper).
    world_scale = get_world_altitude_scale(map_id)
    anchor_leg = raw_leadin if raw_leadin and raw_leadin.get("alt") else main
    if anchor_leg and anchor_leg.get("alt"):
        anchor_alt = anchor_leg["alt"][0]
        _anchor_leg_alt(leadin, world_scale, anchor_alt)
        _anchor_leg_alt(main, world_scale, anchor_alt)

    breakdown: dict[str, float] = {}
    for leg in (leadin, main):
        for surf, dist in _surface_breakdown(leg).items():
            breakdown[surf] = breakdown.get(surf, 0.0) + dist

    xs: list[float] = []
    ys: list[float] = []
    for leg in (leadin, main):
        if leg:
            xs.extend(leg["x"])
            ys.extend(leg["y"])
    bounds = ({"min_x": min(xs), "max_x": max(xs),
               "min_y": min(ys), "max_y": max(ys)} if xs else None)

    breakdown_list = sorted(
        ({"surface": s, "distance_m": round(d, 1),
          "color": SURFACE_COLORS.get(s, SURFACE_COLORS["Unknown"])}
         for s, d in breakdown.items()),
        key=lambda e: -e["distance_m"],
    )

    segments = _route_segment_markers(map_id, route, raw_leadin, main)
    escalators = route.get("escalators", [])
    if mode == "free_ride":
        raw_offset = _route_profile_offset_m({"leadin": route.get("leadin"), "route": route.get("route")})
        segments = [segment for segment in segments if segment["start_distance_m"] >= raw_offset]
        for segment in segments:
            for key in ("start_distance_m", "end_distance_m"):
                segment[key] = round(segment[key] + mode_offset - raw_offset, 2)
        escalators = [passage for passage in escalators if passage.get("leg") != "leadin"]
    if leadin_view:
        escalators = [dict(passage) for passage in escalators]
        shifted_markers = segments + [passage for passage in escalators
                                      if passage.get("leg") == "leadin"]
        for marker in shifted_markers:
            for key in ("start_distance_m", "end_distance_m"):
                if marker.get(key) is not None:
                    marker[key] = round(marker[key] - leadin_view["paddock_distance_m"], 2)

    return {
        "mapID": map_id,
        "world_name": _WORLD_NAMES.get(map_id, f"World {map_id}"),
        "nameHash": name_hash,
        "name": route.get("name", ""),
        "distance_m": route.get("distance_m", 0.0),
        "ascent_m": route.get("ascent_m", 0.0),
        "leadin_distance_m": version.get("leadin_distance_m", route.get("leadin_distance_m", 0.0)),
        "leadin_ascent_m": version.get("leadin_ascent_m", route.get("leadin_ascent_m", 0.0)),
        "sport_type": route.get("sport_type", 0),
        "event_only": route.get("event_only", False),
        "route_mode": mode,
        "available_modes": available_modes,
        "start_geometry_status": version.get("geometry_status"),
        "start_position_source": version.get("position_source"),
        "fallback_paddock_road_id": version.get("fallback_paddock_road_id"),
        "fallback_paddock_basis": version.get("fallback_paddock_basis"),
        "start_locations": route.get("start_locations", {}),
        "route_offset_m": mode_offset,
        "leadin": leadin,
        "route": main,
        "segments": segments,
        "escalators": escalators,
        "road_nodes": route.get("road_nodes", []),
        "breakdown": breakdown_list,
        "bounds": bounds,
    }


def local_to_latlng(map_id: int, x, z) -> list[list[float]] | None:
    """Convert local route coordinates to calibrated ``[lat, lng]`` points."""
    calibration = _load_calibration().get(str(map_id))
    if not calibration:
        return None
    points = np.column_stack([
        np.asarray(x, dtype=float),
        np.asarray(z, dtype=float),
        np.ones(len(x)),
    ])
    latlng = points @ np.asarray(calibration["coef"], dtype=float)
    return latlng.tolist()
