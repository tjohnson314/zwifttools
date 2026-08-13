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
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from shared.world_config import WORLD_CONFIG

_BASE = Path(__file__).parent.parent
SURFACE_DIR = _BASE / "zwift_surfaces"
ROUTE_DIR = _BASE / "zwift_routes"
MAPS_DIR = _BASE / "static" / "maps"
CALIBRATION_FILE = SURFACE_DIR / "world_gps_calibration.json"
STYLE_MAP_FILE = SURFACE_DIR / "style_surface_map.json"

# mapID -> human readable world name (derived from route locKeys).
WORLD_NAMES = {
    1: "Watopia", 2: "Richmond", 3: "London", 4: "New York",
    5: "Innsbruck", 6: "Bologna", 7: "Yorkshire", 8: "Crit City",
    9: "Makuri Islands", 10: "France", 11: "Paris",
    12: "Gravel Mountain", 13: "Scotland",
}

# mapID -> key in shared.world_config.WORLD_CONFIG (for the map PNG + GPS bounds).
MAPID_TO_CONFIG = {
    1: "WATOPIA", 2: "RICHMOND", 3: "LONDON", 4: "NEW_YORK",
    5: "INNSBRUCK", 6: "BOLOGNA", 7: "YORKSHIRE", 8: "CRIT_CITY",
    9: "MAKURI", 10: "FRANCE", 11: "PARIS", 13: "SCOTLAND",
}

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
    return SURFACE_DIR / f"world_{map_id}.json"


def _route_path(map_id: int) -> Path:
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
    for map_id, name in WORLD_NAMES.items():
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


def get_world_surfaces(map_id: int) -> dict:
    """Return the road network for a world plus its route list (metadata only)."""
    world = _load_surface_world(map_id)
    proj = _projection(map_id)

    segments = []
    for seg in world["segments"]:
        X, Y = proj["project"](seg.get("x", []), seg.get("z", []))
        segments.append({
            "surface": _seg_surface(seg),
            "x": [round(float(v), 1) for v in X],
            "y": [round(float(v), 1) for v in Y],
        })

    routes = []
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
        })
    routes.sort(key=lambda r: r["name"].lower())

    surfaces_present = sorted(
        {s["surface"] for s in segments},
        key=lambda s: SURFACE_ORDER.index(s) if s in SURFACE_ORDER else 99,
    )

    return {
        "mapID": map_id,
        "name": WORLD_NAMES.get(map_id, f"World {map_id}"),
        "projection": proj["mode"],
        "background": proj["background"],
        "bounds": proj["bounds"],
        "segments": segments,
        "routes": routes,
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

    surfaces = _match_surfaces(map_id, np.asarray(xs, dtype=float),
                               np.asarray(zs, dtype=float))
    X, Y = _projection(map_id)["project"](xs, zs)
    return {
        "d": [round(v, 1) for v in leg.get("d", [])],
        "alt": [round(v, 2) for v in leg.get("alt", [])],
        "x": [round(float(v), 1) for v in X],
        "y": [round(float(v), 1) for v in Y],
        "surface": surfaces,
    }


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


def get_route(map_id: int, name_hash: int) -> dict | None:
    """Return full geometry + elevation + per-point surface for one route."""
    routes = _load_route_world(map_id).get("routes", [])
    route = next((r for r in routes if r.get("nameHash") == name_hash), None)
    if route is None:
        return None

    leadin = _pack_leg(map_id, route.get("leadin"))
    main = _pack_leg(map_id, route.get("route"))

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

    return {
        "mapID": map_id,
        "world_name": WORLD_NAMES.get(map_id, f"World {map_id}"),
        "nameHash": name_hash,
        "name": route.get("name", ""),
        "distance_m": route.get("distance_m", 0.0),
        "ascent_m": route.get("ascent_m", 0.0),
        "leadin_distance_m": route.get("leadin_distance_m", 0.0),
        "leadin_ascent_m": route.get("leadin_ascent_m", 0.0),
        "sport_type": route.get("sport_type", 0),
        "event_only": route.get("event_only", False),
        "leadin": leadin,
        "route": main,
        "breakdown": breakdown_list,
        "bounds": bounds,
    }
