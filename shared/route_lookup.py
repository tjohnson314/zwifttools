"""
Route Lookup Module
Route metadata sourced directly from the WAD-extracted route index
(``zwift_routes/index.json``, produced by tools/extract_zwift_routes.py).
"""

import json
from functools import lru_cache
from pathlib import Path

# MAP_TO_WORLD_ID is re-exported for consumers (e.g. app.py).
from shared.world_config import WORLD_ID_TO_MAP, MAP_TO_WORLD_ID  # noqa: F401

ROUTE_INDEX_FILE = Path(__file__).parent.parent / "zwift_routes" / "index.json"


def _adapt(entry):
    """Adapt a WAD index entry to the route-info schema used across the app."""
    return {
        "name": entry.get("name", ""),
        "map": WORLD_ID_TO_MAP.get(entry.get("mapID"), ""),
        "distanceInMeters": entry.get("distance_m", 0.0),
        "leadinDistanceInMeters": entry.get("leadin_distance_m", 0.0),
        "ascentInMeters": entry.get("ascent_m", 0.0),
        "leadinAscentInMeters": entry.get("leadin_ascent_m", 0.0),
        "eventOnly": entry.get("event_only", False),
    }


@lru_cache(maxsize=1)
def load_route_cache():
    """Load all routes keyed by route ID (nameHash), adapted from the WAD index."""
    if not ROUTE_INDEX_FILE.exists():
        return None
    with open(ROUTE_INDEX_FILE, encoding="utf-8") as f:
        entries = json.load(f)
    return {str(e["nameHash"]): _adapt(e) for e in entries if "nameHash" in e}


def get_route_info(route_id):
    """Get route info by route ID/signature."""
    routes = load_route_cache()
    if routes is None:
        return None
    return routes.get(str(route_id))


def get_total_race_distance(route_id):
    """Get total race distance (route + lead-in) in km."""
    route = get_route_info(route_id)
    if route is None:
        return None
    return (route["distanceInMeters"] + route["leadinDistanceInMeters"]) / 1000.0

