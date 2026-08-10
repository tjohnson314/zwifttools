"""
Route Lookup Module
Lookup route data from Zwift's game dictionary cache.
"""

import json
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "routes_cache.json"
# WAD-extracted route index; contains newly added routes not yet in routes_cache.json
ROUTE_INDEX_FILE = Path(__file__).parent.parent / "zwift_routes" / "index.json"

# Zwift internal world ID -> map name mapping (from game dictionary segments)
WORLD_ID_TO_MAP = {
    1: "WATOPIA",
    2: "RICHMOND",
    3: "LONDON",
    4: "NEWYORK",
    5: "INNSBRUCK",
    6: "BOLOGNATT",
    7: "YORKSHIRE",
    8: "CRITCITY",
    9: "MAKURIISLANDS",
    10: "FRANCE",
    11: "PARIS",
    13: "SCOTLAND",
    14: "GRAVEL MOUNTAIN",
}

MAP_TO_WORLD_ID = {v: k for k, v in WORLD_ID_TO_MAP.items()}



def load_route_cache():
    """Load routes from local cache."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return None


def _load_route_index():
    """Load the WAD-extracted route index keyed by nameHash (as string)."""
    if not ROUTE_INDEX_FILE.exists():
        return None
    with open(ROUTE_INDEX_FILE, encoding="utf-8") as f:
        entries = json.load(f)
    return {str(e["nameHash"]): e for e in entries if "nameHash" in e}


def _route_info_from_index(route_id):
    """Look up a route in the WAD index and adapt it to the cache schema.

    Covers newly added routes that exist in zwift_routes/index.json but have not
    yet been synced into routes_cache.json.
    """
    index = _load_route_index()
    if not index:
        return None
    entry = index.get(str(route_id))
    if not entry:
        return None
    return {
        "name": entry.get("name", ""),
        "map": WORLD_ID_TO_MAP.get(entry.get("mapID"), ""),
        "distanceInMeters": entry.get("distance_m", 0.0),
        "leadinDistanceInMeters": entry.get("leadin_distance_m", 0.0),
        "ascentInMeters": entry.get("ascent_m", 0.0),
        "leadinAscentInMeters": entry.get("leadin_ascent_m", 0.0),
        "eventOnly": entry.get("event_only", False),
    }


def get_route_info(route_id):
    """Get route info by route ID/signature."""
    routes = load_route_cache()

    if routes is not None:
        info = routes.get(str(route_id))
        if info is not None:
            return info

    # Fall back to the WAD index for routes not yet in routes_cache.json
    return _route_info_from_index(route_id)


def get_total_race_distance(route_id):
    """Get total race distance (route + lead-in) in km."""
    route = get_route_info(route_id)
    
    if route is None:
        return None
    
    total_meters = route["distanceInMeters"] + route["leadinDistanceInMeters"]
    return total_meters / 1000.0

