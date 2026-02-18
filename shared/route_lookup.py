"""
Route Lookup Module
Lookup route data from Zwift's game dictionary cache.
"""

import json
from pathlib import Path

CACHE_FILE = Path(__file__).parent.parent / "routes_cache.json"

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
    
    total_meters = route["distanceInMeters"] + route["leadinDistanceInMeters"]
    return total_meters / 1000.0

