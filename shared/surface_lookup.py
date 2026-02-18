"""
Surface type lookup for Zwift worlds.

Uses polygon data from ZwiftMap to determine the road surface type
at any GPS coordinate, then maps it to a rolling resistance coefficient (CRR).

Surface polygons define non-default areas (cobbles, dirt, wood, etc.).
Any point not inside a polygon is assumed to be Tarmac.
"""

import json
import numpy as np
from pathlib import Path


SURFACE_DATA_PATH = Path(__file__).parent.parent / 'zwiftmap_surfaces' / 'surface_data.json'

# Map detect_world_from_coords names to surface_data.json world keys
WORLD_NAME_MAP = {
    'LONDON': 'london',
    'WATOPIA': 'watopia',
    'NEW_YORK': 'new-york',
    'NEWYORK': 'new-york',
    'INNSBRUCK': 'innsbruck',
    'RICHMOND': 'richmond',
    'FRANCE': 'france',
    'PARIS': 'paris',
    'MAKURI': 'makuri-islands',
    'MAKURIISLANDS': 'makuri-islands',
    'SCOTLAND': 'scotland',
    'YORKSHIRE': 'yorkshire',
    'BOLOGNA': 'bologna',
    'BOLOGNATT': 'bologna',
    'CRIT_CITY': 'crit-city',
    'CRITCITY': 'crit-city',
    'GRAVEL MOUNTAIN': 'gravel-mountain',
}

# Default CRR values for road bikes (from ZwiftInsider research)
DEFAULT_CRR = {
    'Tarmac': 0.004,
    'Brick': 0.0055,
    'Wood': 0.0065,
    'Cobbles': 0.0065,
    'Snow': 0.0075,
    'Dirt': 0.025,
    'Grass': 0.004,   # Grass has no effect on road bikes
    'Sand': 0.004,
    'Gravel': 0.012,
}

_surface_cache = {}

# Map Zwift frame types to surface CRR bike categories
FRAME_TYPE_TO_BIKE_TYPE = {
    'Standard': 'road_bike',
    'TT': 'road_bike',
    'Tron': 'road_bike',
    'Funny': 'road_bike',
    'Hand': 'road_bike',
    'MTB': 'mtb',
    'Gravel': 'gravel_bike',
}


def get_bike_type_for_frame(frame_type: str) -> str:
    """Map a Zwift frame type to the CRR bike category.
    
    Args:
        frame_type: Zwift frame type ('Standard', 'TT', 'Gravel', 'MTB', etc.)
    
    Returns:
        CRR bike category: 'road_bike', 'mtb', or 'gravel_bike'
    """
    return FRAME_TYPE_TO_BIKE_TYPE.get(frame_type, 'road_bike')


def surface_types_to_crr(surface_types: np.ndarray, bike_type: str = 'road_bike') -> np.ndarray:
    """Convert an array of surface type strings to CRR values for a given bike type.
    
    Args:
        surface_types: numpy array of surface type strings (e.g. 'Tarmac', 'Cobbles')
        bike_type: 'road_bike', 'mtb', or 'gravel_bike'
    
    Returns:
        numpy array of CRR values
    """
    data = _load_surface_data()
    crr_table = DEFAULT_CRR.copy()
    if data and bike_type in data.get('crr_values', {}):
        for k, v in data['crr_values'][bike_type].items():
            if v is not None:
                crr_table[k] = v
    
    default_crr = crr_table.get('Tarmac', 0.004)
    return np.array([crr_table.get(st, default_crr) for st in surface_types])


def _load_surface_data():
    """Load and cache surface_data.json."""
    if not SURFACE_DATA_PATH.exists():
        return None
    with open(SURFACE_DATA_PATH) as f:
        return json.load(f)


def _get_world_surfaces(world: str) -> list:
    """Get surface polygons for a world, with caching."""
    world_key = WORLD_NAME_MAP.get(world.upper(), world.lower())

    if world_key in _surface_cache:
        return _surface_cache[world_key]

    data = _load_surface_data()
    if data is None:
        _surface_cache[world_key] = []
        return []

    world_data = data.get('worlds', {}).get(world_key, {})
    surfaces = world_data.get('surfaces', [])

    # Precompute bounding boxes for fast rejection
    for s in surfaces:
        poly = s['polygon']
        lats = [p[0] for p in poly]
        lngs = [p[1] for p in poly]
        s['_bbox'] = (min(lats), max(lats), min(lngs), max(lngs))

    _surface_cache[world_key] = surfaces
    return surfaces


def _point_in_polygon(lat: float, lng: float, polygon: list) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        if ((lng_i > lng) != (lng_j > lng)) and \
           (lat < (lat_j - lat_i) * (lng - lng_i) / (lng_j - lng_i) + lat_i):
            inside = not inside
        j = i
    return inside


def compute_crr_array(lats: np.ndarray, lngs: np.ndarray, world: str,
                      bike_type: str = 'road_bike') -> np.ndarray:
    """
    Compute CRR values for an array of GPS coordinates.

    This is the main entry point for the physics pipeline.
    Returns a numpy array of CRR values, one per coordinate pair.
    
    Args:
        lats: Array of latitudes
        lngs: Array of longitudes
        world: World name (e.g. 'RICHMOND', 'WATOPIA')
        bike_type: 'road_bike', 'mtb', or 'gravel_bike'

    Returns:
        numpy array of CRR values
    """
    surfaces = _get_world_surfaces(world)

    # Load per-bike CRR table
    data = _load_surface_data()
    crr_table = DEFAULT_CRR.copy()
    if data and bike_type in data.get('crr_values', {}):
        for k, v in data['crr_values'][bike_type].items():
            if v is not None:
                crr_table[k] = v

    default_crr = crr_table.get('Tarmac', 0.004)
    n = len(lats)
    crr_values = np.full(n, default_crr)

    if not surfaces:
        return crr_values

    # For each surface polygon, find which points fall inside it
    for s in surfaces:
        surface_crr = crr_table.get(s['type'], default_crr)
        if surface_crr == default_crr:
            continue  # Skip if same as default — no benefit

        bbox = s.get('_bbox')
        if bbox:
            lat_min, lat_max, lng_min, lng_max = bbox
            # Vectorized bounding box filter
            candidates = np.where(
                (lats >= lat_min) & (lats <= lat_max) &
                (lngs >= lng_min) & (lngs <= lng_max)
            )[0]
        else:
            candidates = np.arange(n)

        for idx in candidates:
            if _point_in_polygon(float(lats[idx]), float(lngs[idx]), s['polygon']):
                crr_values[idx] = surface_crr

    return crr_values


def compute_surface_types_array(lats: np.ndarray, lngs: np.ndarray, world: str) -> np.ndarray:
    """
    Compute surface type for each GPS coordinate.

    Returns a numpy array of surface type strings (e.g. 'Tarmac', 'Cobbles').
    This can be later converted to CRR values for any bike type using
    surface_types_to_crr().

    Args:
        lats: Array of latitudes
        lngs: Array of longitudes
        world: World name (e.g. 'RICHMOND', 'WATOPIA')

    Returns:
        numpy array of surface type strings
    """
    surfaces = _get_world_surfaces(world)
    n = len(lats)
    result = np.full(n, 'Tarmac', dtype=object)

    if not surfaces:
        return result

    for s in surfaces:
        bbox = s.get('_bbox')
        if bbox:
            lat_min, lat_max, lng_min, lng_max = bbox
            candidates = np.where(
                (lats >= lat_min) & (lats <= lat_max) &
                (lngs >= lng_min) & (lngs <= lng_max)
            )[0]
        else:
            candidates = np.arange(n)

        for idx in candidates:
            if _point_in_polygon(float(lats[idx]), float(lngs[idx]), s['polygon']):
                result[idx] = s['type']

    return result
