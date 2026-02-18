"""
World configuration for Zwift maps.

Maps each Zwift world to its map image and GPS coordinate bounds
(from zwift-data / ZwiftMap). The bounds define the image overlay:
  - lat_max / lng_min = NW corner (top-left of image)
  - lat_min / lng_max = SE corner (bottom-right of image)
"""

WORLD_CONFIG = {
    'WATOPIA': {
        'slug': 'watopia',
        'image': '/static/maps/watopia.png',
        'lat_min': -11.74087,
        'lat_max': -11.62597,
        'lng_min': 166.87747,
        'lng_max': 167.03255,
        'bg_color': '#0884e2',
    },
    'RICHMOND': {
        'slug': 'richmond',
        'image': '/static/maps/richmond.png',
        'lat_min': 37.5014,
        'lat_max': 37.5774,
        'lng_min': -77.48954,
        'lng_max': -77.394,
        'bg_color': '#7c9938',
    },
    'LONDON': {
        'slug': 'london',
        'image': '/static/maps/london.png',
        'lat_min': 51.4601,
        'lat_max': 51.5362,
        'lng_min': -0.1776,
        'lng_max': -0.0555,
        'bg_color': '#6f992d',
    },
    'NEW_YORK': {
        'slug': 'new-york',
        'image': '/static/maps/new-york.png',
        'lat_min': 40.587913,
        'lat_max': 40.817257,
        'lng_min': -74.022655,
        'lng_max': -73.922165,
        'bg_color': '#bbbbb7',
    },
    'INNSBRUCK': {
        'slug': 'innsbruck',
        'image': '/static/maps/innsbruck.png',
        'lat_min': 47.2055,
        'lat_max': 47.2947,
        'lng_min': 11.3501,
        'lng_max': 11.4822,
        'bg_color': '#7c9938',
    },
    'BOLOGNA': {
        'slug': 'bologna',
        'image': '/static/maps/bologna.png',
        'lat_min': 44.45463821,
        'lat_max': 44.5308037,
        'lng_min': 11.26261748,
        'lng_max': 11.36991729,
        'bg_color': '#b9b9b8',
    },
    'YORKSHIRE': {
        'slug': 'yorkshire',
        'image': '/static/maps/yorkshire.png',
        'lat_min': 53.9491,
        'lat_max': 54.0254,
        'lng_min': -1.632,
        'lng_max': -1.5022,
        'bg_color': '#7c9938',
    },
    'CRIT_CITY': {
        'slug': 'crit-city',
        'image': '/static/maps/crit-city.png',
        'lat_min': -10.4038,
        'lat_max': -10.3657,
        'lng_min': 165.7824,
        'lng_max': 165.8207,
        'bg_color': '#7c9938',
    },
    'MAKURI': {
        'slug': 'makuri-islands',
        'image': '/static/maps/makuri-islands.png',
        'lat_min': -10.85234,
        'lat_max': -10.73746,
        'lng_min': 165.76591,
        'lng_max': 165.88222,
        'bg_color': '#7d9a35',
    },
    'FRANCE': {
        'slug': 'france',
        'image': '/static/maps/france.png',
        'lat_min': -21.7564,
        'lat_max': -21.64155,
        'lng_min': 166.1384,
        'lng_max': 166.26125,
        'bg_color': '#6f992d',
    },
    'PARIS': {
        'slug': 'paris',
        'image': '/static/maps/paris.png',
        'lat_min': 48.82945,
        'lat_max': 48.9058,
        'lng_min': 2.2561,
        'lng_max': 2.3722,
        'bg_color': '#b9b9b9',
    },
    'SCOTLAND': {
        'slug': 'scotland',
        'image': '/static/maps/scotland.png',
        'lat_min': 55.61845,
        'lat_max': 55.67595,
        'lng_min': -5.2802,
        'lng_max': -5.17798,
        'bg_color': '#aba73a',
    },
}


def get_world_map_config(world_name: str) -> dict | None:
    """Get map config for a world name (as returned by detect_world_from_coords)."""
    if world_name and world_name in WORLD_CONFIG:
        return WORLD_CONFIG[world_name]
    return None
