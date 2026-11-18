"""Event route previews built from shipped geometry and native arch crossings."""

import json
import re
from functools import lru_cache

import numpy as np

from shared.event_settings import _arch_powerup_pairs, _distribution, _powerup
from shared.surface_map import ROUTE_DIR, get_route, get_world_background


@lru_cache(maxsize=1)
def _arch_catalog():
    with open(ROUTE_DIR / "arches.json", encoding="utf-8") as handle:
        return json.load(handle)["worlds"]


def _award(tags, arch_id):
    overrides = []
    distributions = []
    for tag in tags:
        key, separator, value = tag.partition("=")
        if not separator:
            continue
        if key == "arch_powerup":
            pairs = _arch_powerup_pairs(value)
            if pairs is None:
                return f"Unknown award: unrecognized arch override: {value}"
            overrides.extend(_powerup(powerup_id) for configured_id, powerup_id in pairs
                             if int(configured_id) == arch_id)
        elif key == "powerup_percent":
            distributions.append(_distribution(value))
    if overrides:
        return "Arch override: " + "; ".join(overrides)
    if distributions:
        return "Fallback distribution: " + "; ".join(distributions)
    return "Feather, Aero, or Draft"


def _polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


@lru_cache(maxsize=32)
def _geometry(map_id, route_id):
    try:
        route = get_route(map_id, route_id, mode="race")
    except FileNotFoundError:
        return {"error": "Route geometry is not available in the local catalog."}
    if route is None:
        return {"error": "Route geometry is not available in the local catalog."}
    samples = []
    for leg_name, offset in (("leadin", 0), ("route", route["route_offset_m"])):
        leg = route.get(leg_name)
        if leg:
            samples.extend((distance + offset, altitude, x, y, leg_name)
                           for distance, altitude, x, y in zip(leg["d"], leg["alt"], leg["x"], leg["y"]))
    if len(samples) < 2:
        return {"error": "Route geometry has too few points for a preview."}
    distances, altitudes, xs, ys, _ = zip(*samples)
    low, high = min(altitudes), max(altitudes)
    total = distances[-1]
    if total <= 0:
        return {"error": "Route geometry has no positive distance axis."}
    scale = min(840 / (max(xs) - min(xs) or 1), 340 / (max(ys) - min(ys) or 1))
    center_x, center_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def map_point(x, y):
        return 450 + (x - center_x) * scale, 200 + (y - center_y) * scale

    def profile_point(distance, altitude):
        return 60 + distance / total * 820, 230 - (altitude - low) / (high - low or 1) * 200

    def at(distance):
        return {
            "map": map_point(float(np.interp(distance, distances, xs)), float(np.interp(distance, distances, ys))),
            "profile": profile_point(distance, float(np.interp(distance, distances, altitudes))),
        }

    paths = {}
    for leg_name in ("leadin", "route"):
        leg_samples = [sample for sample in samples if sample[4] == leg_name]
        paths[leg_name] = {
            "map": _polyline(map_point(sample[2], sample[3]) for sample in leg_samples),
            "profile": _polyline(profile_point(sample[0], sample[1]) for sample in leg_samples),
        }
    background = get_world_background(map_id)["background"]
    if background:
        image_x, image_y = map_point(0, 0)
        background = {**background, "x": image_x, "y": image_y,
                      "width": background["width"] * scale, "height": background["height"] * scale}
    segments = []
    for segment in route["segments"]:
        start, end = segment.get("start_distance_m"), segment.get("end_distance_m")
        if start is None or end is None:
            continue
        segments.append({
            "name": segment["name"], "start_km": start / 1000, "end_km": end / 1000,
            "type": segment["type"],
            "map": _polyline(map_point(point["x"], point["y"]) for point in segment["path"]),
            "profile_x": profile_point(start, low)[0],
            "profile_width": (end - start) / total * 820,
        })
    escalators = []
    for passage in route.get("escalators", []):
        start, end = passage["start_distance_m"], passage["end_distance_m"]
        passage_distances = [start, *(distance for distance in distances if start < distance < end), end]
        escalators.append({
            "name": passage["name"], "start_km": start / 1000, "end_km": end / 1000,
            "trainer_slope_percent": passage["trainer_slope_scale"] * 100,
            "map": _polyline(at(distance)["map"] for distance in passage_distances),
            "profile_x": profile_point(start, low)[0],
            "profile_end_x": profile_point(end, low)[0],
            "profile_width": (end - start) / total * 820,
        })
    try:
        passages = _arch_catalog().get(str(map_id), {}).get(str(route_id))
    except FileNotFoundError:
        passages = None
    arches = []
    for index, passage in enumerate(passages or [], 1):
        arches.append({
            **passage, "number": index, "distance_km": passage["distance_m"] / 1000,
            "effective_id": passage["arch_id"] if passage["arch_id"] is not None else 0,
            **at(passage["distance_m"]),
        })
    return {
        "name": route["name"], "map_id": map_id, "route_id": route_id,
        "paths": paths, "background": background, "segments": segments, "arches": arches,
        "escalators": escalators,
        "arch_error": "Arch crossings are not available in the local catalog." if passages is None else None,
        "total_km": total / 1000, "min_altitude": low, "max_altitude": high,
        "leadin_x": profile_point(route["route_offset_m"], low)[0],
    }


def build_event_route(map_id, route_id, tag_sets, rules):
    if not all(re.fullmatch(r"[0-9]+", str(value)) for value in (map_id, route_id)):
        return {"error": "A valid mapId and routeId are needed for a route preview."}
    preview = _geometry(int(map_id), int(route_id))
    if "error" in preview:
        return preview
    configured_sources = [(title, tags) for title, tags in tag_sets
                          if any(tag.startswith(("arch_powerup=", "powerup_percent=")) for tag in tags)]
    sources = configured_sources or [("Default", [])]
    arches = []
    for arch in preview["arches"]:
        if "NO_POWERUPS" in rules:
            awards = [("Event/pen rules", "Powerups disabled (NO_POWERUPS)")]
            warning = None
        else:
            awards = [(title, _award(tags, arch["effective_id"])) for title, tags in sources]
            warnings = []
            if not arch["enabled"] or not arch["gives_powerup"]:
                warnings.append(
                    "Native data disables this direction or normal powerup awards here. "
                    "Whether the event configuration enables an award is uncertain."
                )
            if arch["distance_m"] == 0:
                warnings.append("This arch is at the preview start; an award on spawning is not established.")
            if any(other["number"] != arch["number"] and
                   abs(other["distance_m"] - arch["distance_m"]) < 0.1
                   for other in preview["arches"]):
                warnings.append("Multiple native arch entities overlap here; award precedence is unknown.")
            warning = " ".join(warnings) or None
        arches.append({**arch, "awards": awards, "warning": warning})
    return {
        **preview, "arches": arches,
        "identical_tags": len(tag_sets) == 1 and tag_sets[0][0] == "Event and pen tags (identical)",
    }
