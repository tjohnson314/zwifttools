"""Extract per-route elevation profiles (lead-in + main lap) directly from the
Zwift game client's compressed world archives.

For each ``assets/Worlds/world*/data_1.wad`` this tool:

1. Decompresses the ``ZWF!`` archive in memory (Zwift's custom LZ format).
2. Reads every ``routes/*.xml`` entry from the archive's table of contents.
3. Parses each route's header plus its ``<leadinhighrescheckpoint>`` and
   ``<highrescheckpoint>`` polylines.
4. Converts the raw world geometry (game units, 100 units == 1 metre) into
    metre-based ``(distance, altitude, x, z)`` profiles. Coordinates are the raw
    values divided by 100 (no clamping). Cumulative planar distance is summed
    directly from the checkpoint geometry (game units -> metres). Zwift's header
    ``distanceInMeters`` over-states the ridden path (mean ~3% across Watopia,
    up to ~13% on pretzel routes); the in-game odometer tracks the geometry, so
    the geometry is the physical distance. Header distances are still stored as
    metadata for reference.

The decompression algorithm is a faithful in-memory port of r3dey3's
``decode_wad.py`` (https://gitlab.com/r3dey3/zwift-utils).

Usage:
    python tools/extract_zwift_routes.py \
        [--zwift-dir "C:\\Program Files (x86)\\Zwift"] \
        [--out zwift_routes] [--max-points 600]
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET


# --------------------------------------------------------------------------- #
# WAD decompression (in-memory port of r3dey3/zwift-utils decode_wad.py)
# --------------------------------------------------------------------------- #
def uncompress(c: bytes, start: int) -> bytearray:
    """Decompress a Zwift ``ZWF!`` LZ stream starting at ``start``."""
    out = bytearray()
    p = start
    n = len(c)

    def simple_chunk(p: int) -> int:
        copy_len = c[p]; p += 1
        if copy_len != 0:
            copy_len += 2
        else:
            copy_len = c[p] | (c[p + 1] << 8); p += 2
        out.extend(c[p:p + copy_len]); p += copy_len
        return p

    p = simple_chunk(p)  # first block is always a literal chunk

    while p < n:
        b0 = c[p]; b1 = c[p + 1]; p += 2
        enc_type = b0 & 0xE0
        if enc_type < 0xC0:
            ret_val = b0 & 3
            copy_len = (b0 >> 5) + 4
            copy_off = b1 | ((b0 & 0xC) << 6)
        elif enc_type == 0xC0:
            b2 = c[p]; p += 1
            ret_val = b1 & 3
            copy_len = (b0 & 0x1F) + 4
            copy_off = b2 | ((b1 & 0xFC) << 6)
        else:  # 0xE0
            b2 = c[p]; p += 1
            copy_len = (b0 & 0xF) + 3
            if (b0 & 0xF) != 0:
                ret_val = b1 & 3
                v15 = (b1 & 0xFC) | (16 * (b0 & 0x10))
                copy_off = b2 | (v15 << 6)
            else:
                copy_len = b1 + 18
                b3 = c[p]; p += 1
                if copy_len <= 0x12:
                    b4 = c[p]; b5 = c[p + 1]; p += 2
                    ret_val = b4 & 3
                    copy_len = b3 | (b2 << 8)
                    v15 = (b4 & 0xFC) | (16 * (b0 & 0x10))
                    copy_off = b5 | (v15 << 6)
                else:
                    ret_val = b2 & 3
                    v15 = (b2 & 0xFC) | (16 * (b0 & 0x10))
                    copy_off = b3 | (v15 << 6)

        s = len(out) - copy_off
        out.extend(out[s:s + copy_len])

        if ret_val != 3 and p < n:
            if ret_val:
                out.extend(c[p:p + ret_val]); p += ret_val
            else:
                p = simple_chunk(p)

    return out


def read_wad_entries(path: str, keep_substrings: tuple[str, ...] | None = None) -> dict[str, bytes]:
    """Decompress ``path`` and return ``{entry_name: data}`` for its TOC.

    ``entry_name`` uses forward slashes (e.g. ``Worlds/world10/routes/routes0.xml``).
    If ``keep_substrings`` is given, only entries whose normalised name contains
    one of the substrings are returned.
    """
    c = open(path, "rb").read()
    sig, version, decomp_size, comp_size = struct.unpack("<LLLL", c[0xF0:0x100])
    buf = uncompress(c, 0x100)
    if len(buf) != decomp_size:
        raise ValueError(
            f"{path}: decompressed {len(buf)} bytes, expected {decomp_size}"
        )

    entries: dict[str, bytes] = {}
    total = len(buf)
    p = 0x2000
    while p < total:
        p += 4  # unknown
        name = buf[p:p + 0x60].strip(b"\x00").decode("ascii", "replace"); p += 0x60
        if p + 8 > total:
            break
        _, size = struct.unpack("<LL", buf[p:p + 8]); p += 8
        p += 0x54
        data = bytes(buf[p:p + size]); p += size
        if size % 0x40:
            p += 0x40 - (size % 0x40)
        norm = name.replace("\\", "/").lstrip("/")
        if keep_substrings is None or any(sub in norm for sub in keep_substrings):
            entries[norm] = data
    return entries


# --------------------------------------------------------------------------- #
# Route XML parsing
# --------------------------------------------------------------------------- #
UNITS_PER_METRE = 100.0  # calibrated: game world units are centimetres
UNSET_STYLE = 31  # 0x1F sentinel in <defaultStyle> -> NORMAL/tarmac

_STYLE_SURFACE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "zwift_surfaces", "style_surface_map.json")


def _load_style_surface() -> dict:
    """Authoritative roadstyle NAME -> surface category map (e.g. SNOW -> Tarmac)."""
    try:
        with open(_STYLE_SURFACE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


def parse_roadstyles(data: bytes) -> list[str]:
    """roadstyle.xml segment names in document order (index == style id)."""
    txt = data.decode("utf-8", "replace")
    return re.findall(r'<segment\s+[^>]*\bstyle="([^"]+)"', txt)


def parse_roads(road_xml: str) -> dict:
    """road_id -> (base_style_id, [(t0, t1, style_id), ...]) from road.xml."""
    roads: dict[int, tuple[int, list]] = {}
    for m in re.finditer(r"<road>(.*?)</road>", road_xml, re.S):
        block = m.group(1)
        idm = re.search(r"<id>(\d+)</id>", block)
        if not idm:
            continue
        dm = re.search(r"<defaultStyle>(\d+)</defaultStyle>", block)
        base = int(dm.group(1)) if dm else UNSET_STYLE
        markers = []
        for em in re.finditer(r'<ent\b[^>]*type="ENTITY_TYPE_ROADMARKER"[^>]*>', block):
            tag = em.group(0)
            # Mirror extract_zwift_surfaces._markers so route surfaces match the
            # network overlay / mod bundle: a marker with only one road-time
            # defaults the missing endpoint to the road boundary; a style-less
            # visible marker is asphalt (style 0); a style-less invisible one is
            # skipped (base shows through); only a marker with NO road-time at
            # all is position-less and dropped.
            t1 = re.search(r'm_roadTime1="([-\d.]+)"', tag)
            t2 = re.search(r'm_roadTime2="([-\d.]+)"', tag)
            if not (t1 or t2):
                continue
            st = re.search(r'm_style="(\d+)"', tag)
            invisible = re.search(r'm_isInvisible="1"', tag) is not None
            if st is not None:
                style = int(st.group(1))
            elif invisible:
                continue
            else:
                style = 0
            a = float(t1.group(1)) if t1 else 0.0
            b = float(t2.group(1)) if t2 else 1.0
            if b < a:
                a, b = b, a
            markers.append((a, b, style))
        roads[int(idm.group(1))] = (base, markers)
    return roads


def _resolve_style_name(rid: int, t: float, roads: dict, styles: list[str]) -> str:
    """Authoritative roadstyle NAME for a checkpoint via road-id + time join.

    Unambiguous even where roads run parallel (unlike spatial nearest-vertex),
    because each checkpoint carries the exact road id and normalised position it
    lies on.
    """
    base, markers = roads.get(rid, (UNSET_STYLE, []))
    style = base
    for a, b, s in markers:
        if a <= t <= b:
            style = s
    if style == UNSET_STYLE or not (0 <= style < len(styles)):
        style = 0
    return styles[style] if 0 <= style < len(styles) else "NORMAL"


def load_multiroot(data: bytes) -> ET.Element:
    """routes/*.xml contains several top-level elements; wrap them in a root."""
    txt = data.decode("utf-8", "replace")
    txt = re.sub(r"<\?xml[^>]*\?>", "", txt, count=1)
    return ET.fromstring("<root>" + txt + "</root>")


def parse_paddock_roads(road_data: bytes, entity_data: bytes | None = None) -> list[dict]:
    """Read marked pen roads in local metres without guessing route paddock IDs."""
    spawn_markers: dict[int, list[dict]] = {}
    if entity_data:
        for entity in load_multiroot(entity_data).iter("ent"):
            if entity.get("type") != "ENTITY_TYPE_SPAWNMARKER":
                continue
            if entity.get("m_roadId") is None or entity.get("m_roadTime") is None:
                continue
            spawn_markers.setdefault(int(entity.get("m_roadId")), []).append({
                "name": entity.get("m_Name") or "",
                "road_time": float(entity.get("m_roadTime")),
            })

    def vector(entity: ET.Element, attribute: str) -> list[float] | None:
        value = entity.get(attribute)
        if value is None:
            return None
        values = [float(component) / UNITS_PER_METRE
                  for component in value.strip("{} ").split(",")]
        if len(values) != 3:
            raise ValueError(f"Invalid {attribute} vector: {value}")
        return values

    paddocks = []
    for road in load_multiroot(road_data).iter("road"):
        if (road.findtext("roadIsPaddock") or "").strip().lower() not in ("true", "1"):
            continue
        road_id = int(road.findtext("id"))
        nodes = []
        for entity in road.findall("ent"):
            if entity.get("type") != "ENTITY_TYPE_ROADNODE":
                continue
            position = vector(entity, "m_pos")
            if position is None:
                continue
            nodes.append({
                "x": position[0], "alt": position[1], "z": position[2],
                "tangent_in": vector(entity, "m_tangentIn"),
                "tangent_out": vector(entity, "m_tangentOut"),
            })
        exit_time = road.findtext("paddockExitRoadTime")
        paddocks.append({
            "road_id": road_id,
            "name": road.findtext("roadName") or "",
            "paddock_exit_road_time": float(exit_time) if exit_time is not None else None,
            "nodes": nodes,
            "spawn_markers": spawn_markers.get(road_id, []),
        })
    return paddocks


def _points(el: ET.Element | None) -> list[tuple[float, float, float]]:
    if el is None:
        return []
    pts = []
    for e in el.findall("entry"):
        try:
            pts.append((float(e.get("x")), float(e.get("y")), float(e.get("z"))))
        except (TypeError, ValueError):
            continue
    return pts


def _leg_points(el: ET.Element | None):
    """Aligned geometry ``[(x, y, z)]`` and per-entry ``(road_id, time)``."""
    pts: list[tuple[float, float, float]] = []
    rts: list[tuple[int, float]] = []
    if el is None:
        return pts, rts
    for e in el.findall("entry"):
        try:
            x, y, z = float(e.get("x")), float(e.get("y")), float(e.get("z"))
        except (TypeError, ValueError):
            continue
        pts.append((x, y, z))
        try:
            rid = int(e.get("road"))
        except (TypeError, ValueError):
            rid = -1
        try:
            tt = float(e.get("time"))
        except (TypeError, ValueError):
            tt = 0.0
        rts.append((rid, tt))
    return pts, rts


def _cumulative_planar(pts: list[tuple[float, float, float]]) -> list[float]:
    """Cumulative horizontal distance (game units) along the polyline."""
    cum = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dz = pts[i][2] - pts[i - 1][2]
        cum.append(cum[-1] + math.hypot(dx, dz))
    return cum


def _resample(dist_m: list[float], pts: list[tuple[float, float, float]], n: int,
              surfaces: list[str] | None = None,
              preserve_positions: list[float] | None = None,
              position_precision: int = 2) -> dict[str, list]:
    """Refine an initial ``n`` samples to retain boundaries and <=1 m 3D error.

    ``surfaces`` (one label per source point) is resampled as a step function:
    each output point takes the surface of the nearer bracketing source point, so
    style boundaries are preserved rather than blended.
    """
    x_m = [p[0] / UNITS_PER_METRE for p in pts]
    alt_m = [p[1] / UNITS_PER_METRE for p in pts]
    z_m = [p[2] / UNITS_PER_METRE for p in pts]
    if (len(dist_m) <= n and not preserve_positions) or len(dist_m) < 2:
        out: dict[str, list] = {
            "d": [round(d, position_precision) for d in dist_m],
            "alt": [round(a, 3) for a in alt_m],
            "x": [round(x, 3) for x in x_m],
            "z": [round(z, 3) for z in z_m],
        }
        if surfaces is not None:
            out["surface"] = list(surfaces)
        return out
    total = dist_m[-1]
    targets = set(dist_m if len(dist_m) <= n else
                  (total * index / (n - 1) for index in range(n)))
    targets.update(position for position in (preserve_positions or [])
                   if dist_m[0] <= position <= total)
    if surfaces is not None:
        for index in range(1, len(surfaces)):
            if surfaces[index] != surfaces[index - 1]:
                targets.update((dist_m[index - 1], dist_m[index]))

    def sample(target):
        index = max(0, min(bisect_right(dist_m, target) - 1, len(dist_m) - 2))
        span = dist_m[index + 1] - dist_m[index]
        weight = (target - dist_m[index]) / span if span else 0.0
        return tuple(values[index] + weight * (values[index + 1] - values[index])
                     for values in (x_m, alt_m, z_m))

    ordered = sorted(targets)
    pending = list(zip(ordered, ordered[1:]))
    while pending:
        start, end = pending.pop()
        first, last = sample(start), sample(end)
        worst_error, worst_position = 1.0, None
        for index in range(bisect_right(dist_m, start), bisect_left(dist_m, end)):
            weight = (dist_m[index] - start) / (end - start)
            error = sum((values[index] - (initial + weight * (final - initial))) ** 2
                        for values, initial, final in zip((x_m, alt_m, z_m), first, last))
            if error > worst_error:
                worst_error, worst_position = error, dist_m[index]
        if worst_position is not None:
            targets.add(worst_position)
            pending.extend(((start, worst_position), (worst_position, end)))

    out_d, out_a, out_x, out_z = [], [], [], []
    out_s: list[str] = []
    j = 0
    for target in sorted(targets):
        while j < len(dist_m) - 2 and dist_m[j + 1] < target:
            j += 1
        d0, d1 = dist_m[j], dist_m[j + 1]
        frac = 0.0 if d1 == d0 else (target - d0) / (d1 - d0)
        alt = alt_m[j] + frac * (alt_m[j + 1] - alt_m[j])
        x = x_m[j] + frac * (x_m[j + 1] - x_m[j])
        z = z_m[j] + frac * (z_m[j + 1] - z_m[j])
        out_d.append(round(target, position_precision))
        out_a.append(round(alt, 3))
        out_x.append(round(x, 3))
        out_z.append(round(z, 3))
        if surfaces is not None:
            out_s.append(surfaces[j + 1] if frac >= 0.5 else surfaces[j])
    res: dict[str, list] = {"d": out_d, "alt": out_a, "x": out_x, "z": out_z}
    if surfaces is not None:
        res["surface"] = out_s
    return res


def build_profile(pts: list[tuple[float, float, float]],
                  max_points: int, surfaces: list[str] | None = None,
                  preserve_distances: list[float] | None = None) -> dict | None:
    """Convert a checkpoint polyline into a metre-based route geometry profile.

    Distance is summed from the actual checkpoint geometry (game units -> metres)
    at full resolution before resampling. Zwift's header ``distanceInMeters``
    over-states the ridden path; the in-game odometer tracks this geometry.
    """
    if len(pts) < 2:
        return None
    cum = _cumulative_planar(pts)
    raw_total = cum[-1]
    if raw_total <= 0:
        return None
    dist_m = [u / UNITS_PER_METRE for u in cum]
    return _resample(dist_m, pts, max_points, surfaces, preserve_distances)


def build_segment_path(pts: list[tuple[float, float, float]], max_points: int,
                       preserve_positions: list[float] | None = None) -> dict | None:
    """Build an adaptive path whose ``d`` retains original WAD checkpoint indices."""
    if len(pts) < 2:
        return None
    return _resample([float(i) for i in range(len(pts))], pts, max_points,
                     preserve_positions=preserve_positions, position_precision=9)


def _f(el: ET.Element, attr: str, default: float = 0.0) -> float:
    try:
        return float(el.get(attr))
    except (TypeError, ValueError):
        return default


def _i(el: ET.Element, attr: str, default: int = 0) -> int:
    try:
        return int(el.get(attr))
    except (TypeError, ValueError):
        return default


def parse_segment_data(root: ET.Element) -> list[dict]:
    """Extract the route's in-game leaderboard segment boundaries.

    Segment percentages are normalized positions in the route checkpoint
    sequence, rather than cumulative physical distance or rider odometer data.
    """
    segment_data = root.find("segment_data")
    if segment_data is None:
        return []
    segments = []
    for entry in segment_data.findall("entry"):
        try:
            segments.append({
                "hash": int(entry.get("hash")),
                "percent_start": float(entry.get("percentStart")),
                "percent_end": float(entry.get("percentEnd")),
            })
        except (TypeError, ValueError):
            continue
    return segments


def _directional_segment_name(name: str, direction: str) -> str:
    if direction == "R" and not re.search(r"\breverse\b", name, re.IGNORECASE):
        return f"{name} Reverse"
    return name


def parse_segment_names(data: bytes) -> dict[int, str]:
    """Return leaderboard segment names keyed by their directional network hash."""
    root = load_multiroot(data)
    names: dict[int, str] = {}
    for element in root.iter():
        attributes = element.attrib
        for direction in ("F", "R"):
            hash_value = attributes.get(f"m_overrideNetworkHash{direction}")
            if not hash_value:
                continue
            name = next((
                attributes.get(key, "").strip()
                for key in (
                    f"m_ArchFriendlyMaleName{direction}",
                    f"m_ArchFriendlyName{direction}",
                    f"m_ArchSegmentName{direction}",
                    f"m_ArchName{direction}",
                    "m_ArchFriendlyMaleName",
                    "m_ArchFriendlyName",
                    "m_ArchSegmentName",
                    "m_ArchName",
                    "m_Name",
                    "m_leaderJersey_Name",
                )
                if attributes.get(key, "").strip()
            ), "")
            if name:
                names.setdefault(int(hash_value), _directional_segment_name(name, direction))
    return names


def parse_segment_definitions(data: bytes) -> list[dict]:
    """Extract directional segment boundaries from timing-arch entities."""
    root = load_multiroot(data)
    definitions = []
    for element in root.iter("ent"):
        attributes = element.attrib
        if attributes.get("type") != "ENTITY_TYPE_TIMINGARCH":
            continue
        try:
            road_id = int(attributes.get("m_roadId", "0"))
            finish_time = float(attributes["m_roadTime"])
        except (KeyError, TypeError, ValueError):
            continue
        lap_arch = (attributes.get("m_bStopwatchRequiresAllCheckpoints") == "1"
                    or attributes.get("m_telemetryArchType") == "4"
                    or "laparch" in attributes.get("m_archGdePath", "").casefold())
        for direction in ("F", "R"):
            try:
                start_time = float(attributes[f"m_startLineSplineTime{direction}"])
                segment_hash = int(attributes[f"m_overrideNetworkHash{direction}"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(start_time) or not math.isfinite(finish_time):
                continue
            name = next((
                attributes.get(key, "").strip()
                for key in (
                    f"m_ArchFriendlyMaleName{direction}",
                    f"m_ArchFriendlyName{direction}",
                    "m_ArchFriendlyMaleName",
                    "m_ArchFriendlyName",
                )
                if attributes.get(key, "").strip()
            ), "")
            if not name:
                continue
            try:
                segment_distance = float(attributes[f"m_ArchSegmentDistance{direction}"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                begin_timer = float(attributes[f"m_beginTimerSplineTime{direction}"])
            except (KeyError, TypeError, ValueError):
                begin_timer = None
            gap = abs(start_time - finish_time)
            definitions.append({
                "hash": segment_hash,
                "name": _directional_segment_name(name, direction),
                "road_id": road_id,
                "start_time": start_time,
                "end_time": finish_time,
                "direction": direction,
                "distance_m": segment_distance * 1000,
                "is_loop": lap_arch and min(gap, abs(1 - gap)) <= 0.0001,
                "begin_timer_time": begin_timer if begin_timer is not None and math.isfinite(begin_timer) else None,
            })
    return definitions


def parse_escalator_definitions(data: bytes) -> list[dict]:
    """Read directional subway bonus-power volumes, not visual props."""
    definitions = []
    for entity in load_multiroot(data).iter("ent"):
        attributes = entity.attrib
        if attributes.get("type") != "ENTITY_TYPE_VOLUME_EVENTTRIGGER":
            continue
        responses = attributes.get("m_selectedResponsesForSave", "").split(";")
        name = attributes.get("m_triggerTagName", "")
        if not name.startswith("Subway") or "Set Bonus Watts" not in responses:
            continue
        parameters = attributes["m_selectedResponseParametersForSave"].split(";")
        definitions.append({
            "name": name,
            "road_id": int(attributes.get("m_roadId", "0")),
            "start_time": float(attributes["m_backRoadTime"]),
            "end_time": float(attributes["m_frontRoadTime"]),
            # Trigger 0/1 means forward/reverse; checkpoint dir 1/0 does.
            "direction": (1 - int(attributes.get("m_triggerDir", "0"))
                          if attributes.get("m_bDirectionalTrigger") == "1" else None),
            "bonus_watts": float(parameters[responses.index("Set Bonus Watts")]),
            "trainer_slope_scale": float(attributes["m_trainerSlopeScale"]),
        })
    return definitions


def derive_escalator_passages(root: ET.Element, definitions: list[dict],
                              leadin_distance_m: float) -> list[dict]:
    """Intersect trigger road-time intervals with full-resolution route edges."""
    passages = []
    for leg_name, offset in (("leadinhighrescheckpoint", 0.0),
                             ("highrescheckpoint", leadin_distance_m)):
        leg = root.find(leg_name)
        if leg is None:
            continue
        entries = leg.findall("entry")
        points, road_times = _leg_points(leg)
        if len(points) != len(entries):
            raise ValueError("Cannot locate escalators on invalid route checkpoints")
        distances = [value / UNITS_PER_METRE for value in _cumulative_planar(points)]
        for definition in definitions:
            low = min(definition["start_time"], definition["end_time"])
            high = max(definition["start_time"], definition["end_time"])
            current = None
            for index in range(len(entries) - 1):
                road0, time0 = road_times[index]
                road1, time1 = road_times[index + 1]
                direction = definition["direction"]
                if (road0 != definition["road_id"] or road1 != road0 or time0 == time1
                        or (direction is not None and (
                            int(entries[index].get("dir", "0")) != direction
                            or int(entries[index + 1].get("dir", "0")) != direction))):
                    current = None
                    continue
                start = max(0.0, min((low - time0) / (time1 - time0),
                                     (high - time0) / (time1 - time0)))
                end = min(1.0, max((low - time0) / (time1 - time0),
                                   (high - time0) / (time1 - time0)))
                if end <= start:
                    current = None
                    continue
                edge_distance = distances[index + 1] - distances[index]
                start_distance = offset + distances[index] + start * edge_distance
                end_distance = offset + distances[index] + end * edge_distance
                if current is not None and math.isclose(
                        current["end_distance_m"], start_distance, abs_tol=1e-6):
                    current["end_distance_m"] = end_distance
                else:
                    current = {
                        **definition, "leg": "leadin" if leg_name == "leadinhighrescheckpoint" else "route",
                        "start_distance_m": start_distance, "end_distance_m": end_distance,
                    }
                    passages.append(current)
    passages.sort(key=lambda passage: passage["start_distance_m"])
    for passage in passages:
        for key in ("start_distance_m", "end_distance_m"):
            passage[key] = round(passage[key], 2)
    return passages


def derive_segment_data(root: ET.Element, definitions: list[dict]) -> list[dict]:
    """Locate complete directional timing-line crossings on native route edges."""
    legs = [root.find("leadinhighrescheckpoint"), root.find("highrescheckpoint")]
    entries = [entry for leg in legs if leg is not None for entry in leg.findall("entry")]
    leadin_count = len(legs[0].findall("entry")) if legs[0] is not None else 0
    main_entries = entries[leadin_count:]
    route_header = root.find("route")
    closed_main = (route_header is not None and _i(route_header, "supportedLaps") > 0
                   and len(main_entries) > 2
                   and all(main_entries[0].get(key) == main_entries[-1].get(key)
                           for key in ("road", "dir"))
                   and all(_f(main_entries[0], key) == _f(main_entries[-1], key)
                           for key in ("time", "x", "y", "z")))
    edges = {}
    for index, (first, last) in enumerate(zip(entries, entries[1:])):
        try:
            road0, road1 = int(first.get("road")), int(last.get("road"))
            time0, time1 = float(first.get("time")), float(last.get("time"))
            dir0, dir1 = int(first.get("dir")), int(last.get("dir"))
        except (TypeError, ValueError):
            continue
        if road0 == road1 and dir0 == dir1 and math.isfinite(time0) and math.isfinite(time1):
            edges.setdefault((road0, dir0), []).append((index, time0, time1))
    derived = []
    for definition in definitions:
        direction = definition.get("direction", "F" if definition["end_time"] > definition["start_time"] else "R")
        checkpoint_direction = 1 if direction == "F" else 0
        is_loop = definition.get("is_loop", False)
        timer_time = definition.get("begin_timer_time")
        timer_boundary = timer_time % 1 if timer_time is not None else None
        start = None
        lap_armed = None
        previous = -2
        previous_passages = len(derived)
        matching_edges = edges.get((definition["road_id"], checkpoint_direction), [])
        for index, time0, time1 in matching_edges:
            if index != previous + 1:
                start = None
                lap_armed = None
            previous = index
            if index == leadin_count - 1:
                continue
            monotonic = time0 != time1 and (time1 > time0) == bool(checkpoint_direction)
            if (is_loop and timer_boundary is None and start is not None and not monotonic
                    and ((time0 > 0.5 > time1) if checkpoint_direction else (time0 < 0.5 < time1))):
                lap_armed = index + 1
            crossings = []
            boundaries = ([(boundary, road_time)
                           for road_time in {definition["start_time"], definition["end_time"]}
                           for boundary in ("start", "end")] if is_loop else
                          [("start", definition["start_time"]), ("end", definition["end_time"])])
            if is_loop and timer_boundary is not None:
                boundaries.append(("arm", timer_boundary))
            for boundary, road_time in boundaries:
                if road_time == time0:
                    weight = 0.0
                elif road_time == time1:
                    weight = 1.0
                elif monotonic:
                    weight = (road_time - time0) / (time1 - time0)
                else:
                    continue
                if 0 <= weight <= 1:
                    crossings.append((weight, boundary, road_time))
            for weight, boundary, road_time in sorted(crossings):
                position = index + weight
                if boundary == "start" and (start is None or not is_loop):
                    start = position
                    start_line_time = road_time
                    lap_armed = None
                elif boundary == "arm" and start is not None and position > start:
                    lap_armed = position
                elif (boundary == "end" and start is not None and position > start
                      and (not is_loop or (lap_armed is not None and position >= lap_armed))):
                    derived.append({
                        "hash": definition["hash"], "name": definition["name"],
                        "road_id": definition["road_id"], "direction": direction,
                        "start_time": definition["start_time"], "end_time": definition["end_time"],
                        "declared_distance_m": definition.get("distance_m"),
                        "boundary_source": "timing_arch",
                        "start_checkpoint": start, "end_checkpoint": position,
                        "percent_start": start / max(len(entries) - 1, 1),
                        "percent_end": position / max(len(entries) - 1, 1),
                        **({"is_loop": True, "effective_start_time": start_line_time,
                            "effective_end_time": road_time,
                            "begin_timer_time": timer_time,
                            "lap_armed_checkpoint": lap_armed} if is_loop else {}),
                    })
                    start = None
                    lap_armed = None
        if (is_loop and closed_main and len(derived) == previous_passages
                and start is not None and leadin_count <= start <= leadin_count + 1
                and lap_armed is not None and start < lap_armed <= len(entries) - 1
                and sum(index >= leadin_count for index, _, _ in matching_edges) == len(main_entries) - 1):
            derived.append({
                "hash": definition["hash"], "name": definition["name"],
                "road_id": definition["road_id"], "direction": direction,
                "start_time": definition["start_time"], "end_time": definition["end_time"],
                "declared_distance_m": definition.get("distance_m"),
                "boundary_source": "closed_route_lap", "is_loop": True,
                "start_checkpoint": leadin_count, "end_checkpoint": len(entries) - 1,
                "percent_start": leadin_count / (len(entries) - 1), "percent_end": 1.0,
                "effective_start_time": start_line_time, "effective_end_time": start_line_time,
                "begin_timer_time": timer_time, "lap_armed_checkpoint": lap_armed,
                "timing_arch_checkpoint": start,
            })
    derived.sort(key=lambda segment: (segment["start_checkpoint"], segment["end_checkpoint"], segment["hash"]))
    return derived


def build_leadin_view(pts, road_times, main_pts, published_distance_m,
                      paddock_road_ids, max_points, surfaces=None,
                      preserve_distances=None) -> dict | None:
    """Omit a paddock prefix only when header error falls from >50 m to <10 m."""
    if not pts or not main_pts or not paddock_road_ids:
        return None
    first_regular = next((index for index, (road_id, _) in enumerate(road_times)
                          if road_id not in paddock_road_ids), len(pts))
    if not first_regular or first_regular == len(pts):
        return None
    if any(road_id in paddock_road_ids for road_id, _ in road_times[first_regular:]):
        return None
    distances = _cumulative_planar(pts + [main_pts[0]])
    full_distance = distances[-1] / UNITS_PER_METRE
    paddock_distance = distances[first_regular] / UNITS_PER_METRE
    remaining_distance = full_distance - paddock_distance
    error_before = abs(full_distance - published_distance_m)
    error_after = abs(remaining_distance - published_distance_m)
    if not (error_before > 50 and error_after < 10):
        return None
    profile = build_profile(pts[first_regular:], max_points,
                            surfaces[first_regular:] if surfaces is not None else None,
                            [distance - paddock_distance for distance in (preserve_distances or [])
                             if distance >= paddock_distance])
    if profile is None:
        return None
    return {
        "profile": profile,
        "full_distance_m": full_distance,
        "distance_m": remaining_distance,
        "paddock_distance_m": paddock_distance,
        "error_before_m": error_before,
        "error_after_m": error_after,
        "excluded_checkpoint_count": first_regular,
        "paddock_road_ids": sorted({road_id for road_id, _ in road_times[:first_regular]}),
    }


def route_start_locations(root):
    header = root.find("route")
    locations = {}
    for mode, tag, prefix in (("free_ride", "spawn_area", "freeRide"),
                              ("meetup", "private_spawn_area", "meetup")):
        area = root.find(tag)
        distance = header.get(f"{prefix}LeadinDistanceInMeters")
        ascent = header.get(f"{prefix}LeadinAscentInMeters")
        if area is not None or distance is not None:
            locations[mode] = {
                "spawn_area": dict(area.attrib) if area is not None else None,
                "leadin_distance_m": float(distance) if distance is not None else None,
                "leadin_ascent_m": float(ascent) if ascent is not None else None,
            }
    return locations


def native_start_paths(root, source="route"):
    lead = root.findall("./leadinhighrescheckpoint/entry")
    main = root.findall("./highrescheckpoint/entry")
    return [(lead + main, source)] if lead or main else []


def _native_crossings(entries, road, direction, road_time):
    crossings = []
    for index, entry in enumerate(entries):
        if entry.get("road") != road or entry.get("dir") != direction:
            continue
        current_time = float(entry.get("time"))
        if current_time == road_time:
            crossings.append(float(index))
        if index + 1 == len(entries):
            continue
        following = entries[index + 1]
        if following.get("road") != road or following.get("dir") != direction:
            continue
        next_time = float(following.get("time"))
        if (next_time - current_time) * (1 if direction == "1" else -1) <= 0:
            continue
        if min(current_time, next_time) < road_time < max(current_time, next_time):
            crossings.append(index + (road_time - current_time) / (next_time - current_time))
    return crossings


def _native_start_slice(paths, area, target):
    if area.get("road") is None or area.get("forward") not in ("0", "1"):
        return None
    try:
        spawn_time = (float(area["starttime"]) + float(area["endtime"])) / 2
        target_time = float(target.get("time"))
        if not math.isfinite(spawn_time) or not math.isfinite(target_time):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    best = None
    for entries, source in paths:
        starts = _native_crossings(entries, area["road"], area["forward"], spawn_time)
        if not starts:
            continue
        ends = _native_crossings(entries, target.get("road"), target.get("dir"), target_time)
        for start in starts:
            end = next((position for position in ends if position >= start), None)
            if end is None:
                continue
            def checkpoint(position):
                index = int(position)
                first = entries[index]
                if position == index:
                    return tuple(float(first.get(key)) for key in ("x", "y", "z")), first
                last = entries[index + 1]
                fraction = position - index
                return tuple(float(first.get(key)) + fraction * (float(last.get(key)) - float(first.get(key)))
                             for key in ("x", "y", "z")), first
            first, first_entry = checkpoint(start)
            last, last_entry = checkpoint(end)
            target_point = tuple(float(target.get(key)) for key in ("x", "y", "z"))
            if math.hypot(last[0] - target_point[0], last[2] - target_point[2]) > 200:
                continue
            middle = entries[math.floor(start) + 1:math.ceil(end)]
            points = [first] + [tuple(float(entry.get(key)) for key in ("x", "y", "z")) for entry in middle] + [last]
            distance = _cumulative_planar(points)[-1]
            if best is None or distance < best[0]:
                best = (distance, points, [first_entry] + middle + [last_entry], source)
    return best


def build_road_paths(entries, points=None):
    if points is None:
        points = [tuple(float(entry.get(key)) for key in ("x", "y", "z")) for entry in entries]
    paths = {}
    current_road = None
    run = []

    def save():
        if current_road is None or not run:
            return
        profile = build_profile(run, 120)
        path = {key: profile[key] for key in ("x", "z")} if profile else {
            "x": [run[0][0] / UNITS_PER_METRE], "z": [run[0][2] / UNITS_PER_METRE]}
        paths.setdefault(str(current_road), []).append(path)

    for index, (entry, point) in enumerate(zip(entries, points)):
        road = int(entry.get("road")) if entry.get("road") is not None else None
        if road != current_road:
            if run:
                run.append(point)
                save()
            run = [points[index - 1], point] if index and current_road is not None else [point]
            current_road = road
        else:
            run.append(point)
    save()
    return paths


def build_route_versions(root, route, max_points=600, reference_paths=(),
                         roads=None, styles=None, surface_map=None):
    main_entries = list(root.findall("./highrescheckpoint/entry"))
    race_entries = list(root.findall("./leadinhighrescheckpoint/entry")) + main_entries
    main_roads = {int(entry.get("road")) for entry in root.findall("./highrescheckpoint/entry")
                  if entry.get("road") is not None}
    race_roads = main_roads | {int(entry.get("road")) for entry in root.findall("./leadinhighrescheckpoint/entry")
                               if entry.get("road") is not None}
    versions = {"race": {"available": True, "geometry_status": "native" if route.get("leadin") else "missing_leadin",
                         "leadin_distance_m": route["leadin_distance_m"],
                         "leadin_ascent_m": route["leadin_ascent_m"], "road_ids": sorted(race_roads),
                         "road_paths": build_road_paths(race_entries)}}
    location = route.get("start_locations", {}).get("free_ride")
    if not location or route.get("event_only"):
        versions["free_ride"] = {"available": False}
        return versions
    version = {**location, "available": True, "leadin": None, "geometry_status": "missing_leadin",
               "road_ids": sorted(main_roads), "road_paths": build_road_paths(main_entries)}
    target = root.find("./highrescheckpoint/entry")
    area = location.get("spawn_area")
    if target is not None and area:
        result = _native_start_slice(native_start_paths(root), area, target)
        if result is None:
            result = _native_start_slice(reference_paths, area, target)
        if result is not None:
            _distance, points, entries, source = result
            surfaces = None
            if roads and styles and surface_map is not None:
                surfaces = [surface_map.get(_resolve_style_name(int(entry.get("road")), float(entry.get("time")), roads, styles),
                                            "Unknown") for entry in entries]
            version.update({"leadin": build_profile(points, max_points, surfaces), "geometry_status": "native",
                            "position_source": "native_checkpoints_spawn_range_midpoint", "source_route": source,
                            "road_ids": sorted(main_roads | {int(entry.get("road")) for entry in entries}),
                            "road_paths": build_road_paths(entries + main_entries, points +
                                [tuple(float(entry.get(key)) for key in ("x", "y", "z")) for entry in main_entries])})
    versions["free_ride"] = version
    return versions


def parse_route(data: bytes, map_id: int, max_points: int,
                roads: dict | None = None, styles: list[str] | None = None,
                surface_map: dict | None = None,
                segment_names: dict[int, str] | None = None,
                segment_definitions: list[dict] | None = None,
                escalator_definitions: list[dict] | None = None,
                paddock_road_ids: set[int] | None = None,
                reference_paths=()) -> dict | None:
    root = load_multiroot(data)
    r = root.find("route")
    if r is None:
        return None
    name = (r.get("name") or "").strip()
    leadin_pts, leadin_rt = _leg_points(root.find("leadinhighrescheckpoint"))
    main_pts, main_rt = _leg_points(root.find("highrescheckpoint"))

    distance_m = _f(r, "distanceInMeters")
    leadin_distance_m = _f(r, "leadinDistanceInMeters")

    def _surfaces(rt: list[tuple[int, float]]) -> list[str] | None:
        # Authoritative per-point surface via the road-id + time join. Robust to
        # parallel roads (where spatial nearest-vertex snaps to the wrong road).
        if not roads or not styles or surface_map is None:
            return None
        return [surface_map.get(_resolve_style_name(rid, t, roads, styles), "Unknown")
                for rid, t in rt]

    native_segments = parse_segment_data(root)
    segment_entries = (derive_segment_data(root, segment_definitions)
                       if segment_definitions is not None else native_segments)
    segment_positions, leadin_boundaries, main_boundaries = [], [], []
    if segment_entries:
        combined_points = leadin_pts + main_pts
        physical_distance = [value / UNITS_PER_METRE for value in _cumulative_planar(combined_points)]
        checkpoint_positions = [float(i) for i in range(len(combined_points))]
        def _distance_at(position: float) -> float:
            index = min(int(position), len(physical_distance) - 2)
            fraction = position - index
            return physical_distance[index] + fraction * (physical_distance[index + 1] - physical_distance[index])
        for segment in segment_entries:
            if ("end_checkpoint" not in segment and segment["percent_end"] == 0 < segment["percent_start"] <= 1
                    and _i(r, "supportedLaps") > 0 and len(main_pts) > 1
                    and main_pts[0] == main_pts[-1]):
                segment["resolved_percent_end"] = 1.0
            for boundary in ("start", "end"):
                fraction = segment.get(f"resolved_percent_{boundary}", segment[f"percent_{boundary}"])
                position = segment.get(f"{boundary}_checkpoint", fraction * checkpoint_positions[-1])
                distance = _distance_at(position)
                segment_positions.append(position)
                segment[f"{boundary}_distance_m"] = round(distance, 2)
                if position <= len(leadin_pts) - 1:
                    leadin_boundaries.append(distance)
                if len(leadin_pts) <= position < checkpoint_positions[-1] and main_pts:
                    main_boundaries.append(distance - physical_distance[len(leadin_pts)])
            if "timing_arch_checkpoint" in segment:
                position = segment["timing_arch_checkpoint"]
                segment_positions.append(position)
                segment["timing_arch_distance_m"] = round(_distance_at(position), 2)
            if segment_names and segment["hash"] in segment_names:
                segment["name"] = segment_names[segment["hash"]]
    route_profile = build_profile(main_pts, max_points, _surfaces(main_rt), main_boundaries)
    leadin_surfaces = _surfaces(leadin_rt)
    leadin_profile = build_profile(leadin_pts, max_points, leadin_surfaces, leadin_boundaries)
    leadin_view = build_leadin_view(
        leadin_pts, leadin_rt, main_pts, leadin_distance_m,
        paddock_road_ids, max_points, leadin_surfaces, leadin_boundaries,
    ) if r.get("leadinDistanceInMeters") is not None else None
    segment_path = build_segment_path(leadin_pts + main_pts, max_points * 2,
                                      segment_positions) if segment_entries else None

    route = {
        "name": name,
        "nameHash": _i(r, "nameHash"),
        "mapID": _i(r, "mapID", map_id),
        "locKey": r.get("locKey") or "",
        "distance_m": round(distance_m, 2),
        "ascent_m": round(_f(r, "ascentInMeters"), 2),
        "leadin_distance_m": round(leadin_distance_m, 2),
        "leadin_ascent_m": round(_f(r, "leadinAscentInMeters"), 2),
        "sport_type": _i(r, "sportType"),
        "event_only": _i(r, "eventOnly") == 1,
        "supported_laps": _i(r, "supportedLaps"),
        "event_paddocks": [int(value) for value in (r.get("eventPaddocks") or "").split(",")
                   if value.strip()],
        "alternate_event_ramp": _i(r, "useAlternateEventRamp")
        if r.get("useAlternateEventRamp") is not None else None,
        "start_locations": route_start_locations(root),
        "leadin": leadin_profile,
        **({"leadin_view": leadin_view} if leadin_view is not None else {}),
        "route": route_profile,
        "leadin_checkpoint_count": len(leadin_pts),
        "route_checkpoint_count": len(main_pts),
        "segments": segment_entries,
        **({"native_segments": native_segments} if segment_definitions is not None else {}),
        "segment_path": segment_path,
        "escalators": derive_escalator_passages(root, escalator_definitions, leadin_distance_m)
        if escalator_definitions else [],
    }
    route["route_versions"] = build_route_versions(root, route, max_points, reference_paths,
                                                    roads, styles, surface_map)
    return route


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def update_segment_distances(out: str) -> dict[int, float]:
    documents = []
    passages = []
    maxima = {}
    for filename in sorted(os.listdir(out)):
        if not (re.fullmatch(r"world_\d+\.json", filename) or filename == "video_routes.json"):
            continue
        path = os.path.join(out, filename)
        with open(path, "rb") as stream:
            original = stream.read()
        data = json.loads(original)
        for route in data.get("routes", []):
            for segment in route.get("segments", []):
                start = segment.get("start_distance_m")
                end = segment.get("end_distance_m")
                if start is None or end is None:
                    continue
                distance = round(end - start, 2)
                segment_hash = int(segment["hash"])
                maxima[segment_hash] = max(maxima.get(segment_hash, distance), distance)
                passages.append((segment, distance, segment_hash))
        documents.append((path, data, original))
    for segment, distance, segment_hash in passages:
        segment["distance_m"] = distance
        segment["max_distance_m"] = maxima[segment_hash]
    for path, data, original in documents:
        updated = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if updated != original:
            with open(path, "wb") as stream:
                stream.write(updated)
    return maxima


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "route"


VIDEO_ROUTES_MAP_ID = 1000
_VIDEO_ROUTE_NAMES = {
    "archesnp": "Arches National Park",
    "benassynt": "Ben Assynt",
    "bonette": "La Bonette",
    "bonetteshort": "La Bonette (Short)",
    "chapmans": "Chapman's Peak",
    "clarencedrive": "Clarence Drive",
    "dutoitskloof": "Du Toitskloof",
    "eggum": "Eggum",
    "etivedown": "Glen Etive",
    "femenia": "Femenia",
    "galibierlautaret": "Galibier from Lautaret",
    "glencanyon": "Glen Canyon",
    "glengarry": "Glengarry",
    "hogsback": "Hogsback",
    "lauvika": "Lauvika",
    "nappvang": "Nappvang",
    "sacalobrad1whisky": "Sa Calobra",
    "teide": "Teide",
    "threerivers": "Three Rivers",
    "wellingrivier": "Wellingrivier",
}


def _video_road(element: ET.Element, cache_key: str) -> dict:
    nodes = []
    for entity in element.findall("ent"):
        if entity.get("type") != "ENTITY_TYPE_ROADNODE":
            continue
        position = [float(value) / UNITS_PER_METRE
                    for value in entity.attrib["m_pos"].strip("{}").split(",")]
        nodes.append({
            "x": position[0], "alt": position[1], "z": position[2],
            "lat": float(entity.attrib["m_lat"]),
            "lng": float(entity.attrib["m_lon"]),
            "video_seconds": float(entity.attrib["m_videoSeconds"]),
            "video_distance_m": float(entity.attrib["m_videoDist"]) / UNITS_PER_METRE,
        })
    if len(nodes) < 2:
        raise ValueError("Video road needs at least two GPS/video nodes")
    markers = []
    for entity in element.findall("ent"):
        if entity.get("type") != "ENTITY_TYPE_ROADMARKER":
            continue
        attributes = entity.attrib
        if not ("m_roadTime1" in attributes or "m_roadTime2" in attributes):
            continue
        style = attributes.get("m_styleName")
        if style is None and attributes.get("m_isInvisible") == "1":
            continue
        low, high = sorted((float(attributes.get("m_roadTime1", 0)),
                            float(attributes.get("m_roadTime2", 1))))
        markers.append((low, high, style or "NORMAL"))
    return {
        "id": int(element.findtext("id")),
        "name": element.findtext("roadName") or "",
        "style": element.findtext("defaultStyleName") or "UNKNOWN",
        "markers": markers, "nodes": nodes, "cache_key": cache_key,
    }


def _video_style(road: dict, road_time: float) -> str:
    style = road["style"]
    for low, high, override in road["markers"]:
        if low <= road_time <= high:
            style = override
    return style


def _video_network(road: dict, surface_map: dict) -> list[dict]:
    nodes = road["nodes"]
    times = [index / (len(nodes) - 1) for index in range(len(nodes))]
    boundaries = sorted({0.0, 1.0, *(
        value for low, high, _ in road["markers"] for value in (low, high)
        if 0 < value < 1
    )})

    def sample(road_time: float, key: str) -> float:
        index = min(bisect_right(times, road_time) - 1, len(nodes) - 2)
        fraction = (road_time - times[index]) / (times[index + 1] - times[index])
        return nodes[index][key] + fraction * (nodes[index + 1][key] - nodes[index][key])

    segments = []
    for low, high in zip(boundaries, boundaries[1:]):
        samples = [low, *(value for value in times if low < value < high), high]
        style = _video_style(road, (low + high) / 2)
        segments.append({
            "road_id": road["id"], "style": style,
            "surface": surface_map.get(style, "Unknown"),
            "x": [sample(value, "x") for value in samples],
            "z": [sample(value, "z") for value in samples],
        })
    return segments


def extract_video_routes(cache_dir: str, out: str, out_surfaces: str) -> int:
    """Publish locally cached Mochi route XML and GPS/video road nodes."""
    route_sources = []
    roads = {}
    for filename in sorted(os.listdir(cache_dir)):
        path = os.path.join(cache_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as stream:
            prefix = stream.read(64).lstrip()
            if not prefix.startswith((b"<VERSION", b"<world")):
                continue
            stream.seek(0)
            data = stream.read()
        root = load_multiroot(data)
        if root.find("route") is not None:
            route_sources.append((filename, data, root))
        for element in root.findall(".//road"):
            road = _video_road(element, filename)
            roads[road["id"]] = road

    surface_map = _load_style_surface()
    routes = []
    used_roads = set()
    for cache_key, data, root in route_sources:
        point_count = max(len(_points(root.find(name))) for name in (
            "leadinhighrescheckpoint", "highrescheckpoint"))
        route = parse_route(data, VIDEO_ROUTES_MAP_ID, max(point_count, 2))
        if not route or not route["route"]:
            continue
        route_road_ids = set()
        for leg_name, xml_name in (("leadin", "leadinhighrescheckpoint"),
                                   ("route", "highrescheckpoint")):
            leg = route[leg_name]
            if not leg:
                continue
            _, road_times = _leg_points(root.find(xml_name))
            route_road_ids.update(road_id for road_id, _ in road_times)
            leg["surface"] = [surface_map.get(_video_style(roads[road_id], time), "Unknown")
                              for road_id, time in road_times]
        source_name = route["name"]
        route["source_name"] = source_name
        route["name"] = _VIDEO_ROUTE_NAMES.get(source_name, source_name.replace("_", " ").title())
        route["source_mapID"] = route["mapID"]
        route["mapID"] = VIDEO_ROUTES_MAP_ID
        route["cache_key"] = cache_key
        route["road_nodes"] = [{"road_id": road_id, "nodes": roads[road_id]["nodes"]}
                               for road_id in sorted(route_road_ids)]
        routes.append(route)
        used_roads.update(route_road_ids)
    if not routes:
        raise ValueError("No usable video routes found in the Mochi cache")
    routes.sort(key=lambda route: route["name"].lower())
    metadata = {"mapID": VIDEO_ROUTES_MAP_ID, "world": "Video Routes",
                "source": "zwift_mochi_cache", "coordinate_system": "zwift_local_m"}
    segments = [segment for road_id in sorted(used_roads)
                for segment in _video_network(roads[road_id], surface_map)]
    for directory, payload in ((out, {**metadata, "routes": routes}),
                               (out_surfaces, {**metadata, "segments": segments})):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "video_routes.json"), "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    update_segment_distances(out)
    return len(routes)


def refresh_route_versions(out, zwift_dir, source_cache_dir=None):
    import pickle
    from tools.analyze_paddock_groups import assign_route, make_groups

    surface_map = _load_style_surface()
    for filename in sorted(os.listdir(out)):
        if not re.fullmatch(r"world_\d+\.json", filename):
            continue
        map_id = int(re.search(r"\d+", filename).group())
        snapshot = os.path.join(source_cache_dir, f"world_{map_id}.pickle") if source_cache_dir else None
        if snapshot and os.path.isfile(snapshot):
            with open(snapshot, "rb") as handle:
                members = pickle.load(handle)
        else:
            members = read_wad_entries(os.path.join(zwift_dir, "assets", "Worlds", f"world{map_id}", "data_1.wad"),
                                       keep_substrings=("/routes/", "road.xml", "roadstyle.xml", "entities.xml"))
        roots = {}
        for source, data in members.items():
            if "/routes/" not in source.lower() or not source.endswith(".xml"):
                continue
            root = load_multiroot(data)
            header = root.find("route")
            if header is not None:
                roots[int(header.get("nameHash"))] = (source, root)
        references = [path for source, root in roots.values() for path in native_start_paths(root, source)]
        road_data = next(data for source, data in members.items() if source.lower().endswith("/road.xml"))
        entity_data = next((data for source, data in members.items() if source.lower().endswith("/entities.xml")), None)
        styles = parse_roadstyles(next(data for source, data in members.items() if source.lower().endswith("/roadstyle.xml")))
        roads = parse_roads(road_data.decode("utf-8", "replace"))
        groups = make_groups(map_id, parse_paddock_roads(road_data, entity_data), entity_data=entity_data, road_data=road_data)
        group_lookup = {group["group"]: group for group in groups}
        path = os.path.join(out, filename)
        with open(path, encoding="utf-8") as handle:
            world = json.load(handle)
        for route in world["routes"]:
            source, root = roots[route["nameHash"]]
            route["start_locations"] = route_start_locations(root)
            route["route_versions"] = build_route_versions(root, route, reference_paths=references,
                                                           roads=roads, styles=styles, surface_map=surface_map)
            assignment = assign_route(root, map_id, groups, source)
            group = group_lookup.get(assignment["observed_group"] or assignment["nearest_group"])
            if group is not None:
                route["route_versions"]["race"].update({
                    "fallback_paddock_road_id": group["representative_road_id"],
                    "fallback_paddock_road_length_m": group["representative_road_length_m"],
                    "fallback_paddock_basis": "longest road in matched group; inferred, not actual pen selection",
                })
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(world, handle, ensure_ascii=False, separators=(",", ":"))
        available = [route for route in world["routes"] if route["route_versions"]["free_ride"].get("available")]
        located = sum(route["route_versions"]["free_ride"]["geometry_status"] == "native" for route in available)
        print(f"world{map_id}: {located}/{len(available)} free-ride approaches; {len(world['routes'])} routes", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift",
                    help="Zwift install directory (contains assets/Worlds).")
    ap.add_argument("--out", default="zwift_routes",
                    help="Output directory for the cached route profiles.")
    ap.add_argument("--max-points", type=int, default=600,
                    help="Initial samples per profile; boundaries and <=1 m error may add points.")
    ap.add_argument("--world", default=None,
                    help="Extract only one world folder, e.g. world11.")
    ap.add_argument("--video-cache", default=None,
                    help="Export video routes only, from a local Zwift/mochi cache directory.")
    ap.add_argument("--out-surfaces", default="zwift_surfaces",
                    help="Surface output directory for video-cache extraction.")
    ap.add_argument("--refresh-route-versions", action="store_true",
                    help="Update only start-location metadata and mode profiles in existing numeric-world exports.")
    ap.add_argument("--source-cache-dir", default=None,
                    help="Optional trusted per-world WAD pickle snapshots for the start-version refresh.")
    args = ap.parse_args()
    if args.refresh_route_versions:
        refresh_route_versions(args.out, args.zwift_dir, args.source_cache_dir)
        return 0

    if args.video_cache:
        count = extract_video_routes(args.video_cache, args.out, args.out_surfaces)
        print(f"Done: {count} video routes -> {args.out}/video_routes.json")
        return 0

    worlds_dir = os.path.join(args.zwift_dir, "assets", "Worlds")
    if not os.path.isdir(worlds_dir):
        print(f"error: worlds directory not found: {worlds_dir}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    index: list[dict] = []
    surface_map = _load_style_surface()

    world_names = sorted(
        (d for d in os.listdir(worlds_dir)
         if d.startswith("world") and os.path.isdir(os.path.join(worlds_dir, d))
         and (args.world is None or d.lower() == args.world.lower())),
        key=lambda d: int(re.sub(r"\D", "", d) or 0),
    )

    existing_index = []
    index_path = os.path.join(args.out, "index.json")
    if args.world and os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            existing_index = json.load(f)
        selected_map = int(re.sub(r"\D", "", args.world) or 0)
        existing_index = [entry for entry in existing_index if entry.get("mapID") != selected_map]

    for world in world_names:
        wad = os.path.join(worlds_dir, world, "data_1.wad")
        if not os.path.isfile(wad):
            continue
        map_id = int(re.sub(r"\D", "", world) or 0)
        try:
            entries = read_wad_entries(
                wad, keep_substrings=("/routes/", "road.xml", "roadstyle.xml", "entities.xml"))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {world}: FAILED to read wad ({exc})", file=sys.stderr)
            continue

        # Authoritative surface data for this world's road network (road-id join).
        road_bytes = next((v for k, v in entries.items()
                           if k.lower().endswith("road.xml")
                           and "roadstyle" not in k.lower()), None)
        style_bytes = next((v for k, v in entries.items()
                            if k.lower().endswith("roadstyle.xml")), None)
        roads = parse_roads(road_bytes.decode("utf-8", "replace")) if road_bytes else {}
        styles = parse_roadstyles(style_bytes) if style_bytes else []
        entity_bytes = next((v for k, v in entries.items()
                     if k.lower().endswith("entities.xml")), None)
        paddock_roads = parse_paddock_roads(road_bytes, entity_bytes) if road_bytes else []
        segment_names = parse_segment_names(entity_bytes) if entity_bytes else {}
        segment_definitions = parse_segment_definitions(entity_bytes) if entity_bytes else []
        escalator_definitions = parse_escalator_definitions(entity_bytes) if entity_bytes else []

        route_files = sorted(
            (nm for nm in entries if "/routes/" in nm and nm.endswith(".xml")),
            key=lambda nm: int(re.sub(r"\D", "", os.path.basename(nm)) or 0),
        )
        if not route_files:
            continue

        routes: list[dict] = []
        reference_paths = [path for source in route_files
                   for path in native_start_paths(load_multiroot(entries[source]), source)]
        for nm in route_files:
            try:
                route = parse_route(entries[nm], map_id, args.max_points,
                                    roads, styles, surface_map, segment_names,
                                    segment_definitions, escalator_definitions,
                                    {road["road_id"] for road in paddock_roads},
                                    reference_paths=reference_paths)
            except Exception as exc:  # noqa: BLE001
                print(f"  {world}/{nm}: parse error ({exc})", file=sys.stderr)
                continue
            if route and route["name"]:
                routes.append(route)

        if not routes:
            continue

        out_file = f"world_{map_id}.json"
        with open(os.path.join(args.out, out_file), "w", encoding="utf-8") as f:
            json.dump({"mapID": map_id, "world": world,
                       "coordinate_system": "zwift_local_m",
                       "paddock_roads": paddock_roads, "routes": routes},
                      f, ensure_ascii=False, separators=(",", ":"))
        print(f"  {world} (map {map_id}): {len(routes)} routes -> {out_file}")

        for route in routes:
            index.append({
                "name": route["name"],
                "nameHash": route["nameHash"],
                "mapID": route["mapID"],
                "world": world,
                "locKey": route["locKey"],
                "distance_m": route["distance_m"],
                "ascent_m": route["ascent_m"],
                "leadin_distance_m": route["leadin_distance_m"],
                "leadin_ascent_m": route["leadin_ascent_m"],
                "sport_type": route["sport_type"],
                "event_only": route["event_only"],
                "event_paddocks": route["event_paddocks"],
                "alternate_event_ramp": route["alternate_event_ramp"],
                "has_leadin": route["leadin"] is not None,
                "has_route": route["route"] is not None,
                "file": out_file,
            })

    index = existing_index + index
    index.sort(key=lambda e: (e["mapID"], e["name"].lower()))
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    update_segment_distances(args.out)
    print(f"\nDone: {len(index)} routes across {len(world_names)} worlds -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
