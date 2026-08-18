"""Extract per-route elevation profiles (lead-in + main lap) directly from the
Zwift game client's compressed world archives.

For each ``assets/Worlds/world*/data_1.wad`` this tool:

1. Decompresses the ``ZWF!`` archive in memory (Zwift's custom LZ format).
2. Reads every ``routes/*.xml`` entry from the archive's table of contents.
3. Parses each route's header plus its ``<leadinhighrescheckpoint>`` and
   ``<highrescheckpoint>`` polylines.
4. Converts the raw world geometry (game units, 100 units == 1 metre) into
    metre-based ``(distance, altitude, x, z)`` profiles. Coordinates are the raw
    values divided by 100 (no clamping). Cumulative planar distance is anchored
    to Zwift's authoritative header distances (``leadinDistanceInMeters`` /
    ``distanceInMeters``) so total distances match the game exactly while the
    checkpoint geometry provides the shape.

The decompression algorithm is a faithful in-memory port of r3dey3's
``decode_wad.py`` (https://gitlab.com/r3dey3/zwift-utils).

Usage:
    python tools/extract_zwift_routes.py \
        [--zwift-dir "C:\\Program Files (x86)\\Zwift"] \
        [--out zwift_routes] [--max-points 600]
"""
from __future__ import annotations

import argparse
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
            st = re.search(r'm_style="(\d+)"', tag)
            t1 = re.search(r'm_roadTime1="([-\d.]+)"', tag)
            t2 = re.search(r'm_roadTime2="([-\d.]+)"', tag)
            if st and t1 and t2:
                a, b = float(t1.group(1)), float(t2.group(1))
                if b < a:
                    a, b = b, a
                markers.append((a, b, int(st.group(1))))
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
              surfaces: list[str] | None = None) -> dict[str, list]:
    """Uniformly resample distance and local-world geometry to at most ``n`` points.

    ``surfaces`` (one label per source point) is resampled as a step function:
    each output point takes the surface of the nearer bracketing source point, so
    style boundaries are preserved rather than blended.
    """
    x_m = [p[0] / UNITS_PER_METRE for p in pts]
    alt_m = [p[1] / UNITS_PER_METRE for p in pts]
    z_m = [p[2] / UNITS_PER_METRE for p in pts]
    if len(dist_m) <= n or len(dist_m) < 2:
        out: dict[str, list] = {
            "d": [round(d, 2) for d in dist_m],
            "alt": [round(a, 3) for a in alt_m],
            "x": [round(x, 3) for x in x_m],
            "z": [round(z, 3) for z in z_m],
        }
        if surfaces is not None:
            out["surface"] = list(surfaces)
        return out
    total = dist_m[-1]
    out_d, out_a, out_x, out_z = [], [], [], []
    out_s: list[str] = []
    j = 0
    for k in range(n):
        target = total * k / (n - 1)
        while j < len(dist_m) - 2 and dist_m[j + 1] < target:
            j += 1
        d0, d1 = dist_m[j], dist_m[j + 1]
        frac = 0.0 if d1 == d0 else (target - d0) / (d1 - d0)
        alt = alt_m[j] + frac * (alt_m[j + 1] - alt_m[j])
        x = x_m[j] + frac * (x_m[j + 1] - x_m[j])
        z = z_m[j] + frac * (z_m[j + 1] - z_m[j])
        out_d.append(round(target, 2))
        out_a.append(round(alt, 3))
        out_x.append(round(x, 3))
        out_z.append(round(z, 3))
        if surfaces is not None:
            out_s.append(surfaces[j + 1] if frac >= 0.5 else surfaces[j])
    res: dict[str, list] = {"d": out_d, "alt": out_a, "x": out_x, "z": out_z}
    if surfaces is not None:
        res["surface"] = out_s
    return res


def build_profile(pts: list[tuple[float, float, float]], header_distance_m: float,
                  max_points: int, surfaces: list[str] | None = None) -> dict | None:
    """Convert a checkpoint polyline into a metre-based route geometry profile."""
    if len(pts) < 2:
        return None
    cum = _cumulative_planar(pts)
    raw_total = cum[-1]
    if raw_total <= 0:
        return None
    # Anchor total distance to the authoritative header value; fall back to the
    # geometry-derived length when the header distance is missing/zero.
    if header_distance_m and header_distance_m > 0:
        scale = header_distance_m / raw_total
    else:
        scale = 1.0 / UNITS_PER_METRE
    dist_m = [u * scale for u in cum]
    return _resample(dist_m, pts, max_points, surfaces)


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


def parse_route(data: bytes, map_id: int, max_points: int,
                roads: dict | None = None, styles: list[str] | None = None,
                surface_map: dict | None = None) -> dict | None:
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

    route_profile = build_profile(main_pts, distance_m, max_points, _surfaces(main_rt))
    leadin_profile = build_profile(leadin_pts, leadin_distance_m, max_points,
                                   _surfaces(leadin_rt))

    return {
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
        "leadin": leadin_profile,
        "route": route_profile,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "route"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift",
                    help="Zwift install directory (contains assets/Worlds).")
    ap.add_argument("--out", default="zwift_routes",
                    help="Output directory for the cached route profiles.")
    ap.add_argument("--max-points", type=int, default=600,
                    help="Max sample points per profile segment.")
    args = ap.parse_args()

    worlds_dir = os.path.join(args.zwift_dir, "assets", "Worlds")
    if not os.path.isdir(worlds_dir):
        print(f"error: worlds directory not found: {worlds_dir}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    index: list[dict] = []
    surface_map = _load_style_surface()

    world_names = sorted(
        (d for d in os.listdir(worlds_dir)
         if d.startswith("world") and os.path.isdir(os.path.join(worlds_dir, d))),
        key=lambda d: int(re.sub(r"\D", "", d) or 0),
    )

    for world in world_names:
        wad = os.path.join(worlds_dir, world, "data_1.wad")
        if not os.path.isfile(wad):
            continue
        map_id = int(re.sub(r"\D", "", world) or 0)
        try:
            entries = read_wad_entries(
                wad, keep_substrings=("/routes/", "road.xml", "roadstyle.xml"))
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

        route_files = sorted(
            (nm for nm in entries if "/routes/" in nm and nm.endswith(".xml")),
            key=lambda nm: int(re.sub(r"\D", "", os.path.basename(nm)) or 0),
        )
        if not route_files:
            continue

        routes: list[dict] = []
        for nm in route_files:
            try:
                route = parse_route(entries[nm], map_id, args.max_points,
                                    roads, styles, surface_map)
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
                       "coordinate_system": "zwift_local_m", "routes": routes},
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
                "has_leadin": route["leadin"] is not None,
                "has_route": route["route"] is not None,
                "file": out_file,
            })

    index.sort(key=lambda e: (e["mapID"], e["name"].lower()))
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    print(f"\nDone: {len(index)} routes across {len(world_names)} worlds -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
