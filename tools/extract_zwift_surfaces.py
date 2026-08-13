"""Extract authoritative road-surface geometry from the Zwift game client.

For each ``assets/Worlds/world*/data_1.wad`` this tool reads ``road.xml`` and
``roadstyle.xml`` and produces, in the SAME local-metre coordinate frame as
``tools/extract_zwift_routes.py`` (game units / 100), a set of road segments
each tagged with a single surface type.

Surface model (see repo memory ``zwift-wad-extraction.md``):

* ``roadstyle.xml`` lists ``<segment style="NAME" .../>`` in document order; the
  0-based index is the style id referenced by ``road.xml``. The list/order is
  per-world, so it is always parsed fresh.
* Each ``<road>`` has a ``<defaultStyle>`` (whole-road base surface; ``31`` is a
  sentinel meaning "unset" -> NORMAL/tarmac) and a spline of
  ``ENTITY_TYPE_ROADNODE`` control points (``m_pos`` in game units).
* ``ENTITY_TYPE_ROADMARKER`` entities override a sub-range of the spline
  (normalised ``m_roadTime1..m_roadTime2`` in ``[0, 1]``) with ``m_style``. This
  encodes short mid-road sectors (a cobbled climb, a dirt patch, ...).

The road spline is resampled at a fixed spacing; each sample's surface is the
base style unless a marker range covers its normalised arc-length position.
Consecutive samples sharing a surface are grouped into output segments.

Usage::

    python tools/extract_zwift_surfaces.py \
        [--zwift-dir "C:\\Program Files (x86)\\Zwift"] \
        [--out zwift_surfaces] [--spacing 8.0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

# Reuse the verified in-memory WAD reader from the route extractor.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries  # noqa: E402


UNITS_PER_METRE = 100.0
UNSET_STYLE = 31  # 0x1F sentinel in <defaultStyle>; treated as NORMAL/tarmac.

# Map a roadstyle NAME to a coarse surface category compatible with
# shared/surface_lookup.py's DEFAULT_CRR keys. Raw style name is also retained
# in the output so the mapping can be reviewed/tuned downstream.
STYLE_TO_SURFACE = {
    "NORMAL": "Tarmac",
    "ONELINE": "Tarmac",
    "WET": "Tarmac",
    "LAVA": "Tarmac",
    "BOXHILL": "Tarmac",
    "TRACK": "Tarmac",
    "DOTHINT": "Tarmac",
    "WOODEN": "Wood",
    "COBBLE": "Cobbles",
    "DIRT": "Dirt",
    "INVISIBLE_DIRT": "Dirt",
    "TRAILDIRT": "Dirt",
    "MOUNTAIN": "Dirt",
    "DESERT": "Dirt",
    "GRAVEL": "Gravel",
    "INVISIBLE_GRAVEL": "Gravel",
    "TRAILGRAVEL": "Gravel",
    "BEACHPATH": "Sand",
    "PACKEDSAND": "Sand",
    "SNOW": "Snow",
}


def parse_roadstyles(data: bytes) -> list[str]:
    """Return roadstyle names in document order (index == style id)."""
    txt = data.decode("utf-8", "replace")
    return re.findall(r'<segment\s+[^>]*\bstyle="([^"]+)"', txt)


def _road_blocks(txt: str):
    """Yield the inner text of each top-level ``<road>`` element."""
    for m in re.finditer(r"<road>(.*?)</road>", txt, re.S):
        yield m.group(1)


def _nodes(block: str) -> list[tuple[float, float]]:
    """ROADNODE positions as local-metre ``(x, z)`` pairs, in spline order."""
    pts: list[tuple[float, float]] = []
    for m in re.finditer(
        r'<ent\b[^>]*type="ENTITY_TYPE_ROADNODE"[^>]*m_pos="\{([-\d.]+),([-\d.]+),([-\d.]+)\}"',
        block,
    ):
        pts.append((float(m.group(1)) / UNITS_PER_METRE,
                    float(m.group(3)) / UNITS_PER_METRE))
    return pts


def _markers(block: str) -> list[tuple[float, float, int]]:
    """Surface-override markers as ``(t0, t1, style)`` with ``t`` in ``[0, 1]``."""
    out: list[tuple[float, float, int]] = []
    for m in re.finditer(r'<ent\b[^>]*type="ENTITY_TYPE_ROADMARKER"[^>]*>', block):
        tag = m.group(0)
        st = re.search(r'm_style="(\d+)"', tag)
        if not st:
            continue
        t1 = re.search(r'm_roadTime1="([-\d.]+)"', tag)
        t2 = re.search(r'm_roadTime2="([-\d.]+)"', tag)
        if not (t1 and t2):
            continue
        a, b = float(t1.group(1)), float(t2.group(1))
        if b < a:
            a, b = b, a
        out.append((a, b, int(st.group(1))))
    return out


def _cumulative(pts: list[tuple[float, float]]) -> list[float]:
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(pts[i][0] - pts[i - 1][0],
                                        pts[i][1] - pts[i - 1][1]))
    return cum


def _sample_at(pts: list[tuple[float, float]], cum: list[float], s: float
               ) -> tuple[float, float]:
    """Point at arc-length ``s`` along the polyline."""
    if s <= 0:
        return pts[0]
    if s >= cum[-1]:
        return pts[-1]
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < s:
            lo = mid + 1
        else:
            hi = mid
    j = max(1, lo)
    d0, d1 = cum[j - 1], cum[j]
    f = 0.0 if d1 == d0 else (s - d0) / (d1 - d0)
    (x0, z0), (x1, z1) = pts[j - 1], pts[j]
    return (x0 + f * (x1 - x0), z0 + f * (z1 - z0))


def _style_at(t: float, base: int, markers: list[tuple[float, float, int]]) -> int:
    """Resolved style at normalised position ``t`` (last covering marker wins)."""
    style = base
    for a, b, s in markers:
        if a <= t <= b:
            style = s
    return style


def resolve_road(block: str, spacing: float, styles: list[str]
                 ) -> list[dict]:
    """Return constant-surface segments for one ``<road>`` block."""
    pts = _nodes(block)
    if len(pts) < 2:
        return []
    cum = _cumulative(pts)
    total = cum[-1]
    if total <= 0:
        return []

    dm = re.search(r"<defaultStyle>(\d+)</defaultStyle>", block)
    base = int(dm.group(1)) if dm else UNSET_STYLE
    if base == UNSET_STYLE or not (0 <= base < len(styles)):
        base = 0  # NORMAL / tarmac
    markers = [(a, b, s) for (a, b, s) in _markers(block) if 0 <= s < len(styles)]

    n = max(2, int(math.ceil(total / spacing)) + 1)
    samples: list[tuple[float, float, int]] = []
    for k in range(n):
        s = total * k / (n - 1)
        x, z = _sample_at(pts, cum, s)
        style = _style_at(s / total, base, markers)
        samples.append((x, z, style))

    # Group consecutive samples that share a style into polyline segments.
    segments: list[dict] = []
    run_style = samples[0][2]
    xs = [samples[0][0]]
    zs = [samples[0][1]]
    for x, z, style in samples[1:]:
        if style != run_style:
            segments.append(_segment(run_style, xs, zs, styles))
            # start next run from the boundary point for continuity
            xs, zs = [xs[-1], x], [zs[-1], z]
            run_style = style
        else:
            xs.append(x)
            zs.append(z)
    segments.append(_segment(run_style, xs, zs, styles))
    return [s for s in segments if s is not None]


def _segment(style_id: int, xs: list[float], zs: list[float], styles: list[str]
             ) -> dict | None:
    if len(xs) < 2:
        return None
    name = styles[style_id] if 0 <= style_id < len(styles) else "NORMAL"
    return {
        "style": name,
        "surface": STYLE_TO_SURFACE.get(name, "Tarmac"),
        "x": [round(v, 2) for v in xs],
        "z": [round(v, 2) for v in zs],
    }


def build_world(data: bytes, styledata: bytes, spacing: float) -> dict:
    styles = parse_roadstyles(styledata)
    txt = data.decode("utf-8", "replace")
    segments: list[dict] = []
    for block in _road_blocks(txt):
        segments.extend(resolve_road(block, spacing, styles))
    return {"styles": styles, "segments": segments}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift")
    ap.add_argument("--out", default="zwift_surfaces")
    ap.add_argument("--spacing", type=float, default=8.0,
                    help="Resample spacing along each road (metres).")
    args = ap.parse_args()

    worlds_dir = os.path.join(args.zwift_dir, "assets", "Worlds")
    if not os.path.isdir(worlds_dir):
        print(f"error: worlds directory not found: {worlds_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    world_names = sorted(
        (d for d in os.listdir(worlds_dir)
         if d.startswith("world") and os.path.isdir(os.path.join(worlds_dir, d))),
        key=lambda d: int(re.sub(r"\D", "", d) or 0),
    )

    index: list[dict] = []
    for world in world_names:
        wad = os.path.join(worlds_dir, world, "data_1.wad")
        if not os.path.isfile(wad):
            continue
        map_id = int(re.sub(r"\D", "", world) or 0)
        try:
            entries = read_wad_entries(wad, keep_substrings=("road.xml", "roadstyle.xml"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {world}: FAILED to read wad ({exc})", file=sys.stderr)
            continue
        road = next((v for k, v in entries.items()
                     if k.lower().endswith("road.xml") and "roadstyle" not in k.lower()), None)
        style = next((v for k, v in entries.items() if k.lower().endswith("roadstyle.xml")), None)
        if road is None or style is None:
            print(f"  {world}: missing road/roadstyle xml", file=sys.stderr)
            continue

        result = build_world(road, style, args.spacing)
        segs = result["segments"]
        out_file = f"world_{map_id}.json"
        with open(os.path.join(args.out, out_file), "w", encoding="utf-8") as f:
            json.dump({"mapID": map_id, "world": world,
                       "coordinate_system": "zwift_local_m",
                       "styles": result["styles"], "segments": segs},
                      f, ensure_ascii=False, separators=(",", ":"))

        by_surface: dict[str, int] = {}
        for s in segs:
            by_surface[s["surface"]] = by_surface.get(s["surface"], 0) + 1
        non_tarmac = {k: v for k, v in by_surface.items() if k != "Tarmac"}
        print(f"  {world} (map {map_id}): {len(segs)} segments -> {out_file}"
              f"  non-tarmac={non_tarmac}")
        index.append({"mapID": map_id, "world": world, "file": out_file,
                      "segments": len(segs), "surfaces": by_surface})

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"\nDone: {len(index)} worlds -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
