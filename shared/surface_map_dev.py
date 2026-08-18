"""
Surface Map (developer view).

Reads ``road.xml`` / ``roadstyle.xml`` straight from the locally installed Zwift
game client (the WAD files under ``assets/Worlds``) and exposes the raw road
network for inspection: every ``<road>`` element, its spline geometry, its raw
XML, and the individual node-to-node segments.

The WAD files only exist on a machine with the Zwift client installed, so this
module (and the routes/UI that use it) are gated behind :func:`is_available` and
are never reachable on the production host.
"""
from __future__ import annotations

import math
import os
import re
import sys
from functools import lru_cache

import numpy as np

from shared.surface_map import (
    WORLD_NAMES, SURFACE_COLORS, SURFACE_ORDER, _projection, _seg_surface,
)

# Reuse the verified WAD reader + road-spline helpers from the extract tools.
_TOOLS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from extract_zwift_routes import read_wad_entries  # noqa: E402
import extract_zwift_surfaces as _exs  # noqa: E402

ZWIFT_DIR = os.environ.get("ZWIFT_DIR", r"C:\Program Files (x86)\Zwift")

_ROAD_BLOCK_RE = re.compile(r"<road>.*?</road>", re.S)
_ID_RE = re.compile(r"<id>(\d+)</id>")
_NAME_RE = re.compile(r"<roadName>(.*?)</roadName>", re.S)
_DEFSTYLE_RE = re.compile(r"<defaultStyle>(\d+)</defaultStyle>")
_LOOPED_RE = re.compile(r"<looped>\s*1\s*</looped>")
_MARKER_ID_RE = re.compile(r'm_markerId="(\d+)"')
_MARKER_TAG_RE = re.compile(r'<ent\b[^>]*type="ENTITY_TYPE_ROADMARKER"[^>]*>')


def _worlds_dir() -> str:
    return os.path.join(ZWIFT_DIR, "assets", "Worlds")


def is_available() -> bool:
    """True when the local Zwift game client (WAD files) is present."""
    return os.path.isdir(_worlds_dir())


@lru_cache(maxsize=1)
def _world_wads() -> dict[int, str]:
    """Map ``mapID`` -> ``data_1.wad`` path for every ``world*`` folder found."""
    out: dict[int, str] = {}
    base = _worlds_dir()
    if not os.path.isdir(base):
        return out
    for name in os.listdir(base):
        if not name.startswith("world"):
            continue
        wad = os.path.join(base, name, "data_1.wad")
        if not os.path.isfile(wad):
            continue
        digits = re.sub(r"\D", "", name)
        if digits:
            out[int(digits)] = wad
    return out


def list_dev_worlds() -> list[dict]:
    """Worlds that have a WAD available and a known name, sorted by name."""
    worlds = [{"mapID": mid, "name": WORLD_NAMES.get(mid, f"World {mid}")}
              for mid in _world_wads() if mid in WORLD_NAMES]
    worlds.sort(key=lambda w: w["name"].lower())
    return worlds


@lru_cache(maxsize=None)
def _read_road_style(map_id: int) -> tuple[str, list[str]]:
    """Return ``(road.xml text, roadstyle names)`` for a world from its WAD."""
    wad = _world_wads().get(map_id)
    if wad is None:
        raise FileNotFoundError(f"No WAD for world {map_id}")
    entries = read_wad_entries(wad, keep_substrings=("road.xml", "roadstyle.xml"))
    road = next((v for k, v in entries.items()
                 if k.lower().endswith("road.xml") and "roadstyle" not in k.lower()),
                None)
    style = next((v for k, v in entries.items()
                  if k.lower().endswith("roadstyle.xml")), None)
    if road is None or style is None:
        raise FileNotFoundError(f"road/roadstyle xml missing for world {map_id}")
    return road.decode("utf-8", "replace"), _exs.parse_roadstyles(style)


def _style_name(styles: list[str], idx: int) -> str:
    return styles[idx] if 0 <= idx < len(styles) else "NORMAL"


def _surface_for(styles: list[str], idx: int) -> str:
    return _seg_surface({"style": _style_name(styles, idx)})


def _dedent(block: str) -> str:
    """Strip the common leading indentation so the raw XML reads cleanly."""
    lines = block.splitlines()
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    trim = min(indents) if indents else 0
    return "\n".join(ln[trim:] if len(ln) >= trim else ln for ln in lines)


def _node_segments(nodes, looped: bool):
    """Per-node outgoing cubic-Bezier polylines in local metres.

    Returns a list of ``(node_index, next_index, [(x, z), ...])`` — one entry per
    node-to-node segment, plus the wrap segment when ``looped``."""
    segs = []
    step = _exs._SPLINE_STEP_M

    def build(i0: int, i1: int):
        (a, _ai, ao) = nodes[i0]
        (b, bi, _bo) = nodes[i1]
        p0, p3 = a, b
        p1 = (a[0] + ao[0], a[1] + ao[1])
        p2 = (b[0] + bi[0], b[1] + bi[1])
        clen = (math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                + math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                + math.hypot(p3[0] - p2[0], p3[1] - p2[1]))
        n = max(1, int(math.ceil(clen / step)))
        return [_exs._bezier(p0, p1, p2, p3, k / n) for k in range(n + 1)]

    for i in range(len(nodes) - 1):
        segs.append((i, i + 1, build(i, i + 1)))
    if looped and len(nodes) >= 2:
        segs.append((len(nodes) - 1, 0, build(len(nodes) - 1, 0)))
    return segs


def _subpath(pts, cum, total, t0, t1):
    """Local-metre sub-polyline of ``pts`` between normalised ``t0..t1``."""
    s0, s1 = t0 * total, t1 * total
    out = [_exs._sample_at(pts, cum, s0)]
    for p, c in zip(pts, cum):
        if s0 < c < s1:
            out.append(p)
    out.append(_exs._sample_at(pts, cum, s1))
    return out


def _project_xy(proj, pts) -> tuple[list[float], list[float]]:
    if not pts:
        return [], []
    X, Y = proj["project"]([p[0] for p in pts], [p[1] for p in pts])
    return ([round(float(v), 1) for v in X], [round(float(v), 1) for v in Y])


def _decimate(xs: list[float], ys: list[float], cap: int = 250):
    if len(xs) <= cap:
        return xs, ys
    stride = (len(xs) + cap - 1) // cap
    dx = xs[::stride]
    dy = ys[::stride]
    if dx[-1] != xs[-1] or dy[-1] != ys[-1]:
        dx.append(xs[-1])
        dy.append(ys[-1])
    return dx, dy


def get_world_roads(map_id: int) -> dict:
    """Overview of every road in a world: metadata + a coarse plot-space path."""
    road_txt, styles = _read_road_style(map_id)
    proj = _projection(map_id)

    roads: list[dict] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for m in _ROAD_BLOCK_RE.finditer(road_txt):
        block = m.group(0)
        rid_m = _ID_RE.search(block)
        if not rid_m:
            continue
        rid = int(rid_m.group(1))
        name_m = _NAME_RE.search(block)
        name = name_m.group(1).strip() if name_m else ""
        nodes = _exs._nodes_full(block)
        looped = _LOOPED_RE.search(block) is not None
        ds_m = _DEFSTYLE_RE.search(block)
        base = int(ds_m.group(1)) if ds_m else _exs.UNSET_STYLE
        if base == _exs.UNSET_STYLE or not (0 <= base < len(styles)):
            base = 0

        if len(nodes) >= 2:
            pts = _exs._spline_points(nodes, looped=looped)
            length_m = _exs._cumulative(pts)[-1]
            xs, ys = _project_xy(proj, pts)
            all_x.extend(xs)
            all_y.extend(ys)
            xs, ys = _decimate(xs, ys)
        else:
            length_m = 0.0
            xs, ys = _project_xy(proj, [n[0] for n in nodes])

        roads.append({
            "id": rid,
            "name": name,
            "node_count": len(nodes),
            "looped": looped,
            "style": _style_name(styles, base),
            "surface": _surface_for(styles, base),
            "length_m": round(length_m, 1),
            "x": xs,
            "y": ys,
        })

    roads.sort(key=lambda r: r["id"])

    if proj["mode"] == "image":
        bounds = proj["bounds"]
    elif all_x:
        bounds = {"min_x": min(all_x), "max_x": max(all_x),
                  "min_y": min(all_y), "max_y": max(all_y)}
    else:
        bounds = proj["bounds"]

    surfaces = sorted({r["surface"] for r in roads},
                      key=lambda s: SURFACE_ORDER.index(s) if s in SURFACE_ORDER else 99)

    return {
        "mapID": map_id,
        "name": WORLD_NAMES.get(map_id, f"World {map_id}"),
        "projection": proj["mode"],
        "background": proj["background"],
        "bounds": bounds,
        "roads": roads,
        "colors": {s: SURFACE_COLORS.get(s, SURFACE_COLORS["Unknown"]) for s in surfaces},
    }


def get_road_detail(map_id: int, road_id: int) -> dict | None:
    """Raw XML, full geometry, per-node segments and markers for one road."""
    road_txt, styles = _read_road_style(map_id)
    proj = _projection(map_id)

    block = None
    for m in _ROAD_BLOCK_RE.finditer(road_txt):
        b = m.group(0)
        rid_m = _ID_RE.search(b)
        if rid_m and int(rid_m.group(1)) == road_id:
            block = b
            break
    if block is None:
        return None

    name_m = _NAME_RE.search(block)
    name = name_m.group(1).strip() if name_m else ""
    nodes = _exs._nodes_full(block)
    looped = _LOOPED_RE.search(block) is not None
    ds_m = _DEFSTYLE_RE.search(block)
    base = int(ds_m.group(1)) if ds_m else _exs.UNSET_STYLE
    if base == _exs.UNSET_STYLE or not (0 <= base < len(styles)):
        base = 0

    full_pts = _exs._spline_points(nodes, looped=looped) if len(nodes) >= 2 \
        else [n[0] for n in nodes]
    fx, fy = _project_xy(proj, full_pts)
    length_m = _exs._cumulative(full_pts)[-1] if len(full_pts) >= 2 else 0.0

    segments = []
    for i0, i1, seg_pts in _node_segments(nodes, looped):
        sx, sy = _project_xy(proj, seg_pts)
        segments.append({"node": i0, "to": i1, "x": sx, "y": sy})

    markers = []
    if len(full_pts) >= 2:
        cum = _exs._cumulative(full_pts)
        total = cum[-1]
        for mm in _MARKER_TAG_RE.finditer(block):
            tag = mm.group(0)
            if total <= 0:
                continue
            st = re.search(r'm_style="(\d+)"', tag)
            t1 = re.search(r'm_roadTime1="([-\d.]+)"', tag)
            t2 = re.search(r'm_roadTime2="([-\d.]+)"', tag)
            mid = _MARKER_ID_RE.search(tag)
            # Some markers omit a road-time (span from road start/to road end) or
            # carry no m_style (width/other markers with no surface override).
            a = float(t1.group(1)) if t1 else 0.0
            b = float(t2.group(1)) if t2 else 1.0
            if b < a:
                a, b = b, a
            has_style = st is not None
            style_idx = int(st.group(1)) if has_style else -1
            sub = _subpath(full_pts, cum, total, a, b)
            mx, my = _project_xy(proj, sub)
            markers.append({
                "markerId": int(mid.group(1)) if mid else None,
                "style": _style_name(styles, style_idx) if has_style else None,
                "surface": _surface_for(styles, style_idx) if has_style else None,
                "t0": round(a, 4),
                "t1": round(b, 4),
                "x": mx,
                "y": my,
            })

    bounds = ({"min_x": min(fx), "max_x": max(fx),
               "min_y": min(fy), "max_y": max(fy)} if fx else None)

    return {
        "mapID": map_id,
        "id": road_id,
        "name": name,
        "looped": looped,
        "one_way": re.search(r"<oneWay>\s*1\s*</oneWay>", block) is not None,
        "node_count": len(nodes),
        "length_m": round(length_m, 1),
        "default_style": _style_name(styles, base),
        "default_surface": _surface_for(styles, base),
        "xml": _dedent(block),
        "full": {"x": fx, "y": fy},
        "segments": segments,
        "markers": markers,
        "bounds": bounds,
        "colors": {s: SURFACE_COLORS.get(s, SURFACE_COLORS["Unknown"])
                   for s in SURFACE_COLORS},
    }
