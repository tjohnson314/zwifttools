"""Throwaway probe #4: carry-forward semantics + verify against known surfaces.

- For roads that have SOME (not all) styled nodes, are the styled nodes contiguous?
- Resolve each road to a dominant style; print roads whose style != NORMAL with
  their spatial center so we can sanity-check against known Watopia surfaces.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from extract_zwift_routes import read_wad_entries  # noqa: E402

ZWIFT = r"C:\Program Files (x86)\Zwift"
STYLES = ["NORMAL","WOODEN","COBBLE","DIRT","ONELINE","WET","MOUNTAIN","LAVA",
          "INVISIBLE_DIRT","BEACHPATH","DESERT","TRAILDIRT","TRAILGRAVEL","TRACK",
          "DOTHINT","GRAVEL","INVISIBLE_GRAVEL","BOXHILL","PACKEDSAND","SNOW"]


def label(v):
    return STYLES[v] if 0 <= v < len(STYLES) else f"#{v}"


def main() -> None:
    world = sys.argv[1] if len(sys.argv) > 1 else "world1"
    wad = os.path.join(ZWIFT, "assets", "Worlds", world, "data_1.wad")
    entries = read_wad_entries(wad)
    txt = next(v for k, v in entries.items() if k.lower().endswith("road.xml")).decode("utf-8", "replace")
    roads = re.findall(r'<road>(.*?)</road>', txt, re.S)

    partial_contig = 0
    partial_noncontig = 0
    for r in roads:
        nodes = re.findall(r'<ent\b[^>]*type="ENTITY_TYPE_ROADNODE"[^>]*>', r)
        styled_idx = [i for i, n in enumerate(nodes) if 'm_style=' in n]
        if styled_idx and len(styled_idx) < len(nodes):
            contig = styled_idx == list(range(styled_idx[0], styled_idx[-1] + 1))
            if contig:
                partial_contig += 1
            else:
                partial_noncontig += 1
    print(f"partially-styled roads: contiguous={partial_contig} noncontiguous={partial_noncontig}")

    # resolve each road to style set and print non-normal roads with center
    print("\n== roads with a non-NORMAL surface ==")
    for ri, r in enumerate(roads):
        dm = re.search(r'<defaultStyle>(\d+)</defaultStyle>', r)
        ds = int(dm.group(1)) if dm else 31
        nodes = re.findall(r'<ent\b[^>]*type="ENTITY_TYPE_ROADNODE"[^>]*>', r)
        mvals = {int(m) for n in nodes for m in re.findall(r'm_style="(\d+)"', n)}
        # effective styles present
        eff = set()
        if 0 <= ds < len(STYLES):
            eff.add(ds)
        eff |= {v for v in mvals if 0 <= v < len(STYLES)}
        eff.discard(0)  # NORMAL
        if not eff:
            continue
        xs, zs = [], []
        for n in nodes:
            p = re.search(r'm_pos="\{([-\d.]+),([-\d.]+),([-\d.]+)\}"', n)
            if p:
                xs.append(float(p.group(1)) / 100)
                zs.append(float(p.group(3)) / 100)
        cx = sum(xs) / len(xs) if xs else 0
        cz = sum(zs) / len(zs) if zs else 0
        idm = re.search(r'<id>(\d+)</id>', r)
        rid = idm.group(1) if idm else "?"
        print(f"  road {rid:>4} default={label(ds):16s} node_styles={sorted(label(v) for v in mvals)} nodes={len(nodes)} center=({cx:.0f},{cz:.0f})")


if __name__ == "__main__":
    main()
