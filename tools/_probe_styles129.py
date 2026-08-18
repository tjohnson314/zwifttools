import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries
from extract_zwift_surfaces import parse_roadstyles

ZWIFT = r"C:\Program Files (x86)\Zwift"
wad = os.path.join(ZWIFT, "assets", "Worlds", "world9", "data_1.wad")
entries = read_wad_entries(wad, keep_substrings=("roadstyle.xml",))
styledata = next(v for k, v in entries.items() if k.lower().endswith("roadstyle.xml"))
styles = parse_roadstyles(styledata)

surf_map = json.load(open('zwift_surfaces/style_surface_map.json', encoding='utf-8'))
# support either {NAME: surface} or nested
if isinstance(next(iter(surf_map.values())), dict):
    surf_map = {k: v.get('surface', v) for k, v in surf_map.items()}

for idx in (2, 4, 5, 10):
    name = styles[idx] if idx < len(styles) else "?"
    surf = surf_map.get(name, "(default) Tarmac")
    print(f"style {idx:2d} -> {name:20s} -> {surf}")
