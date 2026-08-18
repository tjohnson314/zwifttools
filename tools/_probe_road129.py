import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries

ZWIFT = r"C:\Program Files (x86)\Zwift"
wad = os.path.join(ZWIFT, "assets", "Worlds", "world9", "data_1.wad")
entries = read_wad_entries(wad, keep_substrings=("road.xml",))
road = next(v for k, v in entries.items()
            if k.lower().endswith("road.xml") and "roadstyle" not in k.lower())
txt = road.decode("utf-8", "replace")

for m in re.finditer(r"<road>.*?</road>", txt, re.S):
    block = m.group(0)
    idm = re.search(r"<id>(\d+)</id>", block)
    if idm and idm.group(1) == "129":
        # normalise CRLF, collapse blank lines, strip trailing spaces
        clean = block.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln.rstrip() for ln in clean.split("\n") if ln.strip()]
        out = "\n".join(lines) + "\n"
        with open("tools/_road129.xml", "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        nodes = out.count('type="ENTITY_TYPE_ROADNODE"')
        markers = out.count('type="ENTITY_TYPE_ROADMARKER"')
        print(f"wrote {len(out)} chars, {len(lines)} lines, "
              f"{nodes} nodes, {markers} markers, closed={'</road>' in out}")
        break
