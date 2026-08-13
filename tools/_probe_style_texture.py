"""Dump sound + texture folder for every distinct roadstyle across all worlds."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_zwift_routes import read_wad_entries  # noqa: E402

Z = r"C:\Program Files (x86)\Zwift\assets\Worlds"


def main():
    seen = {}
    for w in sorted(os.listdir(Z)):
        p = os.path.join(Z, w, "data_1.wad")
        if not (w.startswith("world") and os.path.isfile(p)):
            continue
        txt = next(v for k, v in read_wad_entries(p, keep_substrings=("roadstyle.xml",)).items()
                   if k.lower().endswith("roadstyle.xml")).decode("utf-8", "replace")
        for m in re.finditer(r"<segment\s+([^>]*?)/>", txt):
            a = m.group(1)
            nm = re.search(r'style="([^"]+)"', a)
            snd = re.search(r'sound="([^"]*)"', a)
            tex = re.search(r'texture="([^",]*)', a)
            if nm and nm.group(1) not in seen:
                t = tex.group(1) if tex else ""
                tt = re.split(r"[\\/]", t)
                folder = tt[-2] if len(tt) >= 2 else t
                seen[nm.group(1)] = (snd.group(1) if snd else "", folder)
    for k in sorted(seen):
        print(f"{k:22s} sound={seen[k][0]:28s} tex={seen[k][1]}")


if __name__ == "__main__":
    main()
