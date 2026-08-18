"""Throwaway: enumerate every <Bike> config field + attributes to check whether
the game marks frames/wheels as retired / no-longer-available in the drop shop."""
import os, sys, collections
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_zwift_bikes as e  # noqa: E402

base = r"C:\Program Files (x86)\Zwift\assets"
cfg = e.read_wad_entries(os.path.join(base, "Bikes", "bikes_config.wad"))

tags = collections.Counter()
attrs = collections.Counter()
ex = {}
n = 0
for name, data in cfg.items():
    if not (name.startswith("bikes/") and name.endswith("Config.xml")):
        continue
    if "/Wheels/" in name or "/Components/" in name:
        continue
    try:
        root = ET.fromstring(e.decrypt(data))
    except ET.ParseError:
        continue
    if root.tag != "Bike":
        continue
    n += 1
    for k in root.attrib:
        attrs[f"@{k}"] += 1
    for ch in root:
        tags[ch.tag] += 1
        ex.setdefault(ch.tag, (ch.text or "").strip())
        for k in ch.attrib:
            attrs[f"{ch.tag}@{k}"] += 1

print("bikes parsed:", n)
print("\n=== child tags ===")
for t, c in sorted(tags.items(), key=lambda x: -x[1]):
    print(f"{c:4} {t:34} e.g. {ex[t]!r}")
print("\n=== attributes ===")
for a, c in sorted(attrs.items(), key=lambda x: -x[1]):
    print(f"{c:4} {a}")

# Flag anything that looks availability-related.
KEYS = ("avail", "retir", "shop", "purchas", "hidden", "deprecat", "enable",
        "disable", "active", "legacy", "obsolete", "sunset", "sale", "store",
        "unlock", "entitle", "visib")
print("\n=== availability-ish tags/attrs ===")
for coll in (tags, attrs):
    for k in coll:
        if any(w in k.lower() for w in KEYS):
            print(f"  {k}  (in {coll[k]} bikes)  e.g. {ex.get(k.split('@')[0], '')!r}")
