"""Three-way frame audit:

  1. ZwiftInsider master-list page  vs  ZI speed-test spreadsheet (zi_sheet.csv)
  2. our extracted game stage-0 stats  vs  spreadsheet stage-0 measurements
  3. frames in our game data that are absent from the ZI master list, with the
     game-asset availability signals (Price / Level / entitlement / halo).

Reuses the matcher in _zi_compare.py and the 2x2 stage-0 solver in
_zi_stage_solve.py so the name-matching and physics stay consistent with the
rest of the pipeline.
"""
import csv, json, os, re, math, importlib.util
from statistics import mean, median, pstdev

ROOT = r"C:\Users\timjo\Documents\Coding\Zwift\zwifttools"
TEMP = os.environ["TEMP"]
BASE = r"C:\Program Files (x86)\Zwift\assets"


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(ROOT, "tools", path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

zc = _load("zc", "_zi_compare.py")
zs = _load("zs", "_zi_stage_solve.py")
ez = _load("ez", "extract_zwift_bikes.py")

# ---------------------------------------------------------------------------
# ZwiftInsider master-list page (https://zwiftinsider.com/frames/), transcribed.
# make | model | price(drops) | level | aero* | weight* | type
# ---------------------------------------------------------------------------
ZI_MASTER_RAW = """
Allied|Able 2022|550000|23|1|3|Gravel
BMC|RoadMachine|344100|18|1|3|Road
BMC|SLR01|759500|39|1|3|Road
BMC|TeamMachine|969200|39|1|3|Road
BMC|Timemachine01|400000|7|4|1|TT
Bridgestone|Anchor RS9s|200000|10|1|4|Road
Brompton|P Line|600000|10|1|1|Funny
CADEX|Tri|1500000|40|4|1|TT
Cannondale|CAAD12|106300|6|1|2|Road
Cannondale|CAAD13|500000|14|3|3|Road
Cannondale|EVO|714500|29|1|4|Road
Cannondale|R4000 Roller Blade|10000000|0|3|2|Halo Road
Cannondale|Super Six Evo|768200|29|2|4|Road
Cannondale|SuperSix Evo LAB71|1750000|40|3|4|Road
Cannondale|SuperSix EVO LAB71 Team|1750000|40|3|4|Road
Cannondale|SuperX LAB71|1000000|28|2|3|Gravel
Cannondale|Synapse|270100|12|1|3|Road
Cannondale|System Six|725100|20|3|3|Road
Canyon|Aeroad|532500|23|3|3|Road
Canyon|Aeroad 2021|1029200|27|3|3|Road
Canyon|Aeroad 2024|1100000|10|3|4|Road
Canyon|Aeroad CFR Alpecin Premier-Tech|1750000|30|3|4|Road
Canyon|Grail|750000|26|1|3|Gravel
Canyon|Grail SLX|800000|10|2|3|Gravel
Canyon|Inflite|200000|8|1|2|Gravel
Canyon|Lux|275000|11|1|1|MTB
Canyon|Speedmax CF SLX Disc|1200000|31|4|1|TT
Canyon|Speedmax CFR|1250000|40|4|2|TT
Canyon|Speedmax|424600|31|4|1|TT
Canyon|Ultimate CFR|350700|12|1|4|Road
Canyon|Ultimate|322500|12|1|4|Road
Cervelo|Aspero 5|1250000|35|3|3|Gravel
Cervelo|Aspero|800000|32|2|3|Gravel
Cervelo|P5|920700|34|4|1|TT
Cervelo|PX-Series|1000000|34|4|1|TT
Cervelo|R5|633600|28|1|4|Road
Cervelo|S3D|415300|15|3|2|Road
Cervelo|S5 2015|1438400|36|3|3|Road
Cervelo|S5 2020|1481000|36|3|3|Road
Cervelo|S5|1800000|40|3|3|Road
Chapter2|Koko|505500|21|1|3|Road
Chapter2|Rere|326400|13|1|3|Road
Chapter2|Tere|199400|6|1|4|Road
Chapter2|Toa|800000|33|3|3|Road
Colnago|V3RS|800000|38|1|3|Road
Cube|Aerium|251700|10|4|1|TT
Cube|Litening C:68x|604200|24|1|4|Road
Cube|Litening|536500|24|1|3|Road
Diamondback|Andean|950000|39|4|1|TT
Factor|One|600000|19|3|3|Road
Felt|AR|714100|23|3|3|Road
Felt|FR|562700|16|1|4|Road
Felt|IA 2.0|750000|16|4|1|TT
Felt|IA|515100|16|4|1|TT
Focus|Izalco Max|712900|22|3|3|Road
Giant|Propel Advanced SL Disc|1102900|35|3|3|Road
Giant|Propel Advanced SL Team|1700000|40|3|4|Road
Giant|Revolt Advanced Pro|475000|23|1|3|Gravel
Giant|TCR Advanced BikeExchange-Jayco Team|543100|27|1|3|Road
Giant|TCR Advanced SL 2021|566100|27|1|3|Road
Giant|TCR Advanced SL 2025|1100000|25|3|4|Road
Lauf|True Grit|125000|5|1|2|Gravel
Liv|Devote Advanced Pro|450000|17|1|3|Gravel
Liv|Langma Advanced SL 2021|511300|24|1|3|Road
Liv|Langma Advanced SL 2025|1000000|8|3|3|Road
Liv|Langma SL Advanced Disc|613200|24|1|3|Road
Moots|Vamoots RCS|275000|8|1|2|Road
Mosaic|RT-1d|250000|11|1|2|Road
Parlee|ESX|153200|5|1|3|Road
Parlee|RZ7|771800|32|1|3|Road
Pinarello|Bolide|618400|28|4|1|TT
Pinarello|Bolide TT|627000|28|4|1|TT
Pinarello|Dogma 65.1|577800|40|1|2|Road
Pinarello|Dogma F 2021|1350000|32|3|4|Road
Pinarello|Dogma F 2024|1750000|40|3|4|Road
Pinarello|Dogma F10|1081900|40|3|3|Road
Pinarello|Dogma F12|1242700|40|3|3|Road
Pinarello|Dogma F8|0|0|3|3|Road
Pinarello|Dogma GR|1100000|30|2|3|Gravel
Pinarello|Dogma X|850000|27|1|3|Road
Pinarello|Espada|10000000|0|4|1|Halo TT
Quintana|Roo V-PR|297400|13|4|1|TT
Ribble|Endurance|505500|21|1|3|Road
Ridley|Helium|272500|15|1|3|Road
Ridley|Noah Fast|950000|33|3|3|Road
Scott|Addict RC|660200|17|3|4|Road
Scott|Foil 2015|676900|26|1|3|Road
Scott|Foil 2023|864600|26|3|3|Road
Scott|Plasma RC Ultimate|600000|19|4|1|TT
Scott|Plasma|528600|19|4|1|TT
Scott|Spark RC|350000|20|1|1|MTB
Scott|Spark RC World Cup|750000|20|1|1|MTB
Specialized|Aethos S-Works|966300|30|2|4|Road
Specialized|Allez|122700|9|1|2|Road
Specialized|Allez Sprint|387000|14|1|3|Road
Specialized|Amira|730400|36|1|3|Road
Specialized|Amira S-Works|966300|36|1|4|Road
Specialized|Crux|950000|35|1|3|Gravel
Specialized|Diverge 4|400000|12|2|2|Gravel
Specialized|Diverge|300000|14|1|2|Gravel
Specialized|Epic S-Works|950000|29|1|1|MTB
Specialized|Project 74|10000000|0|3|2|Halo Road
Specialized|Roubaix|333500|18|1|3|Road
Specialized|Roubaix S-Works|405200|18|1|3|Road
Specialized|Ruby|333500|18|1|3|Road
Specialized|Ruby S-Works|399300|18|1|3|Road
Specialized|S-Works Tarmac SL8|1750000|40|3|4|Road
Specialized|Tarmac SL9|1900000|40|3|4|Road
Specialized|Shiv Disc|1000000|37|4|1|TT
Specialized|Shiv S-Works|910600|37|4|1|TT
Specialized|Shiv|643800|37|4|1|TT
Specialized|Tarmac Pro|856100|36|1|4|Road
Specialized|Tarmac|786300|36|1|4|Road
Specialized|Tarmac SL7|1200000|36|3|4|Road
Specialized|Tarmac SL8|700000|35|3|4|Road
Specialized|Venge 2015|461500|18|3|3|Road
Specialized|Venge S-Works 2019|1200000|37|3|3|Road
Trek|Emonda|0|0|2|4|Road
Trek|Emonda SL|560300|25|1|3|Road
Trek|Madone|1050000|34|3|3|Road
Trek|Speed Concept SLR 9|670500|22|4|1|TT
Trek|Super Caliber|750000|38|1|1|MTB
Uranium|Nuclear|700000|31|2|3|Road
Van Rysel|EDR CF|144100|7|1|3|Road
Van Rysel|RCR Pro|1400000|21|3|4|Road
Van Rysel|RCR-F|1500000|35|3|3|Road
Van Rysel|RCR-X|1200000|40|4|1|TT
Ventum|NS1|750000|30|3|3|Road
Ventum|One|440800|25|4|1|TT
Wilier|Filante SLR ID2|1700000|40|3|4|Road
Zwift|Aero|250000|9|3|3|Road
Zwift|Atomic Cruiser|0|0|2|2|Funny
Zwift|BMX Bandit|0|0|1|1|Funny
Zwift|Buffalo Fahrrad|9500|40|1|1|Road
Zwift|Carbon|55000|3|1|3|Road
Zwift|Concept 1 (Tron)|0|0|3|3|Halo Road
Zwift|Gravel|50000|2|1|2|Gravel
Zwift|Handcycle|0|0|4|1|Recumbent
Zwift|Mountain|50000|2|1|1|MTB
Zwift|MX Rider|0|0|1|1|Funny
Zwift|Recumbent Trike|0|0|4|1|Recumbent
Zwift|Safety|3550000|44|1|2|Road
Zwift|Steel|0|1|1|2|Road
Zwift|TT|60000|4|4|1|TT
"""


def load_zi_master():
    out = []
    for line in ZI_MASTER_RAW.strip().splitlines():
        mk, mdl, price, level, aero, wt, typ = line.split("|")
        out.append({"make": mk, "model": mdl, "name": f"{mk} {mdl}",
                    "price": int(price), "level": int(level),
                    "aero": int(aero), "weight": int(wt), "type": typ})
    return out


def load_sheet_bikes():
    """Every bike row of the ZI speed-test spreadsheet, with its metadata."""
    rows = list(csv.reader(open(os.path.join(ROOT, "zwiftdata", "zi_sheet.csv"), encoding="utf-8-sig")))
    out = []
    for r in rows[2:]:
        if len(r) < 8 or not r[0].strip():
            continue
        def dp(i):
            try:
                return int(r[i].replace(",", ""))
            except (ValueError, IndexError):
                return None
        out.append({"name": r[0].strip(), "wheels": r[1].strip(), "type": r[2].strip(),
                    "price": dp(3), "level": dp(4)})
    return out


# ---------------------------------------------------------------------------
# raw <Bike> configs with availability-relevant fields
# ---------------------------------------------------------------------------
def load_raw_configs():
    import xml.etree.ElementTree as ET
    cfg = ez.read_wad_entries(os.path.join(BASE, "Bikes", "bikes_config.wad"))
    out = {}
    for name, data in cfg.items():
        if not (name.startswith("bikes/") and name.endswith("Config.xml")):
            continue
        if "/Wheels/" in name or "/Components/" in name:
            continue
        try:
            root = ET.fromstring(ez.decrypt(data))
        except ET.ParseError:
            continue
        if root.tag != "Bike":
            continue
        frame_path = root.findtext("Frame") or ""
        folder = ez._folder_from_path(frame_path) if frame_path else None
        if not folder:
            continue
        out[folder] = {
            "make": (root.findtext("Make") or "").strip(),
            "name": (root.findtext("Name") or "").strip(),
            "price": root.findtext("Price"),
            "level": root.findtext("Level"),
            "halo": (root.findtext("HaloBike") or "").strip().upper() == "TRUE",
            "entitlement": root.findtext("EntitlementRequiredToBuy"),
            "type": (root.findtext("Type") or "").strip(),
            "class": (root.findtext("BikeClass") or "").strip(),
        }
    return out


def _key(name):
    return frozenset(zc.toks(name))


def match_sets(a_rows, b_rows, a_key, b_key):
    """Greedy name match a->b using the shared matcher. Returns (matched, only_a, only_b)."""
    b_avail = [{"label": r[b_key], "folder": ""} for r in b_rows]
    matched, only_a = [], []
    used = set()
    for ar in a_rows:
        m, sc = zc.best_match(ar[a_key], b_avail, zi_year=zc.year_of(ar[a_key]))
        if m and sc >= 0.34 and m["label"] not in used:
            used.add(m["label"])
            matched.append((ar, m["label"], sc))
        else:
            only_a.append(ar)
    only_b = [r for r in b_rows if r[b_key] not in used]
    return matched, only_a, only_b


# ===========================================================================
# AUDIT 1 — master-list page vs speed-test spreadsheet
# ===========================================================================
def audit1():
    print("=" * 78)
    print("AUDIT 1 — ZI master-list page  vs  ZI speed-test spreadsheet")
    print("=" * 78)
    master = load_zi_master()
    sheet = load_sheet_bikes()
    print(f"master-list page : {len(master)} frames")
    print(f"spreadsheet      : {len(sheet)} frames")
    matched, only_master, only_sheet = match_sets(master, sheet, "name", "name")
    print(f"matched          : {len(matched)}")
    print(f"\nOn master page but NOT in spreadsheet ({len(only_master)}):")
    for r in sorted(only_master, key=lambda x: x["name"]):
        print(f"   - {r['name']}  ({r['type']}, level {r['level']})")
    print(f"\nIn spreadsheet but NOT on master page ({len(only_sheet)}):")
    for r in sorted(only_sheet, key=lambda x: x["name"]):
        print(f"   - {r['name']}  ({r['type']})")


# ===========================================================================
# AUDIT 2 — our extracted stage-0 vs spreadsheet stage-0 measurements
# ===========================================================================
def audit2():
    print("\n" + "=" * 78)
    print("AUDIT 2 — our game stage-0 stats  vs  spreadsheet stage-0 measurements")
    print("=" * 78)
    recs, theta = zs.build()   # matched bikes with solved per-stage cda/mass
    print(f"climb gradient theta = {math.tan(theta)*100:.2f}%   matched bikes n={len(recs)}")

    def ctx(rc):
        return (rc["m"].get("type") or "?", rc["z"].get("wheels") or "?",
                rc["z"].get("power") or 0, rc["z"].get("bike") or "?",
                rc.get("score") or 0.0)

    # --- WEIGHT: solved stage-0 mass (absolute kg) vs our game total mass ---
    wpairs = []
    for rc in recs:
        solved = rc["mass"][0]
        game = rc["m_game"]           # 75 + frameset_g/1000 + 1.5
        typ, wheel, power, zibike, sco = ctx(rc)
        wpairs.append((rc["m"]["label"], zibike, sco, wheel, power, solved, game, solved - game))
    diffs = [p[7] for p in wpairs]
    print("\n--- WEIGHT (ZI-solved stage-0 total mass vs our game total mass, kg) ---")
    print(f"  n={len(wpairs)}  mean diff={mean(diffs):+.3f}  median={median(diffs):+.3f}"
          f"  RMS={ (sum(d*d for d in diffs)/len(diffs))**0.5 :.3f}  std={pstdev(diffs):.3f}")
    print(f"  Spearman(solved, game) = {zc.spearman([(p[5], p[6]) for p in wpairs]):+.3f}")
    print(f"  {'our frame':26} {'ZI test bike':22} {'sc':>4} {'wheel':14} {'W':>4} {'ZI':>6} {'game':>6} {'diff':>6}")
    for lbl, zb, sco, wheel, power, s, g, d in sorted(wpairs, key=lambda p: -abs(p[7]))[:15]:
        print(f"    {lbl:26.26} {zb:22.22} {sco:4.2f} {wheel:14.14} {power:>4.0f} {s:6.2f} {g:6.2f} {d:+6.2f}")

    # --- CdA: solved stage-0 CdA vs our (frame + test-wheel) bias -----------
    frames = {f["folder"]: f for f in _load_our_full()}
    cpairs = []
    for rc in recs:
        fr = frames.get(rc["m"]["folder"])
        if not fr:
            continue
        # test-wheel CdA bias comes from the solver (TT frames use the wheel's
        # TT bias, others the road bias)
        our_total_bias = (fr.get("frameset_cda_bias_effective") or 0.0) + rc.get("wheel_cda", 0.0)
        typ, wheel, power, zibike, sco = ctx(rc)
        cpairs.append((rc["m"]["label"], zibike, sco, wheel, power, rc["cda"][0], our_total_bias))
    # linear fit solved = a + b*bias  (a = implied rider+bike baseline CdA)
    xs = [p[6] for p in cpairs]; ys = [p[5] for p in cpairs]
    n = len(xs); mx = mean(xs); my = mean(ys)
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    a = my - b*mx
    rows = [(lbl, zb, sco, wheel, power, a + b*bias, solved, solved - (a + b*bias))
            for lbl, zb, sco, wheel, power, solved, bias in cpairs]
    resid = [r[7] for r in rows]
    print("\n--- CdA (ZI-solved stage-0 CdA vs our game-implied CdA, m^2) ---")
    print(f"  n={n}  our-implied = {a:.4f} + {b:.3f}*(frame+wheel bias);  baseline {a:.4f}")
    print(f"  residual std={pstdev(resid):.5f}  R={zc.spearman(list(zip(xs, ys))):+.3f}")
    print(f"  {'our frame':26} {'ZI test bike':22} {'sc':>4} {'wheel':14} {'W':>4} {'ours':>7} {'ZI':>7} {'diff':>8}")
    for lbl, zb, sco, wheel, power, our, zi, r in sorted(rows, key=lambda t: -abs(t[7]))[:15]:
        print(f"    {lbl:26.26} {zb:22.22} {sco:4.2f} {wheel:14.14} {power:>4.0f} {our:7.4f} {zi:7.4f} {r:+8.5f}")


def _load_our_full():
    return json.load(open(os.path.join(ROOT, "zwiftdata", "game_frames.json"),
                          encoding="utf-8-sig"))


_SHEET_CACHE = None
def _sheet_row_for(bike):
    global _SHEET_CACHE
    if _SHEET_CACHE is None:
        _SHEET_CACHE = {r["name"]: r for r in load_sheet_bikes()}
    return _SHEET_CACHE.get(bike)


# ===========================================================================
# AUDIT 3 — our frames not on the ZI master list + asset availability signals
# ===========================================================================
def _matches(name, avail, thresh=0.34):
    m, sc = zc.best_match(name, avail, zi_year=zc.year_of(name))
    return m is not None and sc >= thresh


def audit3():
    print("\n" + "=" * 78)
    print("AUDIT 3 — frames in our game data but NOT on the ZI master list")
    print("=" * 78)
    master = load_zi_master()
    ours = _load_our_full()
    raw = load_raw_configs()
    our_rows = [{"name": f"{f.get('make') or ''} {f.get('name') or ''}".strip(),
                 "folder": f.get("folder"), "type": f.get("type"),
                 "price": f.get("price"), "level": f.get("level")} for f in ours]
    master_avail = [{"label": m["name"], "folder": ""} for m in master]
    sheet_avail = [{"label": r["name"], "folder": ""} for r in load_sheet_bikes()]

    # many-to-one: paint/team variants legitimately map onto one ZI entry, so we
    # DON'T mark a match as "used". A frame is only "extra" if it matches nothing
    # on the master page. We then cross-check the 283-row speed-test spreadsheet.
    in_master = [o for o in our_rows if _matches(o["name"], master_avail)]
    extra = [o for o in our_rows if not _matches(o["name"], master_avail)]
    print(f"our frames: {len(our_rows)}   match master page: {len(in_master)}   "
          f"not on master page: {len(extra)}")

    genuine, in_sheet_only = [], []
    for o in extra:
        (in_sheet_only if _matches(o["name"], sheet_avail) else genuine).append(o)

    def classify(o):
        rc = raw.get(o["folder"], {})
        price = o.get("price"); lvl = o.get("level")
        if rc.get("halo"):
            return "HALO bike (not drop-shop)"
        if rc.get("entitlement"):
            return f"entitlement {rc['entitlement']} required (special unlock)"
        if price == 0:
            return "event/challenge unlock (Price 0)"
        if lvl is not None and lvl < 0:
            return "drop-shop, no XP gate"
        return "drop-shop, XP-gated"

    def show(rows):
        print(f"\n  {'frame':30} {'type':9} {'price×71':>10} {'lvl':>4}  availability signal")
        print("  " + "-" * 84)
        for o in sorted(rows, key=lambda x: x["name"]):
            price = o.get("price")
            drops = f"{price*71:,}" if isinstance(price, int) else "?"
            print(f"  {o['name']:30} {str(o['type']):9} {drops:>10} {str(o['level']):>4}  {classify(o)}")

    print(f"\n>>> Not on master page but PRESENT in speed-test spreadsheet "
          f"({len(in_sheet_only)}) — variants / renames / newer models:")
    show(in_sheet_only)
    print(f"\n>>> GENUINELY absent from all ZI data ({len(genuine)}):")
    show(genuine)

    # asset-wide availability signal summary
    print("\n--- game-asset availability fields (all 163 configs) ---")
    halos = [r["name"] for r in raw.values() if r.get("halo")]
    ents = [(r["name"], r["entitlement"]) for r in raw.values() if r.get("entitlement")]
    freep = [r["name"] for r in raw.values()
             if r.get("price") == "0" and (r.get("level") or "0").lstrip("-").isdigit()
             and int(r.get("level") or 0) < 0]
    print(f"  HaloBike=TRUE ({len(halos)}): {', '.join(halos)}")
    print(f"  EntitlementRequiredToBuy ({len(ents)}): {ents}")
    print(f"  Price=0 & Level<0 (event/challenge unlocks): {len(freep)}")


if __name__ == "__main__":
    audit1()
    audit2()
    audit3()
