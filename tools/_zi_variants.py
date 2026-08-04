"""List frames where OUR data has a single entry but ZI lists multiple variants."""
import importlib.util, os, re, collections

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("zc", os.path.join(HERE, "_zi_compare.py"))
zc = importlib.util.module_from_spec(spec); spec.loader.exec_module(zc)

# unique ZI road frame names (sheet has 150W + 300W rows per bike)
fr = zc.load_frames_csv()
road = [r for r in fr if r["type"].lower() == "road"]
# keep one record per bike name (prefer 300W row for the gap display)
byname = {}
for r in road:
    b = r["bike"]
    if b not in byname or r["power"] == 300:
        byname[b] = r
zi = list(byname.values())

def base_of(name):
    n = zc.norm(name)
    # drop trailing 4-digit year token
    n = re.sub(r"\b20[0-2]\d\b", "", n)
    return " ".join(n.split())

groups = collections.defaultdict(list)
for r in zi:
    groups[base_of(r["bike"])].append(r)

our = zc.load_our_frames()

# generic trim words that are NOT distinctive model identifiers
TRIM = {"advanced", "sl", "slr", "cf", "cfr", "disc", "rim", "pro", "team",
        "edition", "sram", "shimano", "di2", "cc", "aethos"}
YEAR = lambda t: bool(re.fullmatch(r"20[0-2]\d", t))

def dist(tokens):
    return {t for t in tokens if t not in TRIM and not YEAR(t)}

def our_entry_toks(o):
    mk = zc.toks(o["make"] or "")
    nm = zc.toks(o["name"] or "")
    return mk, dist(nm - mk)

def family_match(base_toks, o):
    mk, omod = our_entry_toks(o)
    bmod = dist(base_toks - mk)          # base model tokens minus this brand
    brand_ok = bool(mk & base_toks)
    model_ok = bool(omod & bmod)
    return brand_ok and model_ok

groups_multi = [(b, v) for b, v in groups.items() if len(v) >= 2]

report = []
for base, variants in groups_multi:
    bt = zc.toks(base)
    hits = sorted({o["label"] for o in our if family_match(bt, o)})
    report.append((base, variants, hits))

single = [r for r in report if len(r[2]) == 1]
multi = [r for r in report if len(r[2]) > 1]
none = [r for r in report if len(r[2]) == 0]

def show(r):
    base, variants, hits = r
    ours = ", ".join(hits) if hits else "(none)"
    print(f"* ZI '{base}'  -> {len(variants)} ZI variants | {len(hits)} of ours: {ours}")
    for v in sorted(variants, key=lambda v: (zc.year_of(v['bike']) or 0)):
        print(f"      ZI: {v['bike']:40} flat={v['flat_gap']:>6}  climb={v['climb_gap']:>6}")
    print()

print("=" * 70)
print(f"ZI has MULTIPLE variants, we have EXACTLY ONE entry  ({len(single)} models)")
print("=" * 70 + "\n")
for r in sorted(single, key=lambda r: r[0]):
    show(r)

print("=" * 70)
print(f"ZI has MULTIPLE variants, we also have MULTIPLE entries  ({len(multi)} models)")
print("=" * 70 + "\n")
for r in sorted(multi, key=lambda r: r[0]):
    show(r)

print("=" * 70)
print(f"ZI has MULTIPLE variants, we have NONE  ({len(none)} models)")
print("=" * 70 + "\n")
for r in sorted(none, key=lambda r: r[0]):
    show(r)
