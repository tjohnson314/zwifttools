"""Compare our extracted game weight/CdA against ZwiftInsider speed-test data."""
import csv, json, os, re, statistics

ROOT = r"C:\Users\timjo\Documents\Coding\Zwift\zwifttools"
# cached ZwiftInsider spreadsheets (refresh via tools/cache_zi_sheet.py)
ZI_SHEET = os.path.join(ROOT, "zwiftdata", "zi_sheet.csv")
ZI_WHEELS = os.path.join(ROOT, "zwiftdata", "zi_wheels.csv")

# strip trailing 4-digit model YEARS our folder names carry (…2021, …2026)
_STOP = {"the", "team", "edition", "new", "white", "lava", "pro", "disc",
         "alpecin", "premier", "tech", "bikeexchange", "jayco"}


def canon(s):
    """Punctuation/space/case-insensitive key for tolerant name lookup, so
    'VanRysel RCR X', 'VanRysel RCR-X' and 'Van Rysel RCR-X' all collapse to the
    same key."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def norm(s):
    s = (s or "")
    # split camelCase and letter<->digit boundaries BEFORE lowercasing
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)   # RCRPro -> RCR Pro
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)
    s = s.lower()
    s = s.replace("s-works", "sworks").replace("s works", "sworks")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def toks(s):
    return {t for t in norm(s).split() if t not in _STOP}

def year_of(*strings):
    for s in strings:
        m = re.search(r"\b(20[0-2]\d)\b", s or "")
        if m:
            return int(m.group(1))
    return None

# ---------- load ZI frames sheet ----------
def load_frames_csv():
    path = ZI_SHEET
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header = rows[1]
    out = []
    for r in rows[2:]:
        if len(r) < 22 or not r[0].strip():
            continue
        try:
            power = float(r[7])
        except (ValueError, IndexError):
            continue
        def num(i):
            try:
                return float(r[i])
            except (ValueError, IndexError):
                return None
        out.append({
            "bike": r[0].strip(), "wheels": r[1].strip(), "type": r[2].strip(),
            "power": power, "flat_gap": num(9), "climb_gap": num(21),
        })
    return out

# ---------- load ZI wheels sheet ----------
def load_wheels_csv():
    path = ZI_WHEELS
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    out = []
    for r in rows:
        if len(r) < 11 or r[0].strip() != "Zwift Carbon":
            continue
        def num(i):
            try:
                return float(r[i])
            except (ValueError, IndexError):
                return None
        out.append({"wheel": r[1].strip(), "flat_gap": num(8), "climb_gap": num(10)})
    return out

# ---------- load our data ----------
def load_our_frames():
    d = json.load(open(os.path.join(ROOT, "zwiftdata", "game_frames.json"), encoding="utf-8-sig"))
    out = []
    for f in d:
        w = f.get("frameset_weight_g_effective")
        c = f.get("frameset_cda_bias_effective")
        if w is None or c is None:
            continue
        label = f"{f.get('make') or ''} {f.get('name') or ''}".strip() or f.get("folder")
        out.append({"label": label, "make": f.get("make"), "name": f.get("name"),
                    "folder": f.get("folder"), "type": f.get("type"),
                    "year": year_of(str(f.get("year") or ""), f.get("folder")),
                    "weight": w, "cda": c})
    return out

def load_our_wheels():
    d = json.load(open(os.path.join(ROOT, "zwiftdata", "game_wheels.json"), encoding="utf-8-sig"))
    out = []
    for wh in d:
        w = wh.get("pair_weight_g_effective")
        c = wh.get("pair_cda_bias_effective")
        if w is None or c is None:
            continue
        label = f"{wh.get('brand') or ''} {wh.get('model') or ''}".strip()
        out.append({"label": label, "brand": wh.get("brand"), "model": wh.get("model"),
                    "weight": w, "cda": c})
    return out

# ---------- matching ----------
def _pair_score(zt, o, zi_year):
    """Token-overlap similarity between a ZI name (token set zt) and one of our
    frames o, with model-number and year adjustments. Shared by best_match and
    rank_matches so both use identical scoring."""
    # MAKE must match exactly: every token of our frame's make has to appear in
    # the ZI bike name. Prevents a displaced same-model variant from falling onto
    # a wrong-brand frame (e.g. "Cervelo S5 2020" -> Factor One).
    mk = toks(o.get("make") or "")
    if mk and not (mk <= zt):
        return 0.0
    # include folder-derived tokens: several game_frames records carry a
    # folder-style name (e.g. LivLangma2021, or folder Specialized_Roubaix
    # whose display name is "Ruby S-Works"), which breaks name-only matching.
    ot = toks(o["label"]) | toks(o.get("folder", "") or "")
    if not ot:
        return 0.0
    inter = len(zt & ot)
    if inter == 0:
        return 0.0
    score = inter / max(1, len(zt | ot))
    # boost for model-number token exact match (e.g. 808, s5, sl8)
    nums_z = {t for t in zt if any(ch.isdigit() for ch in t)}
    nums_o = {t for t in ot if any(ch.isdigit() for ch in t)}
    if nums_z and nums_z <= nums_o:
        score += 0.3
    # year compatibility: if both known, reward match / penalise mismatch
    oy = o.get("year")
    if zi_year and oy:
        if zi_year == oy:
            score += 0.5
        else:
            score -= 0.5 * min(1.0, abs(zi_year - oy) / 3.0) + 0.15
    return score


def best_match(zi_name, ours, alias=None, zi_year=None):
    zt = toks(zi_name)
    if alias:
        zt = zt | toks(alias.get(zi_name, ""))
    best, bestscore = None, 0.0
    for o in ours:
        score = _pair_score(zt, o, zi_year)
        if score > bestscore:
            best, bestscore = o, score
    return best, bestscore


def rank_matches(zi_name, ours, alias=None, zi_year=None, min_score=0.0):
    """All candidate frames scoring at/above min_score, sorted best-first.
    Used for global 1:1 assignment where a displaced row needs its next-best."""
    zt = toks(zi_name)
    if alias:
        zt = zt | toks(alias.get(zi_name, ""))
    out = []
    for o in ours:
        score = _pair_score(zt, o, zi_year)
        if score > 0 and score >= min_score:
            out.append((o, score))
    out.sort(key=lambda t: -t[1])
    return out

def spearman(pairs):
    # pairs: list of (x, y)
    n = len(pairs)
    if n < 3:
        return None
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j+1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = (sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry)) ** 0.5
    return num/den if den else None

# frame name aliases (ZI sheet name -> extra tokens)
FRAME_ALIAS = {
}

def analyze(kind, zi_rows, ours, name_key):
    matched = []
    unmatched = []
    used = set()
    for z in zi_rows:
        zy = year_of(z[name_key])
        m, sc = best_match(z[name_key], ours, zi_year=zy)
        if m and sc >= 0.34:
            matched.append((z, m, sc))
        else:
            unmatched.append((z[name_key], sc, m["label"] if m else None))
    print(f"\n===== {kind}: matched {len(matched)} / {len(zi_rows)} ZI rows =====")
    # correlations
    flat_pairs = [(mm["cda"], z["flat_gap"]) for z, mm, _ in matched if z["flat_gap"] is not None]
    climb_pairs = [(mm["weight"], z["climb_gap"]) for z, mm, _ in matched if z["climb_gap"] is not None]
    print(f"  Spearman(our CdA vs ZI flat gap)   = {spearman(flat_pairs):+.3f}  (expect strongly NEGATIVE)")
    print(f"  Spearman(our weight vs ZI climb gap)= {spearman(climb_pairs):+.3f}  (expect strongly NEGATIVE)")
    report_inconsistencies(kind, matched)
    return matched, unmatched

def _rank_map(items, key, reverse):
    ordered = sorted(items, key=key, reverse=reverse)
    return {id(it): i + 1 for i, it in enumerate(ordered)}

def report_inconsistencies(kind, matched):
    # Build list with our stats + ZI gaps
    rows = []
    for z, mm, sc in matched:
        if z["flat_gap"] is None or z["climb_gap"] is None:
            continue
        rows.append({"zi": z, "our": mm, "flat": z["flat_gap"], "climb": z["climb_gap"],
                     "cda": mm["cda"], "weight": mm["weight"], "label": mm["label"]})
    n = len(rows)
    if n < 4:
        return
    # Rank: aero -> lower cda = rank1; ZI flat -> higher gap = rank1
    rk_cda = _rank_map(rows, key=lambda r: r["cda"], reverse=False)
    rk_flat = _rank_map(rows, key=lambda r: r["flat"], reverse=True)
    rk_wt = _rank_map(rows, key=lambda r: r["weight"], reverse=False)
    rk_climb = _rank_map(rows, key=lambda r: r["climb"], reverse=True)
    for r in rows:
        r["d_aero"] = rk_cda[id(r)] - rk_flat[id(r)]
        r["d_climb"] = rk_wt[id(r)] - rk_climb[id(r)]
    print(f"  -- biggest AERO rank disagreements (our-CdA-rank minus ZI-flat-rank), n={n} --")
    for r in sorted(rows, key=lambda r: -abs(r["d_aero"]))[:8]:
        print(f"     {r['label'][:34]:34} ourCdA={r['cda']:+.4f} rk{rk_cda[id(r)]:>2} | ZIflat={r['flat']:+6.1f} rk{rk_flat[id(r)]:>2} | Δ{r['d_aero']:+d}")
    print(f"  -- biggest WEIGHT rank disagreements (our-weight-rank minus ZI-climb-rank) --")
    for r in sorted(rows, key=lambda r: -abs(r["d_climb"]))[:8]:
        print(f"     {r['label'][:34]:34} ourWt={r['weight']:>6.0f}g rk{rk_wt[id(r)]:>2} | ZIclimb={r['climb']:+6.1f} rk{rk_climb[id(r)]:>2} | Δ{r['d_climb']:+d}")

if __name__ == "__main__":
    fr = load_frames_csv()
    fr300 = [r for r in fr if r["power"] == 300 and r["type"].lower() == "road"]
    our_fr = load_our_frames()
    fm, fu = analyze("FRAMES(300W road)", fr300, our_fr, "bike")

    wh = load_wheels_csv()
    our_wh = load_our_wheels()
    wm, wu = analyze("WHEELS(300W)", wh, our_wh, "wheel")

    print("\n--- unmatched ZI frames ---")
    for n, sc, cand in fu:
        print(f"   {n!r} best={cand!r} sc={sc:.2f}")
    print("\n--- unmatched ZI wheels ---")
    for n, sc, cand in wu:
        print(f"   {n!r} best={cand!r} sc={sc:.2f}")
