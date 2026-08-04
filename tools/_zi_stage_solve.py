"""Direct 2x2 steady-state solver for Zwift per-stage frame upgrades from ZwiftInsider speed tests.

At each upgrade stage the flat + climb average-speed tests give two linear equations in the two
unknowns (CdA, total mass), because the power balance is linear in (CdA, m) once the measured
speed is fixed:

    flat  (grade 0):  P(1-DL) = (1/2 rho v_f^3) CdA + (Crr g v_f) m
    climb (grade th): P(1-DL) = (1/2 rho v_c^3) CdA + (g (sin th + Crr cos th) v_c) m

Solve the 2x2 directly (Cramer). The single shared unknown is the climb gradient theta, which is
calibrated once so the solver's stage-0 masses match the authoritative game frame weights.
Climb test = Alpe du Zwift ~8.25% mean; the constant-gradient approximation is within ~1% of the
real varying profile on total time (validated separately), which largely cancels in stage deltas.
"""
import csv, json, os, math, importlib.util
from statistics import mean, pstdev, median

ROOT = r"C:\Users\timjo\Documents\Coding\Zwift\zwifttools"
spec = importlib.util.spec_from_file_location("zc", os.path.join(ROOT, "tools", "_zi_compare.py"))
zc = importlib.util.module_from_spec(spec); spec.loader.exec_module(zc)

RHO, G, DL, CRR, MPH, RIDER_KG, WHEELS_KG = 1.225, 9.8067, 0.025, 0.004, 0.44704, 75.0, 1.5

# ZwiftInsider Crr table (https://zwiftinsider.com/crr/): Zwift models exactly
# three wheel Crr *categories*, and every wheelset within a category shares the
# same per-surface Crr. Both ZI speed tests (the flat run and the Alpe du Zwift
# climb) are ridden entirely on PAVEMENT, so the correct rolling resistance for a
# bike is simply the pavement Crr of its test wheelset's category.
ROAD_CRR, GRAVEL_CRR, MTB_CRR = 0.004, 0.008, 0.009


def _wheel_crr_overrides():
    """Wheels whose own pavement Crr the game overrides (game_wheels.json), so we
    can prefer the authoritative in-game value over the category table."""
    gw = json.load(open(os.path.join(ROOT, "zwiftdata", "game_wheels.json"), encoding="utf-8-sig"))
    out = []
    for w in gw:
        sc = w.get("surface_crr") or {}
        out.append({"label": f"{w.get('brand') or ''} {w.get('model') or ''}".strip(),
                    "folder": "", "pavement": sc.get("pavement")})
    return out


def resolve_crr(wheel_name, overrides):
    """Pavement Crr for a test wheelset: the game's per-wheel override if it has
    one, else the ZwiftInsider category table (gravel 0.008 / MTB 0.009 / road
    0.004)."""
    m, sc = zc.best_match(wheel_name or "", overrides)
    if m and sc >= 0.34 and m.get("pavement") is not None:
        return m["pavement"], "game"
    n = (wheel_name or "").lower()
    if "gravel" in n:
        return GRAVEL_CRR, "gravel"
    if "mountain" in n or "mtb" in n:
        return MTB_CRR, "mtb"
    return ROAD_CRR, "road"


def _wheel_props():
    """Test-wheel physical stats from game_wheels.json: pair mass (kg), the road
    CdA bias, and the TT-specific CdA bias (deep/disc wheels are more aero on TT
    frames, so the game stores a separate, larger bias for them)."""
    gw = json.load(open(os.path.join(ROOT, "zwiftdata", "game_wheels.json"), encoding="utf-8-sig"))
    out = []
    for w in gw:
        mg = w.get("pair_weight_g_effective")
        out.append({"label": f"{w.get('brand') or ''} {w.get('model') or ''}".strip(),
                    "folder": "", "mass_kg": (mg / 1000.0) if mg is not None else None,
                    "cda_road": w.get("pair_cda_bias_effective"),
                    "cda_tt": w.get("pair_cda_bias_tt")})
    return out


def resolve_wheel(wheel_name, frame_type, props):
    """Actual test-wheel mass (kg) and CdA bias for a bike. TT frames get the
    wheel's TT CdA bias (disc/deep wheels help TT bikes more); every other frame
    type gets the standard road bias. Falls back to the nominal 1.5 kg / 0 bias
    if the wheelset can't be matched."""
    m, sc = zc.best_match(wheel_name or "", props)
    if not m or sc < 0.34:
        return WHEELS_KG, 0.0, "default"
    mass = m["mass_kg"] if m["mass_kg"] is not None else WHEELS_KG
    if (frame_type or "").upper() == "TT" and m["cda_tt"] is not None:
        return mass, m["cda_tt"], "tt"
    return mass, (m["cda_road"] or 0.0), "road"


# "Halo" bikes ship with permanently-attached wheels that riders can't swap, so
# ZwiftInsider's free-text wheel label for these is just the bike name (e.g.
# "Cannondale R4000 Roller Blade") and does NOT fuzzy-match the built-in wheel's
# game id (model "CannondalePong"). Resolve those wheels deterministically by
# frame folder instead of by name so their real mass + CdA bias are used.
_BUILTIN_WHEEL_BY_FOLDER = {
    "CannondalePong": ("Cannondale", "CannondalePong"),
    "SpecializedProject74": ("Roval", "RovalProject74"),
    "PinarelloEspada": ("Pinarello", "PinarelloEspada"),
    "Zwift_Concept": ("Zwift", "Zwift_Concept"),
    "Zwift_Concept_Gold": ("Zwift", "Zwift_Concept_Gold"),
}


def _builtin_wheel_props():
    """frame folder -> built-in wheel {mass_kg, cda_road, cda_tt} for halo bikes."""
    gw = json.load(open(os.path.join(ROOT, "zwiftdata", "game_wheels.json"), encoding="utf-8-sig"))
    by_key = {(w.get("brand"), w.get("model")): w for w in gw}
    out = {}
    for folder, key in _BUILTIN_WHEEL_BY_FOLDER.items():
        w = by_key.get(key)
        if not w:
            continue
        mg = w.get("pair_weight_g_effective")
        out[folder] = {"mass_kg": (mg / 1000.0) if mg is not None else None,
                       "cda_road": w.get("pair_cda_bias_effective"),
                       "cda_tt": w.get("pair_cda_bias_tt")}
    return out


def resolve_builtin_wheel(folder, frame_type, builtins):
    """Mass + CdA bias for a halo bike's built-in wheel, or None if the frame has
    no built-in wheel."""
    bw = builtins.get(folder)
    if bw is None:
        return None
    mass = bw["mass_kg"] if bw["mass_kg"] is not None else WHEELS_KG
    if (frame_type or "").upper() == "TT" and bw["cda_tt"] is not None:
        return mass, bw["cda_tt"], "builtin-tt"
    return mass, (bw["cda_road"] or 0.0), "builtin"



def load_zi_full():
    rows = list(csv.reader(open(os.path.join(ROOT, "zwiftdata", "zi_sheet.csv"), encoding="utf-8-sig")))
    hdr = rows[1]
    spd = [i for i, h in enumerate(hdr) if h.strip().endswith("Avg Speed MPH")]
    flat_idx, climb_idx = spd[:6], spd[6:12]
    out, seen = [], set()
    for r in rows[2:]:
        if len(r) < 30 or not r[0].strip():
            continue
        def num(i):
            try: return float(r[i])
            except (ValueError, IndexError): return None
        flats = [num(i) for i in flat_idx]; climbs = [num(i) for i in climb_idx]
        if any(v is None for v in flats) or any(v is None for v in climbs):
            continue
        # collapse duplicate rows for the same bike (e.g. re-tests at another
        # power, or a mixed-spelling row normalized to an existing name): keep
        # the first occurrence so each bike contributes one deterministic row.
        key = zc.canon(r[0])
        if key in seen:
            continue
        seen.add(key)
        out.append({"bike": r[0].strip(), "wheels": r[1].strip(), "power": num(7),
                    "flat": [v * MPH for v in flats], "climb": [v * MPH for v in climbs]})
    return out


def load_our_frames():
    gf = json.load(open(os.path.join(ROOT, "zwiftdata", "game_frames.json"), encoding="utf-8-sig"))
    return [{"label": f"{f.get('make') or ''} {f.get('name') or ''}".strip(),
             "make": f.get("make"),
             "folder": f.get("folder"), "cls": f.get("class"), "path": f.get("upgrade_path"),
             "type": f.get("type"), "year": f.get("year"),
             "weight_g": f.get("frameset_weight_g_effective")} for f in gf]


def solve_stage(theta, v_f, v_c, P, crr=CRR):
    """Return (CdA, mass_kg) from the 2x2 linear system at one stage."""
    Pw = P * (1 - DL)
    a_f, b_f = 0.5 * RHO * v_f**3, crr * G * v_f
    a_c, b_c = 0.5 * RHO * v_c**3, G * (math.sin(theta) + crr * math.cos(theta)) * v_c
    D = a_f * b_c - a_c * b_f
    cda = Pw * (b_c - b_f) / D
    m = Pw * (a_f - a_c) / D
    return cda, m


def calibrate_theta(recs):
    """Find theta so median(stage-0 solved mass - game total mass) = 0."""
    def resid(theta):
        diffs = []
        for rc in recs:
            _, m0 = solve_stage(theta, rc["z"]["flat"][0], rc["z"]["climb"][0],
                                rc["z"]["power"], rc["crr"])
            diffs.append(m0 - rc["m_game"])
        return median(diffs)
    lo, hi = math.atan(0.04), math.atan(0.14)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:   # solved mass too high -> steeper gradient lowers it
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# KNOWN ISSUE — mountain-bike weight: the ZI-solved stage-0 total mass runs
# ~1.3 kg (high-confidence mean) heavier than our game reconstruction for MTBs,
# even after using the actual test-wheel mass. The game's wheel/frame component
# XML carries NO tire mass field (verified in the raw WADs), so Zwift must add
# MTB tire mass (~0.7-0.8 kg/tyre) via a separate tyre/surface subsystem we don't
# extract. theta is calibrated on the road-bike median, which absorbs road tyre
# mass, leaving heavier MTB tyres as a residual. It is constant across the
# upgrade ladder so it cancels in the exported per-stage deltas; only the
# absolute stage-0 MTB weight is affected. Road (STANDARD) weight matches to
# within 0.11 kg std, TT/gravel within ~0.5 kg.


def _wheel_cat(wheel_name):
    """Test-wheel surface category from its name."""
    n = (wheel_name or "").lower()
    if "gravel" in n:
        return "gravel"
    if "mountain" in n or "mtb" in n:
        return "mtb"
    return "road"


def _compatible(frame_type, wheel_name):
    """Reject cross-category pairings (guard against token-bleed mismatches like
    a TT frame matching a gravel-wheel test row): gravel/MTB test wheels only
    pair with gravel/MTB frames, and road wheels never pair with a gravel/MTB
    frame."""
    wc = _wheel_cat(wheel_name)
    ft = (frame_type or "").upper()
    if wc == "mtb":
        return ft == "MOUNTAIN"
    if wc == "gravel":
        return ft == "GRAVEL"
    return ft not in ("MOUNTAIN", "GRAVEL")


def _load_match_map():
    """Hardcoded game-frame-folder -> ZwiftInsider bike name overrides. Returns
    None if the file is absent so build() falls back to the fuzzy matcher (which
    is also how the seed map is generated by tools/build_zi_match_map.py)."""
    p = os.path.join(ROOT, "zwiftdata", "frame_zi_match.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8-sig"))
    return d.get("matches", {})


def _make_rec(z, o, sc, overrides, wprops, builtins):
    crr, crr_src = resolve_crr(z.get("wheels"), overrides)
    bw = resolve_builtin_wheel(o.get("folder"), o.get("type"), builtins)
    if bw is not None:
        wheel_kg, wheel_cda, wheel_src = bw
    else:
        wheel_kg, wheel_cda, wheel_src = resolve_wheel(z.get("wheels"), o.get("type"), wprops)
    return {"z": z, "m": o, "score": sc, "crr": crr, "crr_src": crr_src,
            "wheel_kg": wheel_kg, "wheel_cda": wheel_cda, "wheel_src": wheel_src,
            "m_game": RIDER_KG + o["weight_g"] / 1000 + wheel_kg}


def _finalize(recs):
    theta = calibrate_theta(recs)
    for rc in recs:
        sol = [solve_stage(theta, rc["z"]["flat"][s], rc["z"]["climb"][s],
                           rc["z"]["power"], rc["crr"]) for s in range(6)]
        rc["cda"] = [s[0] for s in sol]
        rc["mass"] = [s[1] for s in sol]
    return recs, theta


def build(use_map=True):
    zi, ours = load_zi_full(), load_our_frames()
    overrides = _wheel_crr_overrides()
    wprops = _wheel_props()
    builtins = _builtin_wheel_props()
    mmap = _load_match_map() if use_map else None
    if mmap is not None:
        # Hardcoded mapping is authoritative: each frame takes the exact ZI test
        # row named in the map (null/missing -> frame stays estimated). Score is
        # the advisory name-similarity, kept only for audits.
        zi_by_name = {zc.canon(z["bike"]): z for z in zi}
        recs = []
        for o in ours:
            if o["weight_g"] is None:
                continue
            zi_name = mmap.get(o["folder"])
            if not zi_name:
                continue
            z = zi_by_name.get(zc.canon(zi_name))
            if z is None or not z.get("power"):
                continue
            sc = zc._pair_score(zc.toks(zi_name), o, zc.year_of(zi_name))
            recs.append(_make_rec(z, o, sc, overrides, wprops, builtins))
        return _finalize(recs)
    # Fuzzy fallback (also used to generate the seed map): enumerate every
    # compatible (ZI row, frame) candidate above threshold, then assign globally
    # 1:1 by descending score so a high-confidence match claims its frame first
    # and weaker duplicates fall through to their next-best available frame.
    cands = []
    for zi_i, z in enumerate(zi):
        if not z["power"]:
            continue
        for o, sc in zc.rank_matches(z["bike"], ours, zi_year=zc.year_of(z["bike"]), min_score=0.34):
            if o["weight_g"] is None or not _compatible(o.get("type"), z.get("wheels")):
                continue
            cands.append((sc, zi_i, o))
    cands.sort(key=lambda t: -t[0])
    used_zi, used_folder, recs = set(), set(), []
    for sc, zi_i, o in cands:
        if zi_i in used_zi or o["folder"] in used_folder:
            continue
        used_zi.add(zi_i)
        used_folder.add(o["folder"])
        recs.append(_make_rec(zi[zi_i], o, sc, overrides, wprops, builtins))
    return _finalize(recs)


def dump_measurements(recs, theta):
    """Write per-bike measured per-stage (dCdA, dWeight_g) deltas keyed by frame
    folder, for the extraction script to prefer over group averages. Keeps the
    single best-scoring ZwiftInsider match per folder."""
    bikes = {}
    for rc in recs:
        folder = rc["m"].get("folder")
        if not folder:
            continue
        prev = bikes.get(folder)
        if prev and prev["score"] >= rc["score"]:
            continue
        deltas = [[round(rc["cda"][s] - rc["cda"][s - 1], 6),
                   round((rc["mass"][s] - rc["mass"][s - 1]) * 1000.0, 1)] for s in range(1, 6)]
        bikes[folder] = {"bike": rc["z"]["bike"], "score": round(rc["score"], 3), "deltas": deltas}
    out = {"theta_pct": round(math.tan(theta) * 100, 3), "n_bikes": len(bikes), "bikes": bikes}
    path = os.path.join(ROOT, "zwiftdata", "frame_upgrade_measurements.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"measurements: {len(bikes)} bikes -> zwiftdata/frame_upgrade_measurements.json")
    return out



def report_transition(recs, s0, s1, title):
    print(f"\n=== {title} (stage {s0} -> {s1}) ===")
    print(f"{'path':9} {'tier':10} {'n':>3} {'dCdA':>10} {'(std)':>9} | {'dWeight g':>10} {'(std)':>8}")
    buckets = {}
    for rc in recs:
        dcda = rc["cda"][s1] - rc["cda"][s0]
        dw = (rc["mass"][s1] - rc["mass"][s0]) * 1000.0
        buckets.setdefault((rc["m"]["path"], rc["m"]["cls"]), []).append((dcda, dw))
    for (path, cls), vals in sorted(buckets.items()):
        cd = [v[0] for v in vals]; wt = [v[1] for v in vals]
        print(f"{str(path):9} {str(cls):10} {len(vals):3} {mean(cd):+10.5f} {pstdev(cd):9.5f} "
              f"| {mean(wt):+10.0f} {pstdev(wt):8.0f}")


if __name__ == "__main__":
    recs, theta = build()
    m0s = [rc["mass"][0] for rc in recs]
    print(f"Calibrated climb gradient: {math.tan(theta)*100:.2f}%   (n={len(recs)} full-data bikes)")
    from collections import Counter
    crr_by = Counter((rc["crr_src"], rc["crr"]) for rc in recs)
    print("Per-bike Crr used: " + ", ".join(f"{src}({crr})={n}" for (src, crr), n in sorted(crr_by.items())))
    wheel_by = Counter(rc["wheel_src"] for rc in recs)
    print(f"Test-wheel CdA source: " + ", ".join(f"{src}={n}" for src, n in sorted(wheel_by.items()))
          + f"  | wheel mass {min(rc['wheel_kg'] for rc in recs):.2f}-{max(rc['wheel_kg'] for rc in recs):.2f} kg")
    print(f"Stage-0 solved mass: median={median(m0s):.2f} kg vs game median="
          f"{median([rc['m_game'] for rc in recs]):.2f} kg")
    print(f"Stage-0 CdA by type: " + ", ".join(
        f"{t}={mean([rc['cda'][0] for rc in recs if rc['m']['type']==t]):.4f}"
        for t in sorted({rc['m']['type'] for rc in recs})))
    report_transition(recs, 2, 3, "Stage 3 = drivetrain upgrade")
    report_transition(recs, 3, 4, "Stage 4")
    report_transition(recs, 4, 5, "Stage 5")
    report_transition(recs, 0, 5, "Cumulative Stage 0->5 (fully upgraded)")
    dump_measurements(recs, theta)
