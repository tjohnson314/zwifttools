"""Seed / rebuild the hardcoded game-frame -> ZwiftInsider name mapping.

Runs the fuzzy matcher (build(use_map=False)) once to propose a folder -> ZI
bike-name mapping for every game frame, writes zwiftdata/frame_zi_match.json,
and prints the entries that should be double-checked by hand. After the seed is
written, the solver reads the JSON as authoritative; edit the JSON by hand.

Re-running this OVERWRITES the JSON (regenerates the seed) -- back up manual
edits first. Usage:
    python tools/build_zi_match_map.py
"""
import importlib.util, json, os, re

ROOT = r"C:\Users\timjo\Documents\Coding\Zwift\zwifttools"
OUT = os.path.join(ROOT, "zwiftdata", "frame_zi_match.json")

_spec = importlib.util.spec_from_file_location("zs", os.path.join(ROOT, "tools", "_zi_stage_solve.py"))
zs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(zs)
zc = zs.zc


def _primary(name, make):
    """First model-identity word: the first token that isn't part of the make
    or a 4-digit year. Folders start with the true model (GiantDefyASL -> 'defy',
    GiantPropelAdvancedSL2026 -> 'propel'), so comparing folder-primary to
    ZI-primary reliably catches same-brand wrong-model matches without being
    fooled by shared trim words like 'Advanced'/'SL'."""
    mk = zc.toks(make or "")
    for t in zc.norm(name).split():
        if t in mk or re.fullmatch(r"(?:19|20)\d{2}", t):
            continue
        return t
    return None


def main():
    recs, _theta = zs.build(use_map=False)          # fuzzy 1:1 proposal
    ours = zs.load_our_frames()
    by_folder = {rc["m"]["folder"]: rc for rc in recs}

    matches, check, estimated = {}, [], []
    for o in sorted(ours, key=lambda x: x["folder"]):
        f = o["folder"]
        rc = by_folder.get(f)
        if rc is None:
            matches[f] = None
            if o.get("weight_g") is not None:
                estimated.append((f, o.get("make")))
            continue
        zi_name = rc["z"]["bike"]
        matches[f] = zi_name
        fp, zp = _primary(f, o.get("make")), _primary(zi_name, o.get("make"))
        if fp != zp:                                   # different model within the same brand
            check.append((f, o.get("make"), zi_name, rc["score"], fp, zp))

    # detect two frames pointing at the same ZI row (shouldn't happen from fuzzy)
    seen = {}
    for f, zi_name in matches.items():
        if zi_name:
            seen.setdefault(zi_name, []).append(f)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}

    payload = {
        "//": ("Hardcoded game-frame-folder -> ZwiftInsider sheet bike name. "
               "null = no ZI test (frame stays estimated). Seed by "
               "tools/build_zi_match_map.py; hand-edit thereafter. The solver "
               "(_zi_stage_solve.build) reads 'matches' as authoritative and "
               "ignores 'review_manually'."),
        "review_manually": [f for f, *_ in check],
        "matches": matches,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    n_matched = sum(1 for v in matches.values() if v)
    print(f"wrote {OUT}")
    print(f"frames: {len(matches)}  matched: {n_matched}  estimated: {len(matches) - n_matched}")
    if dupes:
        print("\n!! DUPLICATE ZI targets (two frames share one test row - fix by hand):")
        for zi_name, fs in dupes.items():
            print(f"   {zi_name}  ->  {fs}")
    print(f"\n===== CHECK: same-brand, possibly-wrong MODEL ({len(check)}) =====")
    for f, mk, zi_name, sc, fp, zp in sorted(check):
        print(f"  [{mk}] {f}  ->  \"{zi_name}\"  (sc {sc:.2f})  model '{fp}' vs zi '{zp}'")
    print(f"\n===== ESTIMATED: no ZI test row (expected for new/novelty frames) ({len(estimated)}) =====")
    for f, mk in sorted(estimated):
        print(f"  [{mk}] {f}")


if __name__ == "__main__":
    main()
