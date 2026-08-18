"""Export the full per-frame per-stage audit stats (weight + CdA, ZI-solved vs our
game-extracted) to a CSV. One row per matched frame, with all six upgrade stages
(0 = un-upgraded ... 5 = fully upgraded). Reuses the shared matcher (_zi_compare)
and the 2x2 stage-0 solver (_zi_stage_solve) so numbers stay consistent with
_zi_master_audit.py."""
import os, json, csv, importlib.util, math
from statistics import mean

ROOT = r"C:\Users\timjo\Documents\Coding\Zwift\zwifttools"
RIDER_KG = 75.0


def _load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(ROOT, "tools", path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


zc = _load("zc", "_zi_compare.py")
zs = _load("zs", "_zi_stage_solve.py")

frames = {f["folder"]: f for f in json.load(
    open(os.path.join(ROOT, "zwiftdata", "game_frames.json"), encoding="utf-8-sig"))
    if "folder" in f}
recs, theta = zs.build()


def frame_stage_arrays(fr):
    """Per-stage (cda_bias, weight_g) for stages 0..5 from the game frame's base
    stats + upgrades ladder."""
    cda = [fr.get("frameset_cda_bias_effective") or 0.0]
    wt = [fr.get("frameset_weight_g_effective") or 0.0]
    by_level = {u.get("level"): u for u in (fr.get("upgrades") or [])}
    for lvl in range(1, 6):
        u = by_level.get(lvl)
        if u is not None:
            cda.append(u.get("cda_bias_effective") if u.get("cda_bias_effective") is not None else cda[-1])
            wt.append(u.get("weight_g_effective") if u.get("weight_g_effective") is not None else wt[-1])
        else:
            cda.append(cda[-1]); wt.append(wt[-1])
    return cda, wt


# ---- CdA linear-fit baseline (implied rider+bike CdA), same as audit2 --------
cpairs = []
for rc in recs:
    fr = frames.get(rc["m"]["folder"])
    if not fr:
        continue
    bias = (fr.get("frameset_cda_bias_effective") or 0.0) + rc.get("wheel_cda", 0.0)
    cpairs.append((rc, rc["cda"][0], bias))
xs = [p[2] for p in cpairs]; ys = [p[1] for p in cpairs]
mx = mean(xs); my = mean(ys)
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
a = my - b * mx

rows = []
for rc in recs:
    fr = frames.get(rc["m"]["folder"], {})
    fcda, fwt = frame_stage_arrays(fr)
    row = {
        "folder": rc["m"].get("folder"),
        "our_frame": rc["m"]["label"],
        "make": fr.get("make"),
        "zi_test_bike": rc["z"].get("bike"),
        "type": rc["m"].get("type"),
        "match_score": round(rc.get("score") or 0.0, 2),
        "wheel": rc["z"].get("wheels"),
        "power_W": rc["z"].get("power"),
    }
    for s in range(6):
        zi_mass = rc["mass"][s]
        zi_cda = rc["cda"][s]
        game_mass = RIDER_KG + fwt[s] / 1000.0 + rc["wheel_kg"]
        game_cda = a + b * (fcda[s] + rc.get("wheel_cda", 0.0))
        row[f"zi_mass_kg_s{s}"] = round(zi_mass, 3)
        row[f"game_mass_kg_s{s}"] = round(game_mass, 3)
        row[f"mass_diff_kg_s{s}"] = round(zi_mass - game_mass, 3)
        row[f"zi_cda_m2_s{s}"] = round(zi_cda, 5)
        row[f"game_cda_m2_s{s}"] = round(game_cda, 5)
        row[f"cda_diff_m2_s{s}"] = round(game_cda - zi_cda, 5)
    rows.append(row)

rows.sort(key=lambda r: (str(r["make"] or ""), str(r["our_frame"] or "")))

cols = ["folder", "our_frame", "make", "zi_test_bike", "type", "match_score",
        "wheel", "power_W"]
for s in range(6):
    cols += [f"zi_mass_kg_s{s}", f"game_mass_kg_s{s}", f"mass_diff_kg_s{s}",
             f"zi_cda_m2_s{s}", f"game_cda_m2_s{s}", f"cda_diff_m2_s{s}"]

out_path = os.path.join(ROOT, "zwiftdata", "frame_stage_audit.csv")
with open(out_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows x {len(cols)} cols to {out_path}")
print(f"theta={math.tan(theta)*100:.2f}%  cda baseline a={a:.4f}  slope b={b:.3f}")
