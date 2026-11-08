import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.distance import cdist

from tools.analyze_route_paddock_leadins import WORLD_NAMES
from tools.extract_zwift_routes import load_multiroot, parse_paddock_roads, read_wad_entries
from tools.extract_zwift_surfaces import _cumulative, _nodes_full, _spline_points


def road_points(road):
    return np.asarray([[point["x"], point["z"]] for point in road["nodes"]], dtype=float).reshape(-1, 2)


def cluster_roads(roads, threshold_m=100):
    groups = [{road["road_id"]} for road in roads]
    points = {road["road_id"]: road_points(road) for road in roads}
    for index, first in enumerate(roads):
        first_points = points[first["road_id"]]
        for second in roads[index + 1:]:
            second_points = points[second["road_id"]]
            if not len(first_points) or not len(second_points):
                continue
            if cdist(first_points, second_points).min() > threshold_m:
                continue
            selected = [group for group in groups if first["road_id"] in group or second["road_id"] in group]
            merged = set().union(*selected)
            groups = [group for group in groups if group not in selected] + [merged]
    return sorted([sorted(group) for group in groups], key=lambda group: group[0])


def point_road_distance(position, road):
    points = road_points(road)
    if not len(points):
        return None
    if len(points) == 1:
        return float(np.linalg.norm(points[0] - position))
    starts, vectors = points[:-1], np.diff(points, axis=0)
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(np.einsum("ij,ij->i", position - starts, vectors), lengths_squared,
                          out=np.zeros(len(vectors)), where=lengths_squared != 0)
    fractions = np.clip(fractions, 0, 1)
    return float(np.linalg.norm(starts + vectors * fractions[:, None] - position, axis=1).min())


def make_groups(map_id, roads, threshold_m=100, entity_data=None, road_data=None):
    lookup = {road["road_id"]: road for road in roads}
    lengths = {road_id: float(np.linalg.norm(np.diff(road_points(road), axis=0), axis=1).sum())
               for road_id, road in lookup.items()}
    if road_data is not None:
        for road in load_multiroot(road_data).iter("road"):
            road_id = int(road.findtext("id"))
            if road_id in lookup:
                nodes = _nodes_full(ET.tostring(road, encoding="unicode"), auto_tangents=True)
                points = _spline_points(nodes, looped=(road.findtext("looped") or "").strip() == "1")
                lengths[road_id] = _cumulative(points)[-1] if len(points) >= 2 else 0.0
    gate_models = defaultdict(list)
    if entity_data:
        for entity in load_multiroot(entity_data).iter("ent"):
            if entity.get("type") == "ENTITY_TYPE_ANIMATED_PADDOCK" and entity.get("m_roadId") is not None:
                gate_models[int(entity.get("m_roadId"))].append({
                    "road_time": entity.get("m_roadTime"), "model": entity.get("m_propHiResGdePath"),
                    "material": entity.get("m_strMaterialTag"),
                })
    memberships = cluster_roads(roads, threshold_m)
    groups = []
    for membership in memberships:
        names = sorted({marker["name"] for road_id in membership
                        for marker in lookup[road_id].get("spawn_markers", []) if marker["name"]})
        candidates = [road_id for road_id in membership if len(lookup[road_id]["nodes"]) >= 2]
        representative = min(candidates, key=lambda road_id: (-lengths[road_id], road_id)) if candidates else None
        groups.append({"world": WORLD_NAMES[map_id], "map_id": map_id,
                       "group": f"{map_id}:{membership[0]}", "roads": membership,
                       "representative_road_id": representative,
                       "representative_road_length_m": lengths[representative] if representative is not None else None,
                       "spawn_names": names, "source_roads": [lookup[road_id] for road_id in membership],
                       "gated_roads": [road_id for road_id in membership if road_id in gate_models],
                       "nongated_roads": [road_id for road_id in membership if road_id not in gate_models],
                       "gate_models": {str(road_id): gate_models[road_id] for road_id in membership if road_id in gate_models}})
    return groups


def route_anchor(root):
    for tag in ("leadinhighrescheckpoint", "leadincheckpoint", "highrescheckpoint", "checkpoint"):
        section = root.find(tag)
        if section is not None:
            entry = section.find("entry")
            if entry is not None and all(entry.get(key) is not None for key in ("x", "z")):
                return tag, entry
    return None, None


def assign_route(root, map_id, groups, source):
    header = root.find("route")
    raw_ids = header.get("eventPaddocks")
    ids = [int(value.strip()) for value in (raw_ids or "").split(",") if value.strip()]
    anchor_kind, anchor = route_anchor(root)
    row = {"world": WORLD_NAMES[map_id], "map_id": map_id, "route": (header.get("name") or "").strip(),
           "name_hash": header.get("nameHash"), "source": source, "paddock_ids": ids,
           "raw_paddock_ids": raw_ids, "alternate_event_ramp": header.get("useAlternateEventRamp"),
           "sport_type": header.get("sportType"),
           "event_only": header.get("eventOnly"), "zwift_event_only": header.get("zwiftEventOnly"),
           "anchor_kind": anchor_kind, "anchor_road": int(anchor.get("road")) if anchor is not None else None,
           "anchor_x_m": float(anchor.get("x")) / 100 if anchor is not None else None,
           "anchor_z_m": float(anchor.get("z")) / 100 if anchor is not None else None,
           "nearest_group": None, "nearest_distance_m": None, "runner_up_group": None,
           "runner_up_distance_m": None, "margin_m": None, "observed_group": None,
           "observed_paddock_roads": [], "nearest_group_distances_m": {}, "issues": []}
    if anchor is None:
        return row
    point = np.asarray([row["anchor_x_m"], row["anchor_z_m"]])
    for group in groups:
        distances = [point_road_distance(point, road) for road in group["source_roads"]]
        available = [value for value in distances if value is not None]
        if available:
            row["nearest_group_distances_m"][group["group"]] = min(available)
        if anchor_kind.startswith("leadin") and row["anchor_road"] in group["roads"]:
            row["observed_group"] = group["group"]
        for tag in ("leadinhighrescheckpoint", "leadincheckpoint"):
            section = root.find(tag)
            if section is not None:
                row["observed_paddock_roads"].extend(int(entry.get("road")) for entry in section.findall("entry")
                                                      if int(entry.get("road")) in group["roads"])
    row["observed_paddock_roads"] = sorted(set(row["observed_paddock_roads"]))
    ranking = sorted(row["nearest_group_distances_m"].items(), key=lambda item: (item[1], item[0]))
    if ranking:
        row["nearest_group"], row["nearest_distance_m"] = ranking[0]
    if len(ranking) > 1:
        row["runner_up_group"], row["runner_up_distance_m"] = ranking[1]
        row["margin_m"] = ranking[1][1] - ranking[0][1]
    return row


def world_members(archive, source_cache_dir=None):
    map_id = int(archive.parent.name.removeprefix("world"))
    snapshot = source_cache_dir / f"world_{map_id}.pickle" if source_cache_dir else None
    if snapshot is not None and snapshot.exists():
        with snapshot.open("rb") as handle:
            return pickle.load(handle)
    return read_wad_entries(str(archive), keep_substrings=("/routes/", "road.xml", "entities.xml"))


def scan_worlds(zwift_dir, source_cache_dir=None, threshold_m=100):
    rows, all_groups, skipped = [], [], []
    archives = sorted((zwift_dir / "assets" / "Worlds").glob("world[0-9]*/data_1.wad"),
                      key=lambda path: int(path.parent.name.removeprefix("world")))
    if not archives:
        raise ValueError("No installed world archives found")
    for archive in archives:
        map_id = int(archive.parent.name.removeprefix("world"))
        if map_id not in WORLD_NAMES:
            continue
        members = world_members(archive, source_cache_dir)
        road_data = next(data for name, data in members.items() if name.lower().endswith("/road.xml"))
        entity_data = next((data for name, data in members.items() if name.lower().endswith("/entities.xml")), None)
        roads = parse_paddock_roads(road_data, entity_data)
        groups = make_groups(map_id, roads, threshold_m, entity_data, road_data)
        partitions = {str(threshold): cluster_roads(roads, threshold) for threshold in (50, 100, 150, 200)}
        for group in groups:
            group["threshold_partitions"] = partitions
            group["stable_membership"] = all(group["roads"] in partition for partition in partitions.values())
        for source, data in sorted(members.items()):
            if "/routes/" not in source.lower() or not source.lower().endswith(".xml"):
                continue
            root = load_multiroot(data)
            if root.find("route") is not None:
                row = assign_route(root, map_id, groups, source)
                row["source_sha256"] = hashlib.sha256(data).hexdigest()
                rows.append(row)
        all_groups.extend(groups)
        print(f"{WORLD_NAMES[map_id]}: {len(roads)} paddock roads; {len(groups)} spatial groups", flush=True)
    return rows, all_groups, skipped


def annotate_consistency(rows, groups):
    references = defaultdict(list)
    for row in rows:
        for paddock_id in set(row["paddock_ids"]):
            references[(row["map_id"], paddock_id)].append(row)
    mappings = []
    for (map_id, paddock_id), entries in sorted(references.items()):
        observed = Counter(row["observed_group"] for row in entries if row["observed_group"] is not None)
        nearest = Counter(row["nearest_group"] for row in entries if row["nearest_group"] is not None)
        group = next(iter(nearest)) if len(nearest) == 1 else None
        mappings.append({"map_id": map_id, "world": WORLD_NAMES[map_id], "paddock_id": paddock_id,
                         "group": group, "observed_groups": dict(observed), "nearest_groups": dict(nearest),
                         "route_count": len(entries), "observed_route_count": sum(observed.values()),
                         "basis": "observed paddock-road start" if observed else "nearest geometry only",
                         "inconsistent_observed": len(observed) > 1, "inconsistent_nearest": len(nearest) > 1})
    lookup = {(item["map_id"], item["paddock_id"]): item for item in mappings}
    for row in rows:
        row["issues"], row["caveats"] = [], []
        row["assigned_group"] = row["nearest_group"]
        row["assignment_basis"] = ("observed paddock-road start" if row["observed_group"] else
                                   "nearest lead-in start" if (row["anchor_kind"] or "").startswith("leadin") else
                                   "nearest main start; lead-in geometry absent" if row["anchor_kind"] else "no anchor geometry")
        row["id_group_candidates"] = sorted({lookup[(row["map_id"], paddock_id)]["group"]
                                             for paddock_id in row["paddock_ids"]
                                             if lookup[(row["map_id"], paddock_id)]["group"] is not None})
        if not row["paddock_ids"]:
            row["caveats"].append("No eventPaddocks IDs in source header; nearest physical group only")
        for paddock_id in row["paddock_ids"]:
            mapping = lookup[(row["map_id"], paddock_id)]
            if mapping["inconsistent_observed"]:
                row["issues"].append(f"ID {paddock_id} occurs at multiple observed paddock groups")
            if mapping["inconsistent_nearest"]:
                row["issues"].append(f"ID {paddock_id} has inconsistent nearest-group matches")
        if row["observed_group"] is not None and row["observed_group"] != row["nearest_group"]:
            row["issues"].append("Observed paddock road and nearest spatial group disagree")
        if len(row["id_group_candidates"]) > 1:
            row["issues"].append("Route IDs span multiple physical groups")
        if row["nearest_distance_m"] is None:
            row["caveats"].append("No group distance could be measured")
        elif row["nearest_distance_m"] > 500:
            row["caveats"].append("First available checkpoint is more than 500 m from nearest paddock group")
        if row["margin_m"] is not None and row["margin_m"] < 50:
            row["caveats"].append("Nearest and second-nearest groups differ by less than 50 m")
        if row["anchor_kind"] in ("highrescheckpoint", "checkpoint"):
            row["caveats"].append("No lead-in checkpoint; using main-route start as weaker spatial evidence")
    for group in groups:
        associated = [row for row in rows if row["assigned_group"] == group["group"]]
        observed = [row for row in rows if row["observed_group"] == group["group"]]
        group["paddock_ids"] = sorted({item["paddock_id"] for item in mappings if item["group"] == group["group"]})
        group["observed_paddock_ids"] = sorted({paddock_id for row in observed for paddock_id in row["paddock_ids"]})
        group["nearest_route_count"] = len(associated)
        group["observed_route_count"] = len(observed)
        witnesses = sorted(associated, key=lambda row: (row["observed_group"] is None, row["nearest_distance_m"], row["route"]))
        group["example_routes"] = [row["route"] for row in witnesses[:3]]
        points = np.concatenate([road_points(road) for road in group["source_roads"] if road["nodes"]])
        group["bounds_m"] = {"min_x": float(points[:, 0].min()), "max_x": float(points[:, 0].max()),
                             "min_z": float(points[:, 1].min()), "max_z": float(points[:, 1].max())}
    return mappings


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def write_reports(rows, groups, mappings, output_dir, checksums, threshold_m):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paddock_group_routes.csv", rows)
    group_rows = [{key: value for key, value in group.items() if key not in ("source_roads", "threshold_partitions", "gate_models")}
                  for group in groups]
    write_csv(output_dir / "paddock_groups.csv", group_rows)
    payload = {"routes": rows, "groups": groups, "id_mappings": mappings,
               "cache_sha256": checksums, "cluster_threshold_m": threshold_m}
    (output_dir / "paddock_group_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    inconsistent = [row for row in rows if row["issues"]]
    missing_ids = [row for row in rows if not row["paddock_ids"]]
    weak = [row for row in rows if row["assignment_basis"] != "observed paddock-road start"]
    lines = ["# All-World Paddock Group Audit", "",
             f"{len(rows)} installed route definitions across {len({row['map_id'] for row in rows})} worlds; "
             f"{len(groups)} physical clusters from {sum(len(group['roads']) for group in groups)} marked paddock roads.",
             f"{len(rows) - len(missing_ids)} routes specify paddock IDs; {len(inconsistent)} have inconsistent group matches; "
             f"{sum(row['observed_group'] is not None and bool(row['paddock_ids']) for row in rows)} have direct paddock-road-start evidence with IDs.",
             "", "## Method And Limits", "",
             f"- Groups are connected components of marked road control points within {threshold_m:g} planar metres. "
             "50/100/150/200 m sensitivity partitions are retained. Scotland's two previously verified groups are the control.",
             "- Match the first original high-resolution lead-in checkpoint to the nearest group; use coarse lead-in, "
             "high-resolution main start, then coarse main start only when earlier sources are absent.",
             "- Nearest distances are planar point-to-control-polyline distances in local metres, not road-network travel "
             "distances or independently sampled Bezier curves. Road-ID membership gives additional direct evidence.",
             "- Compare every repeated/overlapping paddock ID within its world, not just identical complete ID lists. "
             "No majority vote hides disagreements. IDs and road IDs are different namespaces; this identifies groups, not individual ID-to-road pairs.",
             "- Animated TT/BOOST paddock models corroborate gated roads. Marked roads without those models may be "
             "alternate exits or special-purpose starts; their absence does not invalidate them.",
             "- Missing lead-in geometry stays missing. Main-start fallback, >500 m anchor distances, and <50 m "
             "nearest/runner-up margins are review caveats, not proof of bad metadata. Every raw distance is retained.",
             "- Shipped running, event-only and internal definitions are included. No geometry, distances or metadata are rewritten.",
             "", "## World Summary", "", "| World | Routes | Groups | With IDs | Observed Start With IDs | Inconsistent |", "|---|---:|---:|---:|---:|---:|"]
    for map_id in sorted({row["map_id"] for row in rows}):
        world_rows = [row for row in rows if row["map_id"] == map_id]
        lines.append(f"| {WORLD_NAMES[map_id]} | {len(world_rows)} | {sum(group['map_id'] == map_id for group in groups)} | "
                     f"{sum(bool(row['paddock_ids']) for row in world_rows)} | "
                     f"{sum(row['observed_group'] is not None and bool(row['paddock_ids']) for row in world_rows)} | "
                     f"{sum(bool(row['issues']) for row in world_rows)} |")
    lines += ["", "## Physical Groups And Paddock IDs", "",
              "An ID appears below only if at least one installed route references it. Unreferenced groups do not receive guessed IDs.", "",
              "| World | Group | Paddock IDs | Marked Roads | Animated-Gate Roads | Nearest / Observed Routes | Evidence / Caveat |", "|---|---|---|---|---|---:|---|"]

    def compact(values):
        return ", ".join(map(str, values)) if values else "none"

    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    for group in groups:
        evidence = "observed start" if group["observed_paddock_ids"] else "nearest geometry only" if group["paddock_ids"] else "no route-ID references"
        if not group["stable_membership"]:
            evidence += "; threshold-sensitive membership"
        if group["spawn_names"]:
            evidence += "; spawn " + compact(group["spawn_names"])
        if group["example_routes"]:
            evidence += "; e.g. " + compact(group["example_routes"])
        lines.append(f"| {group['world']} | {group['group']} | {compact(group['paddock_ids'])} | {compact(group['roads'])} | "
                     f"{compact(group['gated_roads'])} | {group['nearest_route_count']} / {group['observed_route_count']} | {escape(evidence)} |")
    lines += ["", "## Inconsistent Routes", ""]
    if not inconsistent:
        lines.append("None. Every referenced paddock ID maps to one nearest spatial group, and all observed paddock-road starts agree with that nearest group.")
    else:
        lines += ["| World | Route | IDs | Nearest Group | Observed Group | Issues |", "|---|---|---|---|---|---|"]
        for row in inconsistent:
            lines.append(f"| {row['world']} | {escape(row['route'])} | {compact(row['paddock_ids'])} | {row['nearest_group']} | "
                         f"{row['observed_group']} | {escape('; '.join(row['issues']))} |")
    lines += ["", "## Missing ID Metadata", "", "| World | Route | Nearest Group | Observed Group |", "|---|---|---|---|"]
    for row in missing_ids:
        lines.append(f"| {row['world']} | {escape(row['route'])} | {row['nearest_group']} | {row['observed_group']} |")
    lines += ["", "## Weaker Geometry Evidence", "",
              f"{len(weak)} routes do not start on an observed paddock road; "
              f"{sum(row['anchor_kind'] in ('highrescheckpoint', 'checkpoint') for row in rows)} have no lead-in checkpoints. "
              "Their matches are geometric inference, not recovered event spawn positions.", "",
              "| World | Route | IDs | Group | Distance m | Runner-Up Margin m | Evidence / Caveat |", "|---|---|---|---|---:|---:|---|"]
    for row in rows:
        if row["assignment_basis"] == "observed paddock-road start" and not row["caveats"]:
            continue
        distance = "" if row["nearest_distance_m"] is None else f"{row['nearest_distance_m']:.3f}"
        margin = "" if row["margin_m"] is None else f"{row['margin_m']:.3f}"
        lines.append(f"| {row['world']} | {escape(row['route'])} | {compact(row['paddock_ids'])} | {row['nearest_group']} | "
                     f"{distance} | {margin} | {escape('; '.join([row['assignment_basis']] + row['caveats']))} |")
    lines += ["", "Full per-route matches: paddock_group_routes.csv. Group table: paddock_groups.csv. "
              "All identity counts, original source hashes, gates and sensitivity partitions: paddock_group_analysis.json.", ""]
    (output_dir / "paddock_group_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reports: {len(groups)} groups; {len(rows)} route matches; {len(inconsistent)} inconsistent routes; "
          f"{len(missing_ids)} missing ID headers", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Match every installed route's paddock IDs to its closest physical paddock-road group.")
    parser.add_argument("--zwift-dir", type=Path, default=Path(r"C:\Program Files (x86)\Zwift"))
    parser.add_argument("--source-cache-dir", type=Path, help="Optional trusted per-world pickle snapshots produced by a previous local WAD scan.")
    parser.add_argument("--output-dir", type=Path, default=Path("race_data"))
    parser.add_argument("--cluster-threshold-m", type=float, default=100)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "zwift_routes").glob("*.json"))
    checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    rows, groups, _ = scan_worlds(arguments.zwift_dir, arguments.source_cache_dir, arguments.cluster_threshold_m)
    mappings = annotate_consistency(rows, groups)
    if checksums != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}:
        raise RuntimeError("Route cache changed during the audit; rerun against stable inputs")
    write_reports(rows, groups, mappings, arguments.output_dir, checksums, arguments.cluster_threshold_m)


if __name__ == "__main__":
    main()