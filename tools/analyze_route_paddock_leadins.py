import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from tools.extract_zwift_routes import load_multiroot, read_wad_entries


WORLD_NAMES = {
    1: "Watopia", 2: "Richmond", 3: "London", 4: "New York",
    5: "Innsbruck", 6: "Bologna", 7: "Yorkshire", 8: "Crit City",
    9: "Makuri Islands", 10: "France", 11: "Paris",
    12: "Gravel Mountain", 13: "Scotland",
}


def analyze_leadin(root, paddock_roads):
    header = root.find("route")
    highres = root.find("leadinhighrescheckpoint")
    coarse = root.find("leadincheckpoint")
    main = root.find("highrescheckpoint")
    checkpoints = highres.findall("entry") if highres is not None else []
    coarse_checkpoints = coarse.findall("entry") if coarse is not None else []
    main_checkpoints = main.findall("entry") if main is not None else []
    road_ids = [int(entry.get("road")) for entry in checkpoints]
    coarse_ids = [int(entry.get("road")) for entry in coarse_checkpoints]
    matched = sorted(set(road_ids + coarse_ids) & paddock_roads)
    flags = [road in paddock_roads for road in road_ids]
    first_regular = next((index for index, flag in enumerate(flags) if not flag), len(flags))
    if not matched:
        pattern = "none"
    elif not any(flags):
        pattern = "coarse_only"
    elif all(flags):
        pattern = "all_paddock"
    elif first_regular and not any(flags[first_regular:]):
        pattern = "prefix"
    else:
        pattern = "non_prefix"

    def position(entry):
        return float(entry.get("x")) / 100, float(entry.get("z")) / 100

    lengths = [math.dist(position(first), position(second))
               for first, second in zip(checkpoints, checkpoints[1:])]
    measured = sum(lengths) if checkpoints else None
    gap = (math.dist(position(checkpoints[-1]), position(main_checkpoints[0]))
           if checkpoints and main_checkpoints else None)
    full = measured + gap if measured is not None and gap is not None else None
    excluded = sum(length for length, flag in zip(lengths, flags) if flag)
    remaining = sum(length for length, flag in zip(lengths, flags) if not flag)
    if gap is not None:
        if flags[-1]:
            excluded += gap
        else:
            remaining += gap
    official_text = header.get("leadinDistanceInMeters")
    official = float(official_text) if official_text is not None else None
    before = abs(full - official) if full is not None and official is not None else None
    after = abs(remaining - official) if before is not None and any(flags) else None
    change = after - before if after is not None else None
    comparison = ("unmeasurable" if change is None else
                  "worse" if change > 1e-9 else "better" if change < -1e-9 else "same")
    use_excluded = after is not None and after < before
    best_distance = remaining if use_excluded else full
    best_error = after if use_excluded else before
    best_choice = "unmeasurable" if best_error is None else "exclude" if use_excluded else "include"
    return {
        "route": (header.get("name") or "").strip(),
        "name_hash": header.get("nameHash"),
        "event_only": header.get("eventOnly"),
        "zwift_event_only": header.get("zwiftEventOnly"),
        "event_paddocks": header.get("eventPaddocks"),
        "alternate_event_ramp": header.get("useAlternateEventRamp"),
        "paddock_roads": ";".join(map(str, matched)),
        "pattern": pattern,
        "official_m": official,
        "checkpoint_m": measured,
        "endpoint_gap_m": gap,
        "full_to_start_m": full,
        "paddock_m": excluded if checkpoints and matched else None,
        "without_paddock_m": remaining if full is not None and any(flags) else None,
        "error_full_m": before,
        "error_without_m": after,
        "error_change_m": change,
        "exclusion": comparison if matched else "not_applicable",
        "best_choice": best_choice,
        "best_distance_m": best_distance,
        "best_error_m": best_error,
        "best_delta_m": best_distance - official if best_error is not None else None,
        "highres_count": len(checkpoints),
        "coarse_count": len(coarse_checkpoints),
    }


def scan_worlds(zwift_dir, routes_dir):
    index = json.loads((routes_dir / "index.json").read_text(encoding="utf-8"))
    cached_ids = {(int(route["mapID"]), str(route["nameHash"])) for route in index}
    archives = sorted((zwift_dir / "assets" / "Worlds").glob("world[0-9]*/data_1.wad"),
                      key=lambda path: int(path.parent.name[5:]))
    if not archives:
        raise ValueError(f"No world archives found under {zwift_dir}")
    rows, skipped = [], []
    for archive in archives:
        map_id = int(archive.parent.name[5:])
        entries = read_wad_entries(str(archive), keep_substrings=("/routes/", "road.xml"))
        route_members = {member: data for member, data in entries.items() if "/routes/" in member.lower()}
        if not route_members:
            skipped.append({"map_id": map_id, "reason": "No route XML members"})
            continue
        road_members = [data for member, data in entries.items() if member.lower().endswith("/road.xml")]
        if len(road_members) != 1:
            raise ValueError(f"Expected one road.xml in {archive}; found {len(road_members)}")
        roads = load_multiroot(road_members[0])
        paddocks = {int(road.findtext("id")) for road in roads.iter("road")
                    if (road.findtext("roadIsPaddock") or "").strip().lower() in ("true", "1")}
        world_rows = []
        for member, data in sorted(route_members.items()):
            root = load_multiroot(data)
            if root.find("route") is None:
                continue
            row = analyze_leadin(root, paddocks)
            row = {
                "world": WORLD_NAMES.get(map_id, archive.parent.name),
                "map_id": map_id,
                "in_cached_index": (map_id, row["name_hash"]) in cached_ids,
                **row,
                "source": member,
            }
            world_rows.append(row)
        rows.extend(world_rows)
        print(f"{WORLD_NAMES.get(map_id, archive.parent.name)}: {len(world_rows)} routes, "
              f"{sum(bool(row['paddock_roads']) for row in world_rows)} with paddock lead-ins", flush=True)
    return sorted(rows, key=lambda row: (row["map_id"], row["route"].casefold())), skipped


def write_reports(rows, skipped, output_dir, error_threshold_m=10.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paddock_leadin_analysis.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value
                             for key, value in row.items()})
    affected = [row for row in rows if row["paddock_roads"]]
    comparisons = Counter(row["exclusion"] for row in affected)
    lines = [
        "# Paddock Lead-in Analysis", "",
        f"Scanned {len(rows)} installed route definitions across {len({row['map_id'] for row in rows})} worlds.",
        f"{len(affected)} include marked paddock roads in lead-in checkpoints; "
        f"{sum(row['in_cached_index'] for row in affected)} are in the extracted route index.",
        "",
        "## Method", "",
        "Distances use original high-resolution x/z checkpoints in centimetres, converted to planar metres. "
        "Full approach includes the straight gap from the last lead-in checkpoint to the first main-route checkpoint. "
        "For exclusion, each original interval is attributed to its starting checkpoint's road; "
        "a paddock-to-normal transition is counted with the paddock. "
        "No new connections across omitted sections, geometry changes, or distance scaling are applied.",
        "",
        "Better/worse means smaller/larger absolute error against source leadinDistanceInMeters, "
        "not verified in-game distance or public route availability. "
        "Floating-point ties within 1e-9 m are classified as same. "
        "Missing checkpoint lengths are blank, not zero. "
        "The CSV includes all scanned routes, raw event restrictions, and six-decimal metre values.",
        "",
        "Closest-choice fields select the smaller absolute header error for each route, "
        "retaining the full approach on exact ties. Routes without paddock checkpoints retain their full approach. "
        "This is metadata agreement, not independent proof of the correct in-game start position. "
        "best_delta_m is selected distance minus published distance; negative means shorter.",
        "",
        "## World Summary", "",
        "| World | Scanned | Paddock Lead-ins | Better Excluding | Worse Excluding | Same | Unmeasurable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for map_id in sorted({row["map_id"] for row in rows}):
        world_rows = [row for row in rows if row["map_id"] == map_id]
        world_affected = [row for row in world_rows if row["paddock_roads"]]
        counts = Counter(row["exclusion"] for row in world_affected)
        lines.append(f"| {world_rows[0]['world']} | {len(world_rows)} | {len(world_affected)} | "
                     f"{counts['better']} | {counts['worse']} | {counts['same']} | {counts['unmeasurable']} |")
    measured = [row for row in rows if row["best_error_m"] is not None]
    unresolved = sorted((row for row in measured if row["best_error_m"] > error_threshold_m),
                        key=lambda row: row["best_error_m"], reverse=True)
    lines.extend([
        "", f"Totals: {dict(comparisons)}.", "", "## Closest-Choice Residuals", "",
        f"{len(measured)} routes have measurable lead-ins; {len(rows) - len(measured)} cannot be evaluated. "
        f"After selecting the closer distance, {len(unresolved)} remain more than {error_threshold_m:g} m "
        "from the published figure. All raw errors remain in the CSV; the threshold only filters this table.",
        "", "| Error Greater Than | All Routes | With Paddock Roads |", "|---|---:|---:|",
    ])
    for threshold in (1, 5, 10, 25, 50, 100, 250, 500):
        above = [row for row in measured if row["best_error_m"] > threshold]
        lines.append(f"| {threshold} m | {len(above)} | {sum(bool(row['paddock_roads']) for row in above)} |")
    lines.extend([
        "", f"### Routes Still More Than {error_threshold_m:g} m From Published", "",
        "| World | Route | Paddock Roads | Published m | Closest m | Difference m | Error m | Choice | Event Only | Zwift Event Only |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ])
    for row in unresolved:
        values = [row["world"], row["route"], row["paddock_roads"]]
        values.extend(f"{row[key]:.3f}" for key in
                      ("official_m", "best_distance_m", "best_delta_m", "best_error_m"))
        values.extend([row["best_choice"], row["event_only"] or "", row["zwift_event_only"] or ""])
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    lines.extend(["", "## Coverage Notes", ""])
    for item in skipped:
        lines.append(f"- world{item['map_id']}: {item['reason']}.")
    missing = [row for row in rows if row["highres_count"] == 0 and row["coarse_count"] == 0]
    lines.extend([
        f"- {len(missing)} definitions have no lead-in checkpoints; their metadata cannot be reconciled this way.",
        "- Shipped event-restricted definitions are included; this scan does not certify public availability.",
        "", "## Routes With Paddock Lead-ins", "",
        "| World | Route | Paddock Roads | Published m | Full Approach m | Paddock m | Excluding Paddock m | Effect |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in affected:
        values = [row["world"], row["route"], row["paddock_roads"]]
        values.extend("" if row[key] is None else f"{row[key]:.3f}"
                      for key in ("official_m", "full_to_start_m", "paddock_m", "without_paddock_m"))
        values.append(row["exclusion"])
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    report_path = output_dir / "paddock_leadin_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected = [row for row in rows if row["pattern"] == "prefix"
                and row["error_full_m"] is not None and row["error_without_m"] is not None
                and row["error_full_m"] > 50 and row["error_without_m"] < 10]
    adjustment_lines = [
        "# Selective Paddock Lead-in Adjustments", "",
        f"{len(selected)} routes qualify for the display-only paddock exclusion. "
        "All other routes retain their current view.", "",
        "Only a leading prefix of roads explicitly marked roadIsPaddock is omitted, "
        "and only when the absolute discrepancy drops from strictly over 50 m to strictly under 10 m. "
        "Discrepancies compare original high-resolution planar checkpoints, including the final gap "
        "to the main-route start, against the unrounded published XML lead-in distance. "
        "The paddock-to-normal transition belongs to the paddock. "
        "Published distances, raw geometry, and main-route geometry remain unchanged; "
        "the remaining display profile is rebased to zero without distance scaling.", "",
        "Shipped event-only definitions are included; inclusion does not establish public availability. "
        "All figures below are metres; discrepancies are absolute, not signed.", "",
        "| World | Route | Name Hash | Published | Full Approach | Without Paddock | Discrepancy Before | Discrepancy After |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        values = [row["world"], row["route"], row["name_hash"]]
        values.extend(f"{row[key]:.3f}" for key in (
            "official_m", "full_to_start_m", "without_paddock_m", "error_full_m", "error_without_m"))
        adjustment_lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    adjustments_path = output_dir / "paddock_leadin_adjustments.md"
    adjustments_path.write_text("\n".join(adjustment_lines) + "\n", encoding="utf-8")
    return csv_path, report_path, adjustments_path


def main():
    parser = argparse.ArgumentParser(description="Compare authored lead-ins with and without marked paddock roads.")
    parser.add_argument("--zwift-dir", type=Path, default=Path(r"C:\Program Files (x86)\Zwift"))
    parser.add_argument("--routes-dir", type=Path, default=Path("zwift_routes"))
    parser.add_argument("--output-dir", type=Path, default=Path("race_data"))
    parser.add_argument("--error-threshold-m", type=float, default=10.0,
                        help="List closest-choice residuals greater than this many metres (default: 10).")
    args = parser.parse_args()
    if not math.isfinite(args.error_threshold_m) or args.error_threshold_m < 0:
        parser.error("--error-threshold-m must be finite and nonnegative")
    rows, skipped = scan_worlds(args.zwift_dir, args.routes_dir)
    if not rows:
        raise ValueError("No route definitions were found")
    for path in write_reports(rows, skipped, args.output_dir, args.error_threshold_m):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())