"""Filter a Sauce4Zwift telemetry CSV to a subset of athlete IDs.

Usage:
    python filter_sauce_csv.py input.csv output.csv 12345 67890 ...
    python filter_sauce_csv.py input.csv output.csv --ids-file riders.txt
"""

import argparse
import csv
import sys


def main():
    parser = argparse.ArgumentParser(description="Filter a Sauce telemetry CSV to specific athletes.")
    parser.add_argument("input", help="Path to the input CSV file")
    parser.add_argument("output", help="Path to write the filtered CSV")
    parser.add_argument("athlete_ids", nargs="*", help="Athlete IDs to keep")
    parser.add_argument("--ids-file", help="Text file with one athlete ID per line")
    parser.add_argument("--list", action="store_true", help="List unique athlete IDs and row counts, then exit")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        print("Input CSV is empty.", file=sys.stderr)
        sys.exit(1)

    if "athlete_id" not in fieldnames:
        print("CSV missing 'athlete_id' column.", file=sys.stderr)
        sys.exit(1)

    if args.list:
        counts = {}
        for row in rows:
            aid = row["athlete_id"].strip()
            counts[aid] = counts.get(aid, 0) + 1
        print(f"{'athlete_id':<15} {'rows':>8}")
        print("-" * 25)
        for aid, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{aid:<15} {cnt:>8}")
        print(f"\n{len(counts)} athletes, {len(rows)} total rows")
        return

    keep = set(args.athlete_ids)
    if args.ids_file:
        with open(args.ids_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keep.add(line)

    if not keep:
        print("No athlete IDs specified. Use positional args or --ids-file.", file=sys.stderr)
        print("Tip: use --list to see available athlete IDs.", file=sys.stderr)
        sys.exit(1)

    filtered = [r for r in rows if r["athlete_id"].strip() in keep]
    found = {r["athlete_id"].strip() for r in filtered}
    missing = keep - found

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Wrote {len(filtered)} rows ({len(found)} athletes) to {args.output}")
    if missing:
        print(f"Warning: these IDs were not found: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
