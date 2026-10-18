"""Export numeric cycling workout prescriptions from the installed Zwift client.

Usage: python tools\\extract_zwift_workouts.py [--zwift-dir PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.workouts import parse_workout
from tools.extract_zwift_routes import read_wad_entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "zwiftdata" / "workouts.json"))
    args = parser.parse_args()
    archive = Path(args.zwift_dir) / "assets" / "Workouts" / "workouts.wad"
    workouts = []
    for source, data in read_wad_entries(str(archive)).items():
        if source.lower().endswith(".xml"):
            workout = parse_workout(source, data)
            if workout is not None:
                workouts.append(workout)
    if not workouts:
        raise ValueError("No cycling workouts found in archive.")
    workouts.sort(key=lambda workout: (workout["name"].casefold(), workout["source"]))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"workouts": workouts}, separators=(",", ":"), ensure_ascii=True),
                           encoding="utf-8")
    print(f"Exported {len(workouts)} cycling workouts to {destination}")


if __name__ == "__main__":
    main()
