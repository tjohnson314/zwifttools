"""Check the installed Zwift client version and regenerate routes/frames/wheels
data if it's newer than the version last extracted.

Compares the client's ``sversion`` (see ``extract_zwift_bikes.read_zwift_version``)
against the ``zwift_version`` stamped in ``zwiftdata/game_frames.json``. If they
differ (or ``--force`` is passed), reruns the full pipeline in order:

    1. extract_zwift_routes.py   (zwift_routes/)
    2. extract_zwift_bikes.py    (game_frames.json / game_wheels.json, pass 1)
    3. zi_stage_solve.py         (frame_upgrade_measurements.json)
    4. extract_zwift_bikes.py    (pass 2, bakes in the fresh measurements)

Usage:
    python tools/regenerate_zwift_data.py \
        [--zwift-dir "C:\\Program Files (x86)\\Zwift"] \
        [--out-routes zwift_routes] [--out-data zwiftdata] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from extract_zwift_bikes import read_zwift_version  # noqa: E402


def last_extracted_version(out_data: str) -> str | None:
    """``zwift_version`` stamped in the meta record at the top of
    game_frames.json, or None if the file/field is missing."""
    path = os.path.join(out_data, "game_frames.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data and isinstance(data[0], dict) and "folder" not in data[0]:
        return data[0].get("zwift_version")
    return None


def run(args: list[str]) -> None:
    print(f"$ {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zwift-dir", default=r"C:\Program Files (x86)\Zwift",
                    help="Zwift install directory (contains assets/).")
    ap.add_argument("--out-routes", default="zwift_routes",
                    help="Output directory for route profiles.")
    ap.add_argument("--out-data", default="zwiftdata",
                    help="Output directory for frame/wheel datasets.")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the installed version matches.")
    args = ap.parse_args()

    installed = read_zwift_version(args.zwift_dir)
    if installed is None:
        print("error: could not read installed Zwift version "
              f"from {args.zwift_dir}", file=sys.stderr)
        return 2

    previous = last_extracted_version(args.out_data)
    print(f"installed version: {installed}")
    print(f"last extracted:    {previous or '(none)'}")

    if installed == previous and not args.force:
        print("up to date -- no regeneration needed.")
        return 0

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    run([py, os.path.join(tools_dir, "extract_zwift_routes.py"),
         "--zwift-dir", args.zwift_dir, "--out", args.out_routes])
    run([py, os.path.join(tools_dir, "extract_zwift_bikes.py"),
         "--zwift-dir", args.zwift_dir, "--out", args.out_data])
    run([py, os.path.join(tools_dir, "zi_stage_solve.py")])
    run([py, os.path.join(tools_dir, "extract_zwift_bikes.py"),
         "--zwift-dir", args.zwift_dir, "--out", args.out_data])

    print(f"regenerated for version {installed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
