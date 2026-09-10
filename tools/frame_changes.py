"""Report frame weight and CdA changes in the working tree."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("zwiftdata/game_frames.json")


def load_json(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    return [record for record in records if record.get("folder")]


def load_base(base: str) -> list[dict]:
    raw = subprocess.check_output(
        ["git", "show", f"{base}:{DATA_PATH.as_posix()}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    return [record for record in json.loads(raw) if record.get("folder")]


def value(record: dict, field: str) -> float | int | None:
    return record.get(field)


def format_value(item: object, digits: int = 6) -> str:
    if item is None:
        return "---"
    return f"{item:.{digits}f}" if isinstance(item, float) else str(item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Git revision to compare against.")
    args = parser.parse_args()

    current = {record["folder"]: record for record in load_json(ROOT / DATA_PATH)}
    previous = {record["folder"]: record for record in load_base(args.base)}
    rows = []
    for folder in sorted(current):
        old = previous.get(folder)
        new = current[folder]
        fields = ("frameset_weight_g_effective", "frameset_cda_bias_effective")
        if old is not None and all(value(old, field) == value(new, field) for field in fields):
            continue
        rows.append((new.get("make") or "", new.get("name") or folder, old, new))

    print("| Frame | Previous weight (g) | New weight (g) | Previous CdA | New CdA |")
    print("|---|---:|---:|---:|---:|")
    for _, name, old, new in rows:
        print("| {} | {} | {} | {} | {} |".format(
            name,
            format_value(value(old or {}, "frameset_weight_g_effective"), 0),
            format_value(value(new, "frameset_weight_g_effective"), 0),
            format_value(value(old or {}, "frameset_cda_bias_effective")),
            format_value(value(new, "frameset_cda_bias_effective")),
        ))
    print(f"\n{len(rows)} changed frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())