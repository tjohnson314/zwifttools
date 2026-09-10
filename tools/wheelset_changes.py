"""Report wheelset weight and CdA changes in the working tree."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path("zwiftdata/game_wheels.json")


def load_json(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    return [record for record in records if record.get("model")]


def load_base(base: str) -> list[dict]:
    raw = subprocess.check_output(
        ["git", "show", f"{base}:{DATA_PATH.as_posix()}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    return [record for record in json.loads(raw) if record.get("model")]


def key(record: dict) -> tuple[str, str]:
    return record.get("brand") or "", record["model"]


def format_value(item: object, digits: int = 6) -> str:
    if item is None:
        return "---"
    return f"{item:.{digits}f}" if isinstance(item, float) else str(item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Git revision to compare against.")
    args = parser.parse_args()

    current = {key(record): record for record in load_json(ROOT / DATA_PATH)}
    previous = {key(record): record for record in load_base(args.base)}
    rows = []
    fields = ("pair_weight_g_effective", "pair_cda_bias_effective")
    for wheel_key in sorted(current):
        old = previous.get(wheel_key)
        new = current[wheel_key]
        if old is not None and all(old.get(field) == new.get(field) for field in fields):
            continue
        rows.append((wheel_key, old, new))

    print("| Wheelset | Previous weight (g) | New weight (g) | Previous CdA | New CdA |")
    print("|---|---:|---:|---:|---:|")
    for (brand, model), old, new in rows:
        print("| {} | {} | {} | {} | {} |".format(
            new.get("name") or f"{brand} {model}".strip(),
            format_value((old or {}).get("pair_weight_g_effective"), 0),
            format_value(new.get("pair_weight_g_effective"), 0),
            format_value((old or {}).get("pair_cda_bias_effective")),
            format_value(new.get("pair_cda_bias_effective")),
        ))
    print(f"\n{len(rows)} changed wheelset(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())