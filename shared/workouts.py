"""Numeric cycling workout definitions and recorded power validation."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

CATALOG_PATH = Path(__file__).resolve().parent.parent / "zwiftdata" / "workouts.json"
MAX_WORKOUT_BYTES = 1024 * 1024
MAX_WORKOUT_SEGMENTS = 10000


class _WorkoutTreeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError("Workout XML must not contain a DOCTYPE declaration.")


def _number(attributes: dict, key: str) -> float:
    value = float(attributes[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite workout attribute: {key}")
    return value


def parse_workout(source: str, data: bytes) -> dict | None:
    """Keep names and numeric prescriptions, not copyrighted coaching text."""
    root = ET.fromstring(data, parser=ET.XMLParser(target=_WorkoutTreeBuilder()))
    workout = root.find("workout")
    if root.tag != "workout_file" or workout is None or (root.findtext("sportType") or "bike").lower() not in ("bike", "ride"):
        return None
    if (root.findtext("durationType") or "time").strip().lower() != "time":
        return None
    segments = []
    warnings = set()
    elapsed = 0.0
    ftp_override = None
    if root.find("ftpOverride") is not None:
        ftp_override = _number({"ftp": root.findtext("ftpOverride")}, "ftp")
        if ftp_override <= 0:
            raise ValueError(f"{source}: invalid fixed FTP reference")
        warnings.add(f"This test uses a fixed {ftp_override:g} W reference, not your profile FTP. Intensity still scales its targets.")

    def power(attributes, fixed="Power", low="PowerLow", high="PowerHigh"):
        if fixed in attributes:
            value = _number(attributes, fixed)
            return value, value
        if low in attributes and high in attributes:
            return _number(attributes, low), _number(attributes, high)
        warnings.add("Some blocks have no explicit FTP-based target (for example zone-only targets); these are unscored.")
        return None, None

    def append(kind, duration, low, high, ramp=False):
        nonlocal elapsed
        if duration < 0:
            raise ValueError(f"{source}: negative duration")
        if duration == 0:
            return
        if len(segments) >= MAX_WORKOUT_SEGMENTS:
            raise ValueError(f"Workout exceeds the {MAX_WORKOUT_SEGMENTS:,}-block limit.")
        if not math.isfinite(elapsed + duration):
            raise ValueError("Workout total duration must be finite.")
        segments.append({
            "kind": kind, "start_sec": elapsed, "end_sec": elapsed + duration,
            "power_low": low, "power_high": high, "ramp": ramp,
        })
        elapsed += duration

    for step in workout:
        kind = step.tag.lower()
        attributes = step.attrib
        if kind in ("textevent", "restday"):
            continue
        if kind == "intervalst":
            repeat = _number(attributes, "Repeat")
            if repeat < 0 or not repeat.is_integer():
                raise ValueError(f"{source}: invalid interval repeat")
            if repeat * 2 + len(segments) > MAX_WORKOUT_SEGMENTS:
                raise ValueError(f"Workout exceeds the {MAX_WORKOUT_SEGMENTS:,}-block limit.")
            on = power(attributes, "OnPower", "PowerOnLow", "PowerOnHigh")
            off = power(attributes, "OffPower", "PowerOffLow", "PowerOffHigh")
            for _ in range(int(repeat)):
                append("Effort", _number(attributes, "OnDuration"), *on)
                append("Recovery", _number(attributes, "OffDuration"), *off)
        elif kind in ("warmup", "cooldown", "ramp", "steadystate", "solidstate"):
            low, high = power(attributes)
            append(
                {"warmup": "Warmup", "cooldown": "Cooldown", "ramp": "Ramp"}.get(kind, "Steady"),
                _number(attributes, "Duration"), low, high,
                ramp=kind in ("warmup", "cooldown", "ramp"),
            )
            if "Quantize" in attributes:
                warnings.add("Quantized ramps are shown as continuous ramps.")
        elif kind in ("freeride", "maxeffort"):
            append("Free ride" if kind == "freeride" else "Max effort",
                   _number(attributes, "Duration"), None, None)
        else:
            raise ValueError(f"{source}: unsupported step {step.tag}")
        if "ramptest" in attributes or "EndAtRoadTime" in attributes:
            warnings.add("This plan has a variable finish; align the nominal schedule manually.")
    if not segments:
        return None
    return {
        "id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
        "name": root.findtext("name") or Path(source).stem,
        "source": source,
        "duration_sec": elapsed,
        "ftp_override": ftp_override,
        "segments": segments,
        "warnings": sorted(warnings),
    }


@lru_cache(maxsize=1)
def load_workouts() -> dict[str, dict]:
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    return {workout["id"]: workout for workout in catalog["workouts"]}


def workout_index() -> list[dict]:
    return [
        {key: workout[key] for key in ("id", "name", "source", "duration_sec")}
        for workout in load_workouts().values()
    ]


def power_telemetry(data: dict) -> dict:
    """Preserve raw watts, including zeros and unusual values, and sample gaps."""
    times = data.get("timeInSec")
    powers = data.get("powerInWatts")
    if not isinstance(times, list) or not isinstance(powers, list) or not times or len(times) != len(powers):
        raise ValueError("Activity has missing or mismatched time/power samples.")
    previous = -math.inf
    for timestamp, watts in zip(times, powers):
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp <= previous):
            raise ValueError("Activity timestamps must be finite and strictly increasing.")
        if watts is not None and (isinstance(watts, bool) or not isinstance(watts, (int, float))
                                  or not math.isfinite(watts)):
            raise ValueError("Activity power samples must be finite numbers or null.")
        previous = timestamp
    return {"time_sec": times, "power_watts": powers}
