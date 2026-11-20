"""Packaged frame upgrade definitions and cumulative performance effects."""
from __future__ import annotations

import json
import math
from pathlib import Path


def validate_definition(definition: dict) -> None:
    if (not isinstance(definition, dict) or definition.get("type") != "bike_level_ups"
            or not isinstance(definition.get("id"), str) or not definition["id"]):
        raise ValueError("Invalid bike upgrade definition")
    levels = definition.get("levelups")
    if not isinstance(levels, list) or len(levels) != 5:
        raise ValueError(f"Expected five upgrade levels for {definition['id']}")
    operations = {
        "cdabias": "absolute", "weight": "absolute", "resistance": "absolute",
        "drops": "percentage", "xp": "percentage",
    }
    for level in levels:
        if not isinstance(level, dict):
            raise ValueError(f"Invalid upgrade level for {definition['id']}")
        rewards = level.get("rewards")
        if not isinstance(rewards, list) or not rewards:
            raise ValueError(f"Missing upgrade rewards for {definition['id']}")
        for reward in rewards:
            if not isinstance(reward, dict) or not isinstance(reward.get("buff"), dict):
                raise ValueError(f"Invalid upgrade reward for {definition['id']}")
            buff = reward["buff"]
            amount = buff.get("amount")
            if (not isinstance(buff.get("attr"), str) or buff["attr"] not in operations
                    or buff.get("op") != operations[buff["attr"]]
                    or buff.get("scope") != "equipped"
                    or isinstance(amount, bool) or not isinstance(amount, (int, float))
                    or not math.isfinite(amount)):
                raise ValueError(f"Unsupported upgrade reward for {definition['id']}: {reward}")


def load_upgrade_definitions(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    definitions = data["levelup_configs"]
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError(f"No bike upgrade definitions in {path}")
    for identifier, definition in definitions.items():
        validate_definition(definition)
        if definition["id"] != identifier:
            raise ValueError(f"Mismatched bike upgrade ID: {identifier}")
    return definitions


def upgrade_effects(definitions: dict[str, dict], level_up: str | None,
                    level: int) -> tuple[float, float, float]:
    """Cumulative aero (m^2), weight (g), and Crr reduction at a frame level."""
    if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 5:
        raise ValueError("Bike upgrade level must be an integer from 0 to 5")
    if not level_up:
        return 0.0, 0.0, 0.0
    if level_up not in definitions:
        raise ValueError(f"Unknown bike upgrade path: {level_up}")
    effects = {"cdabias": 0.0, "weight": 0.0, "resistance": 0.0}
    for stage in definitions[level_up]["levelups"][:level]:
        for reward in stage["rewards"]:
            buff = reward["buff"]
            if buff["attr"] in effects:
                effects[buff["attr"]] += buff["amount"]
    return effects["cdabias"], effects["weight"], effects["resistance"]
