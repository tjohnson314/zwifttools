import json

import pytest

from bike_comparison.upgrades import load_upgrade_definitions, upgrade_effects
from tools import extract_zwift_upgrades as upgrades


def definition():
    return {
        "type": "bike_level_ups", "id": "distance_entry",
        "levelups": [
            {"tasks": [["distance", {"goal": 100 * index}]], "rewards": [
                {"buff": {"attr": attr, "op": operation, "scope": "equipped", "amount": amount}},
            ]}
            for index, (attr, operation, amount) in enumerate([
                ("cdabias", "absolute", .004), ("weight", "absolute", 600),
                ("resistance", "absolute", .0003), ("drops", "percentage", .05),
                ("xp", "percentage", .05),
            ], 1)
        ],
    }


def source_entries():
    return {
        upgrades.CONFIG_ENTRY: json.dumps({
            "type": "bike_levels_config", "prices": [["ENTRY", [1, 2, 3, 4, 5]]],
            "levelup_configs": ["DistanceEntry.json"], "make_group_unlocks": [],
        }).encode(),
        "bikes/levels/DistanceEntry.json": json.dumps(definition()).encode(),
    }


def test_export_preserves_raw_definitions_and_runtime_effects(monkeypatch, tmp_path):
    entries = source_entries()
    monkeypatch.setattr(upgrades, "read_wad_entries", lambda *args, **kwargs: entries)
    monkeypatch.setattr(upgrades, "read_zwift_version", lambda _: "1.126.0 (169052)")

    exported = upgrades.extract_upgrades("client", str(tmp_path))
    path = tmp_path / "game_upgrades.json"
    assert json.loads(path.read_text()) == exported
    assert exported["zwift_version"] == "1.126.0 (169052)"
    assert exported["config"] == json.loads(entries[upgrades.CONFIG_ENTRY])
    assert exported["levelup_configs"] == {"distance_entry": definition()}
    definitions = load_upgrade_definitions(path)
    assert upgrade_effects(definitions, "distance_entry", 0) == (0, 0, 0)
    assert upgrade_effects(definitions, "distance_entry", 2) == (.004, 600, 0)
    assert upgrade_effects(definitions, "distance_entry", 3) == (.004, 600, .0003)
    assert upgrade_effects(definitions, "distance_entry", 5) == (.004, 600, .0003)
    with pytest.raises(ValueError, match="Unknown bike upgrade path"):
        upgrade_effects(definitions, "missing", 3)


@pytest.mark.parametrize("failure", ["missing", "unsupported", "nonfinite", "duplicate", "malformed"])
def test_invalid_definition_preserves_existing_export(monkeypatch, tmp_path, failure):
    entries = source_entries()
    data = definition()
    if failure == "missing":
        del entries["bikes/levels/DistanceEntry.json"]
    elif failure == "duplicate":
        config = json.loads(entries[upgrades.CONFIG_ENTRY])
        config["levelup_configs"].append("DistanceEntry.json")
        entries[upgrades.CONFIG_ENTRY] = json.dumps(config).encode()
    elif failure == "malformed":
        data["levelups"][0]["rewards"] = [None]
        entries["bikes/levels/DistanceEntry.json"] = json.dumps(data).encode()
    else:
        buff = data["levelups"][0]["rewards"][0]["buff"]
        buff["attr" if failure == "unsupported" else "amount"] = (
            "unknown" if failure == "unsupported" else float("nan")
        )
        entries["bikes/levels/DistanceEntry.json"] = json.dumps(data).encode()
    monkeypatch.setattr(upgrades, "read_wad_entries", lambda *args, **kwargs: entries)
    monkeypatch.setattr(upgrades, "read_zwift_version", lambda _: "new")
    destination = tmp_path / "game_upgrades.json"
    destination.write_text("existing")

    with pytest.raises(ValueError):
        upgrades.extract_upgrades("client", str(tmp_path))

    assert destination.read_text() == "existing"


@pytest.mark.parametrize("level", [-1, 6, 2.5, True])
def test_invalid_upgrade_level_fails_explicitly(level):
    with pytest.raises(ValueError, match="integer from 0 to 5"):
        upgrade_effects({"distance_entry": definition()}, "distance_entry", level)
