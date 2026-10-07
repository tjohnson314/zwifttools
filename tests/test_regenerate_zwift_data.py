import os
import sys

import pytest

from tools import regenerate_zwift_data


@pytest.mark.parametrize(
    "extra_args,previous,surface_dir",
    [
        ([], "old", "zwift_surfaces"),
        (["--out-surfaces", "custom-surfaces"], "old", "custom-surfaces"),
        (["--force"], "new", "zwift_surfaces"),
    ],
)
def test_regeneration_refreshes_road_network(monkeypatch, extra_args, previous, surface_dir):
    commands = []
    monkeypatch.setattr(sys, "argv", ["regenerate_zwift_data.py", "--zwift-dir", "client", *extra_args])
    monkeypatch.setattr(regenerate_zwift_data, "read_zwift_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "last_extracted_version", lambda _: previous)
    monkeypatch.setattr(regenerate_zwift_data, "run", commands.append)

    assert regenerate_zwift_data.main() == 0
    assert [os.path.basename(command[1]) for command in commands] == [
        "extract_zwift_routes.py",
        "extract_zwift_surfaces.py",
        "extract_zwift_bikes.py",
        "zi_stage_solve.py",
        "extract_zwift_bikes.py",
    ]
    assert commands[1][2:] == ["--zwift-dir", "client", "--out", surface_dir]


def test_matching_version_skips_regeneration(monkeypatch):
    commands = []
    monkeypatch.setattr(sys, "argv", ["regenerate_zwift_data.py"])
    monkeypatch.setattr(regenerate_zwift_data, "read_zwift_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "last_extracted_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "run", commands.append)

    assert regenerate_zwift_data.main() == 0
    assert commands == []