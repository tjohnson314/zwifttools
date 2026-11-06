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
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
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


@pytest.mark.parametrize("explicit_cache", [False, True])
@pytest.mark.parametrize("extra_args,previous,route_dir,surface_dir", [
    ([], "old", "zwift_routes", "zwift_surfaces"),
    (["--out-routes", "custom-routes", "--out-surfaces", "custom-surfaces"],
     "old", "custom-routes", "custom-surfaces"),
    (["--force"], "new", "zwift_routes", "zwift_surfaces"),
])
def test_regeneration_refreshes_video_routes(monkeypatch, tmp_path, explicit_cache,
                                            extra_args, previous, route_dir, surface_dir):
    commands = []
    local_app_data = tmp_path / "local app data"
    cache = tmp_path / "custom cache" if explicit_cache else local_app_data / "Zwift" / "mochi"
    cache.mkdir(parents=True)
    if explicit_cache:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
    else:
        monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    cache_args = ["--video-cache", str(cache)] if explicit_cache else []
    monkeypatch.setattr(sys, "argv", [
        "regenerate_zwift_data.py", "--zwift-dir", "client", *extra_args, *cache_args,
    ])
    monkeypatch.setattr(regenerate_zwift_data, "read_zwift_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "last_extracted_version", lambda _: previous)
    monkeypatch.setattr(regenerate_zwift_data, "run", commands.append)

    assert regenerate_zwift_data.main() == 0
    assert [os.path.basename(command[1]) for command in commands] == [
        "extract_zwift_routes.py", "extract_zwift_surfaces.py", "extract_zwift_routes.py",
        "extract_zwift_bikes.py", "zi_stage_solve.py", "extract_zwift_bikes.py",
    ]
    assert commands[0][2:] == ["--zwift-dir", "client", "--out", route_dir]
    assert commands[1][2:] == ["--zwift-dir", "client", "--out", surface_dir]
    assert commands[2][2:] == [
        "--video-cache", str(cache), "--out", route_dir, "--out-surfaces", surface_dir,
    ]


@pytest.mark.parametrize("local_app_data", [None, "missing-local-data"])
def test_missing_default_video_cache_keeps_existing_exports(monkeypatch, tmp_path, capsys, local_app_data):
    commands = []
    if local_app_data is None:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
    else:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / local_app_data))
    monkeypatch.setattr(sys, "argv", ["regenerate_zwift_data.py"])
    monkeypatch.setattr(regenerate_zwift_data, "read_zwift_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "last_extracted_version", lambda _: "old")
    monkeypatch.setattr(regenerate_zwift_data, "run", commands.append)

    assert regenerate_zwift_data.main() == 0
    assert len(commands) == 5
    assert all("--video-cache" not in command for command in commands)
    assert "existing video exports left unchanged" in capsys.readouterr().err


def test_missing_explicit_video_cache_fails_before_extraction(monkeypatch, tmp_path, capsys):
    commands = []
    monkeypatch.setattr(sys, "argv", [
        "regenerate_zwift_data.py", "--video-cache", str(tmp_path / "missing cache"),
    ])
    monkeypatch.setattr(regenerate_zwift_data, "read_zwift_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "last_extracted_version", lambda _: "old")
    monkeypatch.setattr(regenerate_zwift_data, "run", commands.append)

    with pytest.raises(SystemExit) as error:
        regenerate_zwift_data.main()

    assert error.value.code == 2
    assert commands == []
    assert "video cache directory does not exist" in capsys.readouterr().err


def test_matching_version_skips_regeneration(monkeypatch):
    commands = []
    monkeypatch.setattr(sys, "argv", ["regenerate_zwift_data.py"])
    monkeypatch.setattr(regenerate_zwift_data, "read_zwift_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "last_extracted_version", lambda _: "new")
    monkeypatch.setattr(regenerate_zwift_data, "run", commands.append)

    assert regenerate_zwift_data.main() == 0
    assert commands == []