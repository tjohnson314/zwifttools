import json

from tools import route_changes


def run_report(monkeypatch, tmp_path, capsys, previous, current):
    data_path = tmp_path / "index.json"
    data_path.write_text(json.dumps(current), encoding="utf-8")
    monkeypatch.setattr(route_changes, "DATA_PATH", data_path)
    monkeypatch.setattr(route_changes, "load_base", lambda base: previous)
    monkeypatch.setattr("sys.argv", ["route_changes"])
    assert route_changes.main() == 0
    return capsys.readouterr().out


def test_reports_renames_separately_from_new_routes(monkeypatch, tmp_path, capsys):
    previous = [
        {"mapID": 2, "nameHash": 123, "name": "Old name"},
        {"mapID": 2, "nameHash": 456, "name": "Unchanged", "distance_m": 100},
    ]
    current = [
        {"mapID": 2, "nameHash": 123, "name": "New name", "world": "world2"},
        {"mapID": 2, "nameHash": 456, "name": "Unchanged", "distance_m": 200},
        {"mapID": 2, "nameHash": 789, "name": "Added", "world": "world2",
         "distance_m": 1000, "leadin_distance_m": 200, "ascent_m": 10,
         "leadin_ascent_m": 2},
    ]
    output = run_report(monkeypatch, tmp_path, capsys, previous, current)
    assert "| Richmond (world2) | Old name | New name |" in output
    assert "| Richmond (world2) | Added | 1.20 | 12.00 |" in output
    assert "1 new route(s)" in output
    assert "1 renamed route(s)" in output
    assert "Unchanged" not in output


def test_rename_matching_is_world_specific_and_sorted(monkeypatch, tmp_path, capsys):
    previous = [
        {"mapID": 2, "nameHash": 123, "name": "Old Z"},
        {"mapID": 1, "nameHash": 123, "name": "Old B"},
        {"mapID": 1, "nameHash": 456, "name": "Old A"},
    ]
    current = [
        {"mapID": 2, "nameHash": 123, "name": "Z"},
        {"mapID": 1, "nameHash": 123, "name": "b"},
        {"mapID": 1, "nameHash": 456, "name": "A"},
    ]
    output = run_report(monkeypatch, tmp_path, capsys, previous, current)
    assert output.index("| Watopia (world1) | Old A | A |") < output.index("| Watopia (world1) | Old B | b |")
    assert output.index("| Watopia (world1) | Old B | b |") < output.index("| Richmond (world2) | Old Z | Z |")
    assert "0 new route(s)" in output
    assert "3 renamed route(s)" in output


def test_no_hash_keeps_name_based_identity(monkeypatch, tmp_path, capsys):
    previous = [{"mapID": 1, "name": "Old name"}]
    current = [{"mapID": 1, "name": "New name"}]
    output = run_report(monkeypatch, tmp_path, capsys, previous, current)
    assert "1 new route(s)" in output
    assert "0 renamed route(s)" in output


def test_unknown_world_keeps_raw_identifier(monkeypatch, tmp_path, capsys):
    previous = [{"mapID": 99, "nameHash": 123, "name": "Old"}]
    current = [
        {"mapID": 99, "nameHash": 123, "name": "Renamed", "world": "world99"},
        {"mapID": 99, "nameHash": 456, "name": "Added", "world": "world99"},
    ]
    output = run_report(monkeypatch, tmp_path, capsys, previous, current)
    assert "| world99 | Old | Renamed |" in output
    assert "| world99 | Added |" in output
