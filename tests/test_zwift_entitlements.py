import json

import pytest

from tools import extract_zwift_entitlements as entitlements


def record(identifier, paths=None):
    return {
        "entitlementId": identifier,
        "shortName": f"ENTITLEMENT_{identifier}",
        "category": "CYCLING_JERSEY",
        "associatedPaths": ["Humans\\Jerseys\\kit.tgax"] if paths is None else paths,
        "unlockData": {"garageName": "LOC_KIT_NAME"},
    }


def registry_bytes(records):
    return json.dumps({
        "type": "entitlements", "version": 2, "id": "main", "lastUpdated": 1234,
        "entitlements": records,
    }).encode()


def mock_archives(monkeypatch, records, spinner=None):
    registry = {entitlements.REGISTRY_ENTRY: registry_bytes(records)}
    spinners = {entitlements.SPINNER_ENTRY: spinner or b"""
        <SpinnerDefinitions>
          <Prize entitlement="2"/><Prize entitlement="9"/><Prize entitlement="9"/>
          <Prize entitlement="ENTITLEMENT_NONE"/><Prize drops="100"/>
        </SpinnerDefinitions>
    """}

    def read(path, keep_substrings):
        return registry if path.endswith("globalproj.wad") else spinners

    monkeypatch.setattr(entitlements, "read_wad_entries", read)
    monkeypatch.setattr(entitlements, "read_zwift_version", lambda _: "1.126.0 (169052)")
    return registry, spinners


def test_export_maps_ids_preserves_raw_records_and_reports_unresolved_ids(monkeypatch, tmp_path, capsys):
    raw = [record(10), record(2, ["Humans/kit.xml", None])]
    raw[0]["type"] = "JERSEY"
    mock_archives(monkeypatch, raw)

    exported = entitlements.extract_entitlements("client", str(tmp_path))

    assert json.loads((tmp_path / "game_entitlements.json").read_text()) == exported
    assert exported["zwift_version"] == "1.126.0 (169052)"
    assert exported["coverage"] == "packaged_registry_only"
    assert list(exported["entitlements"]) == ["2", "10"]
    assert exported["entitlements"] == {"2": raw[1], "10": raw[0]}
    assert exported["unresolved_entitlement_ids"] == [9]
    assert exported["invalid_associated_path_ids"] == [2]
    assert exported["registry_metadata"]["lastUpdated"] == 1234
    assert "invalid associated paths retained: 2" in capsys.readouterr().err


@pytest.mark.parametrize("records,error", [
    ([record(2), record(2)], "Duplicate entitlement ID"),
    ([record(True)], "Invalid entitlement ID"),
    ([record(-1)], "Invalid entitlement ID"),
    ([], "no records"),
])
def test_invalid_registry_does_not_overwrite_export(monkeypatch, tmp_path, records, error):
    destination = tmp_path / "game_entitlements.json"
    destination.write_text("existing")
    mock_archives(monkeypatch, records)

    with pytest.raises(ValueError, match=error):
        entitlements.extract_entitlements("client", str(tmp_path))

    assert destination.read_text() == "existing"


@pytest.mark.parametrize("missing", [entitlements.REGISTRY_ENTRY, entitlements.SPINNER_ENTRY])
def test_missing_source_is_an_error(monkeypatch, tmp_path, missing):
    registry, spinners = mock_archives(monkeypatch, [record(2)])
    registry.pop(missing, None)
    spinners.pop(missing, None)

    with pytest.raises(ValueError, match="Missing"):
        entitlements.extract_entitlements("client", str(tmp_path))

    assert not (tmp_path / "game_entitlements.json").exists()


def test_missing_version_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(entitlements, "read_zwift_version", lambda _: None)

    with pytest.raises(ValueError, match="Could not read installed Zwift version"):
        entitlements.extract_entitlements("client", str(tmp_path))

    assert not (tmp_path / "game_entitlements.json").exists()


def test_unknown_symbolic_spinner_reference_is_not_silently_discarded(monkeypatch, tmp_path):
    mock_archives(monkeypatch, [record(2)], b"""
        <SpinnerDefinitions><Prize entitlement="ENTITLEMENT_UNKNOWN"/></SpinnerDefinitions>
    """)

    with pytest.raises(ValueError, match="Unsupported spinner entitlement reference"):
        entitlements.extract_entitlements("client", str(tmp_path))

    assert not (tmp_path / "game_entitlements.json").exists()
