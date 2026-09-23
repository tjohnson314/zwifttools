import json
from pathlib import Path

from app import app


CATALOG_FILE = Path(__file__).parent.parent / "zrl_event_catalog.json"


def test_zrl_catalog_is_complete_and_subgroups_are_unique():
    with CATALOG_FILE.open(encoding="utf-8-sig") as handle:
        catalog = json.load(handle)
    rows = catalog["rows"]
    available = [row for row in rows if row["available"]]
    subgroup_ids = [row["event_subgroup_id"] for row in available]
    replay_activity_ids = [row["replay_activity_id"] for row in available]

    assert catalog["row_count"] == len(rows) == 221
    assert catalog["available_count"] == len(available) == 215
    assert catalog["unavailable_count"] == 6
    assert len(subgroup_ids) == len(set(subgroup_ids))
    assert len(replay_activity_ids) == len(set(replay_activity_ids)) == 215
    assert all(activity_id.isdigit() for activity_id in replay_activity_ids)
    assert all(
        row["replay_activity_id"] is None
        for row in rows
        if not row["available"]
    )
    assert next(
        (row["event_subgroup_id"], row["replay_activity_id"])
        for row in rows
        if row["wtrl_class"] == "2350A10" and row["race"] == "1"
    ) == (7354731, "2232938456282087472")


def test_zrl_catalog_endpoint_is_public():
    response = app.test_client().get("/api/zrl/catalog")

    assert response.status_code == 200
    assert response.get_json()["row_count"] == 221