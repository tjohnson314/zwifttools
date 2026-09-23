import json
from collections import Counter
from pathlib import Path

from shared.wtrl_teams import add_team_scores, load_wtrl_teams


WORKSPACE_ROOT = Path(__file__).parent.parent


def test_scraped_subgroup_assigns_all_nonzero_wtrl_activities():
    source, assignments = load_wtrl_teams(7354731)

    assert source["class"] == "2350A10"
    assert len(source["teams"]) == 10
    assert len(assignments) == 47
    assert assignments["2232943121044406288"] == "Racing Without Borders Apex"


def test_team_assignments_cover_every_available_catalog_race():
    with (WORKSPACE_ROOT / "zrl_event_catalog.json").open(
        encoding="utf-8-sig"
    ) as handle:
        catalog = json.load(handle)
    with (WORKSPACE_ROOT / "zrl_wtrl_teams.json").open(
        encoding="utf-8"
    ) as handle:
        team_data = json.load(handle)

    available_subgroups = {
        str(row["event_subgroup_id"])
        for row in catalog["rows"]
        if row["available"]
    }
    assert set(team_data) == available_subgroups
    assert len(team_data) == 215

    for subgroup_id, entry in team_data.items():
        activity_ids = [
            rider["activity_id"]
            for team in entry["teams"]
            for rider in team["riders"]
        ]
        assert entry["teams"], subgroup_id
        assert all(team["riders"] for team in entry["teams"]), subgroup_id
        assert all(count == 1 for count in Counter(activity_ids).values()), subgroup_id


def test_lime_b1_has_eleven_five_rider_teams():
    source, assignments = load_wtrl_teams(7354734)

    assert source["class"] == "2350B10"
    assert len(source["teams"]) == 11
    assert all(len(team["riders"]) == 5 for team in source["teams"])
    assert len(assignments) == 55
    assert assignments["2232941382010470400"] == "RTB ÆRO"


def test_team_totals_use_local_rider_scores_only():
    result = {
        "summary": [
            {
                "activity_id": "2232943121044406288",
                "name": "Njal",
                "fal": 10,
                "fts": 15,
                "fin": 20,
                "podium": 8,
                "total": 53,
            },
            {
                "activity_id": "2232939839311921184",
                "name": "Ian",
                "fal": 9,
                "fts": 12,
                "fin": 19,
                "podium": 6,
                "total": 46,
            },
        ]
    }

    add_team_scores(result, 7354731)

    assert result["teams"] == [{
        "name": "Racing Without Borders Apex",
        "rider_count": 2,
        "riders": [
            {"activity_id": "2232943121044406288", "name": "Njal", "total": 53},
            {"activity_id": "2232939839311921184", "name": "Ian", "total": 46},
        ],
        "fal": 19,
        "fts": 27,
        "fin": 39,
        "podium": 14,
        "total": 99,
        "rank": 1,
    }]
    assert result["team_assignment"]["matched_riders"] == 2
    assert result["team_assignment"]["unmatched_riders"] == 0