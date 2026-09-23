"""WTRL team assignments joined to locally calculated ZRL scores."""

import json
from pathlib import Path


TEAMS_FILE = Path(__file__).parent.parent / "zrl_wtrl_teams.json"
SCORE_FIELDS = ("fal", "fts", "fin", "podium", "total")


def load_wtrl_teams(subgroup_id):
    """Return cached WTRL metadata and activity ID -> team assignment."""
    if not TEAMS_FILE.exists():
        return None, {}
    with TEAMS_FILE.open(encoding="utf-8") as handle:
        entry = json.load(handle).get(str(subgroup_id))
    if entry is None:
        return None, {}

    assignments = {}
    for team in entry.get("teams", []):
        for rider in team.get("riders", []):
            activity_id = str(rider.get("activity_id") or "")
            if activity_id and activity_id != "0":
                assignments[activity_id] = team["name"]
    return entry, assignments


def add_team_scores(result, subgroup_id):
    """Attach WTRL team names and aggregate only locally calculated scores."""
    source, assignments = load_wtrl_teams(subgroup_id)
    if source is None:
        result["teams"] = []
        result["team_assignment"] = {
            "available": False,
            "matched_riders": 0,
            "unmatched_riders": len(result.get("summary", [])),
        }
        return result

    teams = {}
    unmatched = []
    for rider in result.get("summary", []):
        activity_id = str(rider.get("activity_id") or rider.get("rider_id") or "")
        team_name = assignments.get(activity_id)
        rider["team"] = team_name
        if team_name is None:
            unmatched.append({
                "activity_id": activity_id,
                "name": rider.get("name", ""),
            })
            continue
        team = teams.setdefault(team_name, {
            "name": team_name,
            "rider_count": 0,
            "riders": [],
            **{field: 0 for field in SCORE_FIELDS},
        })
        team["rider_count"] += 1
        team["riders"].append({
            "activity_id": activity_id,
            "name": rider.get("name", ""),
            "total": rider.get("total", 0),
        })
        for field in SCORE_FIELDS:
            team[field] += rider.get(field, 0) or 0

    team_rows = list(teams.values())
    team_rows.sort(key=lambda team: (
        -team["total"], -team["fin"], -team["fal"], -team["fts"], team["name"]
    ))
    for rank, team in enumerate(team_rows, start=1):
        team["rank"] = rank
        team["riders"].sort(key=lambda rider: (-rider["total"], rider["name"]))

    result["teams"] = team_rows
    result["team_assignment"] = {
        "available": True,
        "source": source.get("source"),
        "season": source.get("season"),
        "class": source.get("class"),
        "race": source.get("race"),
        "matched_riders": sum(team["rider_count"] for team in team_rows),
        "unmatched_riders": len(unmatched),
        "unmatched": unmatched,
    }
    return result