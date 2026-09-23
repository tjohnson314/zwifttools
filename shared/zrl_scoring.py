"""Scoring for Zwift Racing League points races and Race of Truth events."""

from collections import defaultdict


FTS_POINTS = (15, 12, 10, 8, 6, 5, 4, 3, 2, 1)
PODIUM_POINTS = (10, 8, 6, 4, 2)


def score_points_race(riders, segment_passes=()):
    """Return rider scores using the WTRL points-race scoring rules.

    ``riders`` is an iterable of mappings with ``id``, ``finished`` and optional
    ``disqualified`` fields. ``finish_order`` is the order in which finishers
    crossed the line. Each item in ``segment_passes`` has ``rider_id`` and
    ``elapsed_seconds``; items are grouped by ``segment_id`` and pass number.
    """
    riders = list(riders)
    rider_ids = {rider["id"] for rider in riders}
    starters = len(riders)
    eligible = {
        rider["id"]
        for rider in riders
        if rider.get("finished") and not rider.get("disqualified", False)
    }
    scores = {
        rider["id"]: {
            "rider_id": rider["id"],
            "fal": 0,
            "fts": 0,
            "fin": 0,
            "podium": 0,
            "total": 0,
        }
        for rider in riders
    }

    # FAL is awarded on every pass, with the starting field size as the winner's value.
    passes = defaultdict(list)
    for result in segment_passes:
        if result.get("include_fal", True) and result["rider_id"] in rider_ids:
            passes[(result["segment_id"], result.get("pass", 1))].append(result)
    for results in passes.values():
        for place, result in enumerate(
            sorted(results, key=lambda item: item["crossing_seconds"]), start=1
        ):
            rider_id = result["rider_id"]
            if rider_id in eligible:
                scores[rider_id]["fal"] += starters - place + 1

    # FTS is the ten fastest recorded times per segment per race, not per pass.
    by_segment = defaultdict(list)
    for result in segment_passes:
        if result.get("include_fts", True) and result["rider_id"] in rider_ids:
            by_segment[result["segment_id"]].append(result)
    for results in by_segment.values():
        for result, points in zip(sorted(results, key=lambda item: item["elapsed_seconds"]), FTS_POINTS):
            if result["rider_id"] in eligible:
                scores[result["rider_id"]]["fts"] += points

    finish_order = [
        rider["id"]
        for rider in sorted(
            (rider for rider in riders if rider["id"] in eligible and rider.get("finish_order") is not None),
            key=lambda rider: rider["finish_order"],
        )
    ]
    for place, rider_id in enumerate(finish_order, start=1):
        scores[rider_id]["fin"] = starters - place + 1
        if place <= len(PODIUM_POINTS):
            scores[rider_id]["podium"] = PODIUM_POINTS[place - 1]

    for score in scores.values():
        score["total"] = score["fal"] + score["fts"] + score["fin"] + score["podium"]
    return sorted(scores.values(), key=lambda score: (-score["total"], -score["fin"], -score["fal"], -score["fts"], score["rider_id"]))