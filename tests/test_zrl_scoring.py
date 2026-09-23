from shared.zrl_scoring import score_points_race


def test_points_race_scoring_removes_non_finishers_intermediate_points():
    riders = [
        {"id": "a", "finished": True, "finish_order": 1},
        {"id": "b", "finished": True, "finish_order": 2},
        {"id": "c", "finished": False, "finish_order": None},
    ]
    segment_passes = [
        {"segment_id": "sprint", "pass": 1, "rider_id": "c", "crossing_seconds": 10, "elapsed_seconds": 10},
        {"segment_id": "sprint", "pass": 1, "rider_id": "a", "crossing_seconds": 20, "elapsed_seconds": 20},
        {"segment_id": "sprint", "pass": 1, "rider_id": "b", "crossing_seconds": 30, "elapsed_seconds": 30},
    ]

    scores = score_points_race(riders, segment_passes)

    assert scores == [
        {"rider_id": "a", "fal": 2, "fts": 12, "fin": 3, "podium": 10, "total": 27},
        {"rider_id": "b", "fal": 1, "fts": 10, "fin": 2, "podium": 8, "total": 21},
        {"rider_id": "c", "fal": 0, "fts": 0, "fin": 0, "podium": 0, "total": 0},
    ]