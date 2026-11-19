import json

import pandas as pd
import pytest

from tools import analyze_racing_predictions as analysis


def group():
    return pd.DataFrame({
        "profileId": [1, 2, 3], "eventId": [100] * 3, "eventSubgroupId": [1000] * 3,
        "penLabel": ["A"] * 3, "rank": [1, 3, 5], "position": [1, 2, 3],
        "previousScore": [900, 600, 300], "ratingBefore": [300, 600, 900],
        "expectedPosition": [1, 2, 3], "veloBefore": [900, 600, 300],
    })


def test_perfect_and_reversed_scores_are_measured_within_subset():
    metrics = {row["model"]: row for row in analysis.evaluate_group(group())}
    assert metrics["ZRS"]["pairCredit"] == 3
    assert metrics["ZRS"]["spearman"] == pytest.approx(1)
    assert metrics["ZRS"]["absoluteErrorSum"] == 0
    assert metrics["ZRS"]["winnerCredit"] == 1
    assert metrics["ZwiftRacing ratingBefore"]["pairCredit"] == 0
    assert metrics["ZwiftRacing ratingBefore"]["spearman"] == pytest.approx(-1)
    assert metrics["ZwiftRacing ratingBefore"]["absoluteErrorSum"] == 4
    assert metrics["ZwiftRacing expectedPosition"]["absoluteErrorSum"] == 0


def test_predictor_ties_receive_half_pair_credit_and_split_winner_credit():
    frame = group()
    frame["previousScore"] = 500
    metrics = analysis.evaluate_group(frame)[0]
    assert metrics["pairCredit"] == 1.5
    assert metrics["spearman"] is None
    assert metrics["winnerCredit"] == pytest.approx(1 / 3)


def test_actual_ties_excluded_from_pairs():
    frame = group()
    frame["rank"] = [1, 1, 3]
    metrics = analysis.evaluate_group(frame)[0]
    assert metrics["pairs"] == 2
    assert metrics["pairCredit"] == 2


def test_missing_actual_winner_not_replaced_with_best_available_finisher():
    frame = group()
    frame["rank"] += 1
    assert analysis.evaluate_group(frame)[0]["winnerCredit"] is None


def test_raw_scores_are_not_clipped_and_after_scores_are_not_used():
    frame = group()
    frame["previousScore"] = [12000, 0, -100]
    frame["newScore"] = [-10, 0, 900]
    assert analysis.evaluate_group(frame)[0]["pairCredit"] == 3


def test_full_analysis_uses_matching_finishers_and_serializes_report(tmp_path, capsys):
    frame = group()
    zrs = frame[analysis.KEYS + ["previousScore", "rank"]].copy()
    velo = frame[analysis.KEYS + [
        "ratingBefore", "position", "expectedPosition",
    ]].copy()
    zrs["newScore"] = 123
    velo["ratingAfter"] = 456
    metadata = {
        "eventId": 100,
        "pens": [{"name": "A", "riders": [
            {"riderId": int(row.profileId), "velo": {"elo": {"before": row.veloBefore}}}
            for row in frame.itertuples()
        ]}],
    }
    (tmp_path / "100.json").write_text(json.dumps(metadata), encoding="utf-8")
    report, stats, cohort = analysis.analyze(zrs, velo, tmp_path)
    assert report["coverage"]["analyzedResults"] == 3
    assert report["matchedComparison"]["ZRS"]["pairAccuracy"] == 1
    assert report["matchedComparison"]["ZwiftRacing ratingBefore"]["pairAccuracy"] == 0
    assert len(stats) == 4
    assert len(cohort) == 3
    json.dumps(report, allow_nan=False)
    zrs_path, velo_path = tmp_path / "zrs.csv", tmp_path / "velo.csv"
    zrs.to_csv(zrs_path, index=False)
    velo.to_csv(velo_path, index=False)
    output = tmp_path / "analysis"
    assert analysis.main([
        "--zrs", str(zrs_path), "--velo", str(velo_path),
        "--cache", str(tmp_path), "--output-dir", str(output),
    ]) == 0
    captured = capsys.readouterr()
    log = (output / "analysis.log").read_text(encoding="utf-8")
    assert captured.out in log
    assert captured.err.splitlines()[-1] in log
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    assert (output / "subgroup_metrics.csv").exists()
    assert (output / "matched_predictions.csv").exists()


def test_summary_unobserved_winner_and_constant_predictor_serialize_as_null():
    frame = group()
    frame["rank"] += 1
    frame["previousScore"] = 500
    summary = analysis.summarize(pd.DataFrame(analysis.evaluate_group(frame)))
    assert summary["ZRS"]["winnerPickAccuracy"] is None
    assert summary["ZRS"]["meanSubgroupSpearman"] is None
    json.dumps(summary, allow_nan=False)
