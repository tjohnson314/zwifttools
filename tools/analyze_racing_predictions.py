"""Compare pre-race rating predictions within matched Tour of Watopia subgroups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.console_log import run_logged  # noqa: E402

KEYS = ["profileId", "eventId", "eventSubgroupId"]
GROUPS = ["eventId", "eventSubgroupId"]
MODELS = {
    "ZRS": ("previousScore", False),
    "ZwiftRacing ratingBefore": ("ratingBefore", False),
    "ZwiftRacing expectedPosition": ("expectedPosition", True),
    "vElo elo.before": ("veloBefore", False),
}


def load_metadata(frame: pd.DataFrame, cache: Path) -> pd.DataFrame:
    rows = []
    for event_id in sorted(frame.eventId.unique()):
        path = cache / f"{event_id}.json"
        detail = json.loads(path.read_text(encoding="utf-8"))
        if int(detail["eventId"]) != event_id:
            raise ValueError(f"Event cache ID mismatch: {path}")
        for pen in detail["pens"]:
            for rider in pen["riders"]:
                elo = (rider.get("velo") or {}).get("elo") or {}
                rows.append({
                    "profileId": rider["riderId"], "eventId": event_id,
                    "didNotFinish": rider.get("didNotFinish") is True,
                    "veloBefore": elo.get("before"),
                    "penLabel": pen.get("name"),
                })
    metadata = pd.DataFrame(rows)
    if metadata.duplicated(["eventId", "profileId"]).any():
        raise ValueError("A rider appears in multiple pens; metadata join would be ambiguous.")
    result = frame.merge(
        metadata, on=["eventId", "profileId"], how="left", validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("Some exported riders have no cached DNF metadata.")
    return result.drop(columns="_merge")


def evaluate_group(group: pd.DataFrame, actual: str = "rank") -> list[dict]:
    n = len(group)
    outcome = group[actual].to_numpy(dtype=float)
    actual_rank = group[actual].rank(method="average").to_numpy()
    left, right = np.triu_indices(n, 1)
    outcome_difference = np.sign(outcome[left] - outcome[right])
    comparable = outcome_difference != 0
    records = []
    for name, (column, ascending) in MODELS.items():
        prediction = group[column].rank(method="average", ascending=ascending).to_numpy()
        prediction_difference = np.sign(prediction[left] - prediction[right])
        correct = prediction_difference == outcome_difference
        credit = correct.astype(float)
        credit[prediction_difference == 0] = 0.5
        error = np.abs(prediction - actual_rank)
        centered_actual = actual_rank - actual_rank.mean()
        centered_prediction = prediction - prediction.mean()
        denominator = np.linalg.norm(centered_actual) * np.linalg.norm(centered_prediction)
        correlation = (
            float(np.dot(centered_actual, centered_prediction) / denominator)
            if denominator else None
        )
        top = prediction == prediction.min()
        real_winner = outcome == 1
        winner_credit = (
            float(np.sum(top & real_winner) / np.sum(top)) if real_winner.any() else None
        )
        records.append({
            "eventId": int(group.eventId.iloc[0]),
            "eventSubgroupId": int(group.eventSubgroupId.iloc[0]),
            "penLabel": group.penLabel.iloc[0],
            "model": name, "riders": n,
            "pairs": int(comparable.sum()),
            "pairCredit": float(credit[comparable].sum()),
            "spearman": correlation,
            "absoluteErrorSum": float(error.sum()),
            "normalizedErrorSum": float((error / (n - 1)).sum()),
            "winnerCredit": winner_credit,
        })
    return records


def summarize(stats: pd.DataFrame) -> dict:
    def mean_or_none(values: pd.Series) -> float | None:
        value = values.mean()
        return float(value) if pd.notna(value) else None

    summaries = {}
    for name, rows in stats.groupby("model", sort=False):
        pairs = int(rows.pairs.sum())
        summaries[name] = {
            "subgroups": len(rows),
            "riderResults": int(rows.riders.sum()),
            "pairs": pairs,
            "pairAccuracy": float(rows.pairCredit.sum() / pairs) if pairs else None,
            "meanSubgroupPairAccuracy": mean_or_none(rows.pairCredit / rows.pairs),
            "meanSubgroupSpearman": mean_or_none(rows.spearman),
            "rankMAE": float(rows.absoluteErrorSum.sum() / rows.riders.sum()),
            "normalizedRankMAE": float(rows.normalizedErrorSum.sum() / rows.riders.sum()),
            "winnerPickAccuracy": mean_or_none(rows.winnerCredit),
            "subgroupsWithObservedWinner": int(rows.winnerCredit.notna().sum()),
        }
    return summaries


def bootstrap_difference(stats: pd.DataFrame, contender: str) -> dict:
    aggregates = stats.groupby(["eventId", "model"])[["pairCredit", "pairs"]].sum()
    base = aggregates.xs("ZRS", level="model")
    other = aggregates.xs(contender, level="model").reindex(base.index)
    rng = np.random.default_rng(20261027)
    draws = rng.integers(0, len(base), size=(2000, len(base)))
    base_values = base.to_numpy()
    other_values = other.to_numpy()
    base_sums = base_values[draws].sum(axis=1)
    other_sums = other_values[draws].sum(axis=1)
    differences = other_sums[:, 0] / other_sums[:, 1] - base_sums[:, 0] / base_sums[:, 1]
    return {
        "difference": float(other.pairCredit.sum() / other.pairs.sum()
                            - base.pairCredit.sum() / base.pairs.sum()),
        "eventBootstrap95CI": np.quantile(differences, [0.025, 0.975]).tolist(),
        "bootstrapReplicates": 2000,
    }


def analyze(zrs: pd.DataFrame, velo: pd.DataFrame, cache: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    for name, frame in [("ZRS", zrs), ("ZwiftRacing", velo)]:
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f"{name} has missing or duplicate join keys.")
    velo = load_metadata(velo, cache)
    joined = zrs.merge(velo, on=KEYS, how="inner", validate="one_to_one")
    score_columns = [column for column, _ in MODELS.values()]
    finite = np.isfinite(joined[score_columns + ["rank", "position"]]).all(axis=1)
    finished = joined.loc[~joined.didNotFinish & finite].copy()
    finished = finished.loc[finished["rank"].gt(0) & finished.position.gt(0)]
    sizes = finished.groupby(GROUPS).profileId.transform("size")
    cohort = finished.loc[sizes >= 3].copy()
    if cohort.empty:
        raise ValueError("No complete matched finisher subgroups with at least three riders.")
    stats = pd.DataFrame([
        row for _, group in cohort.groupby(GROUPS) for row in evaluate_group(group)
    ])
    other_outcome = pd.DataFrame([
        row for _, group in cohort.groupby(GROUPS) for row in evaluate_group(group, "position")
    ])
    agreements = disagreements = ties = 0
    for _, group in cohort.groupby(GROUPS):
        left, right = np.triu_indices(len(group), 1)
        zwift_rank = group["rank"].to_numpy()
        zr_rank = group.position.to_numpy()
        zwift_sign = np.sign(zwift_rank[left] - zwift_rank[right])
        zr_sign = np.sign(zr_rank[left] - zr_rank[right])
        valid = (zwift_sign != 0) & (zr_sign != 0)
        agreements += int(((zwift_sign == zr_sign) & valid).sum())
        disagreements += int(((zwift_sign != zr_sign) & valid).sum())
        ties += int((~valid).sum())
    full_finishers = velo.loc[
        ~velo.didNotFinish & velo.expectedPosition.notna() & velo.position.notna()
    ]
    published_errors = (full_finishers.expectedPosition - full_finishers.position).abs()
    comparisons = {
        name: bootstrap_difference(stats, name) for name in MODELS if name != "ZRS"
    }
    per_pen = {
        str(label): summarize(rows)
        for label, rows in stats.groupby("penLabel")
    }
    report = {
        "method": {
            "cohort": "Same matched finishers with all four predictors present; at least 3 per subgroup.",
            "outcome": "Zwift rank; reranked within the matched subgroup subset.",
            "prediction": "Descending pre-race score, or ascending published expectedPosition; average ranks for ties.",
            "pairAccuracy": "Correct within-subgroup rider ordering; half credit for predictor ties; outcome ties excluded.",
            "normalizedRankMAE": "Absolute predicted rank error / (matched subgroup size - 1).",
            "winnerPick": "Highest-ranked matched predictor picks the actual rank-1 rider, only where observed; tie-split credit.",
            "bootstrap": "Paired resampling of events (all pens together), fixed seed, 2000 replicates; repeat riders can remain dependent.",
            "leakage": "Only before-race ratings used, never newScore or ratingAfter.",
            "DNF": "Only cached didNotFinish == true is excluded; absent flags count as not marked DNF.",
        },
        "coverage": {
            "zrsResults": len(zrs), "zwiftRacingResults": len(velo),
            "events": int(zrs.eventId.nunique()), "matchedResults": len(joined),
            "matchedDNFs": int(joined.didNotFinish.sum()),
            "matchedNonfinitePredictorsOrOutcomes": int((~finite).sum()),
            "completeMatchedFinishersBeforeSizeFilter": len(finished),
            "analyzedResults": len(cohort),
            "analyzedRiders": int(cohort.profileId.nunique()),
            "analyzedEvents": int(cohort.eventId.nunique()),
            "analyzedSubgroups": len(stats) // len(MODELS),
            "allZwiftRacingDNFs": int(velo.didNotFinish.sum()),
            "matchedRankDisagreements": int(joined["rank"].ne(joined.position).sum()),
            "outcomePairAgreements": agreements, "outcomePairDisagreements": disagreements,
            "outcomeTiedPairs": ties,
        },
        "matchedComparison": summarize(stats),
        "pairAccuracyImprovementOverZRS": comparisons,
        "largerSubgroupsAtLeast10": summarize(stats.loc[stats.riders >= 10]),
        "usingZwiftRacingPositionInstead": summarize(other_outcome),
        "byPenLabel": per_pen,
        "publishedExpectedPositionFullField": {
            "results": len(full_finishers),
            "MAEPlaces": float(published_errors.mean()),
            "medianAbsoluteErrorPlaces": float(published_errors.median()),
            "within5Places": float(published_errors.le(5).mean()),
            "note": "All available non-DNF ZwiftRacing riders; not a head-to-head ZRS comparison.",
        },
    }
    for name, (column, ascending) in MODELS.items():
        cohort[f"predictedRank_{name}"] = cohort.groupby(GROUPS)[column].rank(
            method="average", ascending=ascending
        )
    cohort["actualMatchedRank"] = cohort.groupby(GROUPS)["rank"].rank(method="average")
    return report, stats, cohort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zrs", type=Path, default=Path("race_data") / "tour_of_watopia_2026.csv")
    parser.add_argument("--velo", type=Path, default=Path("race_data") / "tour_of_watopia_2026_velo.csv")
    parser.add_argument("--cache", type=Path, default=Path("race_data") / "zwiftracing_events")
    parser.add_argument("--output-dir", type=Path, default=Path("race_data") / "prediction_analysis")
    parser.add_argument(
        "--log-file", type=Path,
        help="Append terminal output to this file (default: analysis.log in the output directory).",
    )
    args = parser.parse_args(argv)
    return run_logged(args.log_file or args.output_dir / "analysis.log", lambda: run(args))


def run(args: argparse.Namespace) -> int:
    report, stats, cohort = analyze(pd.read_csv(args.zrs), pd.read_csv(args.velo), args.cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    stats.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    cohort.to_csv(args.output_dir / "matched_predictions.csv", index=False)
    print(json.dumps({key: report[key] for key in [
        "coverage", "matchedComparison", "pairAccuracyImprovementOverZRS",
        "largerSubgroupsAtLeast10", "usingZwiftRacingPositionInstead",
        "publishedExpectedPositionFullField",
    ]}, indent=2, allow_nan=False))
    print(f"Saved analysis to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
