import numpy as np
import pytest

from bike_comparison.pacing_planner import RouteProfile
from bike_comparison.physics import speed_from_power
from bike_comparison.ttt_planner import (
    TTTRider,
    _CoursePlanner,
    _Plan,
    _Team,
    _flat_feasible,
    _line_factors,
    plan_ttt_pacing,
    scoring_rider_count,
    solve_flat_rotation,
)


def flat_route(distance_m=8000.0):
    return RouteProfile(
        name="Flat",
        distance_m=np.array([0.0, distance_m]),
        altitude_m=np.array([0.0, 0.0]),
    )


def rider(name, cp, w_prime=20000.0, weight=75.0, height=1.80):
    return TTTRider(name=name, weight_kg=weight, height_m=height, cp_w=cp, w_prime_j=w_prime)


def test_power_driven_traversal_preserves_coasting_momentum():
    riders = [TTTRider(name, 75.0, 1.8, 300.0, 20000.0, cda=0.3)
              for name in ("A", "B", "C", "D")]
    team = _Team(riders, 0.004)
    planner = _CoursePlanner(
        flat_route(200.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    speed = 17.2
    elapsed = 0.0
    for _ in range(50):
        speed, duration, power, braking, peak = planner.advance(
            0, speed, 0.0, 0, np.array([1.0, 0.75, 0.6, 0.6]),
            np.ones(4, dtype=bool), 2.0)
        elapsed += duration
        assert power[0] == 0.0
        assert braking[0] == 0.0
        assert np.all(peak <= 1.5 * team.cp + 1e-6)

    rolling = float(planner.f_static[0, 0])
    aero = float(team.aero_k[0])
    expected = np.sqrt((17.2 ** 2 + rolling / aero) *
                       np.exp(-2.0 * aero * 100.0 / team.mass[0]) - rolling / aero)
    assert speed == pytest.approx(expected, abs=0.05)
    assert speed > 11.2
    assert elapsed > 100.0 / 17.2


@pytest.mark.parametrize("size, expected", [(4, 3), (5, 4), (8, 4)])
def test_scoring_rider_count_follows_wtrl_rules(size, expected):
    assert scoring_rider_count(size) == expected


@pytest.mark.parametrize("size", [3, 9])
def test_scoring_rider_count_rejects_invalid_team_sizes(size):
    with pytest.raises(ValueError, match="4-8 riders"):
        scoring_rider_count(size)


def test_line_factors_put_next_puller_second_and_non_pullers_at_back():
    active = (True, True, True, True)
    pulls = [30.0, 0.0, 30.0, 30.0]

    factors = _line_factors(active, pulls, leader=2)

    assert factors.tolist() == [0.60, 0.60, 1.0, 0.75]


def test_line_factors_use_custom_draft_percentages():
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004,
                 draft_second_pct=10.0, draft_rest_pct=15.0)
    factors = _line_factors((True,) * 4, [30.0, 0.0, 30.0, 30.0], 2, team.draft_factors)

    assert factors.tolist() == [0.85, 0.85, 1.0, 0.9]


@pytest.mark.parametrize("field", ["draft_second_pct", "draft_rest_pct"])
@pytest.mark.parametrize("value", [-1.0, 100.0, float("nan"), float("inf")])
def test_draft_percentages_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        _Team([rider("A", 300)], 0.004, **{field: value})


def test_no_drafting_has_equal_power_without_duplicate_position_pricing(monkeypatch):
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004,
                 draft_second_pct=0.0, draft_rest_pct=0.0)
    planner = _CoursePlanner(
        flat_route(200.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.full(4, 20.0))
    drops = np.full(4, planner.C)
    sim = planner.simulate(np.ones(planner.C), drops, np.ones(4))

    np.testing.assert_allclose(sim.power, 300.0)
    np.testing.assert_allclose(sim.braking_power, 0.0)
    assert np.all(planner.price_efforts(np.full(4, 0.05), sim, drops) >= 0.9)


def test_smaller_draft_reductions_lower_sustainable_rotation_speed():
    riders = [rider(name, 300) for name in ("A", "B", "C", "D")]
    settings = {"min_pull_s": 20.0, "max_pull_s": 20.0}
    default = solve_flat_rotation(riders, **settings)
    reduced = solve_flat_rotation(riders, draft_second_pct=10.0, draft_rest_pct=15.0, **settings)
    none = solve_flat_rotation(riders, draft_second_pct=0.0, draft_rest_pct=0.0, **settings)

    assert default.speed_mps > reduced.speed_mps > none.speed_mps


@pytest.mark.parametrize("margin, expected", [(-3.0 / 20000.0, False), (0.0, False),
                                            (1.0 / 20000.0, True)])
def test_course_feasibility_requires_positive_wbal_margin(monkeypatch, margin, expected):
    planner = object.__new__(_CoursePlanner)
    simulated = type("Sim", (), {"peak_power": np.zeros((1, 4)), "motion_feasible": True})()
    planner.team = type("Team", (), {"cp": np.full(4, 300.0)})()
    planner.max_power_cp_mult = 1.5
    monkeypatch.setattr(planner, "simulate", lambda *args: simulated)
    monkeypatch.setattr(planner, "slack", lambda *args: np.array([margin]))

    sim, slack, feasible = planner._check([], [], [])

    assert sim is simulated
    assert slack[0] == margin
    assert feasible is expected


@pytest.mark.parametrize("pulls", [[30.0, 0.0, 0.0, 0.0], [30.0, 30.0, 30.0, 30.0]])
def test_simulation_caps_power_including_acceleration_and_second_position(monkeypatch, pulls):
    riders = [rider("A", 600), rider("B", 150), rider("C", 180), rider("D", 200)]
    team = _Team(riders, 0.004)
    planner = _CoursePlanner(
        flat_route(600.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=90.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array(pulls))
    efforts = np.linspace(0.5, 2.0, planner.C)

    sim = planner.simulate(efforts, np.full(team.n, planner.C), np.ones(team.n))

    assert np.all(sim.power <= 1.5 * team.cp[None, :] + 1e-6)
    assert np.all(sim.peak_power <= 1.5 * team.cp[None, :] + 1e-6)
    np.testing.assert_array_equal(sim.enter_speeds[1:], sim.exit_speeds[:-1])


def test_flat_rotation_is_sustainable_and_strongest_rider_pulls_longest():
    riders = [rider("A", 320), rider("B", 300), rider("C", 280), rider("D", 260)]

    rotation = solve_flat_rotation(riders)

    team = _Team(riders, 0.004)
    active = (True,) * 4
    assert _flat_feasible(team, active, rotation.pulls_s, rotation.speed_mps, 0.0, 2.0)
    assert not _flat_feasible(team, active, rotation.pulls_s, rotation.speed_mps + 0.05, 0.0, 2.0)
    assert rotation.pulls_s[0] >= rotation.pulls_s[3]
    assert max(rotation.pulls_s) <= 60.0
    assert all(lead > cp for lead, cp in zip(rotation.lead_power_w, [320, 300, 280, 260]))
    assert all(draft < cp for draft, cp in zip(rotation.draft_power_w, [320, 300, 280, 260]))


def test_team_plan_is_feasible_and_faster_than_strongest_solo_rider():
    riders = [rider("A", 320), rider("B", 300), rider("C", 290), rider("D", 280)]

    result = plan_ttt_pacing(flat_route(), riders, allow_drops=False, refine_pulls=False)

    assert result.feasible
    assert result.scoring_rider_count == 3
    assert all(row["min_wbal_j"] >= 0 for row in result.riders)
    solo = speed_from_power(320, 0.0, 75.0, 8.0, riders[0].effective_cda) * 3.6
    assert result.avg_speed_kph > solo
    assert sum(row["time_on_front_s"] for row in result.riders) == pytest.approx(
        result.total_time_seconds, abs=0.5)


def test_dropping_a_weak_rider_never_slows_the_scoring_group():
    riders = [rider("A", 330), rider("B", 320), rider("C", 310), rider("D", 300),
              rider("Weak", 200, w_prime=15000.0)]
    route = flat_route(6000.0)

    no_drop = plan_ttt_pacing(route, riders, allow_drops=False, refine_pulls=False)
    with_drop = plan_ttt_pacing(route, riders, refine_pulls=False)

    assert with_drop.feasible
    assert with_drop.total_time_seconds <= no_drop.total_time_seconds + 0.5
    finishers = [row for row in with_drop.riders if row["finishes"]]
    assert len(finishers) >= with_drop.scoring_rider_count
    assert all(row["name"] == "Weak" for row in with_drop.riders if not row["finishes"])


def test_hilly_plan_keeps_every_rider_positive_and_under_the_power_cap():
    d = np.linspace(0.0, 6000.0, 121)
    route = RouteProfile(name="Hills", distance_m=d,
                         altitude_m=60.0 * np.sin(d / 600.0) + 0.02 * np.maximum(d - 4000.0, 0.0))
    riders = [rider("A", 350), rider("B", 320), rider("C", 300), rider("D", 280),
              rider("E", 240, w_prime=15000.0)]

    result = plan_ttt_pacing(route, riders, max_power_cp_mult=1.5, refine_pulls=False)

    assert result.feasible
    assert all(row["min_wbal_j"] >= 0 for row in result.riders)
    for spec in riders:
        wbal = [w for w in result.wbal_j[spec.name] if w is not None]
        power = [p for p in result.power_w[spec.name] if p is not None]
        assert min(wbal) >= 0
        assert max(power) <= 1.5 * spec.cp_w + 0.5


def test_progress_reports_iterations_and_retains_best_feasible_plan():
    riders = [rider("A", 320), rider("B", 300), rider("C", 290), rider("D", 280)]
    updates = []
    result = plan_ttt_pacing(
        flat_route(1200.0), riders, allow_drops=False, refine_pulls=False,
        min_pull_s=20.0, max_pull_s=20.0, speed_step_mps=1.0,
        search_iterations=2, final_iterations=2, progress_callback=updates.append)

    assert [update["iteration"] for update in updates] == list(range(1, len(updates) + 1))
    for stage in ("Team together", "Final refinement"):
        assert [update["candidate_iteration"] for update in updates
                if update["stage"] == stage and update["step"] == "power search"] == [1, 2]
    plans = [update["plan"] for update in updates if update["plan"] is not None]
    assert plans
    assert all(plan.feasible for plan in plans)
    assert all(row["min_wbal_j"] >= 0 for plan in plans for row in plan.riders)
    times = [plan.total_time_seconds for plan in plans]
    assert times == sorted(times, reverse=True)
    assert result.total_time_seconds <= min(times)


def test_finish_effort_spends_recovered_wbal_without_changing_early_pacing(monkeypatch):
    riders = [rider(name, 300) for name in ("A", "B", "C", "D")]
    team = _Team(riders, 0.004)
    planner = _CoursePlanner(
        flat_route(6000.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=120.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array([30.0, 0.0, 0.0, 0.0]))
    efforts = np.where(planner.end <= 3000.0, 1.2, 2.0 / 3.0)
    drop_chunk = np.full(team.n, planner.C)
    mult = np.ones(team.n)
    sim, slack, feasible = planner._check(efforts, drop_chunk, mult)
    assert feasible
    baseline = planner._tighten(_Plan(efforts, sim, True, float(np.min(slack)), planner.lam),
                                drop_chunk, mult)
    assert baseline is not None
    assert baseline.sim.wbal[-1, 0] > 0.5 * team.wp[0]

    result = planner.finish_effort(baseline, drop_chunk, mult)

    assert result.feasible
    assert result.sim.total_time < baseline.sim.total_time - 5.0
    assert result.sim.wbal[-1, 0] < 0.1 * team.wp[0]
    assert np.min(result.sim.min_b) > 0.0
    assert np.min(planner.slack(result.sim, drop_chunk)) > 0.0
    assert np.all(result.sim.power <= 1.5 * team.cp[None, :] + 1e-6)
    prefix = planner.end <= 3000.0
    np.testing.assert_array_equal(result.sim.speeds[prefix], baseline.sim.speeds[prefix])


def test_momentum_survives_descent_to_flat_without_zero_lead_power(monkeypatch):
    route = RouteProfile(name="Descent to flat", distance_m=np.array([0.0, 100.0, 200.0]),
                         altitude_m=np.array([0.0, -5.0, -5.0]))
    team = _Team([TTTRider(name, 75.0, 1.8, 300.0, 20000.0, cda=0.3)
                  for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        route, team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array([20.0, 0.0, 0.0, 0.0]))
    sim = planner.simulate(np.ones(planner.C), np.full(4, planner.C), np.ones(4),
                           initial_speed_mps=17.2)

    np.testing.assert_array_equal(sim.enter_speeds[1:], sim.exit_speeds[:-1])
    assert np.all(sim.power[:, 0] == pytest.approx(300.0))
    assert np.all(sim.braking_power[:, 0] == 0.0)
    assert sim.exit_speeds[-1] * 3.6 > 48.8
    assert np.all(sim.peak_power <= 450.0 + 1e-6)
    cold = planner.simulate(np.ones(planner.C), np.full(4, planner.C), np.ones(4))
    assert any(not np.array_equal(
        planner.price_efforts(np.full(4, price), sim, np.full(4, planner.C)),
        planner.price_efforts(np.full(4, price), cold, np.full(4, planner.C)))
        for price in (0.0001, 0.0005, 0.001))


def test_momentum_integration_converges_from_slow_start(monkeypatch):
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        flat_route(400.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array([20.0, 0.0, 0.0, 0.0]))
    coarse = planner.simulate(np.ones(planner.C), np.full(4, planner.C), np.ones(4))
    planner.max_step_m = 0.5
    fine = planner.simulate(np.ones(planner.C), np.full(4, planner.C), np.ones(4))

    assert coarse.total_time == pytest.approx(fine.total_time, rel=0.01)
    assert coarse.exit_speeds[-1] == pytest.approx(fine.exit_speeds[-1], abs=0.05)


def test_unpowered_stall_is_not_a_feasible_coasting_plan(monkeypatch):
    route = RouteProfile(name="Climb", distance_m=np.array([0.0, 200.0]),
                         altitude_m=np.array([0.0, 10.0]))
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        route, team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array([20.0, 0.0, 0.0, 0.0]))
    sim, slack, feasible = planner._check(np.zeros(planner.C), np.full(4, planner.C), np.ones(4))

    assert np.all(slack > 0.0)
    assert not sim.motion_feasible
    assert not feasible
