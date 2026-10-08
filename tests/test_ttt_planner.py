import numpy as np
import pytest

from bike_comparison.pacing_planner import RouteProfile
from bike_comparison.physics import speed_from_power
from bike_comparison.ttt_planner import (
    TTTRider,
    _CoursePlanner,
    _Plan,
    _Team,
    _build_result,
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


def test_surface_resistance_affects_group_speed_and_survives_downsampling():
    route = RouteProfile(
        name="Mixed", distance_m=np.array([0.0, 100.0, 200.0, 300.0]),
        altitude_m=np.zeros(4), surfaces=np.array(["Tarmac", "Dirt", "Tarmac", "Tarmac"]))
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        route, team, crr=0.004, max_chunk_m=50.0, reserve_fraction=0.0,
        reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    assert planner.surfaces.tolist() == ["Tarmac", "Tarmac", "Dirt", "Dirt", "Tarmac", "Tarmac"]
    assert planner.crr.tolist() == pytest.approx([0.004, 0.004, 0.016, 0.016, 0.004, 0.004])
    factors = np.array([1.0, 0.75, 0.6, 0.6])
    mask = np.ones(4, dtype=bool)
    tarmac_speed = planner.advance(0, 10.0, 1.0, 0, factors, mask, 50.0)[0]
    dirt_speed = planner.advance(2, 10.0, 1.0, 0, factors, mask, 50.0)[0]
    assert dirt_speed < tarmac_speed
    drop = tuple([planner.C] * 4)
    sim = planner.simulate(np.ones(planner.C), drop, 1.0)
    plan = _Plan(np.ones(planner.C), sim, True, 0.0, planner.lam)
    result = _build_result(route, planner, plan, drop, 1.0, 3, downsample_points=2)
    assert result.surface_type == planner.surfaces.tolist()
    assert result.crr == planner.crr.tolist()
    assert len(result.altitude_m) == len(result.surface_type) == len(result.speed_kph)


@pytest.mark.parametrize("bike_type", ["road_bike", "gravel_bike", "mtb"])
def test_surface_resistance_uses_the_selected_bike_category(bike_type):
    from shared.surface_lookup import surface_types_to_crr

    route = flat_route(200.0)
    route.surfaces = np.array(["Dirt", "Dirt"])
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        route, team, crr=0.004, max_chunk_m=100.0, reserve_fraction=0.0,
        reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1,
        bike_type=bike_type)

    assert planner.crr.tolist() == surface_types_to_crr(np.array(["Dirt", "Dirt"]), bike_type).tolist()


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
    monkeypatch.setattr(planner, "simulate", lambda *args, **kwargs: simulated)
    monkeypatch.setattr(planner, "slack", lambda *args: np.array([margin]))

    sim, slack, feasible = planner._check([], [], [])

    assert sim is simulated
    assert slack[0] == margin
    assert feasible is expected


@pytest.mark.parametrize("pulls", [[30.0, 0.0, 0.0, 0.0], [30.0, 30.0, 30.0, 30.0]])
def test_course_rejects_constant_pulls_over_follower_power_caps(monkeypatch, pulls):
    riders = [rider("A", 600), rider("B", 150), rider("C", 180), rider("D", 200)]
    team = _Team(riders, 0.004)
    planner = _CoursePlanner(
        flat_route(600.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=90.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array(pulls))
    efforts = np.linspace(0.5, 2.0, planner.C)

    sim, _, feasible = planner._check(efforts, np.full(team.n, planner.C), np.ones(team.n))

    assert not feasible
    assert np.any(sim.peak_power > 1.5 * team.cp[None, :] + 1e-6)
    assert sim.pulls[0]["power_w"] == 300.0
    np.testing.assert_array_equal(sim.enter_speeds[1:], sim.exit_speeds[:-1])


def test_constant_pull_crosses_chunks_and_records_actual_duration_and_energy(monkeypatch):
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        flat_route(800.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=60.0, max_pull_s=60.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array([60.0, 0.0, 0.0, 0.0]))
    efforts = np.linspace(0.8, 1.4, planner.C)

    sim = planner.simulate(efforts, np.full(4, planner.C), np.ones(4))

    assert len(sim.pulls) >= 2
    assert sim.pulls[0]["power_w"] == 240.0
    assert sim.pulls[0]["power_wkg"] == 240.0 / 75.0
    assert sim.pulls[0]["duration_s"] == pytest.approx(60.0, abs=1e-7)
    assert 0.0 < sim.pulls[-1]["duration_s"] < 60.0
    ends = np.cumsum(sim.chunk_time)
    np.testing.assert_allclose(sim.power[ends < 60.0, 0], 240.0)
    assert sum(pull["duration_s"] for pull in sim.pulls) == pytest.approx(sim.total_time)
    assert sum(pull["duration_s"] * pull["power_w"] for pull in sim.pulls) == pytest.approx(
        sim.lead_energy[0])
    assert sim.pulls[0]["start_km"] == 0.0
    assert sim.pulls[-1]["end_km"] == pytest.approx(0.8)
    for previous, current in zip(sim.pulls, sim.pulls[1:]):
        assert current["start_time_s"] == pytest.approx(
            previous["start_time_s"] + previous["duration_s"])
        assert current["start_km"] == pytest.approx(previous["end_km"])


def test_non_puller_can_rejoin_rotation_later_without_changing_early_pacing(monkeypatch):
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        flat_route(2400.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=60.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array([20.0, 20.0, 20.0, 0.0]))
    drops = np.full(4, planner.C)
    efforts = np.ones(planner.C)
    baseline = planner.simulate(efforts, drops, np.ones(4))
    start = planner.C // 2

    sim = planner.simulate(efforts, drops, np.ones(4), pull_overrides=((3, start, 20.0),))

    assert not np.any(baseline.front[:, 3])
    assert not np.any(sim.front[:start, 3])
    assert np.any(sim.front[start:, 3])
    np.testing.assert_array_equal(sim.speeds[:start], baseline.speeds[:start])
    assert np.all(sim.pulls_start[:start, 3] == 0.0)
    assert np.all(sim.pulls_start[start:, 3] == 20.0)
    assert np.all(sim.peak_power <= 1.5 * team.cp[None, :] + 1e-6)
    assert np.min(planner.slack(sim, drops)) > 0.0
    result = _build_result(flat_route(2400.0), planner,
                           _Plan(efforts, sim, True, 0.1, planner.lam), drops, np.ones(4), 3, 400)
    assert len(result.phases) == 2
    assert result.phases[0]["pulls_s"]["D"] == 0.0
    assert result.phases[1]["pulls_s"]["D"] == 20.0
    assert any(pull["rider"] == "D" for pull in result.pulls)


@pytest.mark.parametrize("trial_time", [95.0, 105.0])
def test_non_puller_search_checks_late_starts_and_keeps_only_faster_feasible_plans(monkeypatch, trial_time):
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        flat_route(400.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=60.0, pull_step_s=10.0, speed_step_mps=0.1)
    front = np.ones((planner.C, 4))
    front[:, 3] = 0.0
    baseline_sim = type("Sim", (), {"total_time": 100.0, "front": front,
                                    "wbal": np.full((planner.C, 4), 15000.0)})()
    baseline = _Plan(np.ones(planner.C), baseline_sim, True, 0.1, planner.lam)
    checked = []

    def solve(drops, mult, iterations, pull_overrides=()):
        checked.append(pull_overrides)
        start = pull_overrides[-1][1]
        feasible = start != 0
        duration = 90.0 if start == 0 else trial_time if start == planner.C // 2 else 110.0
        sim = type("Sim", (), {"total_time": duration})()
        return _Plan(baseline.efforts, sim, feasible, 0.1, planner.lam, pull_overrides)

    monkeypatch.setattr(planner, "solve", solve)
    monkeypatch.setattr(planner, "finish_effort", lambda plan, *args: plan)

    result = planner.reconsider_non_pullers(baseline, np.full(4, planner.C), np.ones(4), 2)

    assert checked == [((3, start, 20.0),) for start in (0, 2, 3)]
    if trial_time < baseline.sim.total_time:
        assert result.sim.total_time == trial_time
        assert result.pull_overrides == ((3, 2, 20.0),)
    else:
        assert result is baseline


def test_reconsidering_an_idle_rider_improves_a_real_feasible_plan(monkeypatch):
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        flat_route(4000.0), team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=60.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "pulls_for", lambda *args: np.array([20.0, 20.0, 20.0, 0.0]))
    drops = np.full(4, planner.C)
    mult = np.ones(4)
    baseline = planner.finish_effort(planner.solve(drops, mult, 2), drops, mult)

    result = planner.reconsider_non_pullers(baseline, drops, mult, 2)

    assert baseline.feasible and result.feasible
    assert not np.any(baseline.sim.front[:, 3])
    assert np.sum(result.sim.front[:, 3]) > 0.0
    assert result.sim.total_time < baseline.sim.total_time - 0.5
    assert np.min(planner.slack(result.sim, drops)) > 0.0
    assert np.all(result.sim.peak_power <= 1.5 * team.cp[None, :] + 1e-6)


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


def test_scheduled_drop_keeps_a_rider_drafting_while_they_can_hold_the_group(monkeypatch):
    route = flat_route(1000.0)
    team = _Team([rider(name, 300) for name in ("A", "B", "C", "D")], 0.004)
    planner = _CoursePlanner(
        route, team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda active, *args:
                        np.array([20.0 if member else 0.0 for member in active]))
    drops = np.array([planner.C, planner.C, planner.C, 2])
    efforts = np.ones(planner.C)

    sim, slack, feasible = planner._check(efforts, drops, np.ones(4))
    result = _build_result(route, planner, _Plan(efforts, sim, feasible, float(np.min(slack)),
                                               planner.lam), drops, np.ones(4), 3, 400)

    assert feasible
    assert not np.any(sim.front[2:, 3])
    assert result.riders[3]["finishes"]
    assert result.riders[3]["drop_km"] is None
    assert all(value is not None for value in result.wbal_j["D"])
    assert all(value is not None and value > 0.0 for value in result.power_w["D"])
    assert result.phases[-1]["riders"] == ["A", "B", "C", "D"]
    assert result.phases[-1]["pulls_s"]["D"] == 0.0


@pytest.mark.parametrize("mode", ["exhaustion", "recovery", "power_cap", "finish"])
def test_hanging_rider_exit_follows_simulated_power_and_wbal(monkeypatch, mode):
    route = flat_route(1000.0)
    team = _Team([rider(name, 300) for name in ("A", "B", "C")] +
                 [rider("D", 200, w_prime=1000.0)], 0.004)
    planner = _CoursePlanner(
        route, team, crr=0.004, max_chunk_m=100.0,
        reserve_fraction=0.1, reserve_release_m=2000.0, max_power_cp_mult=1.5,
        min_pull_s=20.0, max_pull_s=20.0, pull_step_s=10.0, speed_step_mps=0.1)
    monkeypatch.setattr(planner, "cp_eff", lambda *args: tuple(team.cp))
    monkeypatch.setattr(planner, "pulls_for", lambda active, *args:
                        np.array([20.0 if active[index] and index < 3 else 0.0
                                  for index in range(4)]))

    def advance(chunk, speed, effort, leader, factors, mask, distance, **kwargs):
        trailing_power = 100.0 if chunk < 2 or mode == "finish" else 250.0
        if chunk >= 2 and mode == "power_cap":
            trailing_power = 350.0
        if chunk == 3 and mode == "recovery":
            trailing_power = 100.0
        power = np.where(mask, [300.0, 300.0, 300.0, trailing_power], 0.0)
        return 10.0, distance / 10.0, power, np.zeros(4), power

    monkeypatch.setattr(planner, "advance", advance)
    drops = np.array([planner.C, planner.C, planner.C, 2])
    efforts = np.ones(planner.C)
    sim = planner.simulate(efforts, drops, np.ones(4), initial_speed_mps=10.0)
    result = _build_result(route, planner, _Plan(efforts, sim, True, 0.1, planner.lam),
                           drops, np.ones(4), 3, 400)
    trailing = result.riders[3]

    assert sim.total_time == pytest.approx(100.0)
    assert not np.any(sim.front[:, 3])
    assert np.all(sim.peak_power <= 1.5 * team.cp[None, :] + 1e-6)
    if mode == "finish":
        assert trailing["finishes"]
        assert np.sum(sim.rider_time[:, 3]) == pytest.approx(100.0)
        assert sim.wbal[-1, 3] == pytest.approx(1000.0)
    else:
        expected_time = 20.0 if mode == "power_cap" else 40.0
        if mode == "recovery":
            recovered_balance = 1000.0 - 500.0 * np.exp(-1.0)
            expected_time = 40.0 + recovered_balance / 50.0
        assert not trailing["finishes"]
        assert sim.drop_time[3] == pytest.approx(expected_time)
        assert sim.drop_distance[3] == pytest.approx(expected_time * 10.0)
        assert np.sum(sim.rider_time[:, 3]) == pytest.approx(expected_time)
        assert trailing["drop_time_s"] == round(expected_time, 1)
        assert trailing["drop_km"] == round(expected_time / 100.0, 2)
        last = np.flatnonzero(np.isfinite(sim.wbal[:, 3]))[-1]
        assert np.all(np.isnan(sim.wbal[last + 1:, 3]))
        assert sim.wbal[last, 3] == pytest.approx(1000.0 if mode == "power_cap" else 0.0)
        energy = np.sum(sim.power[:, 3] * sim.rider_time[:, 3])
        assert trailing["avg_power_w"] == round(energy / expected_time, 1)


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
