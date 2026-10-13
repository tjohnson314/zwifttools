"""Tests for the race analyzer (race_replay/race_analyzer.py).

All races here are synthetic: riders on a flat course with scripted speed
changes, so each scenario (drop, near miss, breakaway, split, getaway, bridge,
distance glitch) is known exactly.
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from race_replay import race_analyzer as ra
from race_replay.data_cleaner import RiderData

BASE_SPEED = 12.0          # m/s (43 km/h)
SPACING_SEC = 0.3          # riders this far apart are one bunch
T = 900                    # seconds of racing
BIKE_KG, BIKE_CDA_BIAS = 8.0, 0.0


def make_race(speed_changes=None, n=10, start_offsets_m=None, finish_km=30.0,
              power_w=250.0, weight_kg=70.0, height_cm=178.0):
    """Build a synthetic race.

    ``speed_changes`` maps rider index -> list of (t_from, t_to, delta m/s).
    Riders start in a tight line (``SPACING_SEC`` apart) unless
    ``start_offsets_m`` gives explicit start distances.
    """
    speed = np.full((n, T), BASE_SPEED)
    for i, changes in (speed_changes or {}).items():
        for a, b, delta in changes:
            speed[i, a:b] += delta
    if start_offsets_m is None:
        start_offsets_m = [-i * SPACING_SEC * BASE_SPEED for i in range(n)]
    riders = []
    for i in range(n):
        dist_m = start_offsets_m[i] + np.concatenate([[0.0], np.cumsum(speed[i, 1:])])
        # Power roughly follows speed changes so CP estimates have something to work with.
        power = power_w + 40.0 * (speed[i] - BASE_SPEED)
        data = pd.DataFrame({
            'distance_km': dist_m / 1000.0,
            'speed_kmh': speed[i] * 3.6,
            'power_watts': power,
            'altitude_m': np.full(T, 100.0),
        }, index=pd.Index(np.arange(T, dtype=float), name='time_sec'))
        riders.append(RiderData(rank=i + 1, activity_id=f'a{i}', name=f'Rider {i + 1}', team='',
                                data=data, finish_time_sec=None, weight_kg=weight_kg,
                                height_cm=height_cm))
    elevation = pd.DataFrame({'distance_km': np.linspace(-1, finish_km + 1, 200),
                              'altitude_m': np.full(200, 100.0)})
    return SimpleNamespace(riders=riders, min_time=0.0, max_time=float(T - 1),
                           finish_line_km=finish_km, elevation_profile=elevation,
                           route_name='Synthetic', world=None)


def grid_for(race):
    return ra.build_race_grid(race, BIKE_KG, BIKE_CDA_BIAS)


def events_for(race, rank):
    grid = grid_for(race)
    return ra.detect_separations(grid, grid.ranks.index(rank))


class TestGrouping(unittest.TestCase):
    def test_tight_bunch_is_one_group(self):
        grid = grid_for(make_race())
        k = 100
        self.assertEqual(len(set(grid.group[:, k])), 1)
        self.assertTrue(np.all(grid.group_size[:, k] == 10))
        # Time gaps between neighbours are the spacing we built in.
        self.assertAlmostEqual(np.nanmedian(grid.gap_ahead[1:, k]), SPACING_SEC, places=2)

    def test_gap_over_threshold_starts_new_group(self):
        # Last rider starts 3 s behind the bunch.
        offsets = [-i * SPACING_SEC * BASE_SPEED for i in range(9)] + [-(8 * SPACING_SEC + 3.0) * BASE_SPEED]
        grid = grid_for(make_race(start_offsets_m=offsets))
        self.assertEqual(grid.group_size[9, 100], 1)
        self.assertEqual(grid.group_size[0, 100], 9)
        self.assertGreater(grid.gap_ahead[9, 100], ra.GROUP_GAP_SEC)


class TestSplitRule(unittest.TestCase):
    def test_breakaway_does_not_count(self):
        self.assertFalse(ra._split_matters(1, 10))
        self.assertFalse(ra._split_matters(ra.FLIER_MAX_RIDERS, 10))

    def test_sizeable_front_group_counts_even_if_smaller(self):
        self.assertTrue(ra._split_matters(4, 10))
        self.assertTrue(ra._split_matters(16, 39))       # a front group of 16 from 39

    def test_small_share_of_big_group_does_not_count(self):
        self.assertFalse(ra._split_matters(4, 20))       # 20% < SPLIT_MIN_FRAC


class TestSeparations(unittest.TestCase):
    def test_dropped_rider(self):
        # Last rider loses 1.5 m/s from t=300 and never comes back.
        evs = events_for(make_race({9: [(300, T, -1.5)]}), rank=10)
        self.assertEqual([e['kind'] for e in evs], ['drop'])
        self.assertGreaterEqual(evs[0]['k_sep'], 300)

    def test_near_miss_when_back_on_within_limit(self):
        # Slows for 30 s, then closes the gap within a minute.
        evs = events_for(make_race({9: [(300, 330, -1.5), (330, 370, 1.5)]}), rank=10)
        self.assertEqual([e['kind'] for e in evs], ['near_miss'])
        self.assertIsNotNone(evs[0]['rejoin_k'])

    def test_breakaway_ahead_is_ignored(self):
        # Two riders ride off the front; a rider in the bunch didn't lose contact.
        evs = events_for(make_race({0: [(300, T, 1.5)], 1: [(300, T, 1.5)]}), rank=6)
        self.assertEqual(evs, [])

    def test_missing_a_real_split_counts(self):
        # The front 4 of 10 ride away; riders left behind lost the front group.
        changes = {i: [(300, T, 1.5)] for i in range(4)}
        evs = events_for(make_race(changes), rank=8)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]['kind'], 'drop')

    def test_split_near_finish_is_the_sprint(self):
        # Course ends ~3.9 km after the split, so a split at t=300 (3.6 km) is in the last km.
        race = make_race({9: [(300, T, -1.5)]}, finish_km=4.2)
        evs = events_for(race, rank=10)
        self.assertEqual([e['kind'] for e in evs], ['finish'])

    def test_flicker_is_ignored(self):
        # The gap pokes over the threshold for ~4 s and closes again: not an event.
        race = make_race({9: [(300, 312, -1.5), (312, 318, 3.0)]})
        grid = grid_for(race)
        crossed = np.flatnonzero(grid.gap_ahead[9, 290:340] > ra.GROUP_GAP_SEC)
        self.assertTrue(0 < len(crossed) <= 4)          # it really did cross, briefly
        self.assertEqual(ra.detect_separations(grid, 9), [])


class TestDistanceRepair(unittest.TestCase):
    def test_clean_trace_untouched(self):
        v = np.full(400, 12.0)
        d = np.cumsum(v)
        orig = d.copy()
        self.assertEqual(ra._repair_distance(d, v), [])
        np.testing.assert_array_equal(d, orig)

    def test_jump_then_freeze_is_rebuilt_from_speed(self):
        # Route-alignment glitch: leap 1.2 km ahead, freeze, then drift back to reality.
        v = np.full(600, 12.0)
        truth = np.cumsum(v)
        d = truth.copy()
        d[200:300] = truth[200] + 1200.0                   # jumped ahead and frozen
        d[300:400] = np.linspace(truth[200] + 1200.0, truth[400], 100)   # drifting back
        repairs = ra._repair_distance(d, v)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]['start_k'], 200)
        self.assertLess(np.max(np.abs(d - truth)), 15.0)   # back within the rejoin tolerance

    def test_never_recovers_is_left_and_reported(self):
        v = np.full(400, 12.0)
        d = np.cumsum(v)
        d[200:] += 1200.0                                   # permanent offset
        repairs = ra._repair_distance(d, v)
        self.assertEqual(len(repairs), 1)
        self.assertIsNone(repairs[0]['end_k'])

    def test_grid_records_repairs(self):
        race = make_race()
        df = race.riders[3].data
        df.loc[200:259, 'distance_km'] = df.loc[200, 'distance_km'] + 1.2   # jump + freeze
        grid = grid_for(race)
        self.assertEqual(len(grid.repairs), 1)
        self.assertEqual(grid.repairs[0]['rank'], 4)
        # After repair the rider is back in the bunch the whole time.
        self.assertTrue(np.all(grid.group_size[3, 150:400] == 10))


class TestReserveAndCP(unittest.TestCase):
    def test_reserve_stays_full_at_cp(self):
        bal = ra.w_prime_balance(np.full(600, 300.0), cp=300.0, w_prime=15000.0)
        self.assertAlmostEqual(bal.min(), 15000.0)

    def test_reserve_drains_above_cp_and_refills_below(self):
        power = np.concatenate([np.full(60, 400.0), np.full(300, 150.0)])
        bal = ra.w_prime_balance(power, cp=300.0, w_prime=15000.0)
        self.assertAlmostEqual(bal[59], 15000.0 - 100.0 * 60)
        self.assertGreater(bal[-1], bal[59])
        self.assertLessEqual(bal[-1], 15000.0)

    def test_cp_estimate_from_ideal_efforts(self):
        # Best efforts that exactly follow P = CP + W'/t are recovered.
        cp, wp = 300.0, 18000.0
        power = np.full(3000, 200.0)
        start = 0
        for dur in ra.CP_ESTIMATE_DURATIONS:
            power[start:start + dur] = cp + wp / dur
            start += dur + 400
        est = ra.estimate_cp(power)
        self.assertEqual(est['source'], 'race_estimate')
        self.assertAlmostEqual(est['cp'], cp, delta=3)
        self.assertAlmostEqual(est['w_prime'], wp, delta=600)

    def test_fit_keeps_w_prime_in_range_and_never_empty(self):
        power = np.concatenate([np.full(600, 380.0), np.full(600, 200.0)])
        cp, wp = ra.fit_cp_to_race(power, cp=250.0, w_prime=80000.0)
        self.assertLessEqual(wp, ra.W_PRIME_RANGE_J[1])
        self.assertGreaterEqual(ra.w_prime_balance(power, cp, wp).min(), 0)


class TestGoodMoves(unittest.TestCase):
    def _moves(self, race, rank):
        grid = grid_for(race)
        ri = grid.ranks.index(rank)
        wbal = ra.w_prime_balance(grid.power[ri], 300.0, 15000.0)
        return ra.detect_good_moves(grid, ri, wbal, race.elevation_profile)

    def test_attack_that_sticks_is_a_getaway(self):
        # Rider 1 goes 2 m/s faster for 2 minutes, then rides with the bunch's speed.
        moves = self._moves(make_race({0: [(300, 420, 2.0)]}), rank=1)
        self.assertEqual([m['kind'] for m in moves], ['getaway'])
        self.assertEqual(moves[0]['riders'], 9)
        self.assertGreater(moves[0]['final_gap_sec'], ra.GETAWAY_MIN_GAP_SEC)
        self.assertEqual(moves[0]['status'], 'ok')

    def test_attack_that_gets_caught_is_not_a_good_move(self):
        moves = self._moves(make_race({0: [(300, 420, 2.0), (420, 540, -2.0)]}), rank=1)
        self.assertEqual(moves, [])

    def test_solo_bridge(self):
        # Last rider starts 15 s behind and rides 1 m/s faster until it joins the bunch.
        offsets = [-i * SPACING_SEC * BASE_SPEED for i in range(9)] + [-(8 * SPACING_SEC + 15.0) * BASE_SPEED]
        moves = self._moves(make_race({9: [(0, 180, 1.0)]}, start_offsets_m=offsets), rank=10)
        self.assertEqual([m['kind'] for m in moves], ['bridge'])
        self.assertTrue(moves[0]['solo'])
        self.assertGreaterEqual(moves[0]['gap_closed_sec'], 10)

    def test_sitting_in_while_others_close_is_not_your_bridge(self):
        # Group of 5 (riders 5–9) starts 15 s back; the front rider of it does the chasing.
        offsets = ([-i * SPACING_SEC * BASE_SPEED for i in range(5)]
                   + [-(4 * SPACING_SEC + 15.0 + j * SPACING_SEC) * BASE_SPEED for j in range(5)])
        changes = {i: [(0, 180, 1.0)] for i in range(5, 10)}
        moves = self._moves(make_race(changes, start_offsets_m=offsets), rank=10)
        self.assertEqual(moves, [])


class TestHangOn(unittest.TestCase):
    def test_group_easing_soon_after_drop(self):
        grid = grid_for(make_race())
        cp = 300.0
        pack = np.full(T, 250.0)
        pack[300:330] = 450.0                             # surge lasting 30 s after the split
        wbal = np.full(T, 10000.0)
        ho = ra._hang_on(grid, 0, 300, pack, wbal, cp, 15000.0)
        self.assertTrue(ho['eased'])
        self.assertLessEqual(ho['hang_sec'], 45)          # trailing 30 s average drops under CP
        self.assertGreaterEqual(ho['low_j'], 0)

    def test_group_never_eases(self):
        grid = grid_for(make_race())
        ho = ra._hang_on(grid, 0, 300, np.full(T, 400.0), np.full(T, 5000.0), 300.0, 15000.0)
        self.assertFalse(ho['eased'])


class TestTextHelpers(unittest.TestCase):
    def test_ordinals(self):
        self.assertEqual([ra._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 111)],
                         ['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '23rd', '111th'])

    def test_durations(self):
        self.assertEqual(ra._dur(45), '45s')
        self.assertEqual(ra._dur(162), '2:42')

    def test_fit_count(self):
        self.assertEqual(ra._fit_count({'riders': 6, 'feasible_share': 0.0}),
                         'none of their efforts would have fit your reserve')
        self.assertEqual(ra._fit_count({'riders': 6, 'feasible_share': 4 / 6}, short=True),
                         '4 of their 6 efforts would have fit')
        self.assertEqual(ra._fit_count({'riders': 2, 'feasible_share': 1.0}, short=True),
                         'both of their efforts would have fit')

    def test_work_vs_you(self):
        self.assertEqual(ra._work_vs_you(0.2), 'about the same work as the riders around you')
        self.assertEqual(ra._work_vs_you(2.0, 'you'), '2.0 kJ more work than you')
        self.assertEqual(ra._work_vs_you(-1.4), '1.4 kJ less work than the riders around you')

    def test_margin(self):
        sel = {'intense': True, 'needed_w': 300.0, 'actual_w': 300.0}
        self.assertEqual(ra._margin(sel), 'none')
        self.assertEqual(ra._margin(dict(sel, actual_w=290.0)), 'under')
        self.assertEqual(ra._margin(dict(sel, actual_w=250.0)), 'well_under')
        self.assertEqual(ra._margin(dict(sel, actual_w=330.0)), 'over')
        self.assertIsNone(ra._margin(dict(sel, intense=False)))

    def test_slide_judgement(self):
        drift = {'from_pos': 15, 'to_pos': 33, 'from_size': 34, 'to_size': 33}
        hold = {'min_w_prime_j': -2500.0, 'w_prime_at_start_j': 9000.0, 'extra_kj': 6.5}
        self.assertEqual(ra._slide_judgement(drift, hold)['kind'], 'forced')
        self.assertEqual(ra._slide_judgement(drift, dict(hold, min_w_prime_j=3000.0))['kind'], 'choice')
        self.assertEqual(ra._slide_judgement(drift, dict(hold, min_w_prime_j=-300.0))['kind'], 'edge')
        self.assertIsNone(ra._slide_judgement(None, hold))

    def test_unrealistic_targets(self):
        self.assertTrue(ra._unrealistic(1475, 290))
        self.assertFalse(ra._unrealistic(455, 290))


class TestAnalyzeRider(unittest.TestCase):
    def test_drop_produces_readable_card(self):
        race = make_race({9: [(300, T, -1.5)]})
        result = ra.analyze_rider(grid_for(race), race, rank=10, cp=300, w_prime=15000)
        json.dumps(result)                               # JSON-safe (no NaN / numpy types)
        self.assertEqual(result['cp_model']['source'], 'user')
        drops = [e for e in result['events'] if e['kind'] == 'drop']
        self.assertEqual(len(drops), 1)
        ev = drops[0]
        for key in ('summary', 'story', 'options', 'advice', 'findings'):
            self.assertIn(key, ev)
        self.assertTrue(ev['summary'])
        self.assertTrue(all(o['status'] in ('ok', 'borderline', 'no') for o in ev['options']))
        self.assertIn('data_repairs', result)
        self.assertIn('good_moves', result)

    def test_rider_who_stayed_on_has_no_events(self):
        race = make_race({9: [(300, T, -1.5)]})
        result = ra.analyze_rider(grid_for(race), race, rank=1)
        self.assertEqual(result['events'], [])
        self.assertEqual(result['cp_model']['source'] in ('race_estimate', 'race_fallback'), True)

    def test_unknown_rider(self):
        race = make_race()
        with self.assertRaises(ValueError):
            ra.analyze_rider(grid_for(race), race, rank=99)


class TestAnalysisApi(unittest.TestCase):
    def setUp(self):
        import app as app_module
        self.app_module = app_module
        self.client = app_module.app.test_client()
        self.race_id = 'race_data_synthetic_test'
        app_module._race_data_cache[self.race_id] = make_race({9: [(300, T, -1.5)]})

    def tearDown(self):
        self.app_module._race_data_cache.pop(self.race_id, None)
        for key in [k for k in self.app_module._race_grid_cache if k[0] == self.race_id]:
            self.app_module._race_grid_cache.pop(key)

    def test_analysis_endpoint(self):
        resp = self.client.get(f'/api/race/analysis/{self.race_id}/10?cp=300&w_prime=15000')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['rank'], 10)
        self.assertEqual([e['kind'] for e in data['events']], ['drop'])

    def test_gap_override(self):
        resp = self.client.get(f'/api/race/analysis/{self.race_id}/10?gap_sec=1.0')
        self.assertEqual(resp.get_json()['gap_sec'], 1.0)

    def test_unknown_rank_is_404(self):
        resp = self.client.get(f'/api/race/analysis/{self.race_id}/99')
        self.assertEqual(resp.status_code, 404)

    def test_faq_quotes_live_thresholds(self):
        html = self.client.get('/race-replay').get_data(as_text=True)
        self.assertIn('What counts as a drop or a near miss?', html)
        self.assertIn(f"{ra.GROUP_GAP_SEC:g} s", html)
        self.assertIn(f"{ra.EFFORT_MIN_RESERVE_J / 1000:g} kJ", html)


if __name__ == '__main__':
    unittest.main()
