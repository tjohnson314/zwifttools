from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app as app_module
from race_replay import data_cleaner
from race_replay.data_cleaner import RiderData


@pytest.fixture
def comparison_api(monkeypatch):
    race = SimpleNamespace(riders=[], world='WATOPIA', finish_line_km=1)
    segments = []
    monkeypatch.setattr(app_module, '_get_race_data', lambda _: (race, None))
    monkeypatch.setattr(app_module, '_race_rules', lambda _: set())
    monkeypatch.setattr(app_module, '_race_data_cache', {})
    monkeypatch.setattr(data_cleaner, 'build_replay_route_segments', lambda *_: segments)
    return app_module.app.test_client(), race, segments


def rider(frame, rank=1, finish=10):
    return RiderData(
        rank=rank, activity_id=f'900719925474099{rank}', name=f'Rider {rank}',
        team='', data=frame.set_index('time_sec'), finish_time_sec=finish,
        weight_kg=75, height_cm=180, activity_start_time='2026-11-25T12:00:00Z',
    )


def test_curves_use_exact_dashboard_intervals_and_only_each_riders_race(comparison_api, monkeypatch):
    client, race, _ = comparison_api
    times = np.arange(-2, 14)
    frame = pd.DataFrame({
        'time_sec': times, 'power_watts': np.where((times < 0) | (times > 10), 9999, 2400),
    })
    delayed = frame.copy()
    delayed['time_sec'] += 300
    delayed.loc[delayed['time_sec'] >= 300, 'power_watts'] = -50
    race.riders = [rider(frame), rider(delayed, rank=2, finish=310)]
    monkeypatch.setitem(app_module._race_data_cache, 'test_subgroup_timing', {
        2: {'offset_seconds': 300},
    })

    response = client.get('/api/race/comparisons/test')
    assert response.status_code == 200
    data = response.get_json()
    intervals = app_module._build_power_intervals_seconds()
    assert data['durations_sec'] == intervals
    assert data['durations_label'] == [app_module._format_interval_label(sec) for sec in intervals]
    for row, expected in zip(data['riders'], (2400, -50)):
        assert row['peak_watts'] == [expected if sec <= 11 else None for sec in intervals]
    assert data['riders'][0]['activity_start_time'] == '2026-11-25T12:00:00Z'


def test_power_windows_do_not_bridge_missing_samples_or_clock_gaps(comparison_api):
    client, race, _ = comparison_api
    race.riders = [rider(pd.DataFrame({
        'time_sec': [0, 1, 2, 3, 5, 6],
        'power_watts': [-50, -50, np.nan, 3000, 2000, 2000],
    }))]
    data = client.get('/api/race/comparisons/test').get_json()
    peaks = dict(zip(data['durations_sec'], data['riders'][0]['peak_watts']))
    assert peaks[1] == 3000
    assert peaks[2] == 2000
    assert peaks[3] is None


@pytest.mark.parametrize('geometry_offset_m', [0, .01])
def test_segment_occurrences_keep_pass_identity_when_first_pass_is_missing(comparison_api, geometry_offset_m):
    client, race, segments = comparison_api
    path = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    points = np.asarray([(-10, 0), *path, *path[1:], (10, 0)])
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    frame = pd.DataFrame({
        'time_sec': np.arange(len(points), dtype=float),
        'distance_km': distances / 1000, 'lat': points[:, 0] / 111_320,
        'lng': points[:, 1] / 111_320, 'power_watts': 1500,
        'speed_kmh': 36, 'altitude_m': -np.arange(len(points), dtype=float),
    })
    race.finish_line_km = .82
    race.riders = [rider(frame), rider(frame.iloc[4:], rank=2)]
    for occurrence in range(3):
        geometry = np.asarray(path, dtype=float) / 111_320
        geometry[2, 0] += occurrence * geometry_offset_m / 111_320
        segments.append({
            'segment_id': '1:7', 'name': 'Loop', 'type': 'segment', 'pass': occurrence + 1, 'lap': 1,
            'latlng': geometry.tolist(),
            'start_distance_km': .01 + occurrence * .4,
            'end_distance_km': .41 + occurrence * .4,
        })

    response = client.get('/api/race/comparisons/test')
    assert response.status_code == 200
    data = response.get_json()
    assert [segment['pass'] for segment in data['segments']] == [1, 2]
    first, late = data['riders']
    assert [row['occurrence_id'] for row in first['traversals']] == ['0', '1']
    assert [row['occurrence_id'] for row in late['traversals']] == ['1']
    row = late['traversals'][0]
    assert row['pass'] == 2
    assert row['start_time_sec'] == 5
    assert row['total_time_sec'] == 4
    assert row['distance_m'] == pytest.approx(400)
    assert row['elevation_change_m'] == pytest.approx(-4)
    assert row['avg_power'] == 1500
    assert row['normalized_power'] == 1500
    assert row['initial_speed_kmh'] == 36
    assert row['avg_draft_watts'] < 0


def test_incomplete_gps_does_not_invent_a_traversal_or_block_power(comparison_api):
    client, race, segments = comparison_api
    segments.append({
        'segment_id': '1:7', 'name': 'Sprint', 'pass': 1, 'lap': 1,
        'start_distance_km': .01, 'end_distance_km': .11,
        'latlng': [[0, 0], [100 / 111_320, 0]],
    })
    frame = pd.DataFrame({
        'time_sec': np.arange(13), 'power_watts': 300,
        'distance_km': np.arange(13) / 100, 'lat': np.nan, 'lng': 0,
    })
    race.riders = [rider(frame)]
    data = client.get('/api/race/comparisons/test').get_json()
    assert data['riders'][0]['peak_watts'][0] == 300
    assert data['riders'][0]['traversals'] == []
    assert any('GPS telemetry' in warning for warning in data['warnings'])
    race.riders[0].data = race.riders[0].data.iloc[:0]
    data = client.get('/api/race/comparisons/test').get_json()
    assert all(value is None for value in data['riders'][0]['peak_watts'])
    assert any('no recorded race power' in warning for warning in data['warnings'])


@pytest.mark.parametrize('finish,expected_count', [(10.5, 1), (10.49, 1), (9, 0)])
def test_segment_finish_preserves_crossing_time_with_one_sample_finish_clock_tolerance(
    comparison_api, finish, expected_count,
):
    client, race, segments = comparison_api
    race.finish_line_km = .105
    segments.append({
        'segment_id': '1:7', 'name': 'Finish Sprint', 'pass': 1, 'lap': 1,
        'start_distance_km': .005, 'end_distance_km': .105,
        'latlng': [[0, 0], [100 / 111_320, 0]],
    })
    times = np.arange(13)
    frame = pd.DataFrame({
        'time_sec': times, 'distance_km': times / 100,
        'lat': (times * 10 - 5) / 111_320, 'lng': 0,
        'power_watts': 250, 'speed_kmh': 36,
    })
    race.riders = [rider(frame, finish=finish)]
    data = client.get('/api/race/comparisons/test').get_json()
    rows = data['riders'][0]['traversals']
    assert len(rows) == expected_count
    if rows:
        assert rows[0]['start_time_sec'] == pytest.approx(.5)
        assert rows[0]['end_time_sec'] == pytest.approx(10.5)
        assert rows[0]['total_time_sec'] == pytest.approx(10)


def test_comparison_data_errors_are_explicit(comparison_api, monkeypatch, caplog):
    client, _, _ = comparison_api

    def fail(*_):
        raise ValueError('Invalid segment geometry')

    monkeypatch.setattr(data_cleaner, 'build_replay_route_segments', fail)
    response = client.get('/api/race/comparisons/test')
    assert response.status_code == 422
    assert response.get_json()['error'] == 'Invalid segment geometry'
    assert 'Cannot compare race test' in caplog.text


@pytest.mark.parametrize('offset,expected_count', [(25, 1), (55, 0)])
def test_known_race_course_uses_replay_gps_corridor_for_interior_gates(
    comparison_api, offset, expected_count,
):
    client, race, segments = comparison_api
    segments.append({
        'segment_id': '1:7', 'name': 'Sprint', 'pass': 1, 'lap': 1,
        'start_distance_km': .005, 'end_distance_km': .305,
        'latlng': [[0, 0], [300 / 111_320, 0]],
    })
    x = np.arange(-5, 316, 10)
    points = np.column_stack([x, np.where((x > 80) & (x < 220), offset, 0)])
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    frame = pd.DataFrame({
        'time_sec': np.arange(len(points)), 'distance_km': distances / 1000,
        'lat': points[:, 0] / 111_320, 'lng': points[:, 1] / 111_320,
        'power_watts': 300, 'speed_kmh': 36,
    })
    race.riders = [rider(frame, finish=32)]
    data = client.get('/api/race/comparisons/test').get_json()
    assert len(data['riders'][0]['traversals']) == expected_count
