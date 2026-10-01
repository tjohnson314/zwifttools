import json
from unittest.mock import patch

import pandas as pd

import app as app_module
from race_replay.data_cleaner import CleanedRaceData, RiderData


def _subgroup_race(race_id, activity_id, activity_start_time):
    rider_data = pd.DataFrame(
        {
            'distance_km': [0.0, 1.0],
            'altitude_m': [10.0, 11.0],
        },
        index=pd.Index([0.0, 100.0], name='time_sec'),
    )
    rider = RiderData(
        rank=1,
        activity_id=activity_id,
        name=f'Rider {activity_id}',
        team='',
        data=rider_data,
        finish_time_sec=100.0,
        activity_start_time=activity_start_time,
    )
    elevation = pd.DataFrame({
        'distance_km': [0.0, 1.0],
        'altitude_m': [10.0, 11.0],
    })
    return CleanedRaceData(
        race_id=race_id,
        route_name='Test Route',
        finish_line_km=1.0,
        riders=[rider],
        elevation_profile=elevation,
        min_time=0.0,
        max_time=100.0,
        source_activity_id=activity_id,
    )


def test_chase_subgroups_use_shared_event_clock(tmp_path, monkeypatch):
    race_root = tmp_path / 'race_data'
    event_dir = race_root / 'race_event_123'
    event_dir.mkdir(parents=True)
    subgroup_specs = [
        ('race_data_10', 'D', 10, '2026-09-28T16:00:00.000+0000'),
        ('race_data_11', 'E', 11, '2026-09-28T16:05:00.000+0000'),
    ]
    manifest = {'event_id': 123, 'race_name': 'Test Chase', 'subgroups': []}
    races = {}
    for race_id, label, subgroup_id, start_time in subgroup_specs:
        subgroup_dir = race_root / race_id
        subgroup_dir.mkdir()
        (subgroup_dir / 'race_meta.json').write_text(json.dumps({
            'race_start_time': start_time,
        }))
        manifest['subgroups'].append({
            'race_id': race_id,
            'label': label,
            'subgroup_id': subgroup_id,
        })
        activity_start_time = (
            '2026-09-28T16:00:10.000+0000'
            if label == 'D'
            else '2026-09-28T16:05:10.000+0000'
        )
        races[race_id] = _subgroup_race(
            race_id, f'activity-{label}', activity_start_time
        )
    (event_dir / 'event_manifest.json').write_text(json.dumps(manifest))

    monkeypatch.chdir(tmp_path)
    for key in list(app_module._race_data_cache):
        if key.startswith('race_event_123'):
            app_module._race_data_cache.pop(key)

    with (
        patch(
            'race_replay.data_cleaner.clean_race_data',
            side_effect=lambda path, cache=True: races[path.name],
        ),
        patch('race_replay.data_cleaner.align_riders_to_elevation_profile'),
        patch.object(app_module, '_race_rules', return_value=set()),
        app_module.app.test_request_context(),
    ):
        response = app_module._load_multi_subgroup_race('race_event_123')
        data_response = app_module._build_race_data_response('race_event_123')

    assert response.get_json()['success'] is True
    merged = app_module._race_data_cache['race_event_123']
    assert merged.riders[0].data.index.tolist() == [0.0, 100.0]
    assert merged.riders[0].finish_time_sec == 100.0
    assert merged.riders[1].data.index.tolist() == [300.0, 400.0]
    assert merged.riders[1].finish_time_sec == 400.0
    assert merged.min_time == 0.0
    assert merged.max_time == 400.0
    assert app_module._race_data_cache['race_event_123_start_groups'] == [
        {
            'race_id': 'race_data_10',
            'label': 'D',
            'subgroup_id': 10,
            'start_time': '2026-09-28T16:00:00.000+0000',
            'offset_seconds': 0.0,
        },
        {
            'race_id': 'race_data_11',
            'label': 'E',
            'subgroup_id': 11,
            'start_time': '2026-09-28T16:05:00.000+0000',
            'offset_seconds': 300.0,
        },
    ]
    payload = data_response.get_json()
    assert payload['event_id'] == 123
    assert payload['race_start_time'] == '2026-09-28T16:00:00.000+0000'
    assert [group['offset_seconds'] for group in payload['start_groups']] == [
        0.0, 300.0,
    ]
    riders_by_category = {rider['category']: rider for rider in payload['riders']}
    assert riders_by_category['D']['start_offset_sec'] == 0.0
    assert riders_by_category['E']['start_offset_sec'] == 300.0
    assert riders_by_category['D']['is_late_joiner'] is False
    assert riders_by_category['E']['is_late_joiner'] is False