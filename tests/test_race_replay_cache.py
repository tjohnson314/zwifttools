import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from shared import data_fetcher


@pytest.fixture
def race_api(monkeypatch):
    activity = {
        'eventInfo': {'id': 123, 'eventSubGroupId': 456, 'name': 'Test race'},
        'startDate': '2026-10-23T12:00:00Z',
    }
    monkeypatch.setattr(data_fetcher, 'get_activity_details', Mock(return_value=(activity, None)))
    monkeypatch.setattr(data_fetcher, '_request_with_retry', Mock(return_value=Mock(
        status_code=200, json=lambda: {'eventSubgroups': [{'id': 456}]},
    )))
    monkeypatch.setattr(data_fetcher, 'get_event_subgroups', Mock(return_value=(
        [{'id': 456, 'label': 'A'}], None,
    )))
    monkeypatch.setattr(data_fetcher.blob_storage, 'race_exists_in_blob', Mock(return_value=False))
    monkeypatch.setattr(data_fetcher.blob_storage, 'upload_race_dir', Mock())
    telemetry = {
        'timeInSec': [0, 1, 2],
        'powerInWatts': [100, 200, 300],
        'heartRate': [120, 121, 122],
        'cadencePerMin': [80, 81, 82],
        'speedInCmPerSec': [1000, 1100, 1200],
        'distanceInCm': [0, 1000, 2100],
        'altitudeInCm': [0, 0, 0],
    }
    fetch = Mock(return_value=(telemetry, {
        'startDate': activity['startDate'], 'routeId': 99,
    }, None))
    monkeypatch.setattr(data_fetcher, 'fetch_rider_telemetry', fetch)
    return fetch


def participant(activity_id, rank=1, name='Rider'):
    return {'activity_id': activity_id, 'rank': rank, 'name': name, 'player_id': rank}


def fetch_race(tmp_path, all_subgroups=False, **kwargs):
    function = (data_fetcher.fetch_all_subgroups_from_activity if all_subgroups
                else data_fetcher.fetch_race_from_activity)
    result = function('source', {}, output_base_dir=str(tmp_path), **kwargs)
    assert result[0] is not None, result
    return tmp_path / 'race_data_456'


@pytest.mark.parametrize('all_subgroups', [False, True])
def test_refreshes_roster_but_reuses_activities(tmp_path, monkeypatch, race_api, all_subgroups):
    # Real Zwift IDs exceed exact floating-point integer precision.
    first_id = '2243088276573454336'
    entries = Mock(return_value=([participant(first_id)], None))
    monkeypatch.setattr(data_fetcher, 'get_race_entries', entries)
    directory = fetch_race(tmp_path, all_subgroups)
    (directory / 'cleaned_cache.json').write_text('{}')
    (directory / 'cleaned_data').mkdir()
    (directory / 'cleaned_data' / 'old.parquet').write_text('old')

    entries.return_value = ([
        participant('2243088276573454337'),
        participant(first_id, rank=2, name='Updated name'),
    ], None)
    progress = Mock()
    race_api.reset_mock()
    fetch_race(tmp_path, all_subgroups, progress_callback=progress)

    assert entries.call_count == 2
    race_api.assert_called_once_with('2243088276573454337', {})
    summary = pd.read_csv(directory / 'complete_race_summary.csv', dtype={'activity_id': str})
    assert summary['activity_id'].tolist() == ['2243088276573454337', first_id]
    assert summary['rank'].tolist() == [1, 2]
    assert summary['name'].tolist() == ['Rider', 'Updated name']
    assert (directory / f'rank2_{first_id}.csv').exists()
    assert not (directory / 'cleaned_cache.json').exists()
    assert not (directory / 'cleaned_data').exists()
    assert progress.called


@pytest.mark.parametrize('all_subgroups', [False, True])
def test_failed_activities_are_retried_and_force_refresh_bypasses_cache(
    tmp_path, monkeypatch, race_api, all_subgroups,
):
    monkeypatch.setattr(data_fetcher, 'get_race_entries', Mock(return_value=(
        [participant('101')], None,
    )))
    success = race_api.return_value
    race_api.return_value = (None, None, 'No fitness data available')
    fetch_race(tmp_path, all_subgroups)
    race_api.return_value = success
    fetch_race(tmp_path, all_subgroups)
    assert race_api.call_count == 2
    fetch_race(tmp_path, all_subgroups)
    assert race_api.call_count == 2
    fetch_race(tmp_path, all_subgroups, force_refresh=True)
    assert race_api.call_count == 3


@pytest.mark.parametrize('all_subgroups', [False, True])
def test_blob_snapshot_only_supplies_activity_cache(tmp_path, monkeypatch, race_api, all_subgroups):
    entries = Mock(return_value=([participant('101')], None))
    monkeypatch.setattr(data_fetcher, 'get_race_entries', entries)
    directory = fetch_race(tmp_path, all_subgroups)
    snapshot = {path.name: path.read_bytes() for path in directory.iterdir()}
    for path in directory.iterdir():
        path.unlink()
    monkeypatch.setattr(data_fetcher.blob_storage, 'race_exists_in_blob', Mock(return_value=True))

    def download(race_id, output_dir):
        for name, content in snapshot.items():
            (Path(output_dir) / name).write_bytes(content)
        return True

    monkeypatch.setattr(data_fetcher.blob_storage, 'download_race_dir', download)
    entries.return_value = ([participant('101'), participant('102', rank=2)], None)
    race_api.reset_mock()
    fetch_race(tmp_path, all_subgroups)
    race_api.assert_called_once_with('102', {})
    assert entries.call_count == 2


def test_cleaner_ignores_files_outside_current_successful_roster(tmp_path):
    from race_replay.data_cleaner import clean_race_data

    pd.DataFrame([
        {**participant('101'), 'status': 'No fitness data available'},
    ]).to_csv(tmp_path / 'complete_race_summary.csv', index=False)
    for name in ('rank1_101.csv', 'rank2_102.csv'):
        (tmp_path / name).write_text('stale')
        (tmp_path / name.replace('.csv', '_raw.json')).write_text('{}')
    with pytest.raises(ValueError, match='No rider data found'):
        clean_race_data(tmp_path, cache=False)


def test_saved_category_can_refresh_from_another_categorys_source_activity(tmp_path, monkeypatch, race_api):
    entries = Mock(return_value=([participant('101')], None))
    monkeypatch.setattr(data_fetcher, 'get_race_entries', entries)
    output_dir, message = data_fetcher.fetch_race_from_activity(
        'source', {}, output_base_dir=str(tmp_path), event_subgroup_id=789,
    )
    assert output_dir is not None, message
    assert Path(output_dir).name == 'race_data_789'
    entries.assert_called_once_with(789, {})
    assert (Path(output_dir) / 'complete_race_summary.csv').exists()
    assert not (tmp_path / 'race_data_456').exists()


@pytest.mark.parametrize('authenticated,just_fetched', [(False, False), (True, False), (True, True)])
def test_saved_replay_refreshes_only_when_authenticated(
    tmp_path, monkeypatch, authenticated, just_fetched,
):
    import app as app_module
    from race_replay import data_cleaner

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'race_data' / 'race_data_456'
    directory.mkdir(parents=True)
    (directory / 'race_meta.json').write_text(json.dumps({
        'source_activity_id': '101', 'event_subgroup_id': 456,
    }))
    monkeypatch.setattr(app_module, 'get_headers', lambda: {'Authorization': 'test'} if authenticated else None)
    fetch = Mock(return_value=(str(directory), 'Loaded'))
    monkeypatch.setattr(app_module, 'fetch_race_from_activity', fetch)
    race = Mock(
        route_name='Test', route_slug='test', route_geometry_source='test',
        finish_line_km=10, min_time=0, max_time=100, riders=[],
    )
    monkeypatch.setattr(data_cleaner, 'clean_race_data', Mock(return_value=race))
    monkeypatch.setattr(app_module.blob_storage, 'upload_race_dir', Mock())
    monkeypatch.setattr(app_module, '_race_data_cache', {})
    response = app_module.app.test_client().post('/api/race/load', json={
        'race_id': 'race_data_456', 'refresh_participants': not just_fetched,
    })
    assert response.status_code == 200
    assert fetch.call_count == int(authenticated and not just_fetched)
    if fetch.called:
        assert fetch.call_args.kwargs['event_subgroup_id'] == 456


def test_authenticated_refresh_error_does_not_load_stale_roster(tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'race_data' / 'race_data_456'
    directory.mkdir(parents=True)
    (directory / 'race_meta.json').write_text(json.dumps({'source_activity_id': '101'}))
    monkeypatch.setattr(app_module, 'get_headers', lambda: {'Authorization': 'test'})
    monkeypatch.setattr(app_module, 'fetch_race_from_activity', Mock(return_value=(None, 'API unavailable')))
    response = app_module.app.test_client().post('/api/race/load', json={'race_id': 'race_data_456'})
    assert response.status_code == 502
    assert response.json == {'error': 'API unavailable'}


def test_saved_event_refreshes_all_categories_and_manifest(tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'race_data'
    event = root / 'race_event_123'
    event.mkdir(parents=True)
    (event / 'event_manifest.json').write_text(json.dumps({
        'subgroups': [{'race_id': 'race_data_456'}],
    }))
    subgroup = root / 'race_data_456'
    subgroup.mkdir()
    (subgroup / 'race_meta.json').write_text(json.dumps({'source_activity_id': '101'}))
    monkeypatch.setattr(app_module, 'get_headers', lambda: {'Authorization': 'test'})
    fetch = Mock(return_value=(
        [(str(subgroup), 'A', 456), (str(root / 'race_data_789'), 'B', 789)],
        123, 'Event', None,
    ))
    monkeypatch.setattr(data_fetcher, 'fetch_all_subgroups_from_activity', fetch)
    merged = Mock(return_value={'success': True})
    monkeypatch.setattr(app_module, '_load_multi_subgroup_race', merged)
    monkeypatch.setattr(app_module, '_race_data_cache', {})
    response = app_module.app.test_client().post('/api/race/load', json={'race_id': 'race_event_123'})
    assert response.status_code == 200
    fetch.assert_called_once_with('101', {'Authorization': 'test'}, output_base_dir='race_data')
    merged.assert_called_once_with('race_event_123')
    manifest = json.loads((event / 'event_manifest.json').read_text())
    assert [sg['subgroup_id'] for sg in manifest['subgroups']] == [456, 789]
