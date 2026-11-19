import json

import app as app_module


def test_dirt_roster_page_is_public():
    response = app_module.app.test_client().get('/dirt?team=1854')
    assert response.status_code == 200
    assert b'DIRT Racing Series' in response.data
    assert b'SEASON 14' in response.data
    assert b'id="league"' in response.data
    assert b'id="velo-tab"' in response.data
    assert b'id="power-tab"' in response.data


def test_dirt_roster_api_serves_saved_data_without_a_network_call(tmp_path, monkeypatch):
    path = tmp_path / 'roster.json'
    payload = {'season': 14, 'complete': False, 'teams': [{'name': 'RtB Yaks'}]}
    path.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setattr(app_module, '_DIRT_ROSTER_FILE', path)
    response = app_module.app.test_client().get('/api/dirt/roster')
    assert response.status_code == 200
    assert response.get_json() == payload


def test_missing_roster_returns_an_explicit_unavailable_status(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, '_DIRT_ROSTER_FILE', tmp_path / 'missing.json')
    response = app_module.app.test_client().get('/api/dirt/roster')
    assert response.status_code == 503
    assert 'not been collected' in response.get_json()['error']


def test_stats_api_serves_saved_raw_metrics(tmp_path, monkeypatch):
    path = tmp_path / 'stats.json'
    payload = {'season': 14, 'complete': False, 'riders': {'7681059': {'velo': {'overall': 639.123456, 'sprint': -18.123456}, 'power': {'5': {'live': {'w': 9999.123456}}}}}}
    path.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setattr(app_module, '_DIRT_STATS_FILE', path)
    response = app_module.app.test_client().get('/api/dirt/stats')
    assert response.status_code == 200
    assert response.get_json() == payload


def test_missing_stats_does_not_break_the_roster_page(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, '_DIRT_STATS_FILE', tmp_path / 'missing.json')
    client = app_module.app.test_client()
    assert client.get('/dirt').status_code == 200
    response = client.get('/api/dirt/stats')
    assert response.status_code == 503
    assert 'not been collected' in response.get_json()['error']