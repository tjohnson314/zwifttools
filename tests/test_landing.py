import pytest

import app as app_module


@pytest.mark.parametrize("query", ["", "?tab=tools", "?tab=unknown"])
def test_landing_shows_current_tools_by_default(query, monkeypatch):
    monkeypatch.setattr(app_module, "_surface_map_dev_available", lambda: False)

    response = app_module.app.test_client().get(f"/{query}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'href="/" aria-current="page">Tools</a>' in html
    assert 'href="/?tab=archived"' in html
    assert 'href="/zrl/leaderboard"' not in html
    for path in (
        "/bike-comparison", "/race-replay", "/surface-map", "/my-activities",
        "/power", "/ttt-analysis", "/ride-compare", "/critical-power",
        "/tt-pacing", "/ttt-pacing", "/top-performers",
    ):
        assert f'href="{path}" class="tool-card"' in html
    assert 'href="/surface-map-dev"' not in html
    assert 'href="/top-performers-tt"' not in html


def test_archived_tab_contains_only_zrl_tool():
    response = app_module.app.test_client().get("/?tab=archived")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'href="/?tab=archived" aria-current="page">Archived</a>' in html
    assert 'href="/" >Tools</a>' in html
    assert 'href="/zrl/leaderboard" class="tool-card"' in html
    assert html.count('class="tool-card"') == 1
    assert "Archived tool:" in html


def test_landing_preserves_dev_tool_and_authentication(monkeypatch):
    monkeypatch.setattr(app_module, "_surface_map_dev_available", lambda: True)
    client = app_module.app.test_client()

    assert b'Log in with Zwift' in client.get("/").data
    with client.session_transaction() as session:
        session["tokens"] = {}

    assert b'href="/surface-map-dev"' in client.get("/").data
    for path in ("/", "/?tab=archived"):
        html = client.get(path).get_data(as_text=True)
        assert "Logged in" in html
        assert 'href="/auth/logout"' in html


def test_archived_zrl_page_keeps_existing_url_and_controls(monkeypatch):
    monkeypatch.setattr(app_module, "get_route_info", lambda route_id: None)

    response = app_module.app.test_client().get(
        "/zrl/leaderboard?subgroup_id=7354731")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "ZRL Leaderboard (Archived)" in html
    assert 'href="/?tab=archived" class="back-link"' in html
    assert 'id="subgroup-input" value="7354731"' in html
    assert 'id="load-btn"' in html
    assert 'id="refetch-btn"' in html
