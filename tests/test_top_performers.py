import pytest

from app import app


@pytest.mark.parametrize("query", ["", "?bike=tt"])
def test_combined_top_performers_page(query):
    response = app.test_client().get(f"/top-performers{query}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert html.index('id="roadBtn"') < html.index('id="lvl0Btn"')
    assert html.index('id="ttBtn"') < html.index('id="lvl0Btn"')
    assert 'role="group" aria-labelledby="bikeTypeLabel"' in html
    assert 'src="/static/js/top_performers.js"' in html
    assert "TT-specific" in html


def test_legacy_tt_page_redirects_to_combined_tt_mode():
    client = app.test_client()
    response = client.get("/top-performers-tt")

    assert response.status_code == 302
    assert response.headers["Location"] == "/top-performers?bike=tt"
    assert client.get(response.headers["Location"]).status_code == 200


def test_top_performers_datasets_keep_distinct_types_and_shared_reference():
    client = app.test_client()
    road_response = client.get("/api/top_performers")
    tt_response = client.get("/api/top_performers_tt")

    assert road_response.status_code == tt_response.status_code == 200
    road = road_response.get_json()
    tt = tt_response.get_json()
    assert road["frames"] and tt["frames"]
    assert all(frame["type"] != "TT" for frame in road["frames"])
    assert all(frame["type"] == "TT" for frame in tt["frames"])
    assert road["referenceFrame"] == tt["referenceFrame"]
    assert road["referenceWheel"] == tt["referenceWheel"]
    assert any(wheel["cdaTT"] != wheel["cda"] for wheel in tt["wheels"])
