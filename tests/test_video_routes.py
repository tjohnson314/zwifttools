import json

import pytest

from shared import surface_map
from tools.extract_zwift_routes import extract_video_routes


@pytest.mark.parametrize("key", ["", 'test<&"key'])
def test_surface_map_page_passes_an_escaped_browser_api_key(monkeypatch, key):
    from app import app
    from markupsafe import escape

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", key)
    with app.test_client() as client:
        response = client.get("/surface-map")
    assert response.status_code == 200
    assert f'name="google-maps-api-key" content="{escape(key)}"' in response.get_data(as_text=True)
    assert 'id="satellite-map"' in response.get_data(as_text=True)


@pytest.fixture
def video_world(tmp_path, monkeypatch):
    monkeypatch.setattr(surface_map, "SURFACE_DIR", tmp_path / "surfaces")
    monkeypatch.setattr(surface_map, "ROUTE_DIR", tmp_path / "routes")
    surface_map.SURFACE_DIR.mkdir()
    surface_map.ROUTE_DIR.mkdir()
    surface_map._surface_path(1000).write_text(json.dumps({
        "segments": [{"style": "NORMAL", "x": [0, 100, 200], "z": [0, 20, 30]}],
    }), encoding="utf-8")
    surface_map._route_path(1000).write_text(json.dumps({
        "routes": [{
            "name": "Glen Etive", "nameHash": 2010849247,
            "distance_m": 200, "ascent_m": 15, "sport_type": 1,
            "route": {"d": [0, 100, 200], "alt": [-10, 5, -20],
                      "x": [0, 100, 200], "z": [0, 20, 30],
                      "surface": ["Tarmac", "Tarmac", "Tarmac"]},
        }],
    }), encoding="utf-8")
    caches = (surface_map._load_surface_world, surface_map._load_route_world,
              surface_map._projection)
    for function in caches:
        function.cache_clear()
    yield
    for function in caches:
        function.cache_clear()


def test_video_routes_are_a_separate_surface_map_world(video_world):
    worlds = surface_map.list_worlds()
    assert worlds == [{
        "mapID": 1000, "name": "Video Routes", "route_count": 1,
        "segment_count": 1, "surfaces": ["Tarmac"], "has_map": False,
    }]
    world = surface_map.get_world_surfaces(1000)
    assert world["name"] == "Video Routes"
    assert world["routes"][0]["name"] == "Glen Etive"
    route = surface_map.get_route(1000, 2010849247)
    assert route["world_name"] == "Video Routes"
    assert route["route"]["alt"] == [-10, 5, -20]
    assert surface_map.get_route(1000, 0) is None


def test_video_cache_export_keeps_geometry_gps_and_named_surface_markers(tmp_path):
    cache = tmp_path / "mochi"
    cache.mkdir()
    (cache / "123").write_bytes(b'<VERSION>0</VERSION>'
        b'<route name="etivedown" nameHash="2010849247" mapID="14" '
        b'distanceInMeters="200" ascentInMeters="15" sportType="1"/>'
        b'<highrescheckpoint>'
        b'<entry road="27060" time="0" x="0" y="-1000" z="0"/>'
        b'<entry road="27060" time="0.5" x="10000" y="500" z="2000"/>'
        b'<entry road="27060" time="1" x="20000" y="-2000" z="3000"/>'
        b'</highrescheckpoint>')
    (cache / "456").write_bytes(b'<world><roads><road><id>27060</id>'
        b'<roadName>etivedown</roadName><defaultStyleName>NORMAL</defaultStyleName>'
        b'<ent type="ENTITY_TYPE_ROADNODE" m_pos="{0,-1000,0}" '
        b'm_lat="56.63" m_lon="-4.89" m_videoSeconds="-0.1" m_videoDist="0"/>'
        b'<ent type="ENTITY_TYPE_ROADNODE" m_pos="{20000,-2000,3000}" '
        b'm_lat="56.62" m_lon="-4.98" m_videoSeconds="600" m_videoDist="20000"/>'
        b'<ent type="ENTITY_TYPE_ROADMARKER" m_roadTime1="0.25" '
        b'm_roadTime2="0.75" m_styleName="GRAVEL"/>'
        b'<ent type="ENTITY_TYPE_ROADMARKER" m_roadTime1="0.4" '
        b'm_roadTime2="0.6" m_styleName="WOODEN"/>'
        b'</road></roads></world>')
    (cache / "789").write_bytes(b'not XML: video bytes')
    routes_dir = tmp_path / "routes"
    surfaces_dir = tmp_path / "surfaces"
    assert extract_video_routes(str(cache), str(routes_dir), str(surfaces_dir)) == 1
    data = json.loads((routes_dir / "video_routes.json").read_text(encoding="utf-8"))
    route = data["routes"][0]
    assert data["world"] == "Video Routes"
    assert route["mapID"] == surface_map.VIDEO_ROUTES_MAP_ID
    assert route["source_mapID"] == 14
    assert route["name"] == "Glen Etive"
    assert route["route"]["alt"] == [-10, 5, -20]
    assert route["route"]["surface"] == ["Tarmac", "Wood", "Tarmac"]
    nodes = route["road_nodes"][0]["nodes"]
    assert nodes[0]["video_seconds"] == -0.1
    assert nodes[0]["lat"] == 56.63
    assert nodes[1]["video_distance_m"] == 200
    segments = json.loads((surfaces_dir / "video_routes.json").read_text(encoding="utf-8"))["segments"]
    assert [segment["surface"] for segment in segments] == [
        "Tarmac", "Gravel", "Wood", "Gravel", "Tarmac",
    ]
    assert segments[0]["x"][-1] == segments[1]["x"][0]


def test_empty_video_cache_does_not_overwrite_published_data(tmp_path):
    cache = tmp_path / "mochi"
    cache.mkdir()
    with pytest.raises(ValueError, match="No usable video routes"):
        extract_video_routes(str(cache), str(tmp_path / "routes"), str(tmp_path / "surfaces"))
    assert not (tmp_path / "routes").exists()


def test_published_video_routes_and_glen_etive_data():
    world = surface_map.get_world_surfaces(surface_map.VIDEO_ROUTES_MAP_ID)
    assert world["name"] == "Video Routes"
    assert len(world["routes"]) == 21
    assert set(world["colors"]) == {"Tarmac", "Dirt", "Gravel", "Wood"}
    for metadata in world["routes"]:
        route = surface_map.get_route(1000, metadata["nameHash"])
        assert route["bounds"]
        assert route["road_nodes"]
        for leg_name in ("leadin", "route"):
            leg = route[leg_name]
            if leg:
                assert len({len(leg[key]) for key in ("d", "alt", "x", "y", "surface")}) == 1
    glen = surface_map.get_route(1000, 2010849247)
    assert glen["name"] == "Glen Etive"
    assert glen["distance_m"] == pytest.approx(6221.02)
    assert glen["ascent_m"] == pytest.approx(38.28)
    assert len(glen["route"]["alt"]) == 1245
    assert len(glen["road_nodes"][0]["nodes"]) == 601
    assert glen["road_nodes"][0]["nodes"][0]["video_seconds"] < 0
    assert surface_map.get_world_surfaces(1)["name"] == "Watopia"