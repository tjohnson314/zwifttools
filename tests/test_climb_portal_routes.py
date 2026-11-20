import json

import numpy as np
import pytest

from shared import surface_map
from tools import extract_climb_portal_routes as portal


def portal_road(road_id=10009, start=0, finish=0.5, excluded=False, straight=False, marker=""):
    return f'''<world><roads><road>
        <metadata>
            <m_PortalRoadUserFacingName>Test climb {road_id}</m_PortalRoadUserFacingName>
            <m_PortalRoadInternalName>test_{road_id}</m_PortalRoadInternalName>
            <m_PortalRoadHash>{road_id}</m_PortalRoadHash>
            <m_PortalRoadCourseLength>99900</m_PortalRoadCourseLength>
            <m_PortalRoadCourseAscentF>-1250</m_PortalRoadCourseAscentF>
            <m_PortalRoadExcludeFromDictionary>{int(excluded)}</m_PortalRoadExcludeFromDictionary>
        </metadata>
        <id>{road_id}</id><IsPortalRoad>true</IsPortalRoad>
        <splineType>Bezier</splineType><looped>0</looped><defaultStyle>0</defaultStyle>
        <portalRoadEndGateRoadTime>{finish}</portalRoadEndGateRoadTime>
        <portalRoadPartitions><partition startTime="{start}" endTime="{finish}" gradient="0.1"/>
        </portalRoadPartitions>
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{0,-1000,0}}"
             m_tangentOut="{{0,9000,8000}}" m_bStraight="{int(straight)}"/>
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{10000,2000,0}}"
             m_tangentIn="{{0,-6000,8000}}"/>
        <ent type="ENTITY_TYPE_ROADNODE" m_pos="{{30000,-50000,0}}"/>
        {marker}
    </road></roads></world>'''.encode()


@pytest.mark.parametrize("straight", [False, True])
def test_portal_profile_preserves_native_elevation_and_handles_without_summit_tail(straight):
    climb = portal.parse_portal_climb(portal_road(straight=straight), "road_10009.xml")
    leg = climb["route"]
    assert leg["alt"][0] == -10
    assert leg["alt"][-1] == 20
    assert leg["x"][0] == 0 and leg["x"][-1] == 100
    assert max(leg["x"]) == 100
    assert max(leg["z"]) == pytest.approx(0 if straight else 60, abs=0.1)
    assert leg["d"][-1] == pytest.approx(100 if straight else 174.741908, abs=0.02)
    assert all(a <= b for a, b in zip(leg["d"], leg["d"][1:]))
    assert set(leg["surface"]) == {"Tarmac"}
    assert len({len(leg[key]) for key in ("d", "alt", "x", "z", "surface")}) == 1
    assert climb["distance_m"] == 999
    assert climb["ascent_m"] == -12.5
    assert climb["leadin"] is None


def test_portal_profile_starts_at_first_partition_not_the_road_endpoint():
    leg = portal.parse_portal_climb(portal_road(start=0.25), "road.xml")["route"]
    assert leg["x"][0] == pytest.approx(50)
    assert leg["z"][0] == pytest.approx(60)
    assert leg["alt"][0] == pytest.approx(16.25)
    assert leg["d"][0] == 0
    assert leg["x"][-1] == 100


def test_portal_exclusions_and_invalid_geometry_are_not_published():
    assert portal.parse_portal_climb(portal_road(excluded=True), "excluded.xml") is None
    with pytest.raises(ValueError, match="Invalid portal climb boundaries"):
        portal.parse_portal_climb(portal_road(start=0.75), "invalid.xml")
    marker = '<ent type="ENTITY_TYPE_ROADMARKER" m_style="11" m_roadTime1="0.6"/>'
    assert portal.parse_portal_climb(portal_road(marker=marker), "tail.xml") is not None
    with pytest.raises(ValueError, match="Unresolved portal surface"):
        portal.parse_portal_climb(portal_road(marker=marker.replace("0.6", "0.1")), "unknown.xml")


def test_portal_export_and_surface_map_group_share_only_the_ascent_geometry(tmp_path, monkeypatch):
    entries = {
        f"Worlds/portal/road_{road_id}.xml": portal_road(road_id, excluded=road_id == 10000)
        for road_id in (10000, 10009, 10024)
    }
    monkeypatch.setattr(portal, "read_wad_entries", lambda *args, **kwargs: entries)
    routes_dir, surfaces_dir = tmp_path / "routes", tmp_path / "surfaces"
    assert portal.extract_climb_portal_routes("client", str(routes_dir), str(surfaces_dir)) == 2
    monkeypatch.setattr(surface_map, "ROUTE_DIR", routes_dir)
    monkeypatch.setattr(surface_map, "SURFACE_DIR", surfaces_dir)
    caches = (surface_map._load_surface_world, surface_map._load_route_world, surface_map._projection)
    for function in caches:
        function.cache_clear()
    try:
        worlds = surface_map.list_worlds()
        assert [(world["name"], world["route_count"]) for world in worlds] == [("Climb Portal", 2)]
        world = surface_map.get_world_surfaces(1001)
        assert world["disconnected_routes"] is True
        assert world["satellite_routes"] is False
        assert world["background"] is None
        assert {segment["road_id"] for segment in world["segments"]} == {10009, 10024}
        assert all(route["free_ride_leadin_distance_m"] == 0 for route in world["routes"])
        for mode in ("race", "free_ride"):
            climb = surface_map.get_route(1001, 10009, mode=mode)
            assert climb["route"]["alt"][0] == -10
            assert climb["route"]["alt"][-1] == 20
            assert climb["leadin"] is None
            assert climb["world_name"] == "Climb Portal"
            assert climb["breakdown"][0]["surface"] == "Tarmac"
        assert surface_map.get_route(1001, 10000) is None
        exported = json.loads((surfaces_dir / "climb_portal.json").read_text())
        assert exported["segments"][0]["x"][-1] == 100
    finally:
        for function in caches:
            function.cache_clear()


def test_invalid_portal_archive_does_not_replace_published_files(tmp_path, monkeypatch):
    destination = tmp_path / "climb_portal.json"
    destination.write_text("existing export")
    monkeypatch.setattr(portal, "read_wad_entries", lambda *args, **kwargs: {
        "Worlds/portal/road_10009.xml": portal_road(start=0.75),
    })
    with pytest.raises(ValueError, match="Invalid portal climb boundaries"):
        portal.extract_climb_portal_routes("client", str(tmp_path), str(tmp_path))
    assert destination.read_text() == "existing export"


def test_published_portal_routes_are_served_without_an_installed_client():
    from app import app

    with app.test_client() as client:
        worlds = client.get("/api/surface_map/worlds").get_json()["worlds"]
        assert next(world for world in worlds if world["name"] == "Climb Portal")["route_count"] == 53
        world = client.get("/api/surface_map/world/1001").get_json()
        assert world["disconnected_routes"] and not world["satellite_routes"]
        assert len(world["routes"]) == 53
        for metadata in world["routes"]:
            response = client.get(f'/api/surface_map/route/1001/{metadata["nameHash"]}?mode=free_ride')
            assert response.status_code == 200
            climb = response.get_json()
            assert climb["bounds"] and climb["leadin"] is None
            assert np.isfinite(climb["route"]["alt"]).all()
            assert len(climb["route"]["d"]) == len(climb["route"]["alt"])
            assert climb["route"]["d"][-1] > 0
        assert client.get("/api/surface_map/route/1001/10000").status_code == 404
        assert client.get("/api/surface_map/world/1000").get_json()["satellite_routes"] is True
        assert client.get("/api/surface_map/world/1").get_json()["name"] == "Watopia"
