import pytest

from bike_comparison.bike_data import BikeDatabase, _WHEEL_FRAME_TYPES, _slug


@pytest.fixture(scope="module")
def db():
    return BikeDatabase()


@pytest.mark.parametrize("frame_type,wheel_type", [
    ("Standard", "Standard,TT"), ("TT", "Standard,TT"),
    ("Gravel", "Gravel"), ("MTB", "MTB"),
])
def test_wheels_only_fit_their_bike_category(db, frame_type, wheel_type):
    frame_id = next(key for key, frame in db.frames.items() if frame["frametype"] == frame_type
                    and not any(wheel["wheelownerframeid"] == key for wheel in db.wheels.values()))
    for wheel_id, wheel in db.wheels.items():
        if wheel["wheelownerframeid"]:
            continue
        compatible = wheel["wheelfitsframe"] == wheel_type
        assert (db.get_bike_stats(frame_id, wheel_id) is not None) == compatible
        assert ((frame_id, wheel_id) in db.bikes) == compatible


def test_known_off_road_wheels_publish_the_correct_compatibility(db):
    for (brand, model), frame_type in _WHEEL_FRAME_TYPES.items():
        wheel = db.wheels[_slug(f"{brand}{model}")]
        assert wheel["wheelfitsframe"] == frame_type


def test_fixed_wheels_remain_exclusive_to_their_owner(db):
    road_id = next(key for key, frame in db.frames.items() if frame["frametype"] == "Standard"
                   and not any(wheel["wheelownerframeid"] == key for wheel in db.wheels.values()))
    road_wheel = next(key for key, wheel in db.wheels.items()
                      if wheel["wheelfitsframe"] == "Standard,TT" and not wheel["wheelownerframeid"])
    for wheel_id, wheel in db.wheels.items():
        owner = wheel["wheelownerframeid"]
        if owner not in db.frames:
            continue
        assert db.get_bike_stats(owner, wheel_id) is not None
        assert db.get_bike_stats(owner, road_wheel) is None
        assert db.get_bike_stats(road_id, wheel_id) is None
        assert (road_id, wheel_id) not in db.bikes


def test_wheel_api_exposes_category_filters_and_stats_reject_invalid_pairs(monkeypatch, db):
    import app as app_module

    monkeypatch.setattr(app_module, "get_db", lambda: db)
    client = app_module.app.test_client()
    wheels = client.get("/api/wheels").get_json()
    gravel = next(wheel for wheel in wheels if wheel["fitsFrame"] == "Gravel")
    road = next(wheel for wheel in wheels if wheel["fitsFrame"] == "Standard,TT"
                and not wheel["exclusiveFrameId"])
    gravel_frame = next(key for key, frame in db.frames.items() if frame["frametype"] == "Gravel")
    road_frame = "Zwift_Carbon"

    assert client.get(f'/api/bike_stats?frame_id={road_frame}&wheel_id={gravel["id"]}').status_code == 404
    assert client.get(f'/api/bike_stats?frame_id={gravel_frame}&wheel_id={road["id"]}').status_code == 404
    assert client.get(f'/api/bike_stats?frame_id={gravel_frame}&wheel_id={gravel["id"]}').status_code == 200


@pytest.mark.parametrize("endpoint", ["/api/tt_pacing_plan", "/api/ttt_pacing_plan"])
@pytest.mark.parametrize("frame_type,wheel_type", [("Standard", "Gravel"), ("Gravel", "Standard,TT")])
def test_planner_apis_reject_incompatible_wheels(monkeypatch, db, endpoint, frame_type, wheel_type):
    import app as app_module

    monkeypatch.setattr(app_module, "get_db", lambda: db)
    frame_id = next(key for key, frame in db.frames.items() if frame["frametype"] == frame_type)
    wheel_id = next(key for key, wheel in db.wheels.items()
                    if wheel["wheelfitsframe"] == wheel_type and not wheel["wheelownerframeid"])
    response = app_module.app.test_client().post(endpoint, json={
        "route_name": "Jungle Circuit", "frame_id": frame_id, "wheel_id": wheel_id,
        "rider_weight_kg": 75.0, "rider_height_cm": 180.0, "avg_power_watts": 250.0,
        "riders": [{"name": f"R{index}", "height_cm": 180.0, "weight_kg": 75.0,
                    "cp_w": 250.0, "w_prime_kj": 20.0} for index in range(4)],
    })

    assert response.status_code == 400
    assert "frame/wheel" in response.get_json()["error"]