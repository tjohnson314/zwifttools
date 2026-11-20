import json
from pathlib import Path

import pytest

from bike_comparison.bike_data import BikeDatabase, _WHEEL_FRAME_TYPES, _slug


@pytest.fixture(scope="module")
def db():
    return BikeDatabase()


@pytest.mark.parametrize("path", ["distance_highend", "elevation_highend", "time_highend"])
def test_drivetrain_upgrade_only_changes_rolling_resistance(db, path):
    raw = json.loads((db.DATA_DIR / db.GAME_FRAMES).read_text(encoding="utf-8-sig"))
    frame_id = next(frame["folder"] for frame in raw if frame.get("level_up") == path
                    and not any(w["wheelownerframeid"] == frame["folder"] for w in db.wheels.values()))
    wheel_id = next(wheel_id for wheel_id, wheel in db.wheels.items()
                    if db._wheel_fits_frame(db.frames[frame_id], wheel))
    before = db.get_bike_stats(frame_id, wheel_id, 2)
    after = db.get_bike_stats(frame_id, wheel_id, 3)
    assert before is not None and after is not None

    assert after.cda_bias == before.cda_bias
    assert after.weight_kg == before.weight_kg
    assert after.wheel_cda_bias == before.wheel_cda_bias
    assert after.wheel_weight_g == before.wheel_weight_g
    assert before.crr_reduction == 0
    assert after.crr_reduction == pytest.approx(.0003)


def test_packaged_rewards_replace_baked_frame_ladders_and_do_not_clip_stats(db, tmp_path, monkeypatch):
    frame = {"folder": "Test", "make": "Test", "name": "Bike", "type": "STANDARD",
             "level_up": "distance_entry", "frameset_weight_g_effective": 100,
             "frameset_cda_bias_effective": 0,
             "upgrades": [{"weight_g_effective": 9999, "cda_bias_effective": 9999}]}
    (tmp_path / "game_frames.json").write_text(json.dumps([frame]))
    (tmp_path / "game_wheels.json").write_text("[]")
    (tmp_path / "game_upgrades.json").write_text(json.dumps({
        "levelup_configs": {"distance_entry": db.upgrade_definitions["distance_entry"]},
    }))
    monkeypatch.setattr(BikeDatabase, "DATA_DIR", tmp_path)
    database = BikeDatabase()

    stage3 = database.get_bike_stats("Test", None, 3)
    stage5 = database.get_bike_stats("Test", None, 5)
    assert stage3 is not None and stage5 is not None
    assert stage3.cda_bias == -.004
    assert stage3.frame_weight_g == -500
    assert stage5.cda_bias == stage3.cda_bias
    assert stage5.weight_kg == stage3.weight_kg
    assert stage5.crr_reduction == stage3.crr_reduction


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


@pytest.mark.parametrize("route,prefixes", [
    ("/bike-comparison", ["actual", "alt"]),
    ("/tt-pacing", ["bike"]),
    ("/ttt-pacing", ["bike"]),
    ("/workout-route", ["bike"]),
])
def test_bike_pages_render_all_shared_summary_fields(route, prefixes):
    import app as app_module

    response = app_module.app.test_client().get(route)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert html.count('class="bike-summary"') == len(prefixes)
    for prefix in prefixes:
        fields = [f'id="{prefix}{suffix}"' for suffix in ("Weight", "Cda", "Crr")]
        assert all(html.count(field) == 1 for field in fields)
        assert [html.index(field) for field in fields] == sorted(html.index(field) for field in fields)


@pytest.mark.parametrize("level,expected", [(2, 0), (3, .0003)])
def test_best_bike_results_include_source_resistance_reduction(monkeypatch, db, level, expected):
    import app as app_module

    key = ("Zwift_Carbon", "")
    monkeypatch.setattr(app_module, "get_db", lambda: db)
    monkeypatch.setattr(app_module, "get_bike_stats", db.get_bike_stats)
    monkeypatch.setattr(app_module, "_filter_bike_combos", lambda *args: [(key, db.bikes[key])])
    response = app_module.app.test_client().post("/api/best_bikes", json={
        "upgrade_level": level,
        "actual_bike": {"frame_id": key[0], "wheel_id": key[1], "upgrade_level": level},
        "telemetry": {
            "time_sec": list(range(32)), "power_watts": [250.] * 32,
            "speed_mps": [10.] * 32, "distance_km": [index * .01 for index in range(32)],
        },
    })

    assert response.status_code == 200
    results = response.get_json()["best_bikes"]
    assert len(results) == 1
    assert results[0]["crr_reduction"] == expected


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


@pytest.mark.parametrize("road_styles", [None, {6: {42: {"d": "DIRT", "s": []}}}])
def test_sauce_data_builder_refreshes_both_mods(monkeypatch, tmp_path, db, road_styles):
    from tools import build_power_breakdown_data as builder

    assert {Path(mod_dir).name for mod_dir in builder.MOD_DIRS} == {
        "sauce_mod_power_breakdown", "sauce_mod_nearby_draft",
    }
    mod_dirs = [tmp_path / Path(mod_dir).name for mod_dir in builder.MOD_DIRS]
    for mod_dir in mod_dirs:
        mod_dir.mkdir()
        (mod_dir / "road-styles.mjs").write_text("existing road data\n", encoding="utf-8")
    monkeypatch.setattr(builder, "MOD_DIRS", mod_dirs)
    monkeypatch.setattr(builder, "get_bike_database", lambda: db)
    monkeypatch.setattr(builder, "build_road_styles", lambda style_map: road_styles)

    builder.main()

    models = [(mod_dir / "bike-model.mjs").read_text(encoding="utf-8") for mod_dir in mod_dirs]
    assert models[0] == models[1]
    model = json.loads(models[0].split("export const MODEL = ", 1)[1].removesuffix(";\n"))
    assert model["frames"] == builder.build_frames(db)
    assert model["wheels"] == builder.build_wheels(db)
    assert model["constants"]["AIR_DENSITY"] == builder.AIR_DENSITY
    assert model["constants"]["DRIVETRAIN_LOSS"] == builder.DRIVETRAIN_LOSS
    assert model["crr"] == builder.build_crr_table()
    assert model["styleMap"] == builder.build_style_map()

    roads = [(mod_dir / "road-styles.mjs").read_text(encoding="utf-8") for mod_dir in mod_dirs]
    assert roads[0] == roads[1]
    if road_styles is None:
        assert roads[0] == "existing road data\n"
    else:
        actual = json.loads(roads[0].split("export const ROAD_STYLES = ", 1)[1].removesuffix(";\n"))
        assert actual == json.loads(json.dumps(road_styles))