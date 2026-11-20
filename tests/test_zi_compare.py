import csv

from tools import zi_compare


def test_load_wheels_csv_keeps_300w_road_tests(tmp_path, monkeypatch):
    path = tmp_path / "wheels.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([
            ["Bike", "Wheels", "Price", "Level", "Height", "Weight", "Power",
             "Flat Speed", "Flat Gap", "Climb Speed", "Climb Gap"],
            ["Zwift Carbon", "Road Wheel", "", "", "183", "75", "150",
             "19.2", "50.3", "4.7", "-24.1"],
            ["Zwift TT", "Road Wheel", "", "", "183", "75", "300",
             "26.0", "70.8", "9.0", "-15.1"],
            ["Zwift Gravel", "Gravel Wheel", "", "", "183", "75", "300",
             "23.5", "-147.2", "8.8", "-159.1"],
            ["Zwift Carbon", "Road Wheel", "", "", "183", "75", "300",
             "24.9", "52.7", "9.2", "-14.3"],
        ])
    monkeypatch.setattr(zi_compare, "ZI_WHEELS", str(path))

    assert zi_compare.load_wheels_csv() == [
        {"wheel": "Road Wheel", "flat_gap": 52.7, "climb_gap": -14.3},
    ]
