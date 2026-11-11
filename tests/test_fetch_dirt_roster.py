import pytest

from tools import fetch_dirt_roster as dirt


def test_current_teams_use_the_team_link_and_preserve_source_labels():
    html = '''<table id="teams-table"><tbody>
        <tr data-in-series="false"><td><a href="/riders/1/">Old team</a></td></tr>
        <tr data-in-series="true">
            <td><a href="/riders/1854/">RtB Yaks</a><span>RtB Recruiting</span></td>
            <td><a href="/riders/profile/6667/">Manager</a></td>
            <td> EMEA C </td><td>Tortellini</td><td><span>04 Sapphire</span></td>
        </tr></tbody></table>
        <div data-in-series="true"><a href="/riders/1854/">Mobile duplicate</a></div>'''
    assert dirt.parse_teams(html) == [{
        "team_id": 1854, "name": "RtB Yaks",
        "url": "https://dirtracingseries.com/riders/1854/",
        "time_zone": "EMEA C", "league": "Tortellini", "category": "04 Sapphire",
    }]


@pytest.mark.parametrize("html", ["<html>Login</html>", '<table id="teams-table"></table>'])
def test_unexpected_team_pages_are_not_silently_saved_as_empty(html):
    with pytest.raises(ValueError):
        dirt.parse_teams(html)


def test_roster_excludes_managers_and_reads_only_the_roster_table():
    html = '''<a href="/riders/profile/6667/">Manager</a>
        <table><thead><tr><th>Rider Name</th><th>Team Role</th></tr></thead>
        <tbody><tr><td><a href="/riders/profile/10157/">Timothy Johnson</a></td>
        <td>Rider</td></tr></tbody></table><p>Total: 1 rider(s)</p>'''
    assert dirt.parse_roster(html) == [{
        "rider_id": 10157, "name": "Timothy Johnson",
        "url": f"{dirt.BASE_URL}/riders/profile/10157/", "role": "Rider",
        "zwift_id": None, "profile_fetched": False,
    }]


def test_rider_uses_zwift_id_not_the_dirt_profile_id():
    assert dirt.parse_rider('<p>Zwift ID: 7681059</p><h5>Series 14 Registration</h5>') == {
        "zwift_id": 7681059, "profile_fetched": True,
    }


def test_missing_zwift_id_is_preserved_as_missing():
    assert dirt.parse_rider('<h5>Series 14 Registration</h5>')["zwift_id"] is None


@pytest.mark.parametrize("html", [
    '<form><input type="password"></form>', '<html>Error page</html>',
])
def test_login_and_unexpected_roster_pages_raise(html):
    with pytest.raises(ValueError):
        dirt.parse_roster(html)


def test_different_season_is_rejected():
    with pytest.raises(ValueError, match="Series 14"):
        dirt.parse_rider('<p>Zwift ID: 123</p><h5>Series 15 Registration</h5>')


def test_roster_total_mismatch_is_rejected():
    with pytest.raises(ValueError, match="count"):
        dirt.parse_roster('''<table><thead><tr><th>Rider Name</th><th>Team Role</th>
            </tr></thead><tbody></tbody></table><p>Total: 6 rider(s)</p>''')


def test_collection_resumes_and_reuses_profiles_across_team_assignments(tmp_path):
    teams = [{"team_id": 1, "name": "One", "url": "/one"},
             {"team_id": 2, "name": "Two", "url": "/two"}]
    output = tmp_path / "roster.json"
    collection = dirt.Collection(teams, output)
    roster = '''<table><thead><tr><th>Rider Name</th><th>Team Role</th></tr></thead>
        <tbody><tr><td><a href="/riders/profile/10157/">Timothy Johnson</a></td>
        <td>Rider</td></tr></tbody></table>'''
    pacer = dirt.RequestPacer(0.5)
    assert collection.next_task()["url"] == "/one"
    collection.accept({"url": "/one", "final_url": "/one", "status": 200, "html": roster}, pacer)
    rider_url = collection.next_task()["url"]
    collection.accept({"url": rider_url, "final_url": rider_url, "status": 200,
                       "html": "Zwift ID: 7681059 Series 14 Registration"}, pacer)
    resumed = dirt.Collection([{"team_id": 1}, {"team_id": 2, "name": "Two", "url": "/two"}], output)
    assert resumed.next_task()["url"] == "/two"
    resumed.accept({"url": "/two", "final_url": "/two", "status": 200, "html": roster}, pacer)
    assert resumed.next_task() is None
    snapshot = resumed.snapshot()
    assert snapshot["complete"] is True
    assert snapshot["assignment_count"] == 2
    assert snapshot["rider_count"] == 1
    assert snapshot["missing_zwift_ids"] == 0


def test_checkpoint_retries_a_transient_windows_file_lock(tmp_path, monkeypatch):
    original = dirt.Path.replace
    calls = []

    def replace(path, target):
        calls.append(target)
        if len(calls) == 1:
            raise PermissionError("File is temporarily open")
        return original(path, target)

    monkeypatch.setattr(dirt.Path, "replace", replace)
    monkeypatch.setattr(dirt.time, "sleep", lambda seconds: None)
    output = tmp_path / "roster.json"
    dirt.Collection([], output).save()
    assert output.exists()
    assert len(calls) == 2


def test_missing_ids_do_not_count_unfetched_profiles(tmp_path):
    collection = dirt.Collection([{"team_id": 1}], tmp_path / "roster.json")
    collection.teams[0]["riders"] = [
        {"rider_id": 1, "zwift_id": None, "profile_fetched": False},
        {"rider_id": 2, "zwift_id": None, "profile_fetched": True},
    ]
    assert collection.snapshot()["missing_zwift_ids"] == 1