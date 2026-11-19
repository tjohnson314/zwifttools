import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import pytest

from tools import feature_video as video


EXAMPLE = Path(__file__).parents[1] / "tools" / "feature_videos" / "surface-map.json"


def load(tmp_path, data):
    path = tmp_path / "story.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return video.load_storyboard(path)


def minimal():
    return {"title": "Feature", "scenes": [
        {"id": "intro", "kind": "slide", "title": "Hello", "narration": "Explore."}
    ]}


def test_example_and_copilot_prompt_preserve_browser_steps():
    story = video.load_storyboard(EXAMPLE)
    result = video.prompt(story, "Only describe visible controls.")
    assert "Only describe visible controls." in result
    data = json.loads(result.split("STORYBOARD:\n", 1)[1])
    assert data == json.loads(json.dumps(video.asdict(story)))
    assert data["scenes"][1]["steps"][-1]["value"] == ""


@pytest.mark.parametrize("rate", ["fast", "12%", "+1.5%", 12, ""])
def test_invalid_speech_rates_fail_before_contacting_service(tmp_path, rate):
    data = minimal()
    data["rate"] = rate
    with pytest.raises(ValueError, match="rate"):
        load(tmp_path, data)


@pytest.mark.parametrize("pronunciations", [
    ["Watopia"], {"": "wah TOE pee uh"}, {"Watopia": ""}, {"Watopia": 12},
])
def test_invalid_pronunciations_are_rejected(tmp_path, pronunciations):
    data = minimal()
    data["pronunciations"] = pronunciations
    with pytest.raises(ValueError, match="pronunciations"):
        load(tmp_path, data)


@pytest.mark.parametrize("changes", [
    {"id": "../escape"}, {"kind": "unknown"}, {"narration": ""},
    {"min_seconds": 0}, {"min_seconds": float("inf")}, {"min_seconds": True},
    {"unknown": "typo"}, {"bullets": "not an array"},
    {"screenshot_selector": 1}, {"screenshot_selector": "body"},
    {"path": "/not-a-slide"}, {"steps": [{"action": "click", "selector": "button"}]},
])
def test_invalid_scenes_fail_with_context(tmp_path, changes):
    data = minimal()
    data["scenes"][0].update(changes)
    with pytest.raises(ValueError):
        load(tmp_path, data)


@pytest.mark.parametrize("path", [
    "https://example.com", "//example.com/path", "/\\example.com", "relative", "/\nother",
])
def test_demo_paths_cannot_escape_origin(tmp_path, path):
    data = minimal()
    data["scenes"][0].update(kind="demo", path=path)
    with pytest.raises(ValueError, match="path"):
        load(tmp_path, data)


def test_duplicate_ids_and_invalid_actions_are_rejected(tmp_path):
    data = minimal()
    data["scenes"] *= 2
    with pytest.raises(ValueError, match="unique"):
        load(tmp_path, data)
    data = minimal()
    data["scenes"][0].update(kind="demo", path="/surface-map", steps=[
        {"action": "evaluate", "selector": "body", "value": "alert(1)"}
    ])
    with pytest.raises(ValueError, match="action"):
        load(tmp_path, data)


@pytest.mark.parametrize("position", [
    {"x": -0.1, "y": 0.5}, {"x": 0.5, "y": 1},
    {"x": True, "y": 0.5}, {"x": float("nan"), "y": 0.5},
    {"x": 0.5}, {"x": 0.5, "y": "0.5"}, [0.5, 0.5],
])
def test_invalid_canvas_click_positions_are_rejected(tmp_path, position):
    data = minimal()
    data["scenes"][0].update(kind="demo", path="/surface-map", steps=[
        {"action": "click", "selector": "#map-canvas", "position": position}
    ])
    with pytest.raises(ValueError, match="position"):
        load(tmp_path, data)


def test_canvas_click_positions_require_click_and_survive_prompt_handoff(tmp_path):
    data = minimal()
    step = {"action": "click", "selector": "#map-canvas", "position": {"x": 0.4, "y": 0.6}}
    data["scenes"][0].update(kind="demo", path="/surface-map", steps=[step])
    story = load(tmp_path, data)
    generated = json.loads(video.prompt(story, "").split("STORYBOARD:\n", 1)[1])
    assert load(tmp_path, generated).scenes[0].steps[0].position == video.ClickPosition(0.4, 0.6)
    step["action"] = "hover"
    with pytest.raises(ValueError, match="only supported for click"):
        load(tmp_path, data)


def test_render_rejects_stale_and_modified_assets(tmp_path):
    story = video.load_storyboard(EXAMPLE)
    folder = tmp_path / "capture"
    folder.mkdir()
    files = {}
    for scene in story.scenes:
        if scene.kind == "demo":
            path = folder / f"{scene.id}.png"
            path.write_bytes(b"captured screenshot")
            files[scene.id] = video.digest(path)
    video.write_json(tmp_path / "capture.json",
                     {"storyboard": video.fingerprint(story), "files": files})
    video.check_assets(story, tmp_path, "capture", "png")
    with pytest.raises(ValueError, match="stale"):
        video.check_assets(replace(story, title="Edited script"), tmp_path, "capture", "png")
    path.write_bytes(b"modified screenshot")
    with pytest.raises(ValueError, match="modified"):
        video.check_assets(story, tmp_path, "capture", "png")


def test_failed_speech_invalidates_previous_manifest(tmp_path, monkeypatch):
    story = video.load_storyboard(EXAMPLE)
    (tmp_path / "narrate.json").write_text("old success", encoding="utf-8")

    class FailedSpeech:
        def __init__(self, narration, voice, *, rate):
            assert narration == story.scenes[0].narration
            assert voice == story.voice
            assert rate == story.rate

        async def save(self, path):
            raise RuntimeError("Speech service unavailable")

    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(Communicate=FailedSpeech))
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(video.narrate(story, tmp_path))
    assert not (tmp_path / "narrate.json").exists()


@pytest.mark.parametrize("rate", [None, "+12%", "-5%"])
def test_narration_applies_rate_and_records_it(tmp_path, monkeypatch, rate):
    data = minimal()
    if rate is not None:
        data["rate"] = rate
    story = load(tmp_path, data)
    requested_rates = []

    class Speech:
        def __init__(self, narration, voice, *, rate):
            requested_rates.append(rate)

        async def save(self, path):
            Path(path).write_bytes(b"speech audio")

    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(Communicate=Speech))
    asyncio.run(video.narrate(story, tmp_path))
    expected = "+0%" if rate is None else rate
    assert requested_rates == [expected]
    assert json.loads((tmp_path / "narrate.json").read_text())["rate"] == expected


def test_pronunciation_changes_speech_but_preserves_written_narration(tmp_path, monkeypatch):
    data = minimal()
    original = "Ride Watopia's roads. Watopia, not NewWatopia."
    data["scenes"][0]["narration"] = original
    data["pronunciations"] = {"Watopia": "wah TOE pee uh"}
    story = load(tmp_path, data)
    spoken = []

    class Speech:
        def __init__(self, narration, voice, *, rate):
            spoken.append(narration)

        async def save(self, path):
            Path(path).write_bytes(b"speech audio")

    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(Communicate=Speech))
    asyncio.run(video.narrate(story, tmp_path))
    assert spoken == ["Ride wah TOE pee uh's roads. wah TOE pee uh, not NewWatopia."]
    assert story.scenes[0].narration == original
    manifest = json.loads((tmp_path / "narrate.json").read_text())
    assert manifest["pronunciations"] == data["pronunciations"]


def test_render_uses_decoded_audio_duration_and_writes_provenance(tmp_path, monkeypatch):
    story = video.Storyboard("Feature", video.DEFAULT_VOICE, (
        video.Scene("intro", "slide", "Title", "A longer narration.", min_seconds=1),
        video.Scene("end", "slide", "Done", "End.", min_seconds=5),
    ))
    calls = []

    def ffmpeg(executable, *args):
        calls.append(args)
        destination = Path(args[-1])
        if destination.suffix == ".wav":
            with wave.open(str(destination), "wb") as audio:
                audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * (48000 * 4))
        elif destination.suffix == ".mp4":
            destination.write_bytes(b"encoded video")

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg",
                        SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"))
    monkeypatch.setattr(video, "run_ffmpeg", ffmpeg)
    monkeypatch.setattr(video, "check_assets", lambda *args: {"verified": True})
    monkeypatch.setattr(video, "font_path", lambda requested: "test.ttf")
    monkeypatch.setattr(video, "compose", lambda *args: None)
    video.render(story, tmp_path, False, None)
    manifest = json.loads((tmp_path / "video.json").read_text())
    assert manifest["timeline"][0]["duration"] == 4.6
    assert manifest["timeline"][1]["duration"] == 5
    assert manifest["timeline"][1]["start"] == 4.6
    assert manifest["silent"] is False
    assert manifest["narration"] == {"verified": True}
    assert "00:00:04,600 --> 00:00:09,600" in (tmp_path / "captions.srt").read_text()
    assert all("-af" in args for args in calls if "-loop" in args)
    assert (tmp_path / "video.mp4").exists()


def test_failed_render_invalidates_previous_success(tmp_path, monkeypatch):
    story = video.load_storyboard(EXAMPLE)
    (tmp_path / "video.json").write_text("old success", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace())
    with pytest.raises(FileNotFoundError):
        video.render(story, tmp_path, False, None)
    assert not (tmp_path / "video.json").exists()


def test_text_overflow_is_reported_instead_of_truncated(tmp_path):
    pytest.importorskip("PIL")
    scene = video.Scene("intro", "slide", "Title", "Spoken narration.",
                        bullets=("words " * 1000,))
    with pytest.raises(ValueError, match="does not fit"):
        video.compose(scene, tmp_path / "unused.png", tmp_path / "slide.png",
                      video.font_path(None))


@pytest.mark.parametrize("kind", ["slide", "demo"])
def test_narration_is_not_burned_into_frames(tmp_path, kind):
    pytest.importorskip("PIL")
    from PIL import Image

    screenshot = tmp_path / "screenshot.png"
    Image.new("RGB", (640, 360), "#334455").save(screenshot)
    scene = video.Scene("example", kind, "Title", "Short narration.")
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    font = video.font_path(None)
    video.compose(scene, screenshot, first, font)
    video.compose(replace(scene, narration="Long narration. " * 1000),
                  screenshot, second, font)
    assert first.read_bytes() == second.read_bytes()


def test_cli_validates_without_optional_dependencies(capsys):
    assert video.main(["validate", str(EXAMPLE)]) == 0
    assert "6 scenes" in capsys.readouterr().out
