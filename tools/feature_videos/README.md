# Feature documentation videos

This optional **local CLI** turns an editable JSON storyboard into a 1920x1080,
30 fps H.264/AAC MP4: branded slides, real website screenshots, optional closed
captions, and Microsoft Edge neural voiceover. It does not change the web app
or add production dependencies. Screenshots are stills, not interaction recordings.

## Setup

From the repository root (PowerShell):

```powershell
python -m pip install -r requirements-video.txt
python -m playwright install chromium
```

FFmpeg is supplied by `imageio-ffmpeg`; no system FFmpeg installation is required.
Rendering uses Arial on Windows, DejaVu Sans on Linux, or Arial on macOS.
Use `--font 'C:\path\to\font.ttf'` if none is available.

## 1. Draft and review the AI script

The runnable [Surface Map storyboard](surface-map.json) includes a starter script.
To generate a revised script using Copilot, without an AI API key:

```powershell
python -m tools.feature_video prompt tools\feature_videos\surface-map.json --brief tools\feature_videos\surface-map-brief.txt --output build\surface-map-prompt.txt
```

Open the generated prompt in Copilot, ask it to return the JSON, review its claims
against the feature, and save the response as
`build\surface-map-reviewed.json` (no Markdown fences). This is the explicit
AI-generation/import step: the CLI generates a factual prompt, not AI text itself.
Neither prompt generation nor import makes a network request. Only share public,
non-sensitive feature descriptions with your AI assistant.

```powershell
python -m tools.feature_video validate build\surface-map-reviewed.json
```

For a quick run without rewriting the starter script, use
`tools\feature_videos\surface-map.json` instead of the reviewed file below.

## 2. Capture the website

Start ZwiftTools yourself with `python app.py` in another terminal. Then:

```powershell
python -m tools.feature_video capture build\surface-map-reviewed.json --work-dir build\surface-map-video --base-url http://127.0.0.1:5000
```

Use `--base-url https://zwifttools.azurewebsites.net` for the deployed public site,
or `--headed` to inspect the browser. The pipeline uses a fresh browser context,
without your saved logins or rider selection, at 1600x900 with UTC and en-US.
Each demo starts at its own URL; actions do not carry over between scenes.
Set `screenshot_selector` to capture a specific panel (including content below
the fold), or omit it to capture the viewport. The Surface Map example frames
the route details separately so its elevation chart remains readable.
Screenshots are saved locally and never sent to the speech service.
Review them for loading/error states, sensitive data, and framing before publishing.
The CLI never modifies page data, hides UI, or substitutes synthetic demo data.

Readiness is explicit: add waits for the feature's final UI, especially for canvas
maps or charts. The browser also waits for fonts, document images, and two paint
frames. HTTP errors, failed images, missing selectors, and timeouts fail capture.
Default timeout is 60 seconds; override with `--timeout-ms 120000`.

## 3. Generate voiceover

```powershell
python -m tools.feature_video narrate build\surface-map-reviewed.json --work-dir build\surface-map-video
```

**This command sends each scene's narration text to Microsoft's online Edge
speech service.** No API key is required. Only run it with public, non-sensitive
scripts you are authorized to send. Network access and service availability are
required; failure is reported rather than silently replaced with silence.
Choose an Edge voice using the storyboard's `voice` field; the default is
`en-US-AriaNeural`. Discover voices with `python -m edge_tts --list-voices`.
Set `rate` to a signed percentage such as `"+12%"` for a slightly faster
delivery or `"-5%"` for slower speech. The default is `"+0%"`; the Surface Map
example uses `"+12%"` with the British `en-GB-SoniaNeural` voice.
The rate is recorded in voiceover provenance.
For names that the voice mispronounces, set a speech-only mapping such as
`"pronunciations": {"ZwiftTools": "Zwift Tools"}`. Whole-word, case-sensitive
replacements affect speech only; captions, transcripts, and slide text keep the
original spelling. The mapping is recorded in voiceover provenance.
Review pronunciation and service terms before publishing.

## 4. Render

```powershell
python -m tools.feature_video render build\surface-map-reviewed.json --work-dir build\surface-map-video
```

Rendering is offline. Scene duration is the greater of `min_seconds` and decoded
speech length plus 0.6 seconds, rounded up to a video frame. Audio is never cut
short to fit a slide. Narration is not burned into the images. Captions are
scene-level, not word-timed, and exported separately so they remain optional.
Load the SRT file in your player or upload it alongside the video when publishing
to a platform with closed-caption controls.

For an explicitly silent layout preview, skip narration and render with `--silent`.
This is not a voiceover fallback; the video manifest records that it is silent.
Use short `min_seconds` values in a separate preview storyboard if needed.

Outputs in the work directory:

- `video.mp4`: final video, validated by decoding with FFmpeg
- `captions.srt`: optional scene-level closed captions, provided separately
- `transcript.txt`: readable narration
- `video.json`: scene timing, storyboard fingerprint, asset hashes, capture origin,
  and voiceover provenance
- `capture/`, `narrate/`, `render/`: inspectable screenshots, audio, slides, and clips

Generated files live under the already-ignored `build/` directory by default.
Commands overwrite their own named outputs. Keep distinct work directories for
different videos. Stage manifests are invalidated before a rerun; rendering
rejects changed storyboards, missing assets, or altered screenshots/audio.
Do not publish an older MP4 if the current run failed: `video.json` is written
only after a successful render. To clean up, remove only that video's work directory.

## Add another feature

Copy the example JSON, edit its scenes and factual brief, then repeat the steps.
The format is intentionally small; unknown keys and invalid values are rejected:

| Field | Meaning |
| --- | --- |
| `title` | Video title, required |
| `voice` | Edge voice name, optional |
| `rate` | Signed speech-rate percentage, optional; default `"+0%"` |
| `pronunciations` | Optional map of written names to spoken spellings; speech only |
| `scenes` | Non-empty array in playback order |
| `scenes[].id` | Unique lowercase kebab-case ID, used for asset filenames |
| `kind` | `slide` or `demo` |
| `title`, `narration` | Required non-empty text; narration is spoken and used for optional captions |
| `min_seconds` | Positive finite minimum duration, default 3 |
| `bullets` | Slide-only array of short strings |
| `path` | Demo-only same-site absolute path, including query parameters |
| `steps` | Demo-only ordered browser actions |
| `screenshot_selector` | Optional demo-only element locator; default is viewport |

Each step has `action`, `selector` (a Playwright locator string), and optional
string `value`. Actions: `click`, `fill`, `select` (option value), `hover`,
`wait` (`visible` by default, or `hidden`), and `wait_text` (exact text, including
an empty string). Canvas clicks can include `"position": {"x": 0.4, "y": 0.6}`:
fractions of the element's width and height, each from 0 inclusive to 1 exclusive.
Omit `position` for a normal centered click. Verify canvas coordinates again if
the map projection or initial framing changes. The example clicks an actual road,
shows its route picker, and selects Road to Ruins before showing the profile.
No arbitrary Python, JavaScript, or shell commands are accepted
in a storyboard. Only run reviewed storyboards against sites you control: browser
actions can still submit forms or change application state.

Keep narration around 20-40 words per scene, titles short, and slides to a few
short bullets. Text overflow is an error, never silently clipped. Website
screenshots preserve actual displayed data; the framework does not clamp values.

## Checks

```powershell
python -m pytest tests\test_feature_video.py -q
```

Tests cover storyboard validation, script handoff, asset freshness, speech failure,
and audio-driven rendering. A real capture/render run is also recommended after
changing selectors or layouts.
