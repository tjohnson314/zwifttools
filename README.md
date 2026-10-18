# ZwiftTools

A web app for analyzing Zwift rides. Includes **Bike Comparison**, **Race Replay**, **Ride Compare**, **Workout Analysis**, and **TTT Analysis** tools, plus a collection of **Sauce4Zwift mods** for live in-game overlays.

## Live Site

🌐 https://zwifttools.azurewebsites.net

## Features

The homepage separates current **Tools** from **Archived** tools. The ZRL
Leaderboard is available under **Archived** (`/?tab=archived`) and remains
accessible at its existing `/zrl/leaderboard` URL with event loading and scoring
unchanged.

### Bike Comparison

- Load any Zwift activity by URL/ID or upload a .fit file
- Compare your actual bike against any frame/wheel/upgrade-level combination
- Physics model using Zwift's CdA, Crr, weight, and gradient data
- Estimates drafting savings from your recorded telemetry
- Time range slider to focus analysis on specific sections
- "Find Best Bikes" search with Pareto-optimal filtering
- Power chart showing actual vs. alternative vs. draft savings

### Top Performers

- Compare road or TT frame/wheel combinations at `/top-performers`
- Switch bike type with the **Road / TT** toggle at the top left, above Upgrade Level
- Keep separate frame and wheel selections for each bike type while retaining the upgrade level
- TT mode uses TT-specific wheel drag; both modes use the same stock road reference bike
- Share TT mode with `/top-performers?bike=tt`; the old `/top-performers-tt` URL redirects there

### Race Replay

- Fetches full-field telemetry for every rider in a race from a single activity ID
- Second-by-second playback with adjustable speed (1×–20×)
- **Elevation chart** with two modes:
  - *Riders mode* — individual rider dots on the elevation profile
  - *Peloton mode* — automatic group detection with colored bars, group stats
- **Interactive map** — overhead view of the Zwift world with route line and rider dots, zoom/pan/follow
- **Rider table** — live position, gap, power, 1-min avg, NP, weight, HR, speed, distance; sortable columns, W/kg toggle
- **YouTube stream links** — auto-detects known streamers, links to VOD at the current playback timestamp, handles trimmed VODs
- **Rider analysis** — pick any rider to see where they lost contact (drops and near misses), what it would have taken to stay on, where they spent more than the riders around them, and moves that worked; thresholds are explained in the in-app FAQ
- Shareable URLs via `?activity_id=` parameter
- Server-side caching for fast reloads
- In-app FAQ panels for both Bike Comparison and Race Replay

### Ride Compare

- Compare multiple rides on the same route side-by-side (provide 2+ activity IDs)
- Reuses the Race Replay playback engine — same elevation chart, interactive map, and rider table
- Handles name disambiguation when the same rider appears multiple times
- SSE progress streaming while fetching activities
- Shareable URLs via `?activity_ids=` parameter

### Workout Analysis

- Open `/workout-analysis`, or use an activity's Workout Analysis link in My Activities
- Load an activity by ID or Zwift activity URL and search the bundled built-in time-based cycling workout catalog
- Or choose **Upload custom workout (.zwo)** and load a time-based cycling workout file (up to 1 MiB and 10,000 blocks); uploads are not saved or added to the catalog
- Prefill FTP from the current Zwift profile when available, with a manual override for the FTP used on that ride
- Adjust whole-workout intensity and start offset; negative offsets are supported
- Overlay raw recorded power with planned ramps, steady targets, and target ranges
- Round planned targets to the nearest 5 W after FTP/intensity scaling; ramps step between rounded targets and recorded power stays unchanged
- Compare time-weighted actual/target power per block over recorded coverage only; missing samples are not treated as zero
- Free-ride, max-effort, and unresolved zone-only targets are unscored; quantized ramps are identified as continuous approximations
- FTP tests with a fixed reference use the plan's reference rather than the rider's FTP; distance-based dynamic workouts are excluded
- Select and align plans manually: pauses, skips, variable-duration tests, and mid-ride intensity changes are not reconstructed

The deployable numeric catalog is checked in at `zwiftdata/workouts.json`; the server
does not require Zwift to be installed. Refresh it after a client update with:

```powershell
python tools\extract_zwift_workouts.py --zwift-dir "C:\Program Files (x86)\Zwift"
```

### TTT Analysis

- Analyze team time trial races by event subgroup ID
- Auto-detects teams from rider name tags (e.g. `[V]`, `(RtB)`) or accepts manual team tags
- **Lead Rider Speed** chart — who is on the front at each point on the course, with elevation overlay
- **Lead Rider Power** chart — watts or W/kg, plus pull statistics per rider (pull count, avg duration, total front time)
- **Draft Efficiency** chart — average draft savings as a percentage across connected riders
- **Individual Rider Drilldown** — per-rider power + draft estimate chart
- **Segment Splits** — click on charts to add distance markers and compare teams' elapsed times
- **Drag-and-drop team roster** — reassign riders between teams without re-fetching data
- Distance range slider to focus analysis on any section of the course
- Focus on a single team to see per-rider breakdowns instead of team aggregates

### TTT Pacing Planner

- Plan a WTRL team time trial at `/ttt-pacing` for 4-8 riders in a fixed rotation order
- Edit each rider's CP and W' and choose a route, bike, reserve and maximum power
- Pull durations default to a maximum of 60 seconds
- The leader holds constant power for each pull, including across terrain changes
- Reconsider non-pulling finishers with short pulls from the start, halfway or
   three-quarters through the course, retaining only faster feasible plans
- Set aerodynamic draft reductions for the second rider and third-and-later riders;
   defaults are 25% and 40%, and 0% disables the corresponding draft benefit
- Power drives a forward momentum simulation: speed carries through sections and
   rotation changes instead of being reset to independent chunk targets
- Switch the rider power graph between W and W/kg; export the actual pull schedule
   with durations and both power units as CSV
- Receive live iteration updates while optimization continues; the tables and charts
   show the best feasible plan so far, marked as provisional until completion
- Provisional plans keep W' positive throughout the course and respect each rider's power cap
- W' validation conservatively uses peak power in small integration steps; chart power
   is time-averaged. Tooltips include entry/exit speed, peak power and follower speed-matching
   dissipation, which is separate from pedaling power
- `POST /api/ttt_pacing_plan` accepts `stream: true` for SSE updates; omitting it returns JSON

### Sauce4Zwift Mods

Standalone browser overlays for [Sauce4Zwift](https://www.sauce.llc/), installed by dropping the mod folder into Sauce's mods directory.

| Mod | Directory | Description |
|-----|-----------|-------------|
| **TTT Gaps** | `sauce_mod/` | Live inter-rider gap table for TTT pacing — shows gap distance (color-coded), power, and speed for nearby riders. Configurable target gap and team roster filter. |
| **Gap Forecasting** | `sauce_mod_gap_forecasting/` | Minimal overlay showing the nearest rider gaps ahead and behind with a short forecast. |
| **Surface Telemetry** | `sauce_mod_surface_telemetry/` | Records your telemetry with manual surface type tagging (Tarmac, Cobbles, Brick, Wood, Dirt, Gravel) for CRR mapping. Exports CSV. |
| **Team Race Score** | `sauce_mod_team_race/` | Live team race scoreboard — tracks two teams across categories with 3/2/1 point scoring, gap-to-rival, and power stats. |
| **Race Telemetry** | `sauce_mod_telemetry/` | Records per-second telemetry for all nearby riders (power, HR, speed, cadence, location, draft). Exports rider telemetry + profiles as separate CSVs. |
| **Nearby Draft** | `sauce_mod_nearby_draft/` | Live road map with initials-centered draft donuts at actual rider positions: green effective draft, blue wasted watts, gray remaining maximum draft. Configurable radius (default 50 m) and closest-rider limit (0 = unlimited), with displayed/actual rider counts including you. Zoomable; negative segments are dashed and flagged rather than clipped. Uses each rider's Zwift weight/height, automatic road surfaces, and a configurable shared bike assumption. |

Nearby Draft interpolates all rider positions, displayed distances, and camera motion at approximately 30 Hz using one shared delayed game timestamp. Playback has a one-second buffer anchored to your position history. New or stale riders wait individually until their history covers the displayed timestamp, without hiding the rest of the group or extrapolating positions. Draft values and radius filtering continue to use the actual telemetry.

## Data Sources

- **Bike performance data**: [ZwifterBikes](https://zwifterbikes.web.app) — Cd, weight, and speed measurements for every frame and wheelset
- **Route geometry**: Zwift game WAD checkpoints for lead-ins/laps, calibrated to [ZwiftMap](https://zwiftmap.com) lat/lng and elevation axes
- **Telemetry**: Zwift API — second-by-second power, HR, speed, distance, altitude, GPS coordinates
- **Streamer matching**: YouTube Data API v3 — livestream VOD lookup and trim detection

## Tech Stack

- **Backend**: Flask + Gunicorn
- **Frontend**: Vanilla JS, Plotly.js (charts), Canvas (map), Chart.js (bike comparison)
- **Data processing**: NumPy, Pandas, SciPy (KD-tree route alignment)
- **Auth**: Zwift OAuth 2.0
- **Hosting**: Azure App Service (Linux, Python 3.12, Gunicorn)
- **Analytics pipeline**: Azure Function + Azure Data Explorer (see [ZWIFTGAMES.md](azure_function/ZWIFTGAMES.md))

## Local Development

1. Clone and install (a virtual environment is recommended):
   ```bash
   git clone https://github.com/YOUR_USERNAME/zwifttools.git
   cd zwifttools
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   ```
   `requirements-dev.txt` pulls in `requirements.txt` plus the test tools (pytest). Use plain
   `requirements.txt` if you only want to run the app.

2. Set environment variables (all optional for local dev):
   - `YOUTUBE_API_KEY` — YouTube stream matching for Race Replay
   - `SECRET_KEY` — Flask session secret (auto-generated if unset)
   - `AZURE_STORAGE_CONNECTION_STRING` — Azure Blob Storage for race data caching
   - `AZURE_STORAGE_CONTAINER` — blob container name (default: `race-cache`)
   - `ZWIFT_USERNAME` / `ZWIFT_PASSWORD` — only needed for the Azure Function (headless Zwift auth)

3. Run:
   ```bash
   python app.py
   ```
   Open http://localhost:5000

### Running tests

Python tests (in `tests/`, all synthetic data — no Zwift login or cached races needed):
```bash
python -m pytest tests                          # everything
python -m pytest tests/test_race_analyzer.py    # one file
python -m pytest tests -k near_miss             # tests matching a name
```

JavaScript tests (`tests/*.test.cjs`, need Node 18+, no npm install):
```bash
node --test tests/*.test.cjs
```

> CI (`.github/workflows/deploy.yml`) runs `python -m unittest discover -s tests`, which only picks up
> `unittest.TestCase` classes — pytest-style test functions are skipped there. Run `pytest` locally to
> cover everything.

## Deployment

Deployed to **Azure App Service** via GitHub Actions. The **Azure Function** (Zwift Games ETL) is deployed separately.

### Azure Setup (one-time)

1. Create a Resource Group and App Service in the Azure Portal
2. Download the **Publish Profile** from the App Service
3. Download the **Publish Profile** from the Function App
4. Add these GitHub repo secrets:
   - `AZURE_WEBAPP_NAME` — your app service name (e.g., `zwifttools`)
   - `AZURE_WEBAPP_PUBLISH_PROFILE` — the full XML publish profile for the App Service
   - `AZURE_FUNCTIONAPP_PUBLISH_PROFILE` — the full XML publish profile for the Function App

Pushes to `main` trigger automatic deployment:
- **Web app** changes → `deploy.yml` (ignores `azure_function/`)
- **Azure Function** changes → `deploy-function.yml` (triggers on `azure_function/` changes only)