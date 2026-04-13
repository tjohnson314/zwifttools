# ZwiftTools

A web app for analyzing Zwift racing. Includes **Bike Comparison**, **Race Replay**, **Ride Compare**, and **TTT Analysis** tools, plus a collection of **Sauce4Zwift mods** for live in-game overlays.

## Live Site

🌐 https://zwifttools.azurewebsites.net

## Features

### Bike Comparison

- Load any Zwift activity by URL/ID or upload a .fit file
- Compare your actual bike against any frame/wheel/upgrade-level combination
- Physics model using Zwift's CdA, Crr, weight, and gradient data
- Estimates drafting savings from your recorded telemetry
- Time range slider to focus analysis on specific sections
- "Find Best Bikes" search with Pareto-optimal filtering
- Power chart showing actual vs. alternative vs. draft savings

### Race Replay

- Fetches full-field telemetry for every rider in a race from a single activity ID
- Second-by-second playback with adjustable speed (1×–20×)
- **Elevation chart** with two modes:
  - *Riders mode* — individual rider dots on the elevation profile
  - *Peloton mode* — automatic group detection with colored bars, group stats
- **Interactive map** — overhead view of the Zwift world with route line and rider dots, zoom/pan/follow
- **Rider table** — live position, gap, power, 1-min avg, NP, weight, HR, speed, distance; sortable columns, W/kg toggle
- **YouTube stream links** — auto-detects known streamers, links to VOD at the current playback timestamp, handles trimmed VODs
- Shareable URLs via `?activity_id=` parameter
- Server-side caching for fast reloads
- In-app FAQ panels for both Bike Comparison and Race Replay

### Ride Compare

- Compare multiple rides on the same route side-by-side (provide 2+ activity IDs)
- Reuses the Race Replay playback engine — same elevation chart, interactive map, and rider table
- Handles name disambiguation when the same rider appears multiple times
- SSE progress streaming while fetching activities
- Shareable URLs via `?activity_ids=` parameter

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

### Sauce4Zwift Mods

Standalone browser overlays for [Sauce4Zwift](https://www.sauce.llc/), installed by dropping the mod folder into Sauce's mods directory.

| Mod | Directory | Description |
|-----|-----------|-------------|
| **TTT Gaps** | `sauce_mod/` | Live inter-rider gap table for TTT pacing — shows gap distance (color-coded), power, and speed for nearby riders. Configurable target gap and team roster filter. |
| **Gap Ahead** | `sauce_mod_gap_ahead/` | Minimal overlay showing distance in meters to the nearest rider ahead. |
| **Surface Telemetry** | `sauce_mod_surface_telemetry/` | Records your telemetry with manual surface type tagging (Tarmac, Cobbles, Brick, Wood, Dirt, Gravel) for CRR mapping. Exports CSV. |
| **Team Race Score** | `sauce_mod_team_race/` | Live team race scoreboard — tracks two teams across categories with 3/2/1 point scoring, gap-to-rival, and power stats. |
| **Race Telemetry** | `sauce_mod_telemetry/` | Records per-second telemetry for all nearby riders (power, HR, speed, cadence, location, draft). Exports rider telemetry + profiles as separate CSVs. |

## Data Sources

- **Bike performance data**: [ZwifterBikes](https://zwifterbikes.web.app) — Cd, weight, and speed measurements for every frame and wheelset
- **Route geometry**: [ZwiftMap](https://zwiftmap.com) — lat/lng polylines and elevation profiles for Zwift routes
- **Telemetry**: Zwift API — second-by-second power, HR, speed, distance, altitude, GPS coordinates
- **Streamer matching**: YouTube Data API v3 — livestream VOD lookup and trim detection

## Tech Stack

- **Backend**: Flask + Gunicorn
- **Frontend**: Vanilla JS, Plotly.js (charts), Canvas (map), Chart.js (bike comparison)
- **Data processing**: NumPy, Pandas, SciPy (KD-tree route alignment)
- **Auth**: Zwift OAuth 2.0
- **Hosting**: Azure App Service (Linux, Python 3.12, Gunicorn)
- **Analytics pipeline**: Azure Function + Azure Data Explorer (see [ZWIFTGAMES.md](ZWIFTGAMES.md))

## Local Development

1. Clone and install:
   ```bash
   git clone https://github.com/YOUR_USERNAME/zwifttools.git
   cd zwifttools
   pip install -r requirements.txt
   ```

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
- **Web app** changes → `deploy.yml` (ignores `azure_function/` and `infra/`)
- **Azure Function** changes → `deploy-function.yml` (triggers on `azure_function/` changes only)