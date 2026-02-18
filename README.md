# ZwiftTools

A web app for analyzing Zwift race rides. Includes a **Bike Comparison** tool for evaluating different equipment setups and a **Race Replay** tool for visualizing full-field race data with interactive playback.

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
- **Hosting**: Azure App Service (Linux, Python 3.12)

## Local Development

1. Clone and install:
   ```bash
   git clone https://github.com/YOUR_USERNAME/zwifttools.git
   cd zwifttools
   pip install -r requirements.txt
   ```

2. Set environment variables:
   - `YOUTUBE_API_KEY` — for YouTube stream matching (optional)

3. Run:
   ```bash
   python app.py
   ```
   Open http://localhost:5000

## Deployment

Deployed to **Azure App Service** via GitHub Actions.

### Azure Setup (one-time)

1. Create a Resource Group and App Service in the Azure Portal
2. Download the **Publish Profile** from the App Service
3. Add these GitHub repo secrets:
   - `AZURE_WEBAPP_NAME` — your app service name (e.g., `zwifttools`)
   - `AZURE_WEBAPP_PUBLISH_PROFILE` — the full XML publish profile

Pushes to `main` trigger automatic deployment.