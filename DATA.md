# Data Pipeline Documentation

This document describes the complete data pipeline for every page and API endpoint in the Zwift Tools web app.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Authentication](#authentication)
- [Landing Page](#landing-page)
- [My Activities](#my-activities)
- [Bike Comparison](#bike-comparison)
- [Race Replay](#race-replay)
- [Shared Infrastructure](#shared-infrastructure)

---

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Browser UI  │◄───►│  Flask App   │◄───►│  Zwift REST API   │
│  (Jinja2 +   │     │  (app.py)    │     │  us-or-rly101     │
│   JS/CSS)    │     │              │     │  .zwift.com/api    │
└──────────────┘     └──────┬───────┘     └───────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
      ┌──────────┐  ┌────────────┐  ┌──────────────┐
      │  Local   │  │   Azure    │  │  ZwiftMap     │
      │  Disk    │  │   Blob     │  │  (route data) │
      │ race_data│  │  Storage   │  │  zwiftmap.com │
      └──────────┘  └────────────┘  └──────────────┘
```

All data originates from the **Zwift REST API** (`https://us-or-rly101.zwift.com/api`). Route geometry comes from **ZwiftMap** (`zwiftmap.com`). Processed race data is cached to **local disk** (parquet + JSON) and optionally persisted to **Azure Blob Storage**.

---

## Authentication

### `GET /auth/login`

Renders the login form (`login.html`).

### `POST /auth/login`

**Flow:** Email + password → `get_token_with_password()` → Zwift OAuth token pair → stored in Flask `session['tokens']` as a `ZwiftTokens` dict (access_token, refresh_token, expires_at).

### `GET /auth/callback`

OAuth code exchange flow (CSRF state check → `exchange_code_for_tokens()`). Stores tokens in session.

### `GET /auth/logout`

Clears `session['tokens']`.

### `GET /auth/status`

Returns `{ authenticated: bool, method: 'oauth', expires_at: int }`.

**Token refresh:** `get_session_tokens()` checks `expires_at` on every authenticated request and calls `refresh_access_token()` if expired. All Zwift API calls use `get_headers()` which returns `{ Authorization: 'Bearer <access_token>' }`.

---

## Landing Page

### `GET /`

Renders `landing.html`. Passes `logged_in = 'tokens' in session` to the template so it can show login/logout links. No data pipeline — purely static.

---

## My Activities

### Page: `GET /my-activities`

Renders `my_activities.html`. Redirects to login if not authenticated.

### `GET /api/my_activities`

**Purpose:** Fetch the logged-in user's recent Zwift activities.

**Pipeline:**

1. **Get profile ID** — `GET /api/profiles/me` → extract `id`
2. **Fetch activities** — `GET /api/profiles/{id}/activities?start={start}&limit={limit}`
3. **Transform each activity:**
   - `distanceInMeters` → `distance_km` (÷ 1000)
   - `movingTimeInMs` → `duration_sec` (÷ 1000)
   - Extract `sport`, `name`, `avgWatts`, `calories`, `startDate`
   - Initialize enrichment fields as null: `route_name`, `avg_hr`, `normalized_power`, `is_race`, `race_name`, `event_subgroup_id`
4. **Return** `{ activities: [...], start, limit }`

### `GET /api/activity_enrichment/<activity_id>`

**Purpose:** Enrich a single activity with route name, NP, HR, and race info. Called lazily by the frontend for each visible activity.

**Pipeline:**

1. **Activity details** — `get_activity_details(activity_id, headers)` → `GET /api/activities/{id}`
   - Extract `routeId`, `eventInfo` (subgroup ID, event ID, race name)
   - If no `routeId` but has `eventId`: fetch `GET /api/events/{eventId}` → scan `eventSubgroups` for matching subgroup → get `routeId`
2. **Route lookup** — `get_route_info(routeId)` from `routes_cache.json` → `route_name`
3. **Telemetry** — `fetch_rider_telemetry(activity_id, headers)`:
   - `GET /api/activities/{id}` → extract `fitnessData` URL
   - Fetch raw telemetry JSON (contains `powerInWatts`, `heartRateInBpm`, etc.)
   - `convert_telemetry_to_dataframe()` → DataFrame with `power_watts`, `hr_bpm`, etc.
4. **Compute:**
   - **Normalized Power** — `calculate_normalized_power(power_watts)`: 30-sec rolling avg → 4th power → mean → 4th root
   - **Average HR** — Mean of non-zero `hr_bpm` values
5. **Return** `{ route_name, normalized_power, avg_hr, is_race, race_name, event_subgroup_id, event_id }`

---

## Bike Comparison

### Page: `GET /bike-comparison`

Renders `bike_comparison.html`.

### `GET /api/frames`

**Source:** `BikeDatabase` loaded from `zwiftdata/frames.json` (originally sourced from ZwifterBikes).

**Transform per frame:**
- `frameid` → `id`
- `framemake + framemodel` → `name`
- `frameaero` → `aero`, `frameweight` → `weight`
- `framewheeltype == 'fixed'` → `hasBuiltInWheels`
- `frametype` → `isTT` (bool) + `frameType` (string)
- `framelevel` → `level`

### `GET /api/wheels`

**Source:** `zwiftdata/wheels.json` via `BikeDatabase`.

**Transform per wheel:**
- `wheelid` → `id`
- `wheelmake + wheelmodel` → `name`
- `wheelaero`, `wheelweight`, `wheelfitsframe`, `wheellevel`
- Prepends a synthetic `(Built-in wheels)` entry for fixed-wheel frames

### `GET /api/my_profile`

**Source:** `GET /api/profiles/me`

**Transform:**
- `height` (mm) → `height_cm` (÷ 10)
- `weight` (g) → `weight_kg` (÷ 1000)
- `firstName + lastName` → `name`

### `GET /api/bike_stats`

**Params:** `frame_id`, `wheel_id`, `upgrade_level`

**Source:** `BikeDatabase.get_bike_stats()` → looks up `(frame_id, wheel_id)` in pre-computed combo table from `zwiftdata/bikes.json`. Each combo has arrays of 6 Cd and weight values (upgrade levels 0–5).

**Returns:** `{ frame_id, frame_name, wheel_id, wheel_name, upgrade_level, cd, weight_kg }`

### `POST /api/fetch_activity`

**Purpose:** Fetch full telemetry for an activity to use in bike comparison.

**Pipeline:**

1. **Fetch telemetry** — same as enrichment: `fetch_rider_telemetry()` → `convert_telemetry_to_dataframe()`
2. **Cooldown detection** — `detect_cooldown_start(df)`:
   - Compute race average power (excluding near-zero)
   - Threshold = 40% of average power
   - 5-second rolling average smoothing
   - Scan backwards: find last block of ≥10 consecutive seconds above threshold
   - Return the index where sustained power drops off
3. **Gradient calculation:**
   - `delta_alt / delta_dist` (altitude change over distance change)
   - 5-point uniform filter smoothing
   - Clamped to ±25%
4. **Surface type computation** (GPS-based):
   - `detect_world_from_coords()` → determine Zwift world from mid-ride GPS
   - `compute_surface_types_array(lats, lngs, world)` → ray-cast each GPS point against surface polygons from `surface_data.json`
   - `surface_types_to_crr(surface_types, bike_type)` → convert to rolling resistance coefficients (Tarmac=0.004, Cobbles=0.0065, Dirt=0.025, etc.)
5. **Rider profile** — fetch `GET /api/profiles/{profileId}` for height/weight
6. **Return** `{ telemetry: { time_sec, distance_km, speed_mps, gradient, power_watts, altitude_m, surface_type?, crr? }, cooldown_start_sec, rider_profile?, ... }`

### `POST /api/upload_fit`

**Purpose:** Parse a `.fit file` upload as an alternative to fetching from Zwift API.

**Pipeline:**

1. **Decode FIT file** — `garmin-fit-sdk` decoder with scale/offset/datetime conversion
2. **Auto-decompress** gzip if detected (magic bytes `\x1f\x8b`)
3. **Extract records** — iterate `record_mesgs`, extract power, HR, cadence, speed, distance, altitude, GPS
4. **GPS conversion** — FIT semicircles → degrees (`× 180 / 2^31`)
5. **Same processing as fetch_activity:** cooldown detection, gradient calc, surface types
6. **Return** same telemetry structure

### `POST /api/compare`

**Purpose:** Compare actual vs. hypothetical bike setup using loaded telemetry.

**Pipeline:**

1. **Resolve bike setups** — `get_bike_stats()` for both actual and alternative
2. **Frontal area** — `estimate_frontal_area(height_cm, weight_kg)` using Faria formula: `FA = 0.0293 × H^0.725 × M^0.425 + 0.0604`
3. **Surface-aware CRR** — if `surface_type` column present, compute per-bike-type CRR arrays (road bike vs MTB vs gravel have different rolling resistance on the same surface)
4. **Physics comparison** — `compare_bike_setups()`:
   - For each timestep, compute the power needed on the **alternative** bike to maintain the **same speed** as recorded
   - **With recorded power (energy-balance method):**
     - Split recorded power into: rolling resistance, aerodynamic drag, gravitational force, and a "draft" component (difference between recorded and theoretically required solo power)
     - Apply mass difference to kinetic/potential energy terms
     - Apply Cd difference to aero drag (preserving estimated draft fraction)
   - **Power equation:** `P = (F_gravity + F_rolling + F_aero) × v / (1 - η)` with η = 2.5% drivetrain loss
   - Compute NP for both actual and alternative power series
5. **Return** `{ actual: {name, cd, weight}, alternative: {name, cd, weight}, summary: {total_kj, diff_kj, diff_pct, avg_watts_diff, np_diff}, chart_data: {time_min, distance_km, actual_watts, alternative_watts, watts_difference, draft_watts, gradient} }`

### `POST /api/combo_count`

**Purpose:** Quick count of how many bike combos match the current filters.

**Filters applied by `_filter_bike_combos()`:**
- **Exclude TT frames** — filter by `frametype != 'TT'`
- **Exclude special unlocks** — frames with price=0 (except defaults) or Halo tier; wheels with price=0 (except defaults)
- **Excluded frames list** — explicit blacklist
- **Max rider level** — filter by `framelevel` and `wheellevel` ≤ rider level
- **Pareto frontier** — sort by (Cd, weight), keep only non-dominated combos where lower Cd also means lower weight

### `POST /api/best_bikes`

**Purpose:** Find the top N bike setups that minimize Normalized Power for the loaded telemetry.

**Pipeline:**

1. **Filter combos** — same `_filter_bike_combos()` as above
2. **Always include actual bike** — even if not on Pareto frontier
3. **For the actual bike:** use recorded NP directly (no simulation)
4. **For each alternative combo:**
   - Same physics pipeline as `/api/compare` — `compare_bike_setups()` with surface-aware CRR
   - Record alternative NP, avg power, total kJ
5. **Cost calculation** — `_get_combo_drops_cost()`:
   - Frame price + wheel price + upgrade cost per tier
   - Tier costs: Entry-Level (0–400k), Mid-Range (0–750k), High-End (0–1.9M), Halo (0–10M)
   - Flag `non_shop` for special-acquisition items
6. **Sort by NP** ascending, return top N
7. **Return** `{ best_bikes: [{name, cd, weight_kg, np, avg_power, total_kj, drops_cost, non_shop, is_actual}], combos_searched, actual_np, actual_bike_name }`

---

## Race Replay

### Page: `GET /race-replay`

Renders `race_replay.html`.

### `GET /api/race/fetch_stream`

**Purpose:** Fetch all race data from Zwift API given an activity URL or ID, streaming progress via **Server-Sent Events (SSE)**.**

1. **Extract activity ID** from URL or raw ID
2. **Get activity details** — `GET /api/activities/{id}` → extract `eventInfo.eventSubGroupId`
3. **Check local cache** — does `race_data/race_data_{subgroup_id}/` exist?
4. **Check blob storage** — `race_exists_in_blob(race_id)` → download if available
5. **Fetch event subgroup** — `GET /api/events/subgroups/{subgroupId}` → race name, start time, route ID, segment distance
6. **Fetch race results** — `get_race_entries(subgroup_id, headers)`:
   - Paginated `GET /api/events/subgroups/{id}/results?start={n}&limit=50`
   - Extract rank, name, activity_id, weight, player_id per participant
   - `deduplicate_ranks()` to fix duplicate rank numbers
7. **Fetch all rider telemetry** — 5 concurrent threads, each calls:
   - `fetch_rider_telemetry(activity_id, headers)` → raw telemetry JSON
   - `convert_telemetry_to_dataframe(telem_data)` → DataFrame
   - `save_rider_data()` → writes `rank{n}_{activity_id}.csv` + `rank{n}_{activity_id}_raw.json`
8. **Save metadata:**
   - `complete_race_summary.csv` — rank, activity_id, name, team, weight, player_id, start_date
   - `race_meta.json` — race_name, start_time, event_id, subgroup_id, route_id, segment_distance
9. **Upload to blob storage** — `upload_race_dir()`
10. **Return** `{ race_id: 'race_data_{subgroup_id}' }` (as final SSE message)

Uses a background thread + `queue.Queue` to bridge `progress_callback` → SSE generator. Progress messages: `{ progress: true, current, total, name }`. Final message: `{ success, race_id }` or `{ error }`.

### `GET /api/race/list`

**Purpose:** List all available races (local + blob storage).

**Pipeline:**

1. **In-memory cache** — returns cached result if < 120 seconds old
2. **Scan local `race_data/`** — for each `race_data_*` directory:
   - Read `race_meta.json` → `race_name`
   - Count lines in `complete_race_summary.csv` → `rider_count` (line count - 1 for header)
3. **Merge blob storage** — `blob_storage.list_races()` for any races not found locally
4. **Cache result** for 120 seconds
5. **Return** `{ races: [{ race_id, race_name, rider_count, has_summary }] }`

### `POST /api/race/load`

**Purpose:** Load and clean a race data directory, returning metadata

**Pipeline:**

1. **Check local disk** → if missing, download from blob storage
2. **`clean_race_data(data_path, cache=True)`** — the main cleaning pipeline (see below)
3. **Store in memory** — `_race_data_cache[race_id] = race_data`
4. **Upload cleaned cache to blob** — `upload_race_dir()`
5. **Return** `{ race_id, route_name, route_slug, finish_line_km, rider_count, min_time, max_time, riders: [{ rank, name, team, finish_time_sec }] }`

---

### The `clean_race_data()` Pipeline

This is the core data processing pipeline that transforms raw per-rider telemetry into a unified, time-aligned race dataset.

#### Step 1: Check Cache

If `cleaned_cache.json` exists, `load_from_cache()` reads it and restores all `RiderData` + elevation from parquet files. Returns immediately on cache hit.

#### Step 2: Load Summary

Read `complete_race_summary.csv` for rider metadata: name, team, weight_kg, player_id, activity_start_time.

#### Step 3: Load All Riders

For each `rank*_*.csv` file:
- Find matching `_raw.json`
- Parse rank from filename
- `load_raw_rider_data(csv_path, json_path)`:
  1. `pd.read_csv()` for telemetry
  2. `json.load()` for raw Zwift API response (contains `latlng`, `timeInSec`, `speedInCmPerSec`, `distanceInCm`)
  3. `fix_timestamp_offsets()` — detects server clock jumps in `time_sec` where telemetry was actually continuous (checks distance & speed continuity), corrects by subtracting excess time
  4. Interpolate `lat`/`lng` from JSON timestamps onto CSV timestamps via `scipy.interp1d(fill_value='extrapolate')`
- Attach summary metadata (name, team, weight, player_id, activity_start_time)

#### Step 4: Detect Route

`detect_route()` uses a three-method cascade:
1. **API route_id** — `route_id` from `race_meta.json` → `get_route_info()` from `routes_cache.json`
2. **Name matching** — substring match race title against all known route names (sorted longest-first to avoid partial matches)
3. **GPS + distance matching** — `detect_world_from_coords()` → filter routes by world → compare median of top-10 riders' max distance against each route's total distance → pick closest within 3 km tolerance

#### Step 5: Detect World

`detect_world_from_coords()` — bounding-box lookup across 12 Zwift worlds using mid-ride GPS coordinates of the reference rider (the one with the most data points).

#### Step 6: Align Distances

**If route data available** → `align_riders_to_route()` (preferred path):

##### Pass 1 — Per-Rider GPS Projection

For each rider:

**A. Lead-in Detection (known distance from API):**
- `detect_route()` extracts `leadinDistanceInMeters` from `routes_cache.json` alongside the route name/slug
- This value is passed to `align_riders_to_route()` as `leadin_distance_m`
- **If `leadin_distance_m > 50m`** (known long lead-in):
  - Use `np.searchsorted(raw_traveled_m, leadin_distance_m × 0.8)` to skip ahead to approximate route-join index
  - Search window: 80% to 150% of API lead-in distance (handles GPS jitter / speed variation)
- **Otherwise** (short/unknown lead-in):
  - Search first 1500 GPS points (covers up to ~16 km at typical density)
- Within the search window, compute haversine distance from each point to its nearest route point
- "On route" = deviation < 50 meters
- `leadin_end_idx` = index of first on-route point
- For the first 20 on-route points, compute `offset = route_distance[nearest] - raw_traveled_meters`
- `initial_offset` = median of those offsets (negative = lead-in distance)
- Points before `leadin_end_idx` get assigned `distance = initial_offset + raw_traveled` (typically negative, counting down toward 0)
- This handles lead-ins up to ~12 km (e.g. Richmond Rollercoaster at 12,062m)

**B. KD-tree Alignment (K=8 neighbors):**
- For each GPS point past the lead-in, query the route's `cKDTree` for K=8 nearest route points
- KD-tree uses latitude-scaled coordinates: `[lat, lng × cos(median_lat)]`
- For each of the 8 candidates, compute a **score** = `|candidate_route_distance - expected_distance|` where expected = `initial_offset + raw_traveled_meters`
- On loop routes, also test `± route_total_distance` wrap variants
- Select the candidate with the lowest score (correctly disambiguates where a route crosses itself)

**C. Segment Interpolation:**
- For the best route point, try projecting the GPS point onto the route segment between the best point and each of its two neighbors
- Vector projection: `t = dot(P - A, B - A) / dot(B - A, B - A)`, clamped to [0, 1]
- Interpolated distance: `dist_A + t × (dist_B - dist_A)`
- Deviation: haversine from GPS point to projected point on segment
- If projected deviation < snap-to-point deviation, use the interpolated distance
- This provides sub-route-point precision (~1m vs ~13m route point spacing)

**D. Collect Start Offsets** for loop route unwrapping

##### Pass 2 — Global Unwrapping & Smoothing

**E. Loop Route Unwrapping:**
- `loop_start_offset` = median of all riders' start distances
- For each point, pick the wrap variant (`route_dist + k × route_total`, k ∈ {-2..2}) closest to `expected = loop_start_offset + raw_traveled`
- Rebase: subtract `loop_start_offset` so distance 0 = race start

**F. Distance Smoothing (Zwift distance-delta approach):**
- Uses Zwift's raw distance data (centimeter precision from `distanceInCm`)
- Walking forward: `predicted = smooth[i-1] + raw_delta`
- `smooth[i] = predicted + α × (route_distances[i] - predicted)` with **α = 0.05**
- The 5% drift correction gently pulls toward the route-aligned value to prevent cumulative drift while maintaining sub-meter smoothness
- **Fallback** (no raw distance): use `speed_kmh / 3.6 × dt` with same alpha blending

**G. Write Back:**
- Overwrite `distance_km` = `route_distances / 1000`
- Add `route_deviation_m` column
- Warn if average deviation > 20m

**If no route data** → fallback to `align_rider_distances()`:
- Pick reference rider (most data points)
- `find_course_landmarks()` — detect turn apexes via bearing angle change (sliding window)
- For each other rider, find passage time at each landmark
- Compute per-landmark distance offset vs. reference
- `scipy.interp1d` to interpolate offset across full route
- Apply offsets to correct `distance_km`

#### Step 7: Determine Finish Line

`determine_finish_line()` — four-method cascade:
1. `segment_distance_cm` from `race_meta.json` (÷ 100000 → km)
2. `route_id` → `get_total_race_distance()` from `routes_cache.json`
3. `subgroupResults[0].segmentDistanceInCentimeters` from any rider's raw JSON
4. Median of top 5 ranked riders' maximum `distance_km`

#### Step 8: Compute Global Time Range

- `min_time` = min across all riders' first timestamp
- `max_time` = max across all riders' last timestamp
- (Zwift's full-data API provides telemetry at exactly 1-second intervals, so no interpolation is needed)

#### Step 9: Per-Rider Processing

For each rider:

**A. Cut at Finish:**
- The Zwift race results API only returns finishers (DNF riders are silently excluded), so all riders in our data completed the race
- Detect finish crossing: first timestamp where `distance_km >= finish_line_km - 0.03 km` (30m tolerance for alignment noise)
- Record `finish_time_sec` = that timestamp
- Keep data only up to `finish_time + 5 seconds`, drop everything after

**B. Compute Rolling Resistance (CRR):**
- If world detected: `compute_crr_array(lat, lng, world)` — ray-cast each GPS point against surface polygons
- Per-surface CRR values vary by bike type (road_bike, mtb, gravel_bike): Tarmac=0.004, Cobbles=0.0065, Dirt=0.025, etc.
- Default: 0.004 (tarmac)

**C. Index by Time:**
- Set `time_sec` as DataFrame index for O(1) lookups during playback

#### Step 10: Cap Max Time

Clamp `max_time` to last finisher's time + 10 seconds to exclude cooldown/post-race data.

#### Step 11: Build Elevation Profile

**If route data available:**
- Use `RouteData.distance` and `RouteData.altitude` arrays
- For loop routes: rebase by `loop_start_offset % route_total`, sort by distance
- **Lead-in handling:** If the reference rider has off-route points (deviation=999) at the start, the route data doesn't cover the lead-in terrain. In that case:
  - Extract the ref rider's telemetry altitude for the lead-in distance range
  - Resample at 10m intervals using linear interpolation
  - Offset the loop elevation data to start where the lead-in ends
  - Concatenate: [telemetry lead-in] + [loop route data]
  - This is critical for routes like Richmond Rollercoaster (12 km lead-in along a river valley at 4–55m altitude, vs the loop at 52–65m)
- Tile extra copies to cover multi-lap races (up to max rider distance)

**Fallback:**
- `build_elevation_profile()` from reference rider's `altitude_m` column
- Linear interpolation at 10-meter intervals

#### Step 12: Assemble & Cache

- Read `source_activity_id` from `race_meta.json`
- Build `CleanedRaceData` dataclass
- `save_to_cache()`:
  - Per-rider DataFrames → `cleaned_data/rider_{rank}.parquet`
  - Elevation profile → `cleaned_data/elevation.parquet`
  - Metadata → `cleaned_cache.json` (JSON with rider info, route details, time ranges)

---

### `GET /api/race/data/<race_id>`

**Purpose:** Return the full cleaned race dataset as JSON for the frontend replay player.

**Pipeline:**

1. **Check `_race_data_cache`** — in-memory cache. If missing, re-run `clean_race_data()` or download from blob.
2. **Detect world** if not set (for older cached data)
3. **Build elevation profile JSON** — `{ distance_km: [...], altitude_m: [...] }`
4. **Load race metadata** — `race_meta.json` → `race_start_time`, `event_id`, `event_subgroup_id`
5. **Late joiner detection** — for each rider:
   - Parse race `race_start_time` as ISO 8601 datetime
   - Parse rider's `activity_start_time` as ISO 8601 datetime
   - `is_late_joiner = rider_start_dt > race_start_dt`
6. **Build per-rider JSON** — `{ rank, name, team, activity_id, player_id, weight_kg, is_late_joiner, finish_time_sec, time_sec: [...], distance_km: [...], altitude_m: [...], speed_kmh: [...], power_watts: [...], hr_bpm: [...], lat: [...], lng: [...] }`
7. **World map config** — `get_world_map_config(world)` → map image, GPS bounds, background color
8. **Route polyline** — `load_route_data(route_slug).latlng` → official route lat/lng array
9. **Return** `{ race_id, route_name, route_slug, source_activity_id, race_start_time, event_id, event_subgroup_id, finish_line_km, min_time, max_time, elevation_profile, riders: [...], map_config, route_latlng }`

### `GET /api/race/streams/<race_id>`

**Purpose:** Find YouTube livestream VODs from known streamers who raced.

**Pipeline:**

1. **Check disk cache** — `stream_cache.json` in race directory (avoids YouTube API calls)
2. **Load race metadata** — `race_meta.json` → `race_start_time`
3. **Get rider player IDs** — from `_race_data_cache` or fall back to `cleaned_cache.json` / `complete_race_summary.csv`
4. **`find_matching_streams()`:**
   - Load `streamers.json` — maps Zwift player IDs → YouTube channel handles
   - Cross-reference race participants with known streamers
   - For each matching streamer:
     - `resolve_channel_id(handle)` — YouTube API `channels?forHandle=@handle`
     - **Fast path** — `_find_stream_for_race()`: get uploads playlist (`UU{channelId}`), filter by ±48h publish time, check `liveStreamingDetails.actualStartTime` / `actualEndTime` overlap
     - **Slow fallback** — `_find_stream_for_race_search()`: YouTube search API with `eventType=completed`
     - `_detect_trim_offset()` — compare broadcast duration vs current video duration to compute front-trim offset
   - Return `{ streamer_name, youtube_url, stream_title, offset_seconds }`
5. **Cache to disk** — `stream_cache.json` + upload to blob storage

---

## Shared Infrastructure

### Zwift API (`shared/data_fetcher.py`)

- **Base URL:** `https://us-or-rly101.zwift.com/api`
- **`_request_with_retry()`** — HTTP wrapper with retry on 5xx/429 errors, auto-sets `Accept: application/json`
- **`extract_activity_id()`** — extracts numeric ID from Zwift activity URLs

### Route Lookup (`shared/route_lookup.py`)

- **`routes_cache.json`** — local cache of all Zwift routes with distance, lead-in, world, etc.
- `get_route_info(route_id)` → route dict by numeric ID
- `get_total_race_distance(route_id)` → `(distanceInMeters + leadinDistanceInMeters) / 1000` km
- `WORLD_ID_TO_MAP` / `MAP_TO_WORLD_ID` — int ↔ name mappings

### Route Geometry (`race_replay/data_cleaner.py`)

- **Source:** ZwiftMap (`zwiftmap.com/strava-segments/{id}`)
- `route_strava_segments.json` — maps route slugs to Strava segment IDs
- `fetch_route_from_zwiftmap()` — downloads `latlng.json`, `distance.json`, `altitude.json`
- Cached locally in `zwiftmap_surfaces/{route_slug}_route.json`
- `RouteData` — Nx2 latlng array, cumulative distance array, `cKDTree` for spatial queries

### Surface Lookup (`shared/surface_lookup.py`)

- **`surface_data.json`** — polygon definitions for non-tarmac surfaces per world
- `compute_surface_types_array(lats, lngs, world)` — for each GPS point, ray-cast against polygons to determine surface type
- `surface_types_to_crr(surface_types, bike_type)` — CRR by surface × bike type:

| Surface | Road Bike | MTB   | Gravel |
| ------- | --------- | ----- | ------ |
| Tarmac  | 0.004     | 0.004 | 0.004  |
| Cobbles | 0.0065    | 0.005 | 0.0055 |
| Dirt    | 0.025     | 0.006 | 0.008  |
| Gravel  | 0.020     | 0.007 | 0.009  |

### World Config (`shared/world_config.py`)

Static config for 13 Zwift worlds: map image path, GPS bounding box (lat/lng min/max), background color. Used by the race replay map overlay.

### Blob Storage (`shared/blob_storage.py`)

Azure Blob Storage wrapper — **no-op when `AZURE_STORAGE_CONNECTION_STRING` is not set**.
- `upload_race_dir()` / `download_race_dir()` — persist/restore full race data directories
- `race_exists_in_blob()` — checks for `complete_race_summary.csv` sentinel
- `list_races()` — enumerate all stored races (race_id + name from `race_meta.json`)

### Bike Database (`bike_comparison/bike_data.py`)

- **Source:** `zwiftdata/frames.json`, `wheels.json`, `bikes.json` (from ZwifterBikes)
- `BikeDatabase` — singleton, auto-downloads missing JSON files
- `bikes.json` contains pre-computed combo table: `(frame_id, wheel_id)` → arrays of 6 Cd values and 6 weight values (upgrade levels 0–5)
- `BikeSetup` dataclass: `frame_id`, `frame_name`, `wheel_id`, `wheel_name`, `upgrade_level`, `cd`, `weight_kg`, `frame_type`

### Physics Engine (`bike_comparison/physics.py`)

- **Power equation:** `P = (F_gravity + F_rolling + F_aero) × v / (1 - η)` with η = 0.025 (2.5% drivetrain loss)
  - `F_gravity = total_mass × g × gradient`
  - `F_rolling = total_mass × g × CRR`
  - `F_aero = 0.5 × ρ × CdA × v²` where CdA = Cd × frontal_area
- **`compare_bike_setups()`** — energy-balance method that preserves the draft benefit from real race data when computing hypothetical alternative power

### Normalized Power (`shared/utils.py`)

Dr. Coggan's algorithm:
1. 30-second rolling average of power
2. Raise to 4th power
3. Take mean
4. Take 4th root

Window adapts to actual sample rate when `time_sec` is provided.

### Timestamp Correction (`race_replay/data_cleaner.py`)

- `fix_timestamp_offsets()` — detects Zwift server clock jumps where telemetry data was actually continuous (checks distance & speed continuity across the jump). Subtracts excess time to produce monotonic timestamps.
