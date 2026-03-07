# Zwift Games Data Pipeline

Automated ETL pipeline that captures race telemetry from every Zwift Games event and stores it in Azure Data Explorer (ADX) for OLAP analysis.

## Architecture

```
Zwift API ──> Azure Function (timer, 15 min) ──> Azure Data Explorer (free tier)
                      │                                     │
                      │                                     ├── RiderTelemetry  (~2K rows/rider/race)
                      │                                     ├── RaceSummary     (1 row/rider/race)
                      │                                     └── RaceEvents      (1 row/event)
                      │
                 Managed Identity ──> ADX (ingestor + viewer)
```

## Components

| Component        | Resource               | Details                                                                                        |
| ---------------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| **Compute**      | Azure Function App     | `zwiftgames-func-hc4trbmw4t6kc` on `ASP-zwifttools` (B1, Linux)                                |
| **Database**     | Azure Data Explorer    | Free cluster, `ZwiftGames` database, KQL query language                                        |
| **Storage**      | Storage Account        | `zwiftgamesst` — required by Function App runtime                                              |
| **Monitoring**   | Application Insights   | `zwiftgames-insights` backed by `zwiftgames-logs` Log Analytics                                |
| **Auth (Zwift)** | Password grant         | `ZWIFT_USERNAME` / `ZWIFT_PASSWORD` app settings                                               |
| **Auth (ADX)**   | DefaultAzureCredential | Managed Identity in Azure; Azure CLI locally. Ingestor + Viewer roles on `ZwiftGames` database |

## ETL Flow

The function `zwift_games_etl` runs every 15 minutes:

1. **Discover events** — `POST /api/events/search` returns a protobuf-lite response. A built-in wire-format parser decodes it (no `.proto` file needed) to extract event IDs and subgroups. Each matching event is then enriched via `GET /api/events/{id}` (JSON) for full subgroup details.
2. **Deduplicate** — Queries `RaceEvents` in ADX to get already-ingested event IDs; skips those plus any events that haven't finished yet (start time + 30 min buffer).
3. **Fetch race data** — For each new event, fetches all rider results and second-by-second telemetry using 5 concurrent workers.
4. **Ingest** — Writes data to three KQL tables via queued ingestion:
   - `RaceEvents` — one row per event (metadata)
   - `RaceSummary` — one row per rider per event (result, avg power, NP, etc.)
   - `RiderTelemetry` — one row per rider per second (power, HR, cadence, speed, distance, altitude), chunked in 100K-row batches
5. **Backfill** — Queries `RaceSummary` for riders with non-SUCCESS status in events ingested within the last 24 hours. Re-fetches their telemetry and appends new SUCCESS rows for any that are now available. Riders with HTTP 403 errors (private profiles) are excluded from retry. This ensures riders whose data wasn't ready at initial ingestion are eventually captured.

## Database Schema

### Fact Tables (append-only)

**RiderTelemetry** — Second-by-second rider data
| Column            | Type   | Description                  |
| ----------------- | ------ | ---------------------------- |
| event_subgroup_id | long   | Foreign key to RaceEvents    |
| activity_id       | string | Rider's activity ID          |
| player_id         | long   | Zwift player ID              |
| time_sec          | int    | Seconds since activity start |
| power_watts       | int    | Instantaneous power          |
| hr_bpm            | int    | Heart rate                   |
| cadence_rpm       | int    | Cadence                      |
| speed_kmh         | real   | Speed                        |
| distance_km       | real   | Cumulative distance          |
| altitude_m        | real   | Altitude                     |

**RaceSummary** — Per-rider race results
| Column              | Type     | Description                                       |
| ------------------- | -------- | ------------------------------------------------- |
| event_subgroup_id   | long     | Foreign key to RaceEvents                         |
| activity_id         | string   | Unique activity ID                                |
| rank                | int      | Finishing position                                |
| name                | string   | Rider name                                        |
| weight_kg           | real     | Rider weight                                      |
| player_id           | long     | Zwift player ID                                   |
| file_id             | string   | Telemetry file ID from Zwift                      |
| activity_start_time | datetime | When the rider started                            |
| status              | string   | `SUCCESS`, error message, or `HTTP 403` (private) |
| duration_sec        | real     | Finish time in seconds                            |
| avg_power           | real     | Average power (watts)                             |
| normalized_power    | real     | Normalized power                                  |
| max_power           | real     | Max power                                         |
| avg_hr              | real     | Average heart rate                                |
| data_points         | long     | Number of telemetry samples                       |

> **Note:** Backfilled riders will have two rows — the original error row and a new SUCCESS row. Filter with `| where status == 'SUCCESS'` or deduplicate with `| summarize arg_max(data_points, *) by event_subgroup_id, activity_id`.

**RaceEvents** — One row per event subgroup
| Column            | Type     | Description               |
| ----------------- | -------- | ------------------------- |
| event_subgroup_id | long     | Primary key               |
| event_id          | long     | Parent event ID           |
| race_name         | string   | Event name                |
| race_start_time   | datetime | Scheduled start           |
| world_id          | int      | Zwift world/map ID        |
| route_id          | long     | Route ID                  |
| ingested_at       | datetime | When this row was written |

### Dimension Tables (reference data)

| Table               | Description                                     |
| ------------------- | ----------------------------------------------- |
| Frames              | Bike frame specs (aero, weight, speed rankings) |
| Wheels              | Wheel specs (aero, weight, flat/climb rankings) |
| BikeConfigs         | Frame + wheel combos with CdA and weight arrays |
| Routes              | Route metadata (distance, ascent, difficulty)   |
| RoutePoints         | GPS polyline points per route                   |
| Worlds              | Map bounds and metadata                         |
| SurfaceCrr          | Rolling resistance by surface type              |
| RouteStravaSegments | Route-to-Strava segment mapping                 |

### Stored Function

```kql
// Average telemetry per 50m distance bin for a given race
// Usage: RaceTelemetryByDistance(6773826)
RaceTelemetryByDistance(eventSubgroupId: long)
```

Returns avg power, HR, cadence, speed, and rider count at each 50m segment of a race.

## Example Queries

```kql
// Average power profile across all riders in a race
RaceTelemetryByDistance(6773826)

// Top 10 finishers in a race
RaceSummary
| where event_subgroup_id == 6773826
| top 10 by rank asc
| project rank, name, duration_sec, avg_power, normalized_power

// W/kg distribution for a race
RaceSummary
| where event_subgroup_id == 6773826 and weight_kg > 0
| extend wkg = avg_power / weight_kg
| summarize avg(wkg), percentile(wkg, 50), percentile(wkg, 90)

// All races ingested
RaceEvents
| order by race_start_time desc
| project race_name, race_start_time, event_subgroup_id
```

## Data Volume Estimates

| Metric                            | Estimate                    |
| --------------------------------- | --------------------------- |
| Telemetry rows per rider per race | ~2,000 (30 min race @ 1 Hz) |
| Riders per event subgroup         | ~50-200                     |
| Telemetry rows per event          | ~100K-400K                  |
| Events per day (Zwift Games)      | ~10-50                      |
| Daily telemetry rows              | ~1M-20M                     |
| ADX free tier storage limit       | 100 GB                      |
| Estimated time to fill            | ~1-2 years                  |

## File Structure

```
azure_function/
    function_app.py          # Timer trigger ETL (every 15 min)
    zwift_client.py          # Zwift API client (auth + data fetching)
    kusto_ingest_module.py   # ADX queued ingestion helpers
    ingest_dimension_tables.py # CLI: populate dimension tables in ADX
    manual_ingest.py         # CLI: one-off ingestion of missed events
    host.json                # Function runtime config
    requirements.txt         # Python dependencies
    local.settings.json      # Local dev settings

infra/
    main.bicep               # ARM/Bicep template (Function App, Storage, App Insights)
    parameters.json          # Deployment parameters
    deploy.ps1               # End-to-end deployment script
    kql/
        create_tables.kql    # All table + function definitions
```

## Utility Scripts

### `ingest_dimension_tables.py`

Standalone CLI script that populates all 8 dimension tables in ADX (`Frames`, `Wheels`, `BikeConfigs`, `Routes`, `RoutePoints`, `Worlds`, `SurfaceCrr`, `RouteStravaSegments`). Uses `AzureCliCredential` for local dev auth. Performs a full clear-and-replace for each table.

```bash
cd azure_function
python ingest_dimension_tables.py
```

### `manual_ingest.py`

CLI script for one-off ingestion of missed events by event ID. Fetches subgroups from the Zwift API, deduplicates against ADX, and runs the same ingestion logic as the timer function.

```bash
cd azure_function
python manual_ingest.py 12345 67890
```

## Deployment

The Azure Function is automatically deployed via **GitHub Actions** when changes are pushed to `azure_function/` on `main`. The workflow (`.github/workflows/deploy-function.yml`) uses the `Azure/functions-action` with a publish profile.

**Required secret:** `AZURE_FUNCTIONAPP_PUBLISH_PROFILE` — download from the Function App in the Azure Portal.

### Infrastructure (one-time)

```powershell
cd infra
.\deploy.ps1
```

The script runs four steps:
1. Deploys the Bicep template (Function App, Storage, App Insights)
2. Creates the `ZwiftGames` database in ADX (or prompts for manual creation)
3. Executes all KQL table-creation commands
4. Updates Function App settings with ADX connection URIs

After deployment, set credentials and publish:
```powershell
az functionapp config appsettings set -g zwifttools -n zwiftgames-func-hc4trbmw4t6kc `
    --settings "ZWIFT_USERNAME=..." "ZWIFT_PASSWORD=..." --output none

cd azure_function
func azure functionapp publish zwiftgames-func-hc4trbmw4t6kc --build remote
```

## ADX Cluster

| Property      | Value                                                                    |
| ------------- | ------------------------------------------------------------------------ |
| Cluster URI   | `https://kvcayg2gmw1snvk9sceve0.southcentralus.kusto.windows.net`        |
| Ingestion URI | `https://ingest-kvcayg2gmw1snvk9sceve0.southcentralus.kusto.windows.net` |
| Database      | `ZwiftGames`                                                             |
| Tier          | Free (100 GB storage, 10 GB/day ingestion)                               |
| Query UI      | [https://dataexplorer.azure.com](https://dataexplorer.azure.com)         |

## Estimated Cost

| Resource                         | Monthly Cost        |
| -------------------------------- | ------------------- |
| ADX free cluster                 | $0                  |
| Function App (B1 shared plan)    | Already provisioned |
| Storage Account (minimal usage)  | ~$1                 |
| Application Insights (free tier) | $0                  |
| **Total**                        | **~$1/month**       |
