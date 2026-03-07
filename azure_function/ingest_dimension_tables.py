"""
Populate dimension / reference tables in the ZwiftGames ADX database.

Reads local JSON files and ingests them into the corresponding KQL tables.
Clears each table before ingesting (dimension tables are full-replace).

Usage:
    cd azure_function
    python ingest_dimension_tables.py

Requires env vars (or local.settings.json values):
    KUSTO_CLUSTER_URI, KUSTO_INGEST_URI, KUSTO_DATABASE
"""

import json
import logging
import os
import sys
import glob
import pandas as pd
from azure.identity import AzureCliCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.ingest import QueuedIngestClient, IngestionProperties

# Ensure azure_function is on the path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Paths relative to the workspace root (parent of azure_function/)
WORKSPACE = os.path.dirname(os.path.dirname(__file__))


def _load_local_settings():
    """Load env vars from local.settings.json if present."""
    settings_path = os.path.join(os.path.dirname(__file__), "local.settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            settings = json.load(f)
        for k, v in settings.get("Values", {}).items():
            if k not in os.environ:
                os.environ[k] = v
        logger.info("Loaded local.settings.json")


def _db() -> str:
    return os.environ.get("KUSTO_DATABASE", "ZwiftGames")


def _get_credential():
    return AzureCliCredential()


def get_query_client() -> KustoClient:
    cluster_uri = os.environ["KUSTO_CLUSTER_URI"]
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(cluster_uri, _get_credential())
    return KustoClient(kcsb)


def get_ingest_client() -> QueuedIngestClient:
    ingest_uri = os.environ["KUSTO_INGEST_URI"]
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(ingest_uri, _get_credential())
    return QueuedIngestClient(kcsb)


def clear_table(table_name: str, query_client):
    """Clear all data from a table (for full-replace dimension tables)."""
    cmd = f".clear table {table_name} data"
    query_client.execute_mgmt(_db(), cmd)
    logger.info(f"Cleared table {table_name}")


# ---------------------------------------------------------------------------
# Data loaders — one per dimension table
# ---------------------------------------------------------------------------

def load_frames() -> pd.DataFrame:
    """Load Frames from zwiftdata/frames.json."""
    path = os.path.join(WORKSPACE, "zwiftdata", "frames.json")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for fr in data:
        rows.append({
            "frameid": fr["frameid"],
            "framemake": fr["framemake"],
            "framemodel": fr["framemodel"],
            "frameprice": int(fr["frameprice"]),
            "framelevel": int(fr["framelevel"]),
            "frameaero": int(fr["frameaero"]),
            "frameweight": int(fr["frameweight"]),
            "frametype": fr["frametype"],
            "framewheeltype": fr["framewheeltype"],
            "frameflatspeed": fr["frameflatspeed"],   # dynamic (list)
            "frameclimbspeed": fr["frameclimbspeed"],  # dynamic (list)
        })

    df = pd.DataFrame(rows)
    logger.info(f"Frames: {len(df)} rows")
    return df


def load_wheels() -> pd.DataFrame:
    """Load Wheels from zwiftdata/wheels.json."""
    path = os.path.join(WORKSPACE, "zwiftdata", "wheels.json")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for w in data:
        rows.append({
            "wheelid": w["wheelid"],
            "wheelmake": w["wheelmake"],
            "wheelmodel": w["wheelmodel"],
            "wheelprice": int(w["wheelprice"]),
            "wheellevel": int(w["wheellevel"]),
            "wheelaero": int(w["wheelaero"]),
            "wheelweight": int(w["wheelweight"]),
            "wheelfitsframe": w["wheelfitsframe"],
            "wheelflatspeedonroadframe": float(w["wheelflatspeedonroadframe"]),
            "wheelclimbspeedonroadframe": float(w["wheelclimbspeedonroadframe"]),
            "wheelflatspeedonttframe": float(w["wheelflatspeedonttframe"]),
            "wheelclimbspeedonttframe": float(w["wheelclimbspeedonttframe"]),
            "wheelflatrank": int(w["wheelflatrank"]),
            "wheelclimbrank": int(w["wheelclimbrank"]),
        })

    df = pd.DataFrame(rows)
    logger.info(f"Wheels: {len(df)} rows")
    return df


def load_bike_configs() -> pd.DataFrame:
    """Load BikeConfigs from zwiftdata/bikes.json."""
    path = os.path.join(WORKSPACE, "zwiftdata", "bikes.json")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for b in data:
        rows.append({
            "frameid": b["frameid"],
            "wheelid": b["wheelid"],
            "cd": b["cd"],        # dynamic (list)
            "weight": b["weight"],  # dynamic (list)
        })

    df = pd.DataFrame(rows)
    logger.info(f"BikeConfigs: {len(df)} rows")
    return df


def load_routes() -> pd.DataFrame:
    """Load Routes from routes_cache.json."""
    path = os.path.join(WORKSPACE, "routes_cache.json")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for route_id, r in data.items():
        rows.append({
            "route_id": route_id,
            "name": r["name"],
            "distance_m": float(r["distanceInMeters"]),
            "leadin_distance_m": float(r["leadinDistanceInMeters"]),
            "ascent_m": float(r["ascentInMeters"]),
            "leadin_ascent_m": float(r["leadinAscentInMeters"]),
            "difficulty": r["difficulty"],
            "event_only": r["eventOnly"],
        })

    df = pd.DataFrame(rows)
    logger.info(f"Routes: {len(df)} rows")
    return df


def load_route_points() -> pd.DataFrame:
    """Load RoutePoints from zwiftmap_surfaces/*_route.json files."""
    pattern = os.path.join(WORKSPACE, "zwiftmap_surfaces", "*_route.json")
    files = glob.glob(pattern)

    all_rows = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)

        route_slug = data["routeSlug"]
        route_name = data["routeName"]
        strava_segment_id = data["stravaSegmentId"]
        total_distance_m = data["totalDistance"]

        for idx, point in enumerate(data["latlng"]):
            all_rows.append({
                "route_slug": route_slug,
                "route_name": route_name,
                "strava_segment_id": int(strava_segment_id),
                "total_distance_m": float(total_distance_m),
                "point_index": idx,
                "latitude": float(point[0]),
                "longitude": float(point[1]),
            })

    df = pd.DataFrame(all_rows)
    logger.info(f"RoutePoints: {len(df)} rows from {len(files)} route files")
    return df


def load_worlds() -> pd.DataFrame:
    """Load Worlds from shared/world_config.py WORLD_CONFIG dict."""
    # Import the config directly
    sys.path.insert(0, os.path.join(WORKSPACE, "shared"))
    from world_config import WORLD_CONFIG

    rows = []
    for world_name, cfg in WORLD_CONFIG.items():
        rows.append({
            "world_name": world_name,
            "slug": cfg["slug"],
            "lat_min": cfg["lat_min"],
            "lat_max": cfg["lat_max"],
            "lng_min": cfg["lng_min"],
            "lng_max": cfg["lng_max"],
            "bg_color": cfg["bg_color"],
        })

    df = pd.DataFrame(rows)
    logger.info(f"Worlds: {len(df)} rows")
    return df


def load_surface_crr() -> pd.DataFrame:
    """Load SurfaceCrr from zwiftmap_surfaces/surface_data.json."""
    path = os.path.join(WORKSPACE, "zwiftmap_surfaces", "surface_data.json")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for bike_type, surfaces in data["crr_values"].items():
        for surface_type, crr_value in surfaces.items():
            if crr_value is not None:
                rows.append({
                    "bike_type": bike_type,
                    "surface_type": surface_type,
                    "crr_value": float(crr_value),
                })

    df = pd.DataFrame(rows)
    logger.info(f"SurfaceCrr: {len(df)} rows")
    return df


def load_route_strava_segments() -> pd.DataFrame:
    """Load RouteStravaSegments from route_strava_segments.json."""
    path = os.path.join(WORKSPACE, "route_strava_segments.json")
    with open(path) as f:
        data = json.load(f)

    rows = []
    for route_slug, strava_id in data.items():
        rows.append({
            "route_slug": route_slug,
            "strava_segment_id": int(strava_id),
        })

    df = pd.DataFrame(rows)
    logger.info(f"RouteStravaSegments: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Main — clear + ingest each dimension table
# ---------------------------------------------------------------------------

DIMENSION_TABLES = {
    "Frames": load_frames,
    "Wheels": load_wheels,
    "BikeConfigs": load_bike_configs,
    "Routes": load_routes,
    "RoutePoints": load_route_points,
    "Worlds": load_worlds,
    "SurfaceCrr": load_surface_crr,
    "RouteStravaSegments": load_route_strava_segments,
}

# Column orders matching the KQL CREATE TABLE definitions exactly
DIMENSION_COLUMN_ORDER = {
    "Frames": [
        "frameid", "framemake", "framemodel", "frameprice", "framelevel",
        "frameaero", "frameweight", "frametype", "framewheeltype",
        "frameflatspeed", "frameclimbspeed",
    ],
    "Wheels": [
        "wheelid", "wheelmake", "wheelmodel", "wheelprice", "wheellevel",
        "wheelaero", "wheelweight", "wheelfitsframe",
        "wheelflatspeedonroadframe", "wheelclimbspeedonroadframe",
        "wheelflatspeedonttframe", "wheelclimbspeedonttframe",
        "wheelflatrank", "wheelclimbrank",
    ],
    "BikeConfigs": ["frameid", "wheelid", "cd", "weight"],
    "Routes": [
        "route_id", "name", "distance_m", "leadin_distance_m",
        "ascent_m", "leadin_ascent_m", "difficulty", "event_only",
    ],
    "RoutePoints": [
        "route_slug", "route_name", "strava_segment_id",
        "total_distance_m", "point_index", "latitude", "longitude",
    ],
    "Worlds": [
        "world_name", "slug", "lat_min", "lat_max",
        "lng_min", "lng_max", "bg_color",
    ],
    "SurfaceCrr": ["bike_type", "surface_type", "crr_value"],
    "RouteStravaSegments": ["route_slug", "strava_segment_id"],
}


def ingest_dimension_table(table_name, df, ingest_client):
    """Ingest a DataFrame into a dimension table with correct column ordering."""
    if df.empty:
        logger.warning(f"Skipping empty DataFrame for {table_name}")
        return

    # Reorder columns to match KQL schema
    col_order = DIMENSION_COLUMN_ORDER[table_name]
    df = df[col_order]

    props = IngestionProperties(database=_db(), table=table_name)
    ingest_client.ingest_from_dataframe(df, ingestion_properties=props)
    logger.info(f"Queued {len(df)} rows for ingestion into {table_name}")


def main():
    _load_local_settings()

    query_client = get_query_client()
    ingest_client = get_ingest_client()

    for table_name, loader in DIMENSION_TABLES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {table_name}...")

        # Load data from JSON files
        df = loader()
        if df.empty:
            logger.warning(f"No data for {table_name}, skipping")
            continue

        # Clear existing data (dimension tables are full-replace)
        try:
            clear_table(table_name, query_client)
        except Exception as e:
            logger.warning(f"Could not clear {table_name} (may be empty): {e}")

        # Ingest new data
        ingest_dimension_table(table_name, df, ingest_client)
        logger.info(f"Done: {table_name}")

    logger.info(f"\n{'='*60}")
    logger.info("All dimension tables queued for ingestion.")
    logger.info("Note: Queued ingestion may take a few minutes to complete.")


if __name__ == "__main__":
    main()
