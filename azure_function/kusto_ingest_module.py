"""
Kusto Ingestion Module -- writes data to Azure Data Explorer.

Uses queued ingestion for cost-efficient batch loading.
Falls back to Azure CLI credential for local dev; uses managed identity in Azure.
"""

import os
import logging
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.ingest import QueuedIngestClient, IngestionProperties

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column orders – must match the KQL CREATE TABLE definitions exactly
# ---------------------------------------------------------------------------

TABLE_COLUMN_ORDER = {
    'RaceEvents': [
        'event_subgroup_id', 'event_id', 'race_name', 'race_start_time',
        'world_id', 'route_id', 'ingested_at',
    ],
    'RaceSummary': [
        'event_subgroup_id', 'activity_id', 'rank', 'name', 'weight_kg',
        'player_id', 'file_id', 'activity_start_time', 'status',
        'duration_sec', 'avg_power', 'normalized_power', 'max_power',
        'avg_hr', 'data_points',
    ],
    'RiderTelemetry': [
        'event_subgroup_id', 'activity_id', 'player_id', 'time_sec',
        'power_watts', 'hr_bpm', 'cadence_rpm', 'speed_kmh',
        'distance_km', 'altitude_m',
    ],
}

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_credential():
    """Get Azure credential (managed identity in Azure, CLI locally)."""
    return DefaultAzureCredential()

def get_query_client() -> KustoClient:
    """Get a KQL query client for reading data."""
    cluster_uri = os.environ["KUSTO_CLUSTER_URI"]
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(cluster_uri, _get_credential())
    return KustoClient(kcsb)

def get_ingest_client() -> QueuedIngestClient:
    """Get a queued ingestion client for writing data."""
    ingest_uri = os.environ["KUSTO_INGEST_URI"]
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(ingest_uri, _get_credential())
    return QueuedIngestClient(kcsb)

def _db() -> str:
    return os.environ.get("KUSTO_DATABASE", "ZwiftGames")

# ---------------------------------------------------------------------------
# Ingestion functions
# ---------------------------------------------------------------------------

def ingest_dataframe(table_name: str, df: pd.DataFrame, ingest_client: QueuedIngestClient):
    """
    Ingest a pandas DataFrame into a KQL table.
    
    Reorders columns to match the target table schema (ADX uses ordinal
    mapping by default when ingesting from CSV/DataFrame).
    
    Args:
        table_name: Target KQL table name.
        df: DataFrame whose columns match the table schema.
        ingest_client: Queued ingest client instance.
    """
    if df.empty:
        logger.info(f"Skipping empty DataFrame for {table_name}")
        return

    # Reorder columns to match table schema (ordinal mapping)
    if table_name in TABLE_COLUMN_ORDER:
        expected = TABLE_COLUMN_ORDER[table_name]
        available = [c for c in expected if c in df.columns]
        df = df[available]

    props = IngestionProperties(
        database=_db(),
        table=table_name,
    )
    ingest_client.ingest_from_dataframe(df, ingestion_properties=props)
    logger.info(f"Queued {len(df)} rows for ingestion into {table_name}")


def is_event_ingested(event_subgroup_id: int, query_client: KustoClient) -> bool:
    """
    Check whether a given event_subgroup_id has already been ingested.
    
    Returns True if at least one row exists in RaceEvents for this ID.
    """
    query = f"RaceEvents | where event_subgroup_id == {event_subgroup_id} | take 1 | count"
    result = query_client.execute(_db(), query)
    for row in result.primary_results[0]:
        return row["Count"] > 0
    return False


def get_ingested_event_ids(query_client: KustoClient) -> set:
    """
    Return the set of all event_subgroup_id values already ingested.
    Useful for batch dedup.
    """
    query = "RaceEvents | summarize by event_subgroup_id | project event_subgroup_id"
    result = query_client.execute(_db(), query)
    return {row["event_subgroup_id"] for row in result.primary_results[0]}


def get_incomplete_riders(
    query_client: KustoClient, max_age_hours: int = 24
) -> dict[int, list[dict]]:
    """
    Find riders with non-SUCCESS status in recently ingested events.

    Returns ``{event_subgroup_id: [rider_dicts]}`` for backfill.
    Only considers events whose ``ingested_at`` is within *max_age_hours*.
    """
    query = (
        f"let cutoff = ago({max_age_hours}h);\n"
        "let recent = RaceEvents | where ingested_at > cutoff "
        "| project event_subgroup_id;\n"
        "let already_ok = RaceSummary\n"
        "| where event_subgroup_id in (recent)\n"
        "| where status == 'SUCCESS'\n"
        "| distinct event_subgroup_id, activity_id;\n"
        "RaceSummary\n"
        "| where event_subgroup_id in (recent)\n"
        "| where status != 'SUCCESS'\n"
        "| where status !has 'HTTP 403' // Exclude private riders\n"
        "| join kind=leftanti already_ok on event_subgroup_id, activity_id\n"
        "| project event_subgroup_id, activity_id, player_id, "
        "rank, name, weight_kg"
    )
    result = query_client.execute(_db(), query)
    groups: dict[int, list[dict]] = {}
    for row in result.primary_results[0]:
        sg_id = row["event_subgroup_id"]
        groups.setdefault(sg_id, []).append({
            'activity_id': str(row["activity_id"]),
            'player_id': row["player_id"],
            'rank': row["rank"],
            'name': row["name"],
            'weight_kg': row["weight_kg"],
        })
    return groups
