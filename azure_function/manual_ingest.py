"""
One-off script to manually ingest missed Zwift Games events by event ID.

Usage:
    cd azure_function
    # Ensure env vars are set: ZWIFT_USERNAME, ZWIFT_PASSWORD,
    #   KUSTO_CLUSTER_URI, KUSTO_INGEST_URI, KUSTO_DATABASE
    python manual_ingest.py 5422344 5422345
"""

import os
import sys
import logging
import datetime
import pandas as pd

from azure.identity import AzureCliCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.ingest import QueuedIngestClient

# Reuse existing modules
from zwift_client import fetch_full_race, _request, BASE_URL
from kusto_ingest_module import ingest_dataframe


def _get_local_credential():
    return AzureCliCredential()


def get_local_query_client() -> KustoClient:
    """Build a query client using AzureCliCredential (for local dev)."""
    cluster_uri = os.environ["KUSTO_CLUSTER_URI"]
    cred = _get_local_credential()
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(cluster_uri, cred)
    return KustoClient(kcsb)


def get_local_ingest_client() -> QueuedIngestClient:
    """Build an ingest client using AzureCliCredential (for local dev)."""
    ingest_uri = os.environ["KUSTO_INGEST_URI"]
    cred = _get_local_credential()
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(ingest_uri, cred)
    return QueuedIngestClient(kcsb)


def get_already_ingested(query_client: KustoClient) -> set[int]:
    """Return the set of event_subgroup_ids already present in RaceEvents."""
    db = os.environ.get("KUSTO_DATABASE", "ZwiftGames")
    query = "RaceEvents | summarize by event_subgroup_id | project event_subgroup_id"
    result = query_client.execute(db, query)
    return {row["event_subgroup_id"] for row in result.primary_results[0]}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("manual_ingest_log.txt", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_event_subgroups(event_id: int) -> list[dict]:
    """Fetch subgroups for a given event ID from the Zwift API."""
    resp = _request('GET', f"{BASE_URL}/events/{event_id}", timeout=15)
    if resp.status_code != 200:
        logger.error("Failed to fetch event %s: HTTP %s", event_id, resp.status_code)
        return []

    try:
        data = resp.json()
    except Exception:
        logger.error("Non-JSON response for event %s", event_id)
        return []

    subgroups = data.get('eventSubgroups', [])
    event_name = data.get('name', 'Unknown')
    logger.info("Event %s (%s): %d subgroups", event_id, event_name, len(subgroups))

    # Enrich each subgroup with parent info
    for sg in subgroups:
        sg['parentEventName'] = event_name
        sg['eventId'] = event_id
    return subgroups


def ingest_event_subgroup(sg: dict, ingest_client, utc_now: datetime.datetime):
    """Ingest a single event subgroup (same logic as function_app.py step 3)."""
    sg_id = sg.get('id')
    event_name = sg.get('parentEventName', sg.get('name', 'Unknown'))
    logger.info("Processing subgroup %s: %s", sg_id, event_name)

    event_meta = {
        'event_subgroup_id': sg_id,
        'event_id': sg.get('eventId') or sg_id,
        'race_name': event_name,
        'race_start_time': sg.get('eventSubgroupStart'),
        'world_id': sg.get('mapId'),
        'route_id': sg.get('routeId'),
    }

    telemetry_df, summary_df = fetch_full_race(sg_id, event_meta)

    if summary_df.empty:
        logger.warning("No data for subgroup %s — skipping", sg_id)
        return False

    # Ingest RaceEvents
    race_event_df = pd.DataFrame([{
        'event_subgroup_id': sg_id,
        'event_id': event_meta['event_id'],
        'race_name': event_name,
        'race_start_time': event_meta.get('race_start_time'),
        'world_id': event_meta.get('world_id'),
        'route_id': event_meta.get('route_id'),
        'ingested_at': utc_now.isoformat(),
    }])
    ingest_dataframe('RaceEvents', race_event_df, ingest_client)

    # Ingest RaceSummary
    ingest_dataframe('RaceSummary', summary_df, ingest_client)

    # Ingest RiderTelemetry
    if not telemetry_df.empty:
        chunk_size = 100_000
        for i in range(0, len(telemetry_df), chunk_size):
            chunk = telemetry_df.iloc[i:i + chunk_size]
            ingest_dataframe('RiderTelemetry', chunk, ingest_client)

    success_count = (summary_df['status'] == 'SUCCESS').sum() if 'status' in summary_df.columns else 0
    logger.info(
        "Ingested subgroup %s: %d riders, %d telemetry rows",
        sg_id, success_count, len(telemetry_df),
    )
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} EVENT_ID [EVENT_ID ...]")
        sys.exit(1)

    event_ids = [int(x) for x in sys.argv[1:]]
    utc_now = datetime.datetime.now(datetime.timezone.utc)

    logger.info("Manual ingest for event IDs: %s", event_ids)

    query_client = get_local_query_client()
    ingest_client = get_local_ingest_client()

    # Dedup: skip subgroups already in ADX
    already_ingested = get_already_ingested(query_client)
    logger.info("Already ingested %d subgroups in ADX", len(already_ingested))

    total_subgroups = 0

    for event_id in event_ids:
        subgroups = get_event_subgroups(event_id)
        if not subgroups:
            logger.warning("No subgroups found for event %s", event_id)
            continue

        for sg in subgroups:
            sg_id = sg.get('id')
            if sg_id in already_ingested:
                logger.info("Skipping subgroup %s (already ingested)", sg_id)
                continue
            try:
                ok = ingest_event_subgroup(sg, ingest_client, utc_now)
                if ok:
                    total_subgroups += 1
            except Exception as e:
                logger.error("Failed subgroup %s: %s", sg.get('id'), e, exc_info=True)

    logger.info("Done: ingested %d subgroups across %d events", total_subgroups, len(event_ids))


if __name__ == '__main__':
    main()
