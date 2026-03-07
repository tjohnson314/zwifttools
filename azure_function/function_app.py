"""
Zwift Games ETL -- Azure Function (Timer Trigger)

Runs every 15 minutes. For each Zwift Games event subgroup that has finished
but hasn't been ingested yet, fetches all rider telemetry and ingests it
into the Azure Data Explorer KQL database.
"""

import logging
import datetime
import pandas as pd
import azure.functions as func

from zwift_client import get_zwift_games_events, fetch_full_race, backfill_riders
from kusto_ingest_module import (
    get_query_client,
    get_ingest_client,
    get_ingested_event_ids,
    get_incomplete_riders,
    ingest_dataframe,
)

app = func.FunctionApp()

logger = logging.getLogger(__name__)


@app.timer_trigger(
    schedule="0 */15 * * * *",        # Every 15 minutes
    arg_name="timer",
    run_on_startup=False,
)
def zwift_games_etl(timer: func.TimerRequest) -> None:
    """
    Timer-triggered ETL:
      1. Discover Zwift Games event subgroups
      2. Skip already-ingested events
      3. Fetch race data (participants + telemetry)
      4. Ingest into KQL tables: RaceEvents, RaceSummary, RiderTelemetry
    """
    utc_now = datetime.datetime.now(datetime.timezone.utc)

    if timer.past_due:
        logger.warning("Timer is past due — running catch-up")

    logger.info("Zwift Games ETL triggered at %s", utc_now.isoformat())

    # ── 0. Clients ────────────────────────────────────────────────────────
    query_client = get_query_client()
    ingest_client = get_ingest_client()

    # ── 1. Discover events ────────────────────────────────────────────────
    subgroups = get_zwift_games_events() or []
    if not subgroups:
        logger.info("No Zwift Games events found")

    # ── 2. Filter out future events and already-ingested ones ─────────────
    already_ingested = get_ingested_event_ids(query_client) if subgroups else set()
    logger.info("Already ingested %d events", len(already_ingested))

    candidates = []
    for sg in subgroups:
        sg_id = sg.get('id')
        if not sg_id or sg_id in already_ingested:
            logger.info("  Subgroup %s: skipped (no id or already ingested)", sg_id)
            continue

        # Only process events whose start time has passed (finished races)
        start_val = sg.get('eventSubgroupStart', '')
        sg_name = sg.get('parentEventName', sg.get('name', '?'))
        logger.info("  Subgroup %s [%s]: eventSubgroupStart=%r (type=%s)",
                     sg_id, sg_name, start_val, type(start_val).__name__)
        if start_val:
            try:
                if isinstance(start_val, (int, float)) and start_val > 1_000_000_000_000:
                    # epoch-ms (from protobuf)
                    start_dt = datetime.datetime.fromtimestamp(
                        start_val / 1000, tz=datetime.timezone.utc)
                else:
                    start_dt = datetime.datetime.fromisoformat(
                        str(start_val).replace('+0000', '+00:00'))
                cutoff = start_dt + datetime.timedelta(minutes=30)
                if cutoff > utc_now:
                    logger.info("    => SKIP (future): start=%s cutoff=%s now=%s",
                                start_dt.isoformat(), cutoff.isoformat(), utc_now.isoformat())
                    continue
                else:
                    logger.info("    => READY: start=%s (finished)",
                                start_dt.isoformat())
            except (ValueError, OSError) as exc:
                logger.warning("    => INCLUDE (unparseable start): %s", exc)

        candidates.append(sg)

    if candidates:
        logger.info("Processing %d new event subgroups", len(candidates))
    else:
        logger.info("No new events to ingest")

    # ── 3. Fetch and ingest each event ────────────────────────────────────
    events_ingested = 0

    for sg in candidates:
        sg_id = sg['id']
        event_name = sg.get('parentEventName', sg.get('name', 'Unknown'))
        logger.info("Processing event %s: %s", sg_id, event_name)

        try:
            event_meta = {
                'event_subgroup_id': sg_id,
                'event_id': sg.get('eventId') or sg.get('id'),
                'race_name': event_name,
                'race_start_time': sg.get('eventSubgroupStart'),
                'world_id': sg.get('mapId'),
                'route_id': sg.get('routeId'),
            }

            # Fetch all rider data
            telemetry_df, summary_df = fetch_full_race(sg_id, event_meta)

            if summary_df.empty:
                logger.warning("No data for event %s — skipping", sg_id)
                continue

            # ── Ingest RaceEvents (one row) ───────────────────────────────
            race_event_df = pd.DataFrame([{
                'event_subgroup_id': sg_id,
                'event_id': event_meta.get('event_id'),
                'race_name': event_name,
                'race_start_time': event_meta.get('race_start_time'),
                'world_id': event_meta.get('world_id'),
                'route_id': event_meta.get('route_id'),
                'ingested_at': utc_now.isoformat(),
            }])
            ingest_dataframe('RaceEvents', race_event_df, ingest_client)

            # ── Ingest RaceSummary ────────────────────────────────────────
            ingest_dataframe('RaceSummary', summary_df, ingest_client)

            # ── Ingest RiderTelemetry ─────────────────────────────────────
            if not telemetry_df.empty:
                # Ingest in chunks of 100K rows to stay within ingestion limits
                chunk_size = 100_000
                for i in range(0, len(telemetry_df), chunk_size):
                    chunk = telemetry_df.iloc[i:i + chunk_size]
                    ingest_dataframe('RiderTelemetry', chunk, ingest_client)

            events_ingested += 1
            success_count = (summary_df['status'] == 'SUCCESS').sum() if 'status' in summary_df.columns else 0
            logger.info(
                "Ingested event %s: %d riders, %d telemetry rows",
                sg_id, success_count, len(telemetry_df)
            )

        except Exception as e:
            logger.error("Failed to process event %s: %s", sg_id, e, exc_info=True)
            continue

    logger.info("ETL complete: %d events ingested", events_ingested)

    # ── 4. Backfill incomplete events ─────────────────────────────────────
    try:
        incomplete = get_incomplete_riders(query_client)
        if incomplete:
            total_riders = sum(len(r) for r in incomplete.values())
            logger.info("Backfill: %d subgroups, %d riders with incomplete data",
                        len(incomplete), total_riders)
            backfill_count = 0
            for sg_id, riders in incomplete.items():
                telem_df, summary_df = backfill_riders(sg_id, riders)
                if not summary_df.empty:
                    # Only ingest riders that actually succeeded this time
                    succeeded = summary_df[summary_df['status'] == 'SUCCESS'] if 'status' in summary_df.columns else summary_df
                    if not succeeded.empty:
                        ingest_dataframe('RaceSummary', succeeded, ingest_client)
                        if not telem_df.empty:
                            succeeded_ids = set(succeeded['activity_id'])
                            telem_ok = telem_df[telem_df['activity_id'].isin(succeeded_ids)]
                            chunk_size = 100_000
                            for i in range(0, len(telem_ok), chunk_size):
                                chunk = telem_ok.iloc[i:i + chunk_size]
                                ingest_dataframe('RiderTelemetry', chunk, ingest_client)
                        backfill_count += len(succeeded)
            logger.info("Backfill complete: %d riders recovered", backfill_count)
        else:
            logger.info("Backfill: no incomplete riders to retry")
    except Exception as e:
        logger.error("Backfill error: %s", e, exc_info=True)
