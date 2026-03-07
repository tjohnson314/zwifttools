"""
Zwift API client — adapted for Azure Function (headless, password-grant auth).

Reuses logic from shared/zwift_auth.py and shared/data_fetcher.py but without
session state or Flask dependencies.
"""

import os
import time
import logging
import datetime
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZWIFT_TOKEN_URL = "https://secure.zwift.com/auth/realms/zwift/protocol/openid-connect/token"
CLIENT_ID = "Zwift_Mobile_Link"
BASE_URL = "https://us-or-rly101.zwift.com/api"

_MAX_RETRIES = 3
_RETRY_BACKOFF = [1, 3]
_CONCURRENT_WORKERS = 5

# Headers that mimic the Zwift mobile client (required to avoid 406)
_ZWIFT_HEADERS = {
    'User-Agent': 'Zwift/115 CFNetwork/758.0.2 Darwin/15.0.0',
    'Accept-Encoding': 'gzip',
    'Connection': 'keep-alive',
}

# Module-level token cache
_cached_token = None
_token_expires_at = 0.0


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _get_access_token() -> str:
    """Get a valid Zwift access token, refreshing if needed."""
    global _cached_token, _token_expires_at

    if _cached_token and time.time() < (_token_expires_at - 60):
        return _cached_token

    username = os.environ["ZWIFT_USERNAME"]
    password = os.environ["ZWIFT_PASSWORD"]

    resp = requests.post(ZWIFT_TOKEN_URL, data={
        'client_id': CLIENT_ID,
        'grant_type': 'password',
        'username': username,
        'password': password,
    }, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    _cached_token = data['access_token']
    _token_expires_at = time.time() + data.get('expires_in', 3600)
    logger.info("Zwift token acquired (expires in %ds)", data.get('expires_in', 3600))
    return _cached_token


def _auth_headers() -> dict:
    token = _get_access_token()
    return {
        **_ZWIFT_HEADERS,
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
    }


# ---------------------------------------------------------------------------
# HTTP helper with retry
# ---------------------------------------------------------------------------

def _request(method, url, **kwargs):
    headers = kwargs.pop('headers', {}) or {}
    headers.setdefault('Accept', 'application/json')
    kwargs['headers'] = {**_auth_headers(), **headers}

    last = None
    for attempt in range(_MAX_RETRIES):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get('Retry-After', 5)))
            last = resp
            continue
        if resp.status_code < 500:
            return resp
        last = resp
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BACKOFF[attempt])
    return last


# ---------------------------------------------------------------------------
# Protobuf wire-format decoder
#
# The /api/events/search endpoint returns application/x-protobuf-lite.
# We decode it with a minimal wire-format parser (no extra dependencies).
# ---------------------------------------------------------------------------

def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a protobuf varint starting at *pos*. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise ValueError("Truncated varint")


def _decode_pb(data: bytes) -> dict[int, list]:
    """Parse protobuf wire format into ``{field_number: [values]}``."""
    fields: dict[int, list] = {}
    pos = 0
    end = len(data)
    while pos < end:
        try:
            tag, pos = _decode_varint(data, pos)
        except ValueError:
            break
        wire_type = tag & 0x07
        field_num = tag >> 3
        if wire_type == 0:          # varint
            val, pos = _decode_varint(data, pos)
        elif wire_type == 1:        # 64-bit fixed
            if pos + 8 > end:
                break
            val = int.from_bytes(data[pos:pos + 8], "little")
            pos += 8
        elif wire_type == 2:        # length-delimited (bytes / string / sub-msg)
            length, pos = _decode_varint(data, pos)
            if pos + length > end:
                break
            val = data[pos:pos + length]
            pos += length
        elif wire_type == 5:        # 32-bit fixed
            if pos + 4 > end:
                break
            val = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
        else:
            break  # wire types 3/4 are deprecated
        fields.setdefault(field_num, []).append(val)
    return fields


def _pb_int(fields: dict, num: int, default: int = 0) -> int:
    vals = fields.get(num)
    return vals[0] if vals and isinstance(vals[0], int) else default


def _pb_str(fields: dict, num: int, default: str = "") -> str:
    vals = fields.get(num)
    if vals and isinstance(vals[0], (bytes, bytearray)):
        return vals[0].decode("utf-8", errors="replace")
    return default


def _summarise_pb_fields(fields: dict) -> dict:
    """Build a human-readable summary of protobuf fields for debug logging."""
    out = {}
    for fn in sorted(fields):
        vals = fields[fn]
        v0 = vals[0]
        n = len(vals)
        if isinstance(v0, int):
            out[fn] = f"int={v0} x{n}"
        elif isinstance(v0, (bytes, bytearray)):
            try:
                txt = v0.decode("utf-8")
                if len(txt) > 60:
                    txt = txt[:60] + "…"
                out[fn] = f'str="{txt}" x{n}'
            except UnicodeDecodeError:
                out[fn] = f"bytes({len(v0)}) x{n}"
        else:
            out[fn] = f"{type(v0).__name__} x{n}"
    return out


def _parse_events_protobuf(raw: bytes) -> list[dict]:
    """
    Parse the ``/api/events/search`` protobuf-lite response.

    Returns a list of event dicts with keys:
      id, name, description, eventSubgroups (from protobuf),
      _pb_field_nums (for debugging).

    Protobuf field mapping (from sample data):
      Event: 1=id(varint)  3=name(str)  4=description(str)
      Event subgroups are in whichever repeated embedded-message field
      contains items whose own field-1 is an int (auto-detected).
    """
    top = _decode_pb(raw)
    event_blobs = [v for v in top.get(1, []) if isinstance(v, (bytes, bytearray))]

    if not event_blobs:
        logger.warning("protobuf: no event blobs (top-level fields: %s)",
                       sorted(top.keys()))
        return []

    events: list[dict] = []
    for idx, blob in enumerate(event_blobs):
        ef = _decode_pb(blob)

        event: dict = {
            'id': _pb_int(ef, 1),
            'name': _pb_str(ef, 3),
            'description': _pb_str(ef, 4),
            'eventSubgroups': [],
            'tags': [],
        }

        # --- Debug: log the first event's complete field structure ---
        if idx == 0:
            logger.info("PROTO event[0] fields: %s", _summarise_pb_fields(ef))

        # --- Discover event subgroups ---
        # Subgroups are a *repeated* embedded message whose own field-1
        # is an integer ID.  Usually there are ≥2 (categories A–E).
        for fn in sorted(ef):
            if fn <= 5:
                continue
            vals = ef[fn]
            blobs = [v for v in vals
                     if isinstance(v, (bytes, bytearray)) and len(v) > 20]
            if len(blobs) < 2:
                continue
            try:
                probe = _decode_pb(blobs[0])
                if 1 not in probe or not isinstance(probe[1][0], int):
                    continue
            except Exception:
                continue

            # Looks like embedded sub-messages with IDs → subgroups
            if idx == 0:
                logger.info("PROTO  subgroup candidates in field %d (%d items). "
                            "First sub fields: %s",
                            fn, len(blobs), _summarise_pb_fields(probe))

            for sg_bytes in blobs:
                sgf = _decode_pb(sg_bytes)
                sg: dict = {
                    'id': _pb_int(sgf, 1),
                    'eventId': event['id'],
                }
                # Scan for routeId / worldId / mapId (small ints) …
                for sfn in sorted(sgf):
                    v0 = sgf[sfn][0]
                    if isinstance(v0, int):
                        if v0 > 1_700_000_000_000:          # epoch-ms timestamp
                            sg.setdefault('eventSubgroupStart', v0)
                        elif 1 < v0 < 100_000 and sfn > 1:  # plausible ID
                            sg[f'_int_f{sfn}'] = v0
                    elif isinstance(v0, (bytes, bytearray)):
                        try:
                            s = v0.decode("utf-8")
                            if len(s) <= 2 and s.isalpha():
                                sg['subgroupLabel'] = s
                        except UnicodeDecodeError:
                            pass
                event['eventSubgroups'].append(sg)
            break  # found subgroups — stop scanning fields

        events.append(event)

    logger.info("Parsed %d events from protobuf (%d bytes)", len(events), len(raw))
    return events


# ---------------------------------------------------------------------------
# Zwift Games event discovery
# ---------------------------------------------------------------------------

def get_zwift_games_events() -> list[dict]:
    """
    Discover Zwift Games events and return their subgroups.

    Strategy:
      1. POST /api/events/search → protobuf response (event IDs + names).
      2. Filter for "Zwift Games" by name.
      3. For each matching event, GET /api/events/{id} to try to get a full
         JSON representation (with subgroup details).  If that does not work,
         fall back to the subgroups decoded from the protobuf.

    Returns list of subgroup dicts (one per category A–E per event).
    """
    zwift_games_start = datetime.datetime(2026, 2, 16, tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    start_ms = int(zwift_games_start.timestamp() * 1000)
    end_ms = int((now + datetime.timedelta(hours=1)).timestamp() * 1000)

    url = f"{BASE_URL}/events/search"
    params = {
        'use_subgroup_time': 'true',
        'created_before': end_ms,
        'start': 0,
        'limit': 100,
    }
    body = {
        'eventStartsAfter': start_ms,
        'eventStartsBefore': end_ms,
    }

    # ── 1. Fetch event search results (returns protobuf) ──────────────────
    resp = _request('POST', url, params=params, json=body,
                    headers={'Accept': '*/*'}, timeout=30)

    ct = resp.headers.get('Content-Type', '')
    logger.info("Event search: status=%s ct=%s len=%d",
                resp.status_code, ct, len(resp.content))

    if resp.status_code != 200:
        logger.error("Event search failed: status=%s body=%s",
                     resp.status_code, resp.text[:500])
        return []

    # ── 2. Parse the response ─────────────────────────────────────────────
    if 'protobuf' in ct:
        events = _parse_events_protobuf(resp.content)
    else:
        # Might be JSON — try that first
        try:
            events = resp.json()
            if not isinstance(events, list):
                events = []
        except Exception:
            events = _parse_events_protobuf(resp.content)

    if not events:
        logger.info("No events returned from search")
        return []

    # ── 3. Filter for Zwift Games ─────────────────────────────────────────
    sample = [e.get('name', '?') for e in events[:10]]
    logger.info("Sample event names (first 10 of %d): %s", len(events), sample)

    zwift_games = [
        e for e in events
        if 'zwift games' in e.get('name', '').lower()
    ]
    logger.info("Matched %d Zwift Games events out of %d total",
                len(zwift_games), len(events))

    if not zwift_games:
        return []

    # ── 4. Enrich each event with JSON details (subgroups etc.) ───────────
    for evt in zwift_games:
        eid = evt['id']
        try:
            detail_resp = _request('GET', f"{BASE_URL}/events/{eid}", timeout=15)
            logger.info("GET /api/events/%s => %s (ct=%s)",
                        eid, detail_resp.status_code,
                        detail_resp.headers.get('Content-Type', '?'))
            if detail_resp.status_code == 200:
                try:
                    detail = detail_resp.json()
                    # Merge richer JSON subgroups over the protobuf ones
                    if detail.get('eventSubgroups'):
                        evt['eventSubgroups'] = detail['eventSubgroups']
                    if detail.get('tags'):
                        evt['tags'] = detail.get('tags', [])
                except Exception:
                    pass  # keep protobuf-extracted subgroups
        except Exception as exc:
            logger.warning("GET event %s failed: %s", eid, exc)

    # ── 5. Collect subgroups ──────────────────────────────────────────────
    subgroups = []
    for evt in zwift_games:
        for sg in evt.get('eventSubgroups', []):
            sg['parentEventName'] = evt['name']
            if 'eventId' not in sg:
                sg['eventId'] = evt['id']
            subgroups.append(sg)

    logger.info("Found %d Zwift Games event subgroups", len(subgroups))
    return subgroups


# ---------------------------------------------------------------------------
# Race data fetching (adapted from shared/data_fetcher.py)
# ---------------------------------------------------------------------------

def get_race_entries(event_subgroup_id: int) -> list[dict]:
    """Fetch all race participants from race-results API."""
    all_entries = []
    start = 0
    limit = 50

    while True:
        resp = _request('GET', f"{BASE_URL}/race-results/entries", params={
            'event_subgroup_id': event_subgroup_id,
            'limit': limit,
            'start': start,
        }, timeout=30)

        if resp.status_code != 200:
            logger.error("Race entries error %s for %s", resp.status_code, event_subgroup_id)
            return []

        body = resp.text.strip()
        if not body or not body.startswith(('{', '[')):
            logger.error("Non-JSON race entries for %s", event_subgroup_id)
            return []

        entries = resp.json().get('entries', [])
        if not entries:
            break
        all_entries.extend(entries)
        if len(entries) < limit:
            break
        start += limit

    participants = []
    for entry in all_entries:
        profile = entry.get('profileData', {})
        activity_data = entry.get('activityData', {})
        act_id = activity_data.get('activityId')
        if not act_id:
            continue

        weight_g = profile.get('weightInGrams', 0) or profile.get('weight', 0)
        participants.append({
            'rank': entry.get('rank'),
            'name': f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
            'activity_id': str(act_id),
            'weight_kg': round(weight_g / 1000, 1) if weight_g else 75.0,
            'player_id': entry.get('profileId'),
            'elapsed_ms': activity_data.get('durationInMilliseconds'),
        })

    return participants


def fetch_rider_telemetry(activity_id: str) -> tuple[dict | None, dict | None, str | None]:
    """Fetch telemetry for one rider. Returns (telem_data, activity_data, error)."""
    resp = _request('GET', f"{BASE_URL}/activities/{activity_id}", timeout=30)
    if resp.status_code != 200:
        return None, None, f"Activity {activity_id}: HTTP {resp.status_code}"

    activity_data = resp.json()
    fitness = activity_data.get('fitnessData', {})

    if fitness.get('status') != 'AVAILABLE':
        return None, activity_data, "No fitness data"

    file_url = fitness.get('fullDataUrl') or fitness.get('smallDataUrl')
    if not file_url:
        return None, activity_data, "No telemetry URL"

    file_id = file_url.split('/file/')[-1]
    telem_resp = _request('GET', f"{BASE_URL}/activities/{activity_id}/file/{file_id}", timeout=30)
    if telem_resp.status_code != 200:
        return None, activity_data, f"Telemetry HTTP {telem_resp.status_code}"

    return telem_resp.json(), activity_data, None


def convert_telemetry_to_dataframe(telem_data: dict) -> pd.DataFrame:
    """Convert raw Zwift telemetry JSON to a typed DataFrame."""
    return pd.DataFrame({
        'time_sec': telem_data.get('timeInSec', []),
        'power_watts': telem_data.get('powerInWatts', []),
        'hr_bpm': telem_data.get('heartRate', []),
        'cadence_rpm': telem_data.get('cadencePerMin', []),
        'speed_kmh': [s * 3.6 / 100 for s in telem_data.get('speedInCmPerSec', [])],
        'distance_km': [d / 100000 for d in telem_data.get('distanceInCm', [])],
        'altitude_m': [a / 100 for a in telem_data.get('altitudeInCm', [])],
    })


def _process_rider(event_subgroup_id: int, p: dict) -> tuple[pd.DataFrame | None, dict]:
    """
    Fetch and process telemetry for a single rider.

    Args:
        event_subgroup_id: The event subgroup ID.
        p: Participant dict with keys: activity_id, rank, name, weight_kg, player_id.

    Returns:
        (telemetry_df_or_None, summary_dict)
    """
    telem_data, act_data, error = fetch_rider_telemetry(p['activity_id'])
    if error:
        return None, {
            'event_subgroup_id': event_subgroup_id,
            'activity_id': p['activity_id'],
            'rank': p['rank'],
            'name': p['name'],
            'weight_kg': p['weight_kg'],
            'player_id': p.get('player_id'),
            'status': error,
        }

    df = convert_telemetry_to_dataframe(telem_data)
    df['event_subgroup_id'] = event_subgroup_id
    df['activity_id'] = p['activity_id']
    df['player_id'] = p.get('player_id')

    # Calculate normalized power (30-sec rolling avg of power^4)
    np_val = None
    if len(df) >= 30:
        rolling = df['power_watts'].rolling(30).mean()
        np_val = round((rolling.dropna() ** 4).mean() ** 0.25, 2)

    file_url = ''
    if act_data:
        file_url = act_data.get('fitnessData', {}).get('fullDataUrl', '') or ''
    file_id = file_url.split('/file/')[-1] if file_url else None

    summary = {
        'event_subgroup_id': event_subgroup_id,
        'activity_id': p['activity_id'],
        'rank': p['rank'],
        'name': p['name'],
        'weight_kg': p['weight_kg'],
        'player_id': p.get('player_id'),
        'file_id': file_id,
        'activity_start_time': act_data.get('startDate') if act_data else None,
        'status': 'SUCCESS',
        'duration_sec': float(df['time_sec'].max()) if len(df) > 0 else 0.0,
        'avg_power': round(df['power_watts'].mean(), 1) if len(df) > 0 else 0.0,
        'normalized_power': np_val,
        'max_power': float(df['power_watts'].max()) if len(df) > 0 else 0.0,
        'avg_hr': round(df['hr_bpm'].mean(), 1) if len(df) > 0 else 0.0,
        'data_points': len(df),
    }
    return df, summary


def fetch_full_race(event_subgroup_id: int, event_meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all riders' telemetry for a race event.
    
    Args:
        event_subgroup_id: The event subgroup ID.
        event_meta: Dict with event metadata (race_name, route_id, etc.).
    
    Returns:
        (telemetry_df, summary_df) — ready for Kusto ingestion.
        telemetry_df has columns matching RiderTelemetry table.
        summary_df has columns matching RaceSummary table.
    """
    participants = get_race_entries(event_subgroup_id)
    if not participants:
        logger.warning("No participants for event %s", event_subgroup_id)
        return pd.DataFrame(), pd.DataFrame()

    logger.info("Fetching telemetry for %d riders in event %s", len(participants), event_subgroup_id)

    all_telemetry = []
    all_summaries = []

    with ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        futures = {pool.submit(_process_rider, event_subgroup_id, p): p for p in participants}
        for future in as_completed(futures):
            try:
                telem_df, summary = future.result()
                if telem_df is not None and not telem_df.empty:
                    all_telemetry.append(telem_df)
                all_summaries.append(summary)
            except Exception as e:
                p = futures[future]
                logger.error("Exception fetching %s: %s", p['activity_id'], e)
                all_summaries.append({
                    'event_subgroup_id': event_subgroup_id,
                    'activity_id': p['activity_id'],
                    'rank': p['rank'],
                    'name': p['name'],
                    'weight_kg': p['weight_kg'],
                    'player_id': p.get('player_id'),
                    'status': f'Exception: {e}',
                })

    telemetry_df = pd.concat(all_telemetry, ignore_index=True) if all_telemetry else pd.DataFrame()
    summary_df = pd.DataFrame(all_summaries)

    success = (summary_df['status'] == 'SUCCESS').sum() if len(summary_df) > 0 else 0
    logger.info("Event %s: %d/%d riders fetched successfully", event_subgroup_id, success, len(participants))

    return telemetry_df, summary_df


def backfill_riders(
    event_subgroup_id: int, riders: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-fetch telemetry for specific riders whose initial fetch failed.

    Only returns data for riders whose telemetry is now available.
    Failed riders are silently skipped (their original error rows remain
    in RaceSummary; a new SUCCESS row is appended on success).

    Args:
        event_subgroup_id: The event subgroup ID.
        riders: List of rider dicts (activity_id, player_id, rank, name, weight_kg).

    Returns:
        (telemetry_df, summary_df) — only riders that succeeded this time.
    """
    logger.info("Backfill: attempting %d riders for subgroup %s",
                len(riders), event_subgroup_id)

    all_telemetry: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    with ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        futures = {
            pool.submit(_process_rider, event_subgroup_id, r): r
            for r in riders
        }
        for future in as_completed(futures):
            r = futures[future]
            try:
                telem_df, summary = future.result()
                if telem_df is not None and not telem_df.empty:
                    all_telemetry.append(telem_df)
                    all_summaries.append(summary)
                    logger.info("Backfill OK: rider %s (%s) — %d rows",
                                r['activity_id'], r.get('name', '?'), len(telem_df))
                else:
                    logger.debug("Backfill still unavailable: rider %s",
                                 r['activity_id'])
            except Exception as e:
                logger.error("Backfill exception for %s: %s", r['activity_id'], e)

    telemetry_df = (
        pd.concat(all_telemetry, ignore_index=True) if all_telemetry
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(all_summaries) if all_summaries else pd.DataFrame()

    logger.info("Backfill subgroup %s: %d/%d riders recovered",
                event_subgroup_id, len(all_summaries), len(riders))
    return telemetry_df, summary_df
