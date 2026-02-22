"""
Data Fetcher — unified module for fetching Zwift activity and race data.

Handles:
- Single rider activity + telemetry fetching (for bike comparison)
- Full race fetching: all participants' telemetry (for race replay)
"""

import requests
import pandas as pd
import json
import os
import time as _time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import calculate_normalized_power
from . import blob_storage

BASE_URL = "https://us-or-rly101.zwift.com/api"

# Max retries for transient Zwift API errors (5xx)
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1, 3]  # seconds to wait between retries


def _request_with_retry(method, url, **kwargs):
    """Make an HTTP request with automatic retry on 5xx errors.
    
    Automatically sets Accept: application/json so the Zwift API
    returns JSON instead of protobuf.
    """
    # Ensure we ask for JSON — Zwift returns protobuf by default on some endpoints
    headers = kwargs.get('headers', {})
    if headers is None:
        headers = {}
    else:
        headers = dict(headers)  # don't mutate caller's dict
    headers.setdefault('Accept', 'application/json')
    kwargs['headers'] = headers

    last_response = None
    for attempt in range(_MAX_RETRIES):
        response = requests.request(method, url, **kwargs)
        if response.status_code == 429:
            # Rate limited — back off and retry
            retry_after = int(response.headers.get('Retry-After', 5))
            _time.sleep(retry_after)
            last_response = response
            continue
        if response.status_code < 500:
            return response
        last_response = response
        if attempt < _MAX_RETRIES - 1:
            _time.sleep(_RETRY_BACKOFF[attempt])
    return last_response


# ---------------------------------------------------------------------------
# Activity / telemetry helpers (used by both bike comparison and race replay)
# ---------------------------------------------------------------------------

def extract_activity_id(activity_url_or_id):
    """
    Extract activity ID from a Zwift URL or return the ID directly.
    
    Args:
        activity_url_or_id: Zwift activity URL or activity ID
        
    Returns:
        str: The activity ID
    """
    if 'zwift.com' in str(activity_url_or_id):
        return activity_url_or_id.split('/')[-1].split('?')[0]
    return str(activity_url_or_id)


def get_activity_details(activity_id, headers):
    """
    Fetch activity details from the Zwift API.
    
    Args:
        activity_id: The Zwift activity ID
        headers: Authorization headers
        
    Returns:
        tuple: (activity_data dict, error_message) - activity_data is None on error
    """
    url = f"{BASE_URL}/activities/{activity_id}"
    response = _request_with_retry('GET', url, headers=headers, timeout=30)
    
    if response.status_code != 200:
        body = response.text[:200] if response.text else ''
        return None, f"Error fetching activity: {response.status_code} — {body}"
    
    try:
        return response.json(), None
    except Exception as e:
        return None, f"Failed to parse activity JSON: {e} — body: {(response.text or '')[:200]}"


def fetch_rider_telemetry(activity_id, headers):
    """
    Fetch telemetry data for a single rider.
    
    Args:
        activity_id: The rider's activity ID
        headers: Authorization headers
        
    Returns:
        tuple: (telemetry_data, activity_data, error_message)
               - telemetry_data: dict with raw telemetry
               - activity_data: dict with activity metadata (includes routeId)
               - error_message: string if failed, None on success
    """
    # Get activity details to find file ID
    activity_data, error = get_activity_details(activity_id, headers)
    if error:
        return None, None, error
    
    fitness_data = activity_data.get('fitnessData', {})
    
    if fitness_data.get('status') != 'AVAILABLE':
        return None, activity_data, "No fitness data available"
    
    file_url = fitness_data.get('fullDataUrl') or fitness_data.get('smallDataUrl')
    if not file_url:
        return None, activity_data, "No telemetry file URL"
    
    file_id = file_url.split('/file/')[-1]
    
    # Fetch telemetry
    telem_url = f"{BASE_URL}/activities/{activity_id}/file/{file_id}"
    telem_response = _request_with_retry('GET', telem_url, headers=headers, timeout=30)
    
    if telem_response.status_code != 200:
        body = telem_response.text[:200] if telem_response.text else ''
        return None, activity_data, f"Telemetry error: {telem_response.status_code} — {body}"
    
    try:
        return telem_response.json(), activity_data, None
    except Exception as e:
        return None, activity_data, f"Failed to parse telemetry JSON: {e} — body: {(telem_response.text or '')[:200]}"


def convert_telemetry_to_dataframe(telem_data):
    """
    Convert raw telemetry data to a pandas DataFrame with proper units.
    
    Args:
        telem_data: Raw telemetry dict from API
        
    Returns:
        pd.DataFrame with columns: time_sec, power_watts, hr_bpm, cadence_rpm,
                                   speed_kmh, distance_km, altitude_m
    """
    return pd.DataFrame({
        'time_sec': telem_data.get('timeInSec', []),
        'power_watts': telem_data.get('powerInWatts', []),
        'hr_bpm': telem_data.get('heartRate', []),
        'cadence_rpm': telem_data.get('cadencePerMin', []),
        'speed_kmh': [s * 3.6 / 100 for s in telem_data.get('speedInCmPerSec', [])],
        'distance_km': [d / 100000 for d in telem_data.get('distanceInCm', [])],
        'altitude_m': [a / 100 for a in telem_data.get('altitudeInCm', [])]
    })


# ---------------------------------------------------------------------------
# Race-specific fetching (for race replay)
# ---------------------------------------------------------------------------

def get_race_entries(event_subgroup_id, headers, limit=50):
    """
    Fetch all race participants from the race-results API.
    
    Handles pagination to fetch all entries, not just the first page.
    
    Args:
        event_subgroup_id: The event subgroup ID from activity details
        headers: Authorization headers
        limit: Number of entries per page (API limit)
        
    Returns:
        tuple: (list of participant dicts, error_message) - list is None on error
    """
    all_entries = []
    start = 0
    
    while True:
        url = f"{BASE_URL}/race-results/entries"
        params = {
            'event_subgroup_id': event_subgroup_id,
            'limit': limit,
            'start': start
        }
        response = _request_with_retry('GET', url, headers=headers, params=params, timeout=30)
        
        if response.status_code != 200:
            body = response.text[:200] if response.text else ''
            return None, f"Error fetching race results: {response.status_code} — {body}"
        
        # Guard against empty or non-JSON responses
        body_text = response.text.strip() if response.text else ''
        if not body_text or not body_text.startswith(('{', '[')):
            return None, f"Race results returned non-JSON response (status {response.status_code}): {body_text[:200]}"
        
        try:
            race_data = response.json()
        except Exception as e:
            return None, f"Failed to parse race results JSON: {e} — body: {body_text[:200]}"
        
        entries = race_data.get('entries', [])
        
        if not entries:
            break
        
        all_entries.extend(entries)
        
        if len(entries) < limit:
            break
        
        start += limit
    
    if not all_entries:
        return None, "No race entries found."
    
    # Log the first entry's keys so we know what the API returns
    if all_entries:
        first = all_entries[0]
        print(f"Race result entry keys: {list(first.keys())}")
        ad = first.get('activityData', {})
        if ad:
            print(f"  activityData keys: {list(ad.keys())}")

    participants = []
    for entry in all_entries:
        rank = entry.get('rank')
        profile_data = entry.get('profileData', {})
        activity_data_entry = entry.get('activityData', {})
        name = f"{profile_data.get('firstName', '')} {profile_data.get('lastName', '')}".strip()
        result_activity_id = activity_data_entry.get('activityId')
        weight_grams = profile_data.get('weightInGrams', 0) or profile_data.get('weight', 0)
        weight_kg = round(weight_grams / 1000, 1) if weight_grams else 75.0  # Default 75kg
        player_id = entry.get('profileId')  # Numeric Zwift profile ID

        # Extract race distance from the entry if available
        # The API may provide this in activityData or at the entry level
        segment_distance_cm = entry.get('segmentDistanceInCentimeters')
        elapsed_ms = activity_data_entry.get('durationInMilliseconds') or activity_data_entry.get('elapsedMs')

        if result_activity_id:
            p = {
                'rank': rank,
                'name': name,
                'activity_id': str(result_activity_id),
                'weight_kg': weight_kg,
                'player_id': int(player_id) if player_id else None,
            }
            if segment_distance_cm:
                p['segment_distance_cm'] = segment_distance_cm
            if elapsed_ms:
                p['elapsed_ms'] = elapsed_ms
            participants.append(p)
    
    return participants, None


def deduplicate_ranks(participants):
    """Ensure each participant has a unique rank.
    
    Zwift may assign the same rank to multiple riders (ties, DNFs, etc.).
    This reassigns duplicates to the next available rank to avoid file collisions.
    """
    used_ranks = set()
    for p in participants:
        rank = p['rank']
        while rank in used_ranks:
            rank += 1
        p['rank'] = rank
        used_ranks.add(rank)
    return participants


def save_rider_data(output_dir, rank, activity_id, df, telem_data, route_id=None):
    """
    Save rider telemetry data to CSV and JSON files.
    
    Args:
        output_dir: Directory to save files
        rank: Rider's finishing rank
        activity_id: Rider's activity ID
        df: DataFrame with processed telemetry
        telem_data: Raw telemetry dict
        route_id: Optional route ID to include in JSON
    """
    csv_path = os.path.join(output_dir, f"rank{rank}_{activity_id}.csv")
    df.to_csv(csv_path, index=False)
    
    telem_data['routeId'] = route_id
    json_path = os.path.join(output_dir, f"rank{rank}_{activity_id}_raw.json")
    with open(json_path, 'w') as f:
        json.dump(telem_data, f)


def fetch_race_from_activity(activity_url_or_id, headers, output_base_dir=".", progress_callback=None, force_refresh=False):
    """
    Fetch race data from a Zwift activity URL or ID.
    
    Orchestrates:
    1. Getting activity details to find the event subgroup ID
    2. Fetching all race participants
    3. Downloading telemetry for each participant
    4. Saving all data to files
    
    Results are cached by event_subgroup_id so that different activity IDs
    from the same race reuse the same data.  Set force_refresh=True to
    re-fetch even when cached data exists.
    
    Args:
        activity_url_or_id: Zwift activity URL or activity ID
        headers: Authorization headers dict (e.g. {'Authorization': 'Bearer ...'})
        output_base_dir: Base directory for output files
        progress_callback: Optional callback(current, total, rider_name)
        force_refresh: If True, re-fetch even if data already exists
    
    Returns:
        tuple: (output_dir, message)
    """
    # Step 1: Extract activity ID
    activity_id = extract_activity_id(activity_url_or_id)
    
    # Step 2: Get activity details
    activity_data, error = get_activity_details(activity_id, headers)
    if error:
        return None, error
    
    # Step 3: Get event info
    event_info = activity_data.get('eventInfo', {})
    race_name = event_info.get('name', 'Unknown Race')
    event_subgroup_id = event_info.get('eventSubGroupId')
    
    if not event_subgroup_id:
        return None, "No event subgroup ID found. Is this a race activity?"
    
    # Step 4: Directory keyed by event_subgroup_id (so all riders map to same race)
    output_dir = os.path.join(output_base_dir, f"race_data_{event_subgroup_id}")
    summary_path = os.path.join(output_dir, "complete_race_summary.csv")
    meta_path = os.path.join(output_dir, "race_meta.json")
    
    # Check for existing cached data (local first, then blob storage)
    if not force_refresh:
        # If not on local disk, try downloading from blob storage
        if not os.path.exists(summary_path):
            race_dir_name = f"race_data_{event_subgroup_id}"
            if blob_storage.race_exists_in_blob(race_dir_name):
                os.makedirs(output_dir, exist_ok=True)
                blob_storage.download_race_dir(race_dir_name, output_dir)

        if os.path.exists(summary_path):
            try:
                summary_df = pd.read_csv(summary_path)
                success_count = int((summary_df['status'] == 'SUCCESS').sum())
                # Load race name from metadata if available
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                        race_name = meta.get('race_name', race_name)
                return output_dir, f"Cached: {race_name} ({success_count} riders)"
            except Exception:
                pass  # Fall through to re-fetch
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Delete cleaned cache so it gets regenerated from fresh data
    import shutil
    cleaned_cache_json = os.path.join(output_dir, "cleaned_cache.json")
    cleaned_data_dir = os.path.join(output_dir, "cleaned_data")
    if os.path.exists(cleaned_cache_json):
        os.remove(cleaned_cache_json)
    if os.path.exists(cleaned_data_dir):
        shutil.rmtree(cleaned_data_dir)
    
    # Save race metadata (including start time for stream matching)
    race_start_time = activity_data.get('startDate')  # ISO 8601 UTC (pen join time)
    event_id = event_info.get('id')  # Parent event ID (e.g. 5421702)
    world_id = activity_data.get('worldId')  # Zwift world number
    
    # Extract segment distance from subgroupResults on the source activity
    segment_distance_cm = None
    sgr = activity_data.get('subgroupResults', {})
    top_results = sgr.get('topResults', [])
    if top_results:
        segment_distance_cm = top_results[0].get('segmentDistanceInCentimeters')
        if segment_distance_cm:
            print(f"Race segment distance from subgroupResults: {segment_distance_cm / 100000:.2f} km")
    
    # Get event details (start time, route) from the parent event API
    event_start_time = None
    route_id = None
    if event_id:
        try:
            event_url = f"{BASE_URL}/events/{event_id}"
            event_resp = _request_with_retry('GET', event_url, headers=headers, timeout=15)
            if event_resp.status_code == 200:
                event_data = event_resp.json()
                # Find our subgroup in the event's subgroup list
                for esg in event_data.get('eventSubgroups', []):
                    if esg.get('id') == event_subgroup_id:
                        event_start_time = esg.get('eventSubgroupStart')
                        route_id = esg.get('routeId')
                        if event_start_time:
                            print(f"Event start time: {event_start_time}")
                        if route_id:
                            print(f"Event route ID: {route_id}")
                        break
            else:
                print(f"Could not fetch event {event_id}: {event_resp.status_code}")
        except Exception as e:
            print(f"Could not fetch event details: {e}")
    
    meta = {
        'race_name': race_name,
        'event_id': event_id,
        'event_subgroup_id': event_subgroup_id,
        'source_activity_id': activity_id,
        'race_start_time': event_start_time or race_start_time,
        'activity_start_time': race_start_time,
        'world_id': world_id,
        'route_id': route_id,
    }
    if segment_distance_cm:
        meta['segment_distance_cm'] = segment_distance_cm
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    
    # Step 5: Get all race participants
    participants, error = get_race_entries(event_subgroup_id, headers)
    if error:
        return None, error
    
    # Step 6: Fetch telemetry for each participant (concurrent)
    participants = deduplicate_ranks(participants)
    total_riders = len(participants)
    _CONCURRENT_WORKERS = 5
    
    # Thread-safe progress counter
    _progress_lock = threading.Lock()
    _progress_count = [0]  # mutable container for closure
    
    def fetch_one(p):
        """Fetch a single rider's telemetry. Returns a summary dict."""
        act_id = p['activity_id']
        
        telem_data, act_data, error = fetch_rider_telemetry(act_id, headers)
        
        # Report progress
        with _progress_lock:
            _progress_count[0] += 1
            current = _progress_count[0]
        if progress_callback:
            progress_callback(current, total_riders, p['name'])
        
        if error:
            return {
                'rank': p['rank'],
                'name': p['name'],
                'activity_id': act_id,
                'weight_kg': p.get('weight_kg', 75.0),
                'player_id': p.get('player_id'),
                'status': error
            }
        
        df = convert_telemetry_to_dataframe(telem_data)
        rider_route_id = act_data.get('routeId') if act_data else None
        save_rider_data(output_dir, p['rank'], act_id, df, telem_data, rider_route_id)
        
        file_url = act_data.get('fitnessData', {}).get('fullDataUrl', '') or act_data.get('fitnessData', {}).get('smallDataUrl', '')
        file_id = file_url.split('/file/')[-1] if file_url else None
        
        np_value = calculate_normalized_power(df['power_watts']) if len(df) > 0 else None
        
        return {
            'rank': p['rank'],
            'name': p['name'],
            'activity_id': act_id,
            'weight_kg': p.get('weight_kg', 75.0),
            'player_id': p.get('player_id'),
            'file_id': file_id,
            'activity_start_time': act_data.get('startDate') if act_data else None,
            'status': 'SUCCESS',
            'duration_sec': df['time_sec'].max() if len(df) > 0 else 0,
            'avg_power': round(df['power_watts'].mean(), 1) if len(df) > 0 else 0,
            'normalized_power': np_value,
            'max_power': df['power_watts'].max() if len(df) > 0 else 0,
            'avg_hr': round(df['hr_bpm'].mean(), 1) if len(df) > 0 else 0,
            'data_points': len(df)
        }
    
    # Fetch all riders concurrently, preserving original order
    summary_data = []
    with ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as executor:
        future_to_idx = {
            executor.submit(fetch_one, p): idx
            for idx, p in enumerate(participants)
        }
        results_by_idx = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results_by_idx[idx] = future.result()
            except Exception as e:
                p = participants[idx]
                results_by_idx[idx] = {
                    'rank': p['rank'],
                    'name': p['name'],
                    'activity_id': p['activity_id'],
                    'weight_kg': p.get('weight_kg', 75.0),
                    'player_id': p.get('player_id'),
                    'status': f'Exception: {e}'
                }
        # Reassemble in original participant order
        for idx in range(len(participants)):
            summary_data.append(results_by_idx[idx])
    
    # Step 7: Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "complete_race_summary.csv"), index=False)
    
    success_count = len([s for s in summary_data if s.get('status') == 'SUCCESS'])

    # Persist to blob storage so the cache survives redeployments
    race_dir_name = f"race_data_{event_subgroup_id}"
    blob_storage.upload_race_dir(race_dir_name, output_dir)

    return output_dir, f"Loaded: {race_name} ({success_count} riders)"
