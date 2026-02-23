"""
YouTube livestream matching for Zwift race replays.

Given a list of known streamers (Zwift player IDs → YouTube channels),
finds YouTube livestreams that overlap with a race's time window using
the YouTube Data API v3.

Requires YOUTUBE_API_KEY environment variable to be set.
"""

import json
import os
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


# Cache resolved channel IDs to avoid repeated API calls
_channel_id_cache: Dict[str, str] = {}

STREAMERS_FILE = Path(__file__).parent.parent / 'streamers.json'


def load_streamers() -> List[dict]:
    """Load the known streamers list from streamers.json."""
    if not STREAMERS_FILE.exists():
        return []
    try:
        with open(STREAMERS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('streamers', [])
    except Exception:
        return []


def get_api_key() -> Optional[str]:
    """Get YouTube Data API key from environment."""
    return os.environ.get('YOUTUBE_API_KEY')


def resolve_channel_id(handle: str, api_key: str) -> Optional[str]:
    """Resolve a YouTube @handle to a channel ID using the API.
    
    Results are cached both in memory and back to streamers.json.
    """
    if handle in _channel_id_cache:
        return _channel_id_cache[handle]

    # Strip @ prefix if present
    clean_handle = handle.lstrip('@')

    resp = requests.get('https://www.googleapis.com/youtube/v3/channels', params={
        'part': 'id',
        'forHandle': clean_handle,
        'key': api_key,
    }, timeout=10)

    if resp.ok:
        items = resp.json().get('items', [])
        if items:
            channel_id = items[0]['id']
            _channel_id_cache[handle] = channel_id
            # Persist back to streamers.json
            _save_channel_id(handle, channel_id)
            return channel_id
    else:
        print(f"[YT] Channel resolve failed: {resp.status_code}")

    return None


def _save_channel_id(handle: str, channel_id: str):
    """Update streamers.json with resolved channel ID."""
    try:
        with open(STREAMERS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        for s in data.get('streamers', []):
            if s.get('youtube_handle') == handle:
                s['youtube_channel_id'] = channel_id
        with open(STREAMERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


def find_matching_streams(
    race_start_time: str,
    race_duration_sec: float,
    rider_player_ids: List[int],
    api_key: Optional[str] = None,
) -> List[dict]:
    """
    Find YouTube livestreams from known streamers who were in this race.
    
    Args:
        race_start_time: ISO 8601 timestamp of when the race started
        race_duration_sec: Total duration of the race in seconds
        rider_player_ids: List of Zwift player IDs in the race
        api_key: YouTube Data API key (falls back to env var)
    
    Returns:
        List of dicts with keys: streamer_name, youtube_url, 
        stream_title, offset_seconds (offset from stream start to race start)
    """
    api_key = api_key or get_api_key()
    if not api_key:
        return []

    streamers = load_streamers()
    if not streamers:
        return []

    # Parse race start time
    try:
        race_start = _parse_iso_datetime(race_start_time)
    except Exception as e:
        print(f"Failed to parse race start time '{race_start_time}': {e}")
        return []

    # Find which streamers were in this race
    player_id_set = set(rider_player_ids)
    matching_streamers = [s for s in streamers if s.get('zwift_player_id') in player_id_set]
    
    if not matching_streamers:
        return []

    results = []
    for streamer in matching_streamers:
        # Get or resolve channel ID
        channel_id = streamer.get('youtube_channel_id')
        if not channel_id:
            handle = streamer.get('youtube_handle')
            if handle:
                channel_id = resolve_channel_id(handle, api_key)
            if not channel_id:
                continue

        # Search for completed livestreams around the race time
        stream_info = _find_stream_for_race(
            channel_id, race_start, race_duration_sec, api_key
        )
        
        if stream_info:
            results.append({
                'streamer_name': streamer.get('name', 'Unknown'),
                'zwift_player_id': streamer['zwift_player_id'],
                'youtube_url': f"https://www.youtube.com/watch?v={stream_info['video_id']}",
                'video_id': stream_info['video_id'],
                'stream_title': stream_info['title'],
                'offset_seconds': stream_info['offset_seconds'],
            })

    return results


def _find_stream_for_race(
    channel_id: str,
    race_start: datetime,
    race_duration_sec: float,
    api_key: str,
) -> Optional[dict]:
    """Search a channel for a completed livestream that overlaps the race time.
    
    Uses the channel's uploads playlist (fast) instead of the search API (slow).
    Every YouTube channel has an uploads playlist with ID = channel_id with 'UC' replaced by 'UU'.
    """
    
    # Convert channel ID to uploads playlist ID (UC... -> UU...)
    if channel_id.startswith('UC'):
        uploads_playlist_id = 'UU' + channel_id[2:]
    else:
        # Fallback to slow search if channel ID format is unexpected
        return _find_stream_for_race_search(channel_id, race_start, race_duration_sec, api_key)

    # Fetch recent uploads from the playlist
    t0 = time.time()
    resp = requests.get('https://www.googleapis.com/youtube/v3/playlistItems', params={
        'part': 'snippet',
        'playlistId': uploads_playlist_id,
        'maxResults': 20,
        'key': api_key,
    }, timeout=15)
    print(f"[YT] PlaylistItems API took {time.time()-t0:.1f}s, status={resp.status_code}")

    if not resp.ok:
        print(f"[YT] PlaylistItems failed, falling back to search: {resp.status_code}")
        return _find_stream_for_race_search(channel_id, race_start, race_duration_sec, api_key)

    items = resp.json().get('items', [])
    if not items:
        return None

    # Filter to videos published within ±48h of race (rough filter by publishedAt)
    candidate_ids = []
    for item in items:
        pub = item['snippet'].get('publishedAt')
        if pub:
            try:
                pub_dt = _parse_iso_datetime(pub)
                if abs((pub_dt - race_start).total_seconds()) < 48 * 3600:
                    candidate_ids.append(item['snippet']['resourceId']['videoId'])
            except Exception:
                candidate_ids.append(item['snippet']['resourceId']['videoId'])
        else:
            candidate_ids.append(item['snippet']['resourceId']['videoId'])

    if not candidate_ids:
        return None

    print(f"[YT] {len(candidate_ids)} candidates from playlist (within ±48h)")

    # Get livestream details for candidates
    t1 = time.time()
    detail_resp = requests.get('https://www.googleapis.com/youtube/v3/videos', params={
        'part': 'liveStreamingDetails,snippet,contentDetails',
        'id': ','.join(candidate_ids),
        'key': api_key,
    }, timeout=15)
    print(f"[YT] Videos API took {time.time()-t1:.1f}s, status={detail_resp.status_code}")

    if not detail_resp.ok:
        return None

    race_end = race_start + timedelta(seconds=int(race_duration_sec))

    for video in detail_resp.json().get('items', []):
        live_details = video.get('liveStreamingDetails', {})
        actual_start = live_details.get('actualStartTime')
        actual_end = live_details.get('actualEndTime')

        if not actual_start:
            continue

        stream_start = _parse_iso_datetime(actual_start)
        stream_end = _parse_iso_datetime(actual_end) if actual_end else (stream_start + timedelta(hours=12))

        # Check overlap: stream must have started before race ended
        # and ended after race started
        if stream_start <= race_end and stream_end >= race_start:
            # Offset = how far into the stream the race starts
            offset_seconds = max(0, (race_start - stream_start).total_seconds())
            
            # Detect if VOD was trimmed and adjust offset
            trim_seconds = _detect_trim_offset(video)
            offset_seconds = max(0, offset_seconds - trim_seconds)
            
            return {
                'video_id': video['id'],
                'title': video['snippet']['title'],
                'offset_seconds': int(offset_seconds),
                'stream_start': actual_start,
            }

    return None


def _find_stream_for_race_search(
    channel_id: str,
    race_start: datetime,
    race_duration_sec: float,
    api_key: str,
) -> Optional[dict]:
    """Fallback: use the slow YouTube search API to find livestreams."""
    
    # Search window: streams published within 24h before and after race start
    search_after = (race_start - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')
    search_before = (race_start + timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')

    t0 = time.time()
    resp = requests.get('https://www.googleapis.com/youtube/v3/search', params={
        'part': 'snippet',
        'channelId': channel_id,
        'type': 'video',
        'eventType': 'completed',
        'publishedAfter': search_after,
        'publishedBefore': search_before,
        'maxResults': 10,
        'key': api_key,
    }, timeout=15)
    print(f"[YT] Search API took {time.time()-t0:.1f}s, status={resp.status_code}")

    if not resp.ok:
        print(f"[YT] YouTube search failed: {resp.status_code} {resp.text[:200]}")
        return None

    items = resp.json().get('items', [])
    print(f"[YT] Found {len(items)} candidate streams")
    if not items:
        return None

    # Get livestream details for all candidates
    video_ids = ','.join(item['id']['videoId'] for item in items)
    t1 = time.time()
    detail_resp = requests.get('https://www.googleapis.com/youtube/v3/videos', params={
        'part': 'liveStreamingDetails,snippet,contentDetails',
        'id': video_ids,
        'key': api_key,
    }, timeout=15)
    print(f"[YT] Videos API took {time.time()-t1:.1f}s, status={detail_resp.status_code}")

    if not detail_resp.ok:
        return None

    race_end = race_start + timedelta(seconds=int(race_duration_sec))

    for video in detail_resp.json().get('items', []):
        live_details = video.get('liveStreamingDetails', {})
        actual_start = live_details.get('actualStartTime')
        actual_end = live_details.get('actualEndTime')

        if not actual_start:
            continue

        stream_start = _parse_iso_datetime(actual_start)
        stream_end = _parse_iso_datetime(actual_end) if actual_end else (stream_start + timedelta(hours=12))

        # Check overlap: stream must have started before race ended
        # and ended after race started
        if stream_start <= race_end and stream_end >= race_start:
            # Offset = how far into the stream the race starts
            offset_seconds = max(0, (race_start - stream_start).total_seconds())
            
            # Detect if VOD was trimmed and adjust offset
            trim_seconds = _detect_trim_offset(video)
            offset_seconds = max(0, offset_seconds - trim_seconds)
            
            return {
                'video_id': video['id'],
                'title': video['snippet']['title'],
                'offset_seconds': int(offset_seconds),
                'stream_start': actual_start,
            }

    return None


def _parse_iso_datetime(s: str) -> datetime:
    """Parse an ISO 8601 datetime string to a timezone-aware datetime."""
    # Handle various formats: 2026-02-09T17:36:53.005+0000, 2026-02-09T17:36:53Z
    s = s.replace('+0000', '+00:00').replace('Z', '+00:00')
    # Handle +0000 without colon
    if s[-5] in ('+', '-') and ':' not in s[-5:]:
        s = s[:-2] + ':' + s[-2:]
    return datetime.fromisoformat(s)


def _parse_iso_duration(d: str) -> float:
    """Parse an ISO 8601 duration (e.g. PT1H23M45S) to total seconds."""
    import re
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', d)
    if not m:
        return 0.0
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + s


def _detect_trim_offset(video: dict) -> int:
    """
    Detect if a livestream VOD was trimmed and return the trim amount in seconds.
    
    Compares the original broadcast duration (actualEnd - actualStart) with the
    current video duration (contentDetails.duration). If the video is shorter,
    the difference is assumed to be trimmed from the front.
    
    Returns:
        Number of seconds trimmed from the front (0 if not trimmed or unknown)
    """
    live_details = video.get('liveStreamingDetails', {})
    actual_start = live_details.get('actualStartTime')
    actual_end = live_details.get('actualEndTime')
    content_details = video.get('contentDetails', {})
    duration_str = content_details.get('duration', '')
    
    if not actual_start or not actual_end or not duration_str:
        return 0
    
    try:
        stream_start = _parse_iso_datetime(actual_start)
        stream_end = _parse_iso_datetime(actual_end)
        original_duration = (stream_end - stream_start).total_seconds()
        current_duration = _parse_iso_duration(duration_str)
        
        if current_duration <= 0 or original_duration <= 0:
            return 0
        
        trim = original_duration - current_duration
        # Only consider it a trim if at least 30s was removed
        if trim > 30:
            print(f"[YT] VOD trimmed: original {original_duration:.0f}s, current {current_duration:.0f}s, trim ~{trim:.0f}s")
            return int(trim)
    except Exception as e:
        print(f"[YT] Trim detection failed: {e}")
    
    return 0
