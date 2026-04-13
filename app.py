"""
Zwift Tools Web App.

A unified web app providing:
- Bike Comparison: compare race performance with hypothetical bike setups
- Race Replay: visualize multi-rider race data with interactive playback

Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, session, Response
import numpy as np
import pandas as pd
from pathlib import Path
import os
import logging
import secrets
import requests
import gzip
import json
import time
import queue
import threading
from datetime import datetime
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

from bike_comparison.bike_data import get_bike_database, get_bike_stats
from bike_comparison.physics import compare_bike_setups
from shared.utils import calculate_normalized_power
from shared.data_fetcher import (
    fetch_rider_telemetry, convert_telemetry_to_dataframe,
    fetch_race_from_activity, get_activity_details,
    get_race_entries, save_rider_data, extract_activity_id,
    _request_with_retry, BASE_URL
)
from shared.route_lookup import get_route_info
from shared import blob_storage
from shared.zwift_auth import exchange_code_for_tokens, refresh_access_token, ZwiftTokens, get_token_with_password

try:
    from garmin_fit_sdk import Decoder, Stream
    HAS_FIT_SDK = True
except ImportError:
    HAS_FIT_SDK = False

app = Flask(__name__, template_folder='templates', static_folder='static')

# Use a stable secret key so sessions survive restarts and work across Gunicorn workers
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Global bike database
_db = None

def get_db():
    global _db
    if _db is None:
        _db = get_bike_database()
    return _db


def get_redirect_uri():
    """Get the OAuth redirect URI based on request host."""
    # Use the host from the request, defaulting to localhost
    host = request.host_url.rstrip('/')
    return f"{host}/auth/callback"


def get_session_tokens() -> ZwiftTokens | None:
    """Get tokens from session, refreshing if expired."""
    if 'tokens' not in session:
        return None
    
    try:
        tokens = ZwiftTokens.from_dict(session['tokens'])
        
        # Refresh if expired
        if tokens.is_expired:
            logger.info("Token expired (expires_at=%s, now=%s), refreshing...", tokens.expires_at, time.time())
            tokens = refresh_access_token(tokens.refresh_token)
            session['tokens'] = tokens.to_dict()
        
        return tokens
    except Exception as e:
        # Token refresh failed, user needs to re-login
        logger.exception("get_session_tokens failed")
        session.pop('tokens', None)
        return None


def get_headers():
    """Get authorization headers for Zwift API."""
    tokens = get_session_tokens()
    if tokens:
        return {'Authorization': f'Bearer {tokens.access_token}'}
    
    return None


@app.route('/auth/login')
def auth_login():
    """Show login form for password-based auth."""
    next_url = request.args.get('next', url_for('index'))
    return render_template('login.html', next_url=next_url)


@app.route('/auth/login', methods=['POST'])
def auth_login_post():
    """Handle password-based login."""
    email = request.form.get('email')
    password = request.form.get('password')
    
    if not email or not password:
        return redirect(url_for('auth_login') + '?error=missing')
    
    try:
        tokens = get_token_with_password(email, password)
        session['tokens'] = tokens.to_dict()
        next_url = request.form.get('next') or url_for('index')
        return redirect(next_url)
    except Exception as e:
        return render_template('login_failed.html', error=str(e)), 401


@app.route('/auth/callback')
def auth_callback():
    """Handle OAuth callback from Zwift."""
    # Verify state for CSRF protection
    state = request.args.get('state')
    stored_state = session.pop('oauth_state', None)
    
    if not state or state != stored_state:
        return jsonify({'error': 'Invalid state parameter'}), 400
    
    # Check for errors
    error = request.args.get('error')
    if error:
        error_desc = request.args.get('error_description', 'Unknown error')
        return jsonify({'error': error, 'description': error_desc}), 400
    
    # Exchange code for tokens
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'No authorization code received'}), 400
    
    try:
        redirect_uri = get_redirect_uri()
        tokens = exchange_code_for_tokens(code, redirect_uri)
        session['tokens'] = tokens.to_dict()
        
        # Redirect to main page on success
        return redirect(url_for('index'))
    except Exception as e:
        return jsonify({'error': 'Token exchange failed', 'details': str(e)}), 500


@app.route('/auth/logout')
def auth_logout():
    """Clear session and log out."""
    session.pop('tokens', None)
    return redirect(url_for('index'))


@app.route('/auth/status')
def auth_status():
    """Check authentication status."""
    tokens = get_session_tokens()
    if tokens:
        return jsonify({
            'authenticated': True,
            'method': 'oauth',
            'expires_at': tokens.expires_at
        })
    
    return jsonify({'authenticated': False})


@app.route('/')
def index():
    """Landing page with links to both tools."""
    logged_in = 'tokens' in session
    return render_template('landing.html', logged_in=logged_in)


@app.route('/bike-comparison')
def bike_comparison():
    """Bike comparison tool."""
    return render_template('bike_comparison.html')


@app.route('/race-replay')
def race_replay():
    """Race replay tool."""
    return render_template('race_replay.html')


@app.route('/my-activities')
def my_activities():
    """My Activities page — shows the logged-in user's recent activities."""
    if 'tokens' not in session:
        return redirect(url_for('auth_login', next=url_for('my_activities')))
    return render_template('my_activities.html')


@app.route('/api/my_activities')
def api_my_activities():
    """Fetch the logged-in user's recent activities from the Zwift API."""
    headers = get_headers()
    if not headers:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        # First get the user's profile ID
        profile_resp = _request_with_retry(
            'GET',
            f'{BASE_URL}/profiles/me',
            headers=headers,
            timeout=15,
        )
        if profile_resp.status_code != 200:
            return jsonify({'error': 'Failed to fetch profile'}), profile_resp.status_code

        profile = profile_resp.json()
        profile_id = profile.get('id')
        if not profile_id:
            return jsonify({'error': 'Could not determine profile ID'}), 500

        # Fetch activities — Zwift API: GET /api/profiles/{id}/activities
        start = int(request.args.get('start', 0))
        limit = int(request.args.get('limit', 30))
        resp = _request_with_retry(
            'GET',
            f'{BASE_URL}/profiles/{profile_id}/activities',
            headers=headers,
            params={'start': start, 'limit': limit},
            timeout=30,
        )
        if resp.status_code != 200:
            return jsonify({'error': f'Failed to fetch activities: {resp.status_code}'}), resp.status_code

        raw_activities = resp.json()

        activities = []
        for act in raw_activities:
            activity_id = act.get('id') or act.get('id_str')

            distance_m = act.get('distanceInMeters', 0)

            # movingTimeInMs is the most reliable duration field
            moving_ms = act.get('movingTimeInMs', 0)
            duration_sec = int(moving_ms / 1000) if moving_ms else 0

            sport = act.get('sport', '').upper()
            name = act.get('name', '')
            avg_watts = act.get('avgWatts')

            activities.append({
                'id': str(activity_id) if activity_id else None,
                'name': name,
                'sport': sport,
                'date': act.get('startDate'),
                'distance_km': round(distance_m / 1000, 2) if distance_m else 0,
                'duration_sec': duration_sec,
                'avg_watts': round(avg_watts, 1) if avg_watts else None,
                'calories': act.get('calories'),
                # These will be populated by the enrichment endpoint
                'route_name': None,
                'avg_hr': None,
                'normalized_power': None,
                'is_race': False,
                'race_name': None,
                'event_subgroup_id': None,
            })

        return jsonify({'activities': activities, 'start': start, 'limit': limit})
    except Exception as e:
        logger.exception("Error fetching activities")
        return jsonify({'error': str(e)}), 500


@app.route('/api/activity_enrichment/<activity_id>')
def api_activity_enrichment(activity_id):
    """Fetch enriched data for a single activity — route, NP, HR, race info.

    Reuses the race-replay / bike-comparison data pipeline:
      get_activity_details  →  routeId  →  route name
      fetch_rider_telemetry →  DataFrame →  NP + avg HR
    """
    headers = get_headers()
    if not headers:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        result = {}

        # --- Step 1: Activity details (route, event/race info) ---
        activity_data, error = get_activity_details(activity_id, headers)
        if error:
            return jsonify({'error': error}), 404

        route_id = activity_data.get('routeId')
        route_name = None

        # Race detection
        event_info = activity_data.get('eventInfo') or {}
        event_subgroup_id = (
            event_info.get('eventSubGroupId')
            or event_info.get('eventSubgroupId')
        )
        event_id = event_info.get('id') or event_info.get('eventId')
        result['is_race'] = bool(event_subgroup_id)
        result['race_name'] = event_info.get('name') if result['is_race'] else None
        result['event_subgroup_id'] = event_subgroup_id
        result['event_id'] = event_id

        # If no routeId on the activity, try the event subgroup (races)
        if not route_id and event_id:
            try:
                event_resp = _request_with_retry(
                    'GET',
                    f'{BASE_URL}/events/{event_id}',
                    headers=headers,
                    timeout=15,
                )
                if event_resp.status_code == 200:
                    event_data = event_resp.json()
                    for esg in event_data.get('eventSubgroups', []):
                        if esg.get('id') == event_subgroup_id:
                            route_id = esg.get('routeId')
                            break
            except Exception:
                pass  # Non-critical

        if route_id:
            ri = get_route_info(route_id)
            if ri:
                route_name = ri.get('name')
        result['route_name'] = route_name

        # --- Step 2: Telemetry (NP + avg HR) ---
        telem_data, _, telem_error = fetch_rider_telemetry(int(activity_id), headers)
        if telem_data:
            df = convert_telemetry_to_dataframe(telem_data)
            if len(df) > 0:
                result['normalized_power'] = calculate_normalized_power(df['power_watts'])
                hr_vals = df['hr_bpm']
                hr_nonzero = hr_vals[hr_vals > 0]
                result['avg_hr'] = round(float(hr_nonzero.mean()), 0) if len(hr_nonzero) > 0 else None
            else:
                result['normalized_power'] = None
                result['avg_hr'] = None
        else:
            result['normalized_power'] = None
            result['avg_hr'] = None

        return jsonify(result)
    except Exception as e:
        logger.exception("Error enriching activity")
        return jsonify({'error': str(e)}), 500


def detect_cooldown_start(df: pd.DataFrame, power_col: str = 'power_watts') -> int:
    """
    Detect where the cooldown/post-race period starts.
    
    Scans backwards from the end looking for where sustained race-level power
    transitions to sustained low power. Uses a 10-second consecutive-above-threshold
    check to ignore brief pedal strokes during cooldown.
    
    Returns:
        Index where cooldown starts (rows after this are cooldown)
    """
    if power_col not in df.columns or len(df) < 60:
        return len(df)
    
    power = df[power_col].values
    
    # Race average power (excluding near-zero)
    race_power = power[power > 10]
    if len(race_power) < 10:
        return len(df)
    avg_power = np.mean(race_power)
    
    low_power_threshold = avg_power * 0.4
    
    # Use a 5-second rolling average to smooth single-second noise
    short_window = min(5, len(power) // 4)
    smoothed = pd.Series(power).rolling(window=max(short_window, 1), min_periods=1).mean().values
    above = smoothed >= low_power_threshold
    
    # Scan backwards: find the last point where power was above threshold
    # for at least 10 consecutive seconds (to ignore brief cooldown pedaling)
    sustain_required = 10
    for i in range(len(df) - sustain_required, max(len(df) // 2, 0), -1):
        # Check if indices i..i+sustain_required are all above threshold
        if all(above[i:i + sustain_required]):
            # This is the end of real racing — cooldown starts after this block
            # Find the exact first low point after the sustained block
            for j in range(i + sustain_required, min(i + sustain_required + 30, len(df))):
                if not above[j]:
                    return j
            return min(i + sustain_required, len(df))
    
    return len(df)  # No cooldown detected


@app.route('/api/frames')
def get_frames():
    """Get all frames for dropdown."""
    db = get_db()
    frames = []
    for frame in sorted(db.frames.values(), key=lambda f: f'{f["framemake"]} {f["framemodel"]}'):
        frames.append({
            'id': frame['frameid'],
            'name': f"{frame['framemake']} {frame['framemodel']}",
            'aero': frame['frameaero'],
            'weight': frame['frameweight'],
            'hasBuiltInWheels': frame.get('framewheeltype') == 'fixed',
            'isTT': frame.get('frametype') == 'TT',
            'frameType': frame.get('frametype', 'Standard'),
            'level': int(frame.get('framelevel') or 0)
        })
    return jsonify(frames)


@app.route('/api/wheels')
def get_wheels():
    """Get all wheels for dropdown."""
    db = get_db()
    wheels = [{'id': '', 'name': '(Built-in wheels)', 'aero': '-', 'weight': '-', 'fitsFrame': ''}]
    for wheel in sorted(db.wheels.values(), key=lambda w: f'{w["wheelmake"]} {w["wheelmodel"]}'):
        wheels.append({
            'id': wheel['wheelid'],
            'name': f"{wheel['wheelmake']} {wheel['wheelmodel']}",
            'aero': wheel['wheelaero'],
            'weight': wheel['wheelweight'],
            'fitsFrame': wheel.get('wheelfitsframe', 'Standard,TT'),
            'level': int(wheel.get('wheellevel') or 0)
        })
    return jsonify(wheels)


@app.route('/api/my_profile')
def get_my_profile():
    """Get the logged-in user's height and weight from their Zwift profile."""
    headers = get_headers()
    if not headers:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        resp = requests.get(
            'https://us-or-rly101.zwift.com/api/profiles/me',
            headers=headers,
            timeout=10
        )
        if resp.status_code != 200:
            return jsonify({'error': 'Failed to fetch profile'}), resp.status_code
        
        if not resp.text.strip():
            return jsonify({'error': 'Empty response from Zwift API'}), 502
        
        try:
            profile = resp.json()
        except Exception:
            return jsonify({'error': 'Invalid response from Zwift API'}), 502
        height_mm = profile.get('height', 0)
        weight_g = profile.get('weight', 0)
        
        return jsonify({
            'height_cm': round(height_mm / 10, 1) if height_mm else None,
            'weight_kg': round(weight_g / 1000, 1) if weight_g else None,
            'name': f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
        })
    except Exception as e:
        logger.exception("Error fetching profile")
        return jsonify({'error': str(e)}), 500




@app.route('/api/bike_stats')
def get_bike_stats_api():
    """Get stats for a specific bike combo."""
    frame_id = request.args.get('frame_id')
    wheel_id = request.args.get('wheel_id', '')
    upgrade_level = int(request.args.get('upgrade_level', 0))
    
    if not frame_id:
        return jsonify({'error': 'frame_id required'}), 400
    
    setup = get_bike_stats(frame_id, wheel_id or None, upgrade_level)
    if not setup:
        return jsonify({'error': 'Bike combo not found'}), 404
    
    return jsonify({
        'frame_id': setup.frame_id,
        'frame_name': setup.frame_name,
        'wheel_id': setup.wheel_id,
        'wheel_name': setup.wheel_name,
        'upgrade_level': setup.upgrade_level,
        'cd': round(setup.cd, 4),
        'weight_kg': round(setup.weight_kg, 3)
    })


@app.route('/api/fetch_activity', methods=['POST'])
def fetch_activity():
    """Fetch activity data by ID."""
    data = request.json
    activity_id = data.get('activity_id')
    
    if not activity_id:
        return jsonify({'error': 'activity_id required'}), 400
    
    headers = get_headers()
    if not headers:
        return jsonify({'error': 'Not authenticated. Please log in first.'}), 401
    
    try:
        # Fetch telemetry using existing fetcher
        telem_data, activity_data, error = fetch_rider_telemetry(int(activity_id), headers)
        
        if error:
            return jsonify({'error': error}), 404
        
        if not telem_data:
            return jsonify({'error': 'No telemetry data returned'}), 404
        
        # Convert to DataFrame
        df = convert_telemetry_to_dataframe(telem_data)
        
        # Detect cooldown start (for frontend slider default, but don't prune)
        cooldown_idx = detect_cooldown_start(df)
        cooldown_start_sec = float(df['time_sec'].iloc[cooldown_idx]) if cooldown_idx < len(df) else None
        
        # Calculate gradient from altitude
        if 'altitude_m' in df.columns and len(df) > 1:
            dist_m = df['distance_km'].values * 1000
            alt_m = df['altitude_m'].values
            
            delta_dist = np.diff(dist_m, prepend=dist_m[0])
            delta_alt = np.diff(alt_m, prepend=alt_m[0])
            delta_dist[0] = delta_dist[1] if len(delta_dist) > 1 else 1.0
            delta_dist = np.maximum(delta_dist, 0.1)
            gradient = delta_alt / delta_dist
            
            # Smooth gradient
            gradient = uniform_filter1d(gradient, size=5, mode='nearest')
            gradient = np.clip(gradient, -0.25, 0.25)
        else:
            gradient = np.zeros(len(df))
        
        # Compute per-point surface types from GPS surface polygons
        surface_types = None
        latlng = telem_data.get('latlng', [])
        if latlng and len(latlng) == len(df):
            try:
                from shared.surface_lookup import compute_surface_types_array, surface_types_to_crr
                lats = np.array([pt[0] for pt in latlng])
                lngs = np.array([pt[1] for pt in latlng])
                mid = len(lats) // 2
                from race_replay.data_cleaner import detect_world_from_coords
                world = detect_world_from_coords(float(lats[mid]), float(lngs[mid]))
                if world:
                    surface_types = compute_surface_types_array(lats, lngs, world)
                    non_default = np.sum(surface_types != 'Tarmac')
                    if non_default > 0:
                        logger.info("Surface types: %d points on non-tarmac surfaces", non_default)
            except Exception as e:
                logger.warning("Could not compute surface types: %s", e)

        # Build telemetry response
        telemetry = {
            'time_sec': df['time_sec'].tolist(),
            'distance_km': df['distance_km'].tolist(),
            'speed_mps': (df['speed_kmh'] / 3.6).tolist(),
            'gradient': gradient.tolist(),
            'power_watts': df['power_watts'].tolist(),
            'altitude_m': df['altitude_m'].tolist()
        }
        if surface_types is not None:
            telemetry['surface_type'] = surface_types.tolist()
            # Also include road_bike CRR for backward compatibility
            telemetry['crr'] = surface_types_to_crr(surface_types, 'road_bike').tolist()
        
        # Look up rider profile for height/weight
        # The rider may differ from the logged-in user, so always fetch by profileId
        rider_profile = None
        if activity_data:
            profile_id = activity_data.get('profileId') or activity_data.get('profile', {}).get('id')
            if profile_id:
                try:
                    profile_resp = requests.get(
                        f'https://us-or-rly101.zwift.com/api/profiles/{profile_id}',
                        headers=headers,
                        timeout=10
                    )
                    if profile_resp.status_code == 200:
                        p = profile_resp.json()
                        if p.get('height') and p.get('weight'):
                            rider_profile = {
                                'height_cm': round(p['height'] / 10, 1),
                                'weight_kg': round(p['weight'] / 1000, 1),
                                'name': f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                            }
                except Exception:
                    pass  # Non-critical
            
            # Fallback: try embedded profile data in the activity
            if not rider_profile:
                profile = activity_data.get('profile', {})
                if profile.get('height') and profile.get('weight'):
                    rider_profile = {
                        'height_cm': round(profile['height'] / 10, 1),
                        'weight_kg': round(profile['weight'] / 1000, 1),
                        'name': f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
                    }

        response = {
            'success': True,
            'activity_id': activity_id,
            'points': len(df),
            'cooldown_start_sec': cooldown_start_sec,
            'distance_km': round(df['distance_km'].max(), 2),
            'duration_sec': int(df['time_sec'].max()),
            'telemetry': telemetry
        }
        if rider_profile:
            response['rider_profile'] = rider_profile

        return jsonify(response)
        
    except Exception as e:
        logger.exception("Error fetching activity")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_fit', methods=['POST'])
def upload_fit():
    """Parse an uploaded .fit file and return telemetry data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    fit_file = request.files['file']
    if not fit_file.filename:
        return jsonify({'error': 'No file selected'}), 400

    if not fit_file.filename.lower().endswith('.fit'):
        return jsonify({'error': 'File must be a .fit file'}), 400

    if not HAS_FIT_SDK:
        return jsonify({'error': 'garmin-fit-sdk not installed. Run: pip install garmin-fit-sdk'}), 500

    try:
        # Read file bytes
        file_bytes = fit_file.read()

        # Auto-detect and decompress gzip (common for Zwift .fit exports)
        if file_bytes[:2] == b'\x1f\x8b':
            try:
                file_bytes = gzip.decompress(file_bytes)
            except Exception as gz_err:
                return jsonify({'error': f'File appears gzip-compressed but failed to decompress: {gz_err}'}), 400

        # Check for FIT header: bytes 8-11 should be ".FIT"
        if len(file_bytes) >= 12:
            header_sig = file_bytes[8:12]
            if header_sig != b'.FIT':
                return jsonify({'error': f'Not a valid FIT file (header signature: {header_sig!r})'}), 400

        # Create stream and decode
        stream = Stream.from_byte_array(bytearray(file_bytes))
        decoder = Decoder(stream)
        messages, errors = decoder.read(
            apply_scale_and_offset=True,
            convert_datetimes_to_dates=True,
            convert_types_to_strings=True,
            expand_sub_fields=True,
            expand_components=True,
            merge_heart_rates=True,
        )

        if errors:
            return jsonify({'error': f'FIT decode errors: {errors}'}), 400

        # Extract record messages (per-second telemetry)
        records = messages.get('record_mesgs', [])
        if not records:
            return jsonify({'error': 'No record data found in FIT file'}), 400

        # Build arrays from records
        timestamps = []
        power_list = []
        hr_list = []
        cadence_list = []
        speed_list = []   # m/s from enhanced_speed or speed
        distance_list = []  # metres from distance
        altitude_list = []  # metres from enhanced_altitude or altitude
        lat_list = []
        lng_list = []

        first_ts = None
        for rec in records:
            ts = rec.get('timestamp')
            if ts is None:
                continue
            if first_ts is None:
                first_ts = ts

            # Time in seconds from start
            if hasattr(ts, 'timestamp'):
                t_sec = (ts - first_ts).total_seconds()
            else:
                t_sec = float(ts - first_ts)
            timestamps.append(t_sec)

            power_list.append(rec.get('power', 0) or 0)
            hr_list.append(rec.get('heart_rate', 0) or 0)
            cadence_list.append(rec.get('cadence', 0) or 0)

            # Speed: enhanced_speed (m/s) preferred, fallback to speed
            spd = rec.get('enhanced_speed') or rec.get('speed')
            speed_list.append(float(spd) if spd is not None else 0.0)

            # Distance in metres
            dist = rec.get('distance')
            distance_list.append(float(dist) if dist is not None else 0.0)

            # Altitude: enhanced_altitude preferred, fallback to altitude
            alt = rec.get('enhanced_altitude') or rec.get('altitude')
            altitude_list.append(float(alt) if alt is not None else 0.0)

            # GPS coordinates
            lat = rec.get('position_lat')
            lng = rec.get('position_long')
            # FIT stores in semicircles; convert to degrees
            if lat is not None and isinstance(lat, (int, float)) and abs(lat) > 180:
                lat = lat * (180 / 2**31)
            if lng is not None and isinstance(lng, (int, float)) and abs(lng) > 180:
                lng = lng * (180 / 2**31)
            lat_list.append(float(lat) if lat is not None else None)
            lng_list.append(float(lng) if lng is not None else None)

        if len(timestamps) < 5:
            return jsonify({'error': 'FIT file has too few data points'}), 400

        # Build DataFrame for cooldown detection and gradient calc
        time_arr = np.array(timestamps)
        power_arr = np.array(power_list)
        speed_arr = np.array(speed_list)
        dist_arr = np.array(distance_list) / 1000.0  # m -> km
        alt_arr = np.array(altitude_list)

        df = pd.DataFrame({
            'time_sec': time_arr,
            'power_watts': power_arr,
            'speed_kmh': speed_arr * 3.6,
            'distance_km': dist_arr,
            'altitude_m': alt_arr,
        })

        # Detect cooldown start
        cooldown_idx = detect_cooldown_start(df)
        cooldown_start_sec = float(df['time_sec'].iloc[cooldown_idx]) if cooldown_idx < len(df) else None

        # Calculate gradient
        if len(df) > 1:
            dist_m = dist_arr * 1000
            delta_dist = np.diff(dist_m, prepend=dist_m[0])
            delta_alt = np.diff(alt_arr, prepend=alt_arr[0])
            delta_dist[0] = delta_dist[1] if len(delta_dist) > 1 else 1.0
            delta_dist = np.maximum(delta_dist, 0.1)
            gradient = delta_alt / delta_dist

            gradient = uniform_filter1d(gradient, size=5, mode='nearest')
            gradient = np.clip(gradient, -0.25, 0.25)
        else:
            gradient = np.zeros(len(df))

        telemetry = {
            'time_sec': df['time_sec'].tolist(),
            'distance_km': df['distance_km'].tolist(),
            'speed_mps': speed_arr.tolist(),
            'gradient': gradient.tolist(),
            'power_watts': df['power_watts'].tolist(),
            'altitude_m': df['altitude_m'].tolist(),
        }

        # Compute per-point surface types from GPS surface polygons
        if any(x is not None for x in lat_list):
            try:
                from shared.surface_lookup import compute_surface_types_array, surface_types_to_crr
                from race_replay.data_cleaner import detect_world_from_coords
                lats = np.array([x if x is not None else 0.0 for x in lat_list])
                lngs = np.array([x if x is not None else 0.0 for x in lng_list])
                mid = len(lats) // 2
                world = detect_world_from_coords(float(lats[mid]), float(lngs[mid]))
                if world:
                    surface_types = compute_surface_types_array(lats, lngs, world)
                    telemetry['surface_type'] = surface_types.tolist()
                    telemetry['crr'] = surface_types_to_crr(surface_types, 'road_bike').tolist()
                    non_default = int(np.sum(surface_types != 'Tarmac'))
                    if non_default > 0:
                        logger.info("FIT surface types: %d points on non-tarmac surfaces", non_default)
            except Exception as e:
                logger.warning("Could not compute surface types from FIT GPS: %s", e)

        return jsonify({
            'success': True,
            'source': 'fit_upload',
            'filename': fit_file.filename,
            'points': len(df),
            'cooldown_start_sec': cooldown_start_sec,
            'distance_km': round(float(df['distance_km'].max()), 2),
            'duration_sec': int(df['time_sec'].max()),
            'telemetry': telemetry,
        })

    except Exception as e:
        logger.exception("Error parsing FIT file")
        return jsonify({'error': str(e)}), 500


def estimate_frontal_area(height_cm: float, weight_kg: float, position: str = 'drops') -> float:
    """
    Estimate cyclist frontal area based on height and weight.
    
    Uses the Faria formula (drops position), which is the same formula
    ZwifterBikes uses when deriving their Cd values. Using a consistent
    frontal area formula ensures CdA = Cd × FA matches the original
    wind-tunnel / velodrome calibration data.
    
    Formula: FA = 0.0293 × H^0.725 × M^0.425 + 0.0604
    where H = height in metres, M = mass in kg
    
    Args:
        height_cm: Rider height in cm
        weight_kg: Rider weight in kg
        position: Riding position (currently unused; Faria formula is for drops)
        
    Returns:
        Frontal area in m²
    """
    height_m = height_cm / 100
    
    # Faria frontal area formula (drops position)
    # Consistent with ZwifterBikes Cd derivation methodology
    frontal_area = 0.0293 * (height_m ** 0.725) * (weight_kg ** 0.425) + 0.0604
    
    return frontal_area


@app.route('/api/compare', methods=['POST'])
def compare_bikes():
    """Compare two bike setups using telemetry data."""
    data = request.json
    
    # Get bike setups
    actual = get_bike_stats(
        data['actual']['frame_id'],
        data['actual'].get('wheel_id') or None,
        data['actual'].get('upgrade_level', 0)
    )
    
    alternative = get_bike_stats(
        data['alternative']['frame_id'],
        data['alternative'].get('wheel_id') or None,
        data['alternative'].get('upgrade_level', 0)
    )
    
    if not actual or not alternative:
        return jsonify({'error': 'Invalid bike setup'}), 400
    
    # Get telemetry
    telemetry_data = data.get('telemetry')
    if not telemetry_data:
        return jsonify({'error': 'No telemetry data provided. Please load an activity or upload a .fit file first.'}), 400
    telemetry = pd.DataFrame(telemetry_data)
    
    rider_weight = float(data.get('rider_weight', 75))
    rider_height = float(data.get('rider_height', 175))  # cm
    alt_rider_weight = float(data.get('alt_rider_weight', rider_weight))
    alt_rider_height = float(data.get('alt_rider_height', rider_height))
    
    # Calculate frontal areas from height and weight using Du Bois formula
    frontal_area = estimate_frontal_area(rider_height, rider_weight, 'drops')
    alt_frontal_area = estimate_frontal_area(alt_rider_height, alt_rider_weight, 'drops')
    
    # Use per-point CRR if available from surface data, adjusted by bike type
    crr = 0.004
    alt_crr = None
    if 'surface_type' in telemetry.columns:
        from shared.surface_lookup import surface_types_to_crr, get_bike_type_for_frame
        surface_types = telemetry['surface_type'].values
        actual_bike_type = get_bike_type_for_frame(actual.frame_type)
        alt_bike_type = get_bike_type_for_frame(alternative.frame_type)
        crr = surface_types_to_crr(surface_types, actual_bike_type)
        if alt_bike_type != actual_bike_type:
            alt_crr = surface_types_to_crr(surface_types, alt_bike_type)
    elif 'crr' in telemetry.columns:
        crr = telemetry['crr'].values

    # Run comparison
    result = compare_bike_setups(
        telemetry=telemetry,
        rider_weight_kg=rider_weight,
        actual_setup=actual,
        alternative_setup=alternative,
        crr=crr,
        frontal_area=frontal_area,
        alt_rider_weight_kg=alt_rider_weight,
        alt_frontal_area=alt_frontal_area,
        alt_crr=alt_crr
    )
    
    # Prepare response data for charts
    return jsonify({
        'actual': {
            'name': str(actual),
            'cd': actual.cd,
            'weight_kg': actual.weight_kg
        },
        'alternative': {
            'name': str(alternative),
            'cd': alternative.cd,
            'weight_kg': alternative.weight_kg
        },
        'summary': {
            'total_actual_kj': round(result.total_actual_kj, 1),
            'total_alternative_kj': round(result.total_alternative_kj, 1),
            'diff_kj': round(result.total_alternative_kj - result.total_actual_kj, 1),
            'diff_pct': round((result.total_alternative_kj - result.total_actual_kj) / result.total_actual_kj * 100, 2),
            'avg_watts_diff': round(result.avg_watts_difference, 1),
            'actual_np': round(result.actual_np, 1),
            'alternative_np': round(result.alternative_np, 1),
            'np_diff': round(result.alternative_np - result.actual_np, 1)
        },
        'chart_data': {
            'time_min': (result.time_sec / 60).tolist(),
            'distance_km': result.distance_km.tolist(),
            'actual_watts': result.actual_watts.tolist(),
            'alternative_watts': result.alternative_watts.tolist(),
            'watts_difference': result.watts_difference.tolist(),
            'draft_watts': result.draft_watts.tolist(),
            'gradient': (result.gradient * 100).tolist()
        }
    })


# Frame/wheel IDs that are always available by default (not special unlocks)
# Frames: Zwift Steel (given at account creation), Zwift Handcycle (given at account creation)
_DEFAULT_FRAME_IDS = {'F093', 'F111'}
# Wheels: Classic (default), Buffalo Fahrad/Gravel/Safety (auto-included with frame purchases)
_DEFAULT_WHEEL_IDS = {'W038', 'W037', 'W039', 'W041'}

# Cumulative upgrade costs (Drops) per tier per level (from ZwiftInsider)
# Same across Distance/Duration/Elevation categories
_UPGRADE_COSTS = {
    'Entry-Level': [0, 25000, 75000, 150000, 250000, 400000],
    'Mid-Range':   [0, 50000, 150000, 300000, 500000, 750000],
    'High-End':    [0, 100000, 300000, 650000, 1150000, 1900000],
    'Halo':        [0, 400000, 1200000, 2400000, 5000000, 10000000],
}

def _get_combo_drops_cost(db, frame_id, wheel_id, upgrade_level=0):
    """Get total Drops cost for a frame + wheel + upgrades combination."""
    frame = db.frames.get(frame_id, {})
    frame_price = int(frame.get('frameprice') or 0)
    
    wheel_price = 0
    if wheel_id:
        wheel = db.wheels.get(wheel_id, {})
        wheel_price = int(wheel.get('wheelprice') or 0)
    
    # Add upgrade costs based on frame tier
    upgrade_cost = 0
    if upgrade_level > 0:
        tier = frame.get('frameupgradelevel', '')
        costs = _UPGRADE_COSTS.get(tier)
        if costs and upgrade_level < len(costs):
            upgrade_cost = costs[upgrade_level]
    
    # Check if frame or wheel has special unlock requirements
    frame_tier = frame.get('frameupgradelevel', '')
    non_shop = ((frame_price == 0 and frame_id not in _DEFAULT_FRAME_IDS)
                or (frame_tier == 'Halo')
                or (wheel_id and wheel_price == 0 and wheel_id not in _DEFAULT_WHEEL_IDS))

    total = frame_price + wheel_price + upgrade_cost
    return total if total > 0 else 0, non_shop


def _filter_bike_combos(db, exclude_tt, use_pareto, upgrade_level, max_rider_level=None, excluded_frames=None, exclude_special=False):
    """Filter bike combos by TT exclusion, rider level, excluded frames, and Pareto frontier. Returns filtered list."""
    bike_combos = list(db.bikes.items())
    
    if exclude_tt:
        tt_frame_ids = {fid for fid, frame in db.frames.items() 
                        if frame.get('frametype') == 'TT'}
        bike_combos = [(k, v) for k, v in bike_combos 
                       if k[0] not in tt_frame_ids]
    
    # Filter out bikes with special unlock requirements (price=0 or Halo tier)
    # but keep default frames/wheels that every player has
    if exclude_special:
        special_frame_ids = {fid for fid, frame in db.frames.items()
                             if (int(frame.get('frameprice') or 0) == 0
                                 and fid not in _DEFAULT_FRAME_IDS)
                             or frame.get('frameupgradelevel') == 'Halo'}
        special_wheel_ids = {wid for wid, wheel in db.wheels.items()
                             if int(wheel.get('wheelprice') or 0) == 0
                             and wid not in _DEFAULT_WHEEL_IDS}
        bike_combos = [(k, v) for k, v in bike_combos
                       if k[0] not in special_frame_ids
                       and (not k[1] or k[1] not in special_wheel_ids)]
    
    # Filter out excluded frames
    if excluded_frames:
        excluded_set = set(excluded_frames)
        bike_combos = [(k, v) for k, v in bike_combos if k[0] not in excluded_set]
    
    # Filter by max rider level (XP unlock level)
    if max_rider_level is not None:
        filtered = []
        for (frame_id, wheel_id), combo in bike_combos:
            frame = db.frames.get(frame_id, {})
            frame_lvl = int(frame.get('framelevel', '0') or '0')
            if frame_lvl > max_rider_level:
                continue
            if wheel_id:  # Only check wheel level for non-built-in wheels
                wheel = db.wheels.get(wheel_id, {})
                wheel_lvl = int(wheel.get('wheellevel', '0') or '0')
                if wheel_lvl > max_rider_level:
                    continue
            filtered.append(((frame_id, wheel_id), combo))
        bike_combos = filtered
    
    if use_pareto:
        bike_stats = []
        for (frame_id, wheel_id), combo in bike_combos:
            cd = combo['cd'][upgrade_level]
            wt = combo['weight'][upgrade_level]
            bike_stats.append((cd, wt, frame_id, wheel_id))
        
        bike_stats.sort(key=lambda x: (x[0], x[1]))
        
        pareto = []
        min_weight = float('inf')
        for cd, wt, fid, wid in bike_stats:
            if wt < min_weight:
                pareto.append((fid, wid))
                min_weight = wt
        
        pareto_set = set(pareto)
        bike_combos = [(k, v) for k, v in bike_combos if (k[0], k[1]) in pareto_set]
    
    return bike_combos


@app.route('/api/combo_count', methods=['POST'])
def combo_count():
    """Quick count of bike combos after applying filters."""
    data = request.json
    db = get_db()
    upgrade_level = int(data.get('upgrade_level', 0))
    exclude_tt = data.get('exclude_tt', True)
    use_pareto = data.get('use_pareto', True)
    max_rider_level = data.get('max_rider_level')
    if max_rider_level is not None:
        max_rider_level = int(max_rider_level)
    excluded_frames = data.get('excluded_frames', [])
    exclude_special = data.get('exclude_special', True)
    
    bike_combos = _filter_bike_combos(db, exclude_tt, use_pareto, upgrade_level, max_rider_level, excluded_frames, exclude_special)
    
    # Account for actual bike being added if not already in list
    actual_bike_data = data.get('actual_bike')
    if actual_bike_data:
        actual_key = (actual_bike_data['frame_id'], actual_bike_data.get('wheel_id') or '')
        if actual_key not in [k for k, v in bike_combos] and actual_key in db.bikes:
            bike_combos.append((actual_key, db.bikes[actual_key]))
    
    return jsonify({'count': len(bike_combos)})


@app.route('/api/best_bikes', methods=['POST'])
def find_best_bikes():
    """
    Find the top N bike setups that minimize Normalized Power for the given telemetry.
    
    Uses Pareto frontier optimization to limit search to non-dominated bikes.
    Uses compare_bike_setups() for consistent calculation with timeline comparison.
    """
    data = request.json
    top_n = int(data.get('top_n', 10))
    upgrade_level = int(data.get('upgrade_level', 0))
    rider_weight = float(data.get('rider_weight', 75))
    rider_height = float(data.get('rider_height', 175))
    exclude_tt = data.get('exclude_tt', True)
    use_pareto = data.get('use_pareto', True)
    max_rider_level = data.get('max_rider_level')
    if max_rider_level is not None:
        max_rider_level = int(max_rider_level)
    excluded_frames = data.get('excluded_frames', [])
    exclude_special = data.get('exclude_special', True)
    
    # Get telemetry data
    telemetry_data = data.get('telemetry')
    if not telemetry_data:
        return jsonify({'error': 'No telemetry data provided'}), 400
    
    telemetry = pd.DataFrame(telemetry_data)
    if len(telemetry) < 30:
        return jsonify({'error': 'Need at least 30 data points'}), 400
    
    # Calculate frontal area
    frontal_area = estimate_frontal_area(rider_height, rider_weight, 'drops')
    
    # Get database
    db = get_db()
    
    # Build list of bike combos to evaluate
    bike_combos = _filter_bike_combos(db, exclude_tt, use_pareto, upgrade_level, max_rider_level, excluded_frames, exclude_special)
    
    # Always include the actual bike if specified (even if not on Pareto frontier)
    actual_bike_data = data.get('actual_bike')
    if actual_bike_data:
        actual_frame_id = actual_bike_data['frame_id']
        actual_wheel_id = actual_bike_data.get('wheel_id') or ''
        actual_key = (actual_frame_id, actual_wheel_id)
        
        # Check if actual bike is already in the list
        if actual_key not in [k for k, v in bike_combos]:
            # Add it from the full database
            if actual_key in db.bikes:
                bike_combos.append((actual_key, db.bikes[actual_key]))
    
    # Extract telemetry arrays
    time_sec = np.array(telemetry['time_sec'])
    
    # Get actual recorded power (includes drafting benefit)
    actual_power = None
    if 'power_watts' in telemetry.columns:
        actual_power = np.array(telemetry['power_watts'])
    
    if actual_power is None or len(actual_power) == 0:
        return jsonify({'error': 'Telemetry must include power_watts for bike comparison'}), 400
    
    # Calculate actual NP for reference
    actual_np = calculate_normalized_power(actual_power, time_sec)
    
    # Get the actual bike that was used (from user selection)
    actual_bike_data = data.get('actual_bike')
    if not actual_bike_data:
        return jsonify({'error': 'Actual bike must be specified for comparison'}), 400
    
    actual_bike = get_bike_stats(
        actual_bike_data['frame_id'],
        actual_bike_data.get('wheel_id') or None,
        actual_bike_data.get('upgrade_level', 0)
    )
    
    if not actual_bike:
        return jsonify({'error': 'Could not find actual bike in database'}), 400
    
    actual_bike_name = str(actual_bike)
    actual_upgrade_level = actual_bike_data.get('upgrade_level', 0)
    
    # Iterate through selected bike combinations using compare_bike_setups
    combos_searched = len(bike_combos)
    results = []
    
    for (frame_id, wheel_id), combo in bike_combos:
        try:
            setup = db.get_bike_stats(frame_id, wheel_id, upgrade_level)
            if not setup:
                continue
            
            # Check if this is the actual bike (same frame, wheel, AND upgrade level)
            is_actual = (frame_id == actual_bike.frame_id and 
                         (wheel_id or '') == (actual_bike.wheel_id or '') and
                         upgrade_level == actual_upgrade_level)
            
            if is_actual:
                # For the actual bike, use the recorded NP exactly
                np_value = actual_np
                avg_power = float(np.mean(actual_power))
                dt = np.diff(time_sec, prepend=time_sec[0])
                dt[0] = dt[1] if len(dt) > 1 else 1.0
                total_kj = float(np.sum(actual_power * dt) / 1000)
            else:
                # Compute per-bike-type CRR from surface types
                crr = 0.004
                alt_crr = None
                if 'surface_type' in telemetry.columns:
                    from shared.surface_lookup import surface_types_to_crr, get_bike_type_for_frame
                    surface_types = telemetry['surface_type'].values
                    actual_bike_type = get_bike_type_for_frame(actual_bike.frame_type)
                    alt_bike_type = get_bike_type_for_frame(setup.frame_type)
                    crr = surface_types_to_crr(surface_types, actual_bike_type)
                    if alt_bike_type != actual_bike_type:
                        alt_crr = surface_types_to_crr(surface_types, alt_bike_type)
                elif 'crr' in telemetry.columns:
                    crr = telemetry['crr'].values

                # Use compare_bike_setups for consistent calculation with timeline comparison
                result = compare_bike_setups(
                    telemetry=telemetry,
                    rider_weight_kg=rider_weight,
                    actual_setup=actual_bike,
                    alternative_setup=setup,
                    crr=crr,
                    frontal_area=frontal_area,
                    alt_crr=alt_crr
                )
                np_value = result.alternative_np
                avg_power = float(np.mean(result.alternative_watts))
                total_kj = result.total_alternative_kj
            
            entry = {
                'frame_id': frame_id,
                'wheel_id': wheel_id or '',
                'frame_name': setup.frame_name,
                'wheel_name': setup.wheel_name,
                'name': f"{setup.frame_name} + {setup.wheel_name}",
                'cd': round(setup.cd, 4),
                'weight_kg': round(setup.weight_kg, 3),
                'np': round(np_value, 1),
                'avg_power': round(avg_power, 1),
                'total_kj': round(total_kj, 1),
                'is_actual': is_actual,
            }
            drops_cost, non_shop = _get_combo_drops_cost(db, frame_id, wheel_id, upgrade_level)
            entry['drops_cost'] = drops_cost
            entry['non_shop'] = non_shop
            results.append(entry)
        except Exception as e:
            # Skip combos that fail
            continue
    
    # Sort by NP (lowest first) and take top N
    results.sort(key=lambda x: x['np'])
    top_bikes = results[:top_n]
    
    return jsonify({
        'success': True,
        'combos_searched': combos_searched,
        'total_combos': len(results),
        'upgrade_level': upgrade_level,
        'excluded_tt': exclude_tt,
        'used_pareto': use_pareto,
        'actual_np': round(actual_np, 1),
        'actual_bike_name': actual_bike_name,
        'best_bikes': top_bikes
    })


# ---------------------------------------------------------------------------
# Ride Compare — compare arbitrary activities on the same route
# ---------------------------------------------------------------------------

@app.route('/ride-compare')
def ride_compare():
    """Ride compare tool — compare multiple rides on the same route."""
    return render_template('ride_compare.html')


@app.route('/api/ride_compare/fetch_stream')
def api_ride_compare_fetch_stream():
    """SSE endpoint — fetches telemetry for a list of activity IDs."""
    headers = get_headers()
    if not headers:
        def err_gen():
            yield f"data: {json.dumps({'error': 'Not authenticated. Please log in first.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    activity_ids_raw = request.args.get('activity_ids', '').strip()
    if not activity_ids_raw:
        def err_gen():
            yield f"data: {json.dumps({'error': 'activity_ids required (comma-separated)'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    # Parse comma/space/newline separated activity IDs or URLs
    import re as _re
    raw_parts = _re.split(r'[,\s]+', activity_ids_raw)
    activity_ids = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        aid = extract_activity_id(part)
        if aid:
            activity_ids.append(aid)

    if len(activity_ids) < 2:
        def err_gen():
            yield f"data: {json.dumps({'error': 'At least 2 activity IDs are required.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    # Build a stable compare_id from sorted activity IDs
    import hashlib
    sorted_ids = sorted(set(activity_ids))
    compare_id = 'ride_compare_' + hashlib.sha256(','.join(sorted_ids).encode()).hexdigest()[:16]

    q = queue.Queue()

    def run_fetch():
        try:
            race_data_dir = Path('race_data')
            race_data_dir.mkdir(exist_ok=True)
            output_dir = race_data_dir / compare_id

            # Check for existing data
            summary_path = output_dir / 'complete_race_summary.csv'
            if summary_path.exists():
                try:
                    summary_df = pd.read_csv(summary_path)
                    success_count = int((summary_df['status'] == 'SUCCESS').sum())
                    q.put({'success': True, 'race_id': compare_id,
                           'message': f'Cached: {success_count} rides loaded'})
                    return
                except Exception:
                    pass

            output_dir.mkdir(exist_ok=True)

            # Delete any stale cleaned cache
            import shutil
            cleaned_cache_json = output_dir / 'cleaned_cache.json'
            cleaned_data_dir = output_dir / 'cleaned_data'
            if cleaned_cache_json.exists():
                cleaned_cache_json.unlink()
            if cleaned_data_dir.exists():
                shutil.rmtree(cleaned_data_dir)

            total = len(activity_ids)
            summary_data = []
            route_id = None
            first_start_time = None

            for idx, act_id in enumerate(activity_ids):
                rank = idx + 1
                q.put({'progress': True, 'current': rank, 'total': total,
                       'name': f'Activity {act_id}'})

                telem_data, act_data, error = fetch_rider_telemetry(act_id, headers)

                if error:
                    summary_data.append({
                        'rank': rank,
                        'name': f'Activity {act_id}',
                        'activity_id': act_id,
                        'weight_kg': 75.0,
                        'player_id': None,
                        'status': error,
                    })
                    continue

                df = convert_telemetry_to_dataframe(telem_data)
                rider_route_id = act_data.get('routeId') if act_data else None
                save_rider_data(str(output_dir), rank, act_id, df, telem_data, rider_route_id)

                # Use first rider's route for the comparison
                if route_id is None and rider_route_id:
                    route_id = rider_route_id

                # Get rider name from profile (embedded or top-level)
                profile = act_data.get('profile', {}) if act_data else {}
                first_name = profile.get('firstName', '')
                last_name = profile.get('lastName', '')
                name = f"{first_name} {last_name}".strip()
                if not name and act_data:
                    # Fallback: try top-level playerName or name field
                    name = act_data.get('playerName', '') or act_data.get('name', '')
                name = name or f'Rider {rank}'
                weight_grams = profile.get('weightInGrams', 0) or profile.get('weight', 0) or 0
                weight_kg = round(weight_grams / 1000, 1) if weight_grams else 75.0
                player_id = profile.get('id') or (act_data.get('profileId') or act_data.get('playerId') if act_data else None)

                start_date = act_data.get('startDate') if act_data else None
                if first_start_time is None and start_date:
                    first_start_time = start_date

                file_url = act_data.get('fitnessData', {}).get('fullDataUrl', '') or act_data.get('fitnessData', {}).get('smallDataUrl', '') if act_data else ''
                file_id = file_url.split('/file/')[-1] if file_url else None

                np_value = calculate_normalized_power(df['power_watts']) if len(df) > 0 else None

                summary_data.append({
                    'rank': rank,
                    'name': name,
                    'activity_id': act_id,
                    'weight_kg': weight_kg,
                    'player_id': int(player_id) if player_id else None,
                    'file_id': file_id,
                    'activity_start_time': start_date,
                    'status': 'SUCCESS',
                    'duration_sec': df['time_sec'].max() if len(df) > 0 else 0,
                    'avg_power': round(df['power_watts'].mean(), 1) if len(df) > 0 else 0,
                    'normalized_power': np_value,
                    'max_power': df['power_watts'].max() if len(df) > 0 else 0,
                    'avg_hr': round(df['hr_bpm'].mean(), 1) if len(df) > 0 else 0,
                    'data_points': len(df),
                })

            # Save summary CSV
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(str(summary_path), index=False)

            # Save metadata — use first start time and detected route
            meta = {
                'race_name': 'Ride Comparison',
                'event_id': None,
                'event_subgroup_id': None,
                'source_activity_id': activity_ids[0],
                'race_start_time': first_start_time,
                'activity_start_time': first_start_time,
                'route_id': route_id,
                'is_ride_compare': True,
            }
            with open(str(output_dir / 'race_meta.json'), 'w') as f:
                json.dump(meta, f)

            # Disambiguate names: if the same name appears multiple times,
            # append the activity date or a counter
            name_counts = {}
            for s in summary_data:
                n = s['name']
                name_counts[n] = name_counts.get(n, 0) + 1

            dup_counters = {}
            used_suffixes = {}  # name -> set of suffixes already used
            for s in summary_data:
                n = s['name']
                if name_counts[n] > 1:
                    if n not in used_suffixes:
                        used_suffixes[n] = set()
                    # Prefer using activity date for disambiguation
                    suffix = None
                    start = s.get('activity_start_time')
                    if start:
                        try:
                            dt = datetime.fromisoformat(start.replace('Z', '+00:00').replace('+0000', '+00:00'))
                            date_suffix = dt.strftime('%Y-%m-%d')
                            if date_suffix not in used_suffixes[n]:
                                suffix = date_suffix
                            else:
                                # Same date: add time
                                suffix = dt.strftime('%Y-%m-%d %H:%M')
                        except Exception:
                            pass
                    if suffix is None or suffix in used_suffixes[n]:
                        dup_counters[n] = dup_counters.get(n, 0) + 1
                        suffix = f'#{dup_counters[n]}'
                    used_suffixes[n].add(suffix)
                    s['name'] = f"{n} ({suffix})"

            # Re-save summary with disambiguated names
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(str(summary_path), index=False)

            # Clear cleaned cache so it picks up disambiguated names
            _race_data_cache.pop(compare_id, None)

            success_count = len([s for s in summary_data if s.get('status') == 'SUCCESS'])
            q.put({'success': True, 'race_id': compare_id,
                   'message': f'Loaded: {success_count} rides for comparison'})
        except Exception as e:
            logger.exception("Error in ride_compare fetch")
            q.put({'error': str(e)})
        finally:
            q.put(None)

    thread = threading.Thread(target=run_fetch, daemon=True)
    thread.start()

    def event_stream():
        while True:
            msg = q.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return Response(event_stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---------------------------------------------------------------------------
# Race Replay API Routes
# ---------------------------------------------------------------------------

# In-memory cache of loaded race data (keyed by race_id)
_race_data_cache = {}


@app.route('/api/race/fetch_stream')
def api_race_fetch_stream():
    """SSE endpoint — streams progress while fetching race data."""
    headers = get_headers()
    if not headers:
        logger.warning("SSE fetch_stream: No auth headers. Session keys: %s", list(session.keys()))
        def err_gen():
            yield f"data: {json.dumps({'error': 'Not authenticated. Please log in first.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    activity_url_or_id = request.args.get('activity_id', '').strip()
    if not activity_url_or_id:
        def err_gen():
            yield f"data: {json.dumps({'error': 'activity_id required'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    force_refresh = request.args.get('force_refresh', '').lower() in ('1', 'true')
    all_subgroups = request.args.get('all_subgroups', '').lower() in ('1', 'true')

    # Use a queue to bridge the progress_callback (in a thread) to the SSE generator
    q = queue.Queue()

    def run_fetch():
        try:
            race_data_dir = Path('race_data')
            race_data_dir.mkdir(exist_ok=True)

            def on_progress(current, total, name, type="rider"):
                logger.info("  [%d/%d] Fetching %s...", current, total, name)
                q.put({'progress': True, 'current': current, 'total': total, 'name': name, 'type': type})

            if all_subgroups:
                from shared.data_fetcher import fetch_all_subgroups_from_activity
                results, event_id, race_name, error = fetch_all_subgroups_from_activity(
                    activity_url_or_id, headers,
                    output_base_dir=str(race_data_dir),
                    progress_callback=on_progress,
                    force_refresh=force_refresh
                )
                if error:
                    q.put({'error': error})
                else:
                    race_id = f"race_event_{event_id}"
                    # Save a manifest that lists all subgroup dirs + labels
                    manifest = {
                        'event_id': event_id,
                        'race_name': race_name,
                        'subgroups': [
                            {'race_id': Path(d).name, 'label': label, 'subgroup_id': sg_id}
                            for d, label, sg_id in results
                        ],
                    }
                    manifest_dir = race_data_dir / race_id
                    manifest_dir.mkdir(exist_ok=True)
                    with open(manifest_dir / 'event_manifest.json', 'w') as f:
                        json.dump(manifest, f)

                    total_riders = sum(1 for d, _, _ in results if (Path(d) / 'complete_race_summary.csv').exists())
                    _race_data_cache.pop(race_id, None)
                    _race_list_cache = None
                    q.put({
                        'success': True,
                        'race_id': race_id,
                        'message': f"Loaded: {race_name} — {len(results)} categories",
                    })
            else:
                output_dir, message = fetch_race_from_activity(
                    activity_url_or_id, headers,
                    output_base_dir=str(race_data_dir),
                    progress_callback=on_progress,
                    force_refresh=force_refresh
                )

                if not output_dir:
                    q.put({'error': message})
                else:
                    race_id = Path(output_dir).name
                    # Clear in-memory cleaned data cache so it gets regenerated
                    _race_data_cache.pop(race_id, None)
                    # Invalidate race list cache so the new race appears immediately
                    _race_list_cache = None
                    q.put({'success': True, 'race_id': race_id, 'message': message})
        except Exception as e:
            logger.exception("Error in fetch_stream background thread")
            q.put({'error': str(e)})
        finally:
            q.put(None)  # sentinel: stream done

    thread = threading.Thread(target=run_fetch, daemon=True)
    thread.start()

    def event_stream():
        while True:
            msg = q.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return Response(event_stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


_race_list_cache = None
_race_list_cache_time = 0
_RACE_LIST_TTL = 120  # seconds

@app.route('/api/race/list')
def api_race_list():
    """List available race data directories (local + blob storage).

    Results are cached in memory for up to 120 seconds to avoid
    re-reading CSV files and querying blob storage on every request.
    """
    global _race_list_cache, _race_list_cache_time
    now = time.time()
    if _race_list_cache is not None and (now - _race_list_cache_time) < _RACE_LIST_TTL:
        return jsonify({'races': _race_list_cache})

    race_dirs = []
    seen_ids = set()
    race_data_dir = Path('race_data')
    if race_data_dir.exists():
        for d in sorted(race_data_dir.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith('race_data_'):
                summary_file = d / 'complete_race_summary.csv'
                meta_file = d / 'race_meta.json'
                info = {
                    'race_id': d.name,
                    'has_summary': summary_file.exists(),
                    'race_name': None,
                }
                # Load race name from metadata
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)
                            info['race_name'] = meta.get('race_name')
                    except Exception:
                        pass
                if summary_file.exists():
                    try:
                        # Count lines directly — much faster than reading full CSV with pandas
                        with open(summary_file) as f:
                            rider_count = sum(1 for _ in f) - 1  # subtract header
                        info['rider_count'] = max(rider_count, 0)
                    except Exception:
                        pass
                race_dirs.append(info)
                seen_ids.add(d.name)

    # Merge in any races only in blob storage
    for blob_info in blob_storage.list_races():
        if blob_info['race_id'] not in seen_ids:
            race_dirs.append({
                'race_id': blob_info['race_id'],
                'has_summary': True,
                'race_name': blob_info.get('race_name'),
            })

    _race_list_cache = race_dirs
    _race_list_cache_time = now
    return jsonify({'races': race_dirs})


@app.route('/api/race/load', methods=['POST'])
def api_race_load():
    """Load/clean a race data directory and return metadata."""
    data = request.json
    race_id = data.get('race_id')
    if not race_id:
        return jsonify({'error': 'race_id required'}), 400

    # Multi-subgroup event (race_event_XXXX) — load each subgroup and merge
    if race_id.startswith('race_event_'):
        return _load_multi_subgroup_race(race_id)

    data_path = Path('race_data') / race_id
    if not data_path.exists():
        # Try downloading from blob storage
        if blob_storage.race_exists_in_blob(race_id):
            data_path.parent.mkdir(exist_ok=True)
            data_path.mkdir(parents=True, exist_ok=True)
            blob_storage.download_race_dir(race_id, data_path)
    if not data_path.exists():
        return jsonify({'error': f'Race data not found: {race_id}'}), 404

    try:
        from race_replay.data_cleaner import clean_race_data
        race_data = clean_race_data(data_path, cache=True)
        _race_data_cache[race_id] = race_data

        # Upload cleaned cache to blob so it persists across deploys
        blob_storage.upload_race_dir(race_id, data_path)

        return jsonify({
            'success': True,
            'race_id': race_id,
            'route_name': race_data.route_name or 'Unknown',
            'route_slug': race_data.route_slug,
            'finish_line_km': round(float(race_data.finish_line_km), 3),
            'rider_count': len(race_data.riders),
            'min_time': int(race_data.min_time),
            'max_time': int(race_data.max_time),
            'riders': [
                {
                    'rank': int(r.rank),
                    'name': r.name,
                    'team': r.team,
                    'finish_time_sec': float(r.finish_time_sec) if r.finish_time_sec is not None else None,
                }
                for r in race_data.riders
            ]
        })
    except Exception as e:
        logger.exception("Error loading race")
        return jsonify({'error': str(e)}), 500


def _load_multi_subgroup_race(race_id):
    """Load and merge multiple subgroups for a combined event view."""
    manifest_path = Path('race_data') / race_id / 'event_manifest.json'
    if not manifest_path.exists():
        return jsonify({'error': f'Event manifest not found: {race_id}'}), 404

    with open(manifest_path) as f:
        manifest = json.load(f)

    from race_replay.data_cleaner import clean_race_data, CleanedRaceData, RiderData

    # Clean each subgroup and collect riders with category labels
    all_riders = []
    merged_route_name = None
    merged_route_slug = None
    merged_finish = 0
    merged_min_time = float('inf')
    merged_max_time = float('-inf')
    merged_elevation = None
    merged_world = None
    base_rank = 0

    for sg_info in manifest['subgroups']:
        sg_race_id = sg_info['race_id']
        label = sg_info['label']
        sg_path = Path('race_data') / sg_race_id

        if not sg_path.exists():
            if blob_storage.race_exists_in_blob(sg_race_id):
                sg_path.mkdir(parents=True, exist_ok=True)
                blob_storage.download_race_dir(sg_race_id, sg_path)
        if not sg_path.exists():
            logger.warning("Subgroup data not found: %s", sg_race_id)
            continue

        try:
            sg_data = clean_race_data(sg_path, cache=True)
        except Exception as e:
            logger.warning("Could not clean subgroup %s: %s", sg_race_id, e)
            continue

        # Use the first subgroup's route as the reference
        if merged_route_name is None:
            merged_route_name = sg_data.route_name
            merged_route_slug = sg_data.route_slug
            merged_elevation = sg_data.elevation_profile
            merged_world = sg_data.world
        merged_finish = max(merged_finish, sg_data.finish_line_km)
        merged_min_time = min(merged_min_time, sg_data.min_time)
        merged_max_time = max(merged_max_time, sg_data.max_time)

        # Re-number ranks globally so they don't collide across categories
        for r in sg_data.riders:
            all_riders.append((r, label, base_rank + r.rank))
        if sg_data.riders:
            base_rank += max(r.rank for r in sg_data.riders)

    if not all_riders:
        return jsonify({'error': 'No rider data found across subgroups'}), 404

    # Resolve source_activity_id from subgroup metadata
    merged_source_activity_id = None
    for sg_info in manifest['subgroups']:
        sg_meta_path = Path('race_data') / sg_info['race_id'] / 'race_meta.json'
        if sg_meta_path.exists():
            with open(sg_meta_path) as mf:
                sg_meta = json.load(mf)
            if sg_meta.get('source_activity_id'):
                merged_source_activity_id = str(sg_meta['source_activity_id'])
                break

    # Build a merged CleanedRaceData and cache it
    merged_riders = []
    for r, label, new_rank in all_riders:
        merged_riders.append(RiderData(
            rank=new_rank,
            activity_id=r.activity_id,
            name=r.name,
            team=r.team,
            data=r.data,
            finish_time_sec=r.finish_time_sec,
            weight_kg=r.weight_kg,
            player_id=r.player_id,
            activity_start_time=r.activity_start_time,
            ttt_time_offset=r.ttt_time_offset,
        ))

    merged = CleanedRaceData(
        race_id=race_id,
        route_name=merged_route_name,
        finish_line_km=merged_finish,
        riders=merged_riders,
        elevation_profile=merged_elevation,
        min_time=merged_min_time,
        max_time=merged_max_time,
        source_activity_id=merged_source_activity_id,
        route_slug=merged_route_slug,
        world=merged_world,
    )
    _race_data_cache[race_id] = merged

    # Also store the category mapping for use in api_race_data
    _race_data_cache[race_id + '_categories'] = {
        new_rank: label for _, label, new_rank in all_riders
    }

    return jsonify({
        'success': True,
        'race_id': race_id,
        'route_name': merged.route_name or 'Unknown',
        'route_slug': merged.route_slug,
        'finish_line_km': round(float(merged.finish_line_km), 3),
        'rider_count': len(merged.riders),
        'min_time': int(merged.min_time),
        'max_time': int(merged.max_time),
        'riders': [
            {
                'rank': int(r.rank),
                'name': r.name,
                'team': r.team,
                'finish_time_sec': float(r.finish_time_sec) if r.finish_time_sec is not None else None,
                'category': _race_data_cache[race_id + '_categories'].get(int(r.rank)),
            }
            for r in merged.riders
        ],
    })


@app.route('/api/race/data/<race_id>')
def api_race_data(race_id):
    """Get full cleaned race data (all riders, all timestamps) as JSON."""
    if race_id not in _race_data_cache:
        # Try loading
        data_path = Path('race_data') / race_id
        if not data_path.exists():
            # Try downloading from blob storage
            if blob_storage.race_exists_in_blob(race_id):
                data_path.parent.mkdir(exist_ok=True)
                data_path.mkdir(parents=True, exist_ok=True)
                blob_storage.download_race_dir(race_id, data_path)
        if not data_path.exists():
            return jsonify({'error': f'Race data not found: {race_id}'}), 404
        try:
            from race_replay.data_cleaner import clean_race_data
            race_data = clean_race_data(data_path, cache=True)
            _race_data_cache[race_id] = race_data
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    race_data = _race_data_cache[race_id]

    # Detect world if not already set (e.g. older cached data)
    world = race_data.world
    if not world and race_data.riders:
        ref = max(race_data.riders, key=lambda r: len(r.data))
        df0 = ref.data.reset_index()
        if 'lat' in df0.columns and 'lng' in df0.columns:
            mid = len(df0) // 2
            from race_replay.data_cleaner import detect_world_from_coords
            world = detect_world_from_coords(float(df0['lat'].iloc[mid]), float(df0['lng'].iloc[mid]))

    # Build elevation profile
    elev = race_data.elevation_profile
    elevation_profile = {
        'distance_km': elev['distance_km'].tolist(),
        'altitude_m': elev['altitude_m'].tolist(),
    }

    # Load race metadata (start time, event IDs) — needed for late joiner detection
    race_start_time = None
    event_id = None
    event_subgroup_id = None
    meta_path = Path('race_data') / race_id / 'race_meta.json'
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            race_start_time = meta.get('race_start_time')
            event_id = meta.get('event_id')
            event_subgroup_id = meta.get('event_subgroup_id')
        except Exception:
            pass

    # Build rider data — each rider's full time series
    riders = []
    def safe_list(series):
        """Convert pandas Series to list, replacing NaN with None for valid JSON."""
        return [None if pd.isna(v) else v for v in series]

    # Detect late joiners by comparing each rider's activity start time against
    # the official race start time.  Anyone who started their activity after the
    # race gun is flagged.
    race_start_dt = None
    if race_start_time:
        try:
            # Parse the ISO 8601 race start time
            ts = race_start_time.replace('+0000', '+00:00').replace('Z', '+00:00')
            race_start_dt = datetime.fromisoformat(ts)
        except Exception:
            pass

    # Category mapping (populated by multi-subgroup load)
    cat_map = _race_data_cache.get(race_id + '_categories', {})

    for i, r in enumerate(race_data.riders):
        df = r.data.reset_index()  # time_sec is the index

        # Late joiner: activity started after the race start
        is_late_joiner = False
        if race_start_dt and r.activity_start_time:
            try:
                ts = r.activity_start_time.replace('+0000', '+00:00').replace('Z', '+00:00')
                rider_start_dt = datetime.fromisoformat(ts)
                is_late_joiner = rider_start_dt > race_start_dt
            except Exception:
                pass
        rider_json = {
            'rank': int(r.rank),
            'name': r.name,
            'team': r.team,
            'category': cat_map.get(int(r.rank)),
            'activity_id': str(r.activity_id),
            'player_id': int(r.player_id) if r.player_id else None,
            'weight_kg': float(r.weight_kg),
            'is_late_joiner': is_late_joiner,
            'finish_time_sec': float(r.finish_time_sec) if r.finish_time_sec is not None else None,
            'ttt_time_offset': round(float(r.ttt_time_offset), 1) if r.ttt_time_offset is not None else None,
            'time_sec': df['time_sec'].tolist(),
            'distance_km': safe_list(df['distance_km']),
            'altitude_m': safe_list(df['altitude_m']) if 'altitude_m' in df.columns else [],
            'speed_kmh': safe_list(df['speed_kmh']) if 'speed_kmh' in df.columns else [],
            'power_watts': safe_list(df['power_watts']) if 'power_watts' in df.columns else [],
            'hr_bpm': safe_list(df['hr_bpm']) if 'hr_bpm' in df.columns else [],
            'lat': safe_list(df['lat']) if 'lat' in df.columns else [],
            'lng': safe_list(df['lng']) if 'lng' in df.columns else [],
        }
        riders.append(rider_json)

    # Build world map configuration
    map_config = None
    if world:
        from shared.world_config import get_world_map_config
        map_config = get_world_map_config(world)

    # Load official route polyline from ZwiftMap (if available)
    route_latlng = None
    if race_data.route_slug:
        try:
            from race_replay.data_cleaner import load_route_data
            rd = load_route_data(race_data.route_slug)
            if rd is not None:
                route_latlng = rd.latlng.tolist()
        except Exception as e:
            logger.warning("Could not load route latlng: %s", e)

    # Collect dev warnings for localhost debugging
    dev_warnings = []
    if race_data.route_slug:
        from race_replay.data_cleaner import ROUTE_STRAVA_SEGMENTS
        if race_data.route_slug not in ROUTE_STRAVA_SEGMENTS:
            dev_warnings.append(f"Missing Strava segment ID for route: {race_data.route_slug}")
    try:
        from shared.youtube_streams import get_api_key
        if not get_api_key():
            dev_warnings.append('YOUTUBE_API_KEY environment variable is not set')
    except Exception:
        dev_warnings.append('Could not check YouTube API key')

    return jsonify({
        'race_id': race_id,
        'route_name': race_data.route_name,
        'route_slug': race_data.route_slug,
        'source_activity_id': str(race_data.source_activity_id) if race_data.source_activity_id else None,
        'race_start_time': race_start_time,
        'event_id': event_id,
        'event_subgroup_id': event_subgroup_id,
        'finish_line_km': float(race_data.finish_line_km),
        'min_time': int(race_data.min_time),
        'max_time': int(race_data.max_time),
        'elevation_profile': elevation_profile,
        'riders': riders,
        'map_config': map_config,
        'route_latlng': route_latlng,
        'dev_warnings': dev_warnings,
    })


@app.route('/api/race/streams/<race_id>')
def api_race_streams(race_id):
    """Find YouTube livestreams from known streamers who were in this race."""
    try:
        from shared.youtube_streams import find_matching_streams, get_api_key

        # Check for cached stream results first (avoids slow YouTube API calls)
        stream_cache_path = Path('race_data') / race_id / 'stream_cache.json'
        # Ensure race data is available locally (download from blob if needed)
        race_dir = Path('race_data') / race_id
        if not race_dir.exists() and blob_storage.race_exists_in_blob(race_id):
            race_dir.mkdir(parents=True, exist_ok=True)
            blob_storage.download_race_dir(race_id, race_dir)
        if stream_cache_path.exists():
            with open(stream_cache_path, encoding='utf-8') as f:
                cached = json.load(f)
            return jsonify(cached)

        if not get_api_key():
            return jsonify({'streams': [], 'note': 'YOUTUBE_API_KEY not configured'})

        # Load race metadata for start time
        meta_path = Path('race_data') / race_id / 'race_meta.json'
        if not meta_path.exists():
            return jsonify({'streams': [], 'error': 'Race metadata not found'})

        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

        race_start_time = meta.get('race_start_time')
        if not race_start_time:
            return jsonify({'streams': [], 'note': 'No race start time available'})

        # Get player IDs from in-memory cache or fall back to disk
        if race_id in _race_data_cache:
            race_data = _race_data_cache[race_id]
            player_ids = [r.player_id for r in race_data.riders if r.player_id]
            race_duration = race_data.max_time - race_data.min_time
        else:
            # Fall back to reading from cleaned_cache.json or summary CSV
            cache_path = Path('race_data') / race_id / 'cleaned_cache.json'
            summary_path = Path('race_data') / race_id / 'complete_race_summary.csv'
            player_ids = []
            race_duration = 3600  # Default 1h if unknown

            if cache_path.exists():
                with open(cache_path, encoding='utf-8') as f:
                    cache_meta = json.load(f)
                player_ids = [r['player_id'] for r in cache_meta.get('riders', []) if r.get('player_id')]
                race_duration = cache_meta.get('max_time', 3600) - cache_meta.get('min_time', 0)
            elif summary_path.exists():
                import csv
                with open(summary_path, encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pid = row.get('player_id')
                        if pid:
                            try:
                                player_ids.append(int(float(pid)))
                            except (ValueError, TypeError):
                                pass

            if not player_ids:
                return jsonify({'streams': [], 'note': 'No player IDs available'})

        streams = find_matching_streams(
            race_start_time=race_start_time,
            race_duration_sec=race_duration,
            rider_player_ids=player_ids,
        )

        # Cache results to disk so subsequent loads are instant
        result = {'streams': streams}
        if streams:
            with open(stream_cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            # Persist stream cache to blob storage
            blob_storage.upload_file(race_id, stream_cache_path)

        return jsonify(result)
    except Exception as e:
        logger.exception("Error fetching streams")
        return jsonify({'streams': [], 'error': str(e)})


# ---------------------------------------------------------------------------
# TTT Analysis
# ---------------------------------------------------------------------------

# TTT bike setup: Cadex Tri (F112) + DT Swiss ARC 1100 DICUT 85/Disc (W055), level 5
TTT_FRAME_ID = 'F112'
TTT_WHEEL_ID = 'W055'
TTT_UPGRADE_LEVEL = 5

_ttt_cache = {}  # subgroup_id -> processed rider data for reassignment


@app.route('/ttt-analysis')
def ttt_analysis():
    """Team Time Trial analysis page."""
    return render_template('ttt_analysis.html')


@app.route('/api/ttt/fetch', methods=['POST'])
def api_ttt_fetch():
    """Fetch and analyze a TTT race.

    Accepts JSON with ``subgroup_id`` (the event subgroup ID).

    Steps:
    1. Fetch race entries and telemetry for every rider.
    2. Group riders into teams by race segment start time (>60 s gap).
    3. Compute per-second draft estimates assuming Cadex Tri + ARC 85/Disc L5.
    4. Build per-team time-series arrays for the chart (lead speed, avg draft
       efficiency, max distance) — only including riders connected to the team
       (within 10 s of the leader).

    Returns JSON with ``teams`` list, each containing metadata and chart arrays.
    """
    headers = get_headers()
    if not headers:
        return jsonify({'error': 'Not authenticated. Please log in first.'}), 401

    data = request.json or {}
    raw_id = str(data.get('subgroup_id', '')).strip()
    if not raw_id:
        return jsonify({'error': 'subgroup_id is required'}), 400

    try:
        event_subgroup_id = int(raw_id)
    except ValueError:
        return jsonify({'error': 'subgroup_id must be a number'}), 400

    # ---- Fetch event metadata via the first participant's activity ----
    event_name = 'TTT Race'
    route_id = None
    event_distance_km = None

    # ---- Fetch race entries ----
    participants, error = get_race_entries(event_subgroup_id, headers)
    if error:
        return jsonify({'error': error}), 502

    # ---- Resolve event name from first participant's activity ----
    first_act = None
    if participants:
        first_act, _ = get_activity_details(participants[0]['activity_id'], headers)
        if first_act:
            ei = first_act.get('eventInfo', {})
            event_name = ei.get('name', event_name)
            eid = ei.get('id')
            if eid:
                try:
                    ev_resp = _request_with_retry('GET', f'{BASE_URL}/events/{eid}', headers=headers, timeout=15)
                    if ev_resp.status_code == 200:
                        ev = ev_resp.json()
                        event_name = ev.get('name', event_name)
                        for sg in ev.get('eventSubgroups', []):
                            if sg.get('id') == event_subgroup_id:
                                label = sg.get('subgroupLabel', '')
                                if label:
                                    event_name = f'{event_name} — {label}'
                                route_id = sg.get('routeId')
                                sg_laps = sg.get('laps') or 1
                                # Use routes_cache to get race distance (route × laps + lead-in)
                                if route_id:
                                    from shared.route_lookup import get_route_info as _get_ri
                                    ri = _get_ri(route_id)
                                    if ri and ri.get('distanceInMeters'):
                                        leadin = ri.get('leadinDistanceInMeters', 0)
                                        event_distance_km = round((ri['distanceInMeters'] * sg_laps + leadin) / 1000, 2)
                                break
                except Exception:
                    pass

    # ---- Group riders into teams by name tags ----
    import re
    from collections import defaultdict

    raw_tags = data.get('team_tags', '').strip()

    def _extract_bracket_contents(name):
        """Extract all content inside [...] or (...) in a rider name."""
        return re.findall(r'[\[\(]([^)\]]+)[\]\)]', name)

    def _rider_has_tag(name, tag):
        """Check if tag appears as a word inside any bracketed section of name."""
        for content in _extract_bracket_contents(name):
            # Split on common separators (|, /, space) and strip each part
            parts = [p.strip() for p in re.split(r'[|/]', content)]
            if tag in parts:
                return True
        return False

    if raw_tags:
        # User provided explicit tags — use those
        team_tags = [t.strip() for t in raw_tags.split(',') if t.strip()]
    else:
        # Auto-detect: collect every simple tag that appears on >= 2 riders
        from collections import Counter
        tag_counts = Counter()
        for p in participants:
            for content in _extract_bracket_contents(p['name']):
                parts = [pt.strip() for pt in re.split(r'[|/]', content)]
                for part in set(parts):
                    if part:
                        tag_counts[part] += 1
        team_tags = [tag for tag, cnt in tag_counts.most_common() if cnt >= 2]

    if not team_tags:
        return jsonify({'error': 'Could not detect team tags. Please provide them manually (e.g. V, RtB, SZ).'}), 400

    # Assign each rider to their team (first matching tag wins)
    team_map = defaultdict(list)  # tag -> [participant, ...]
    for p in participants:
        for tag in team_tags:
            if _rider_has_tag(p['name'], tag):
                team_map[tag].append(p)
                break  # assign to first matching team only

    if not team_map:
        return jsonify({'error': 'No riders matched any team tags.'}), 400

    # ---- Fetch telemetry for ALL riders (concurrent) ----
    # We fetch everyone so unassigned riders can be reassigned via
    # drag-and-drop without re-fetching telemetry.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ttt_setup = get_bike_stats(TTT_FRAME_ID, TTT_WHEEL_ID, TTT_UPGRADE_LEVEL)
    if not ttt_setup:
        return jsonify({'error': 'Could not load TTT bike data'}), 500

    def fetch_one(p):
        act_id = p['activity_id']
        telem_data, act_data, err = fetch_rider_telemetry(act_id, headers)
        if err:
            return {**p, 'telemetry': None, 'telem_raw': None, 'error': err}

        df = convert_telemetry_to_dataframe(telem_data)

        height_cm = p.get('height_cm')
        if not height_cm and act_data and act_data.get('profile'):
            height_cm = act_data['profile'].get('heightInCentimeters')

        return {
            **p,
            'telemetry': df,
            'telem_raw': telem_data,
            'height_cm': height_cm or 175,
            'error': None,
        }

    rider_lookup = {}  # activity_id -> fetched rider data
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_one, p): p for p in participants}
        for fut in as_completed(futures):
            result = fut.result()
            if result['telemetry'] is not None:
                rider_lookup[result['activity_id']] = result

    if not rider_lookup:
        return jsonify({'error': 'No telemetry data could be fetched.'}), 502

    # ---- 5. Compute draft estimate for ALL riders ----
    from bike_comparison.physics import calculate_power_for_speed, DRIVETRAIN_LOSS, AIR_DENSITY

    all_processed = {}  # activity_id -> processed rider dict
    for act_id, r in rider_lookup.items():
        df = r['telemetry']
        if len(df) < 10:
            continue

        weight_kg = r.get('weight_kg', 75.0)
        height_cm = r.get('height_cm', 175)
        frontal_area = estimate_frontal_area(height_cm, weight_kg, 'drops')
        cda = ttt_setup.cd * frontal_area

        time_sec = df['time_sec'].values.astype(float)
        speed_mps = (df['speed_kmh'].values / 3.6).astype(float)
        distance_m = (df['distance_km'].values * 1000).astype(float)
        altitude_m = df['altitude_m'].values.astype(float)
        power = df['power_watts'].values.astype(float)

        dist_diff = np.diff(distance_m, prepend=distance_m[0])
        dist_diff[0] = dist_diff[1] if len(dist_diff) > 1 else 1
        dist_diff = np.maximum(dist_diff, 0.1)
        alt_diff = np.diff(altitude_m, prepend=altitude_m[0])
        alt_diff[0] = 0

        total_mass = weight_kg + ttt_setup.weight_kg

        crr = 0.004
        f_rolling = crr * total_mass * 9.8067
        f_aero = 0.5 * AIR_DENSITY * cda * speed_mps ** 2
        safe_speed = np.maximum(speed_mps, 0.5)
        resistance_power_solo = (f_rolling + f_aero) * safe_speed / (1 - DRIVETRAIN_LOSS)

        dt = np.diff(time_sec, prepend=time_sec[0])
        dt[0] = dt[1] if len(dt) > 1 else 1
        dt = np.maximum(dt, 0.01)
        ke = 0.5 * total_mass * speed_mps ** 2
        pe = total_mass * 9.8067 * altitude_m
        total_energy = ke + pe
        energy_change_rate = np.zeros_like(total_energy)
        energy_change_rate[:-1] = np.diff(total_energy) / dt[1:]
        if len(energy_change_rate) > 1:
            energy_change_rate[-1] = energy_change_rate[-2]

        draft_watts = resistance_power_solo - power + energy_change_rate / (1 - DRIVETRAIN_LOSS)
        window = min(5, len(draft_watts))
        if window > 1:
            kernel = np.ones(window) / window
            draft_watts = np.convolve(draft_watts, kernel, mode='same')

        all_processed[act_id] = {
            'name': r['name'],
            'activity_id': act_id,
            'weight_kg': weight_kg,
            'time_sec': time_sec,
            'speed_mps': speed_mps,
            'distance_m': distance_m,
            'altitude_m': altitude_m,
            'power': power,
            'draft_watts': draft_watts,
        }

    # ---- 5b. Route-project riders for accurate distance alignment ----
    # Reuse the battle-tested align_riders_to_route() from race_replay
    route_data = None
    leadin_m = 0.0
    if route_id:
        from shared.route_lookup import get_route_info as _get_ri
        ri = _get_ri(route_id)
        if ri:
            route_slug = ri['name'].lower().replace(' ', '-').replace("'", "")
            leadin_m = ri.get('leadinDistanceInMeters', 0.0)
            from race_replay.data_cleaner import load_route_data, align_riders_to_route
            route_data = load_route_data(route_slug)

    if route_data:
        from scipy.interpolate import interp1d
        import pandas as pd

        # Build rider dicts in the format align_riders_to_route expects
        alignment_riders = []
        act_id_order = []  # track which act_id maps to which alignment rider

        for act_id, rd in all_processed.items():
            r = rider_lookup[act_id]
            raw = r.get('telem_raw') or {}
            latlng_raw = raw.get('latlng', [])
            times_raw = raw.get('timeInSec', [])

            if not latlng_raw or len(latlng_raw) != len(times_raw):
                continue

            # Interpolate latlng to the rider's sample times
            lats_raw = np.array([ll[0] for ll in latlng_raw])
            lngs_raw = np.array([ll[1] for ll in latlng_raw])
            lat_interp = interp1d(times_raw, lats_raw, fill_value='extrapolate', bounds_error=False)
            lng_interp = interp1d(times_raw, lngs_raw, fill_value='extrapolate', bounds_error=False)

            adf = pd.DataFrame({
                'lat': lat_interp(rd['time_sec']),
                'lng': lng_interp(rd['time_sec']),
                'distance_km': rd['distance_m'] / 1000.0,
                'time_sec': rd['time_sec'],
                'speed_kmh': rd['speed_mps'] * 3.6,
            })
            alignment_riders.append({'data': adf, 'rank': len(alignment_riders) + 1})
            act_id_order.append(act_id)

        if alignment_riders:
            aligned, _ = align_riders_to_route(
                alignment_riders, route_data,
                leadin_distance_m=leadin_m,
                detect_late_joiners=False,  # TTT: teams start at staggered times, not late joiners
            )
            # Copy projected distances back into all_processed.
            # On loop courses, route alignment breaks down after the
            # finish line (the route segment is only 1 lap, so post-
            # finish GPS can't be matched and distance drifts backwards).
            # Fix: at the finish line, compute the offset between aligned
            # and raw Zwift odometer distance, then use raw + offset for
            # everything past the finish.  The raw odometer is always
            # monotonically increasing.
            race_finish_m = event_distance_km * 1000 if event_distance_km else None
            for i, act_id in enumerate(act_id_order):
                adf = aligned[i]['data']
                aligned_dist = adf['distance_km'].values * 1000.0
                raw_dist = all_processed[act_id]['distance_m']
                if len(aligned_dist) != len(raw_dist):
                    logger.warning("TTT: rider %s alignment length mismatch (%d vs %d), using raw distance",
                                   all_processed[act_id]['name'], len(aligned_dist), len(raw_dist))
                    continue

                # Fix post-finish distances using raw odometer + offset
                if race_finish_m is not None:
                    # Find where the rider first reaches the finish distance
                    past_finish = np.where(aligned_dist >= race_finish_m)[0]
                    if len(past_finish) > 0:
                        finish_idx = past_finish[0]
                        offset = aligned_dist[finish_idx] - raw_dist[finish_idx]
                        # Replace everything after the finish with raw + offset
                        aligned_dist[finish_idx + 1:] = raw_dist[finish_idx + 1:] + offset

                all_processed[act_id]['route_distance_m'] = aligned_dist

    # ---- 5c. Per-rider time offset + resample to integer seconds ----
    # Use the shared compute_ttt_time_offset() + resample logic from the
    # data cleaner — same algorithm the race replay page uses.
    finish_line_m = None
    if event_distance_km:
        finish_line_m = event_distance_km * 1000
    else:
        for p in participants:
            sd = p.get('segment_distance_cm')
            if sd:
                finish_line_m = sd / 100.0  # cm -> m
                break

    if finish_line_m and route_data is not None:
        from race_replay.data_cleaner import compute_ttt_time_offset
        finish_lat = route_data.latlng[-1, 0]
        finish_lng = route_data.latlng[-1, 1]

        elapsed_lookup = {}
        for p in participants:
            ems = p.get('elapsed_ms')
            if ems:
                elapsed_lookup[str(p['activity_id'])] = ems

        offsets_applied = 0
        for act_id, rd in all_processed.items():
            ems = elapsed_lookup.get(act_id)
            if ems is None:
                continue

            # Get raw latlng for this rider
            r = rider_lookup[act_id]
            raw = r.get('telem_raw') or {}
            latlng_raw = raw.get('latlng', [])
            times_raw = raw.get('timeInSec', [])
            if not latlng_raw or len(latlng_raw) != len(times_raw):
                continue

            lats_raw = np.array([ll[0] for ll in latlng_raw])
            lngs_raw = np.array([ll[1] for ll in latlng_raw])
            times_raw_arr = np.array(times_raw, dtype=float)
            dists_raw = np.array(raw.get('distanceInCm', []), dtype=float) / 100.0

            if len(dists_raw) != len(times_raw_arr):
                continue

            offset = compute_ttt_time_offset(
                times=times_raw_arr,
                distances_m=dists_raw,
                lats=lats_raw,
                lngs=lngs_raw,
                finish_lat=finish_lat,
                finish_lng=finish_lng,
                finish_distance_m=finish_line_m,
                official_time_sec=ems / 1000.0,
            )
            if offset is not None:
                rd['time_sec'] = rd['time_sec'] - offset
                offsets_applied += 1
                logger.info("TTT offset for %s: %.2fs", rd['name'], offset)

        if offsets_applied:
            logger.info("TTT: applied time offsets to %d / %d riders",
                        offsets_applied, len(all_processed))

    # Resample all riders to integer-second grid (same as data_cleaner).
    # After TTT offset, timestamps are fractional; this makes all riders
    # directly comparable at the same integer seconds.
    for act_id, rd in all_processed.items():
        t = rd['time_sec']
        int_start = int(np.ceil(t[0]))
        int_end = int(np.floor(t[-1]))
        if int_end <= int_start:
            continue
        int_times = np.arange(int_start, int_end + 1, dtype=float)
        for key in ('speed_mps', 'distance_m', 'altitude_m', 'power', 'draft_watts'):
            if key in rd:
                rd[key] = np.interp(int_times, t, rd[key])
        if 'route_distance_m' in rd:
            rd['route_distance_m'] = np.interp(int_times, t, rd['route_distance_m'])
        rd['time_sec'] = int_times

    # ---- 6. Build team assignments and aggregate ----
    team_assignments = {}  # activity_id -> tag
    for tag, riders in team_map.items():
        for p in riders:
            if p['activity_id'] in all_processed:
                team_assignments[p['activity_id']] = tag

    team_results = _build_ttt_team_results(all_processed, team_assignments, team_tags)

    assigned_ids = set(team_assignments.keys())
    unassigned = [
        {'name': rd['name'], 'activity_id': act_id, 'weight_kg': rd['weight_kg']}
        for act_id, rd in all_processed.items()
        if act_id not in assigned_ids
    ]

    race_distance_km = event_distance_km or _get_race_distance_km(participants, team_results)

    # Cache for reassignment (drag-and-drop)
    _ttt_cache[event_subgroup_id] = {
        'event_name': event_name,
        'ttt_bike': str(ttt_setup),
        'all_processed': all_processed,
        'team_tags': team_tags,
        'participants': participants,
        'race_distance_km': race_distance_km,
        'team_assignments': dict(team_assignments),
    }

    return jsonify({
        'event_name': event_name,
        'teams': team_results,
        'ttt_bike': str(ttt_setup),
        'race_distance_km': race_distance_km,
        'unassigned': unassigned,
        'team_tags': team_tags,
    })


def _build_ttt_team_results(all_processed, team_assignments, tag_order):
    """Aggregate per-rider processed data into per-team chart arrays.

    ``all_processed`` maps activity_id -> rider dict with numpy arrays.
    ``team_assignments`` maps activity_id -> team tag string.
    ``tag_order`` is the list of tags in display order.
    """
    # Group riders by tag
    teams_by_tag = {}
    for act_id, tag in team_assignments.items():
        if act_id in all_processed:
            teams_by_tag.setdefault(tag, []).append(all_processed[act_id])

    team_results = []
    for tag in tag_order:
        rider_data = teams_by_tag.get(tag, [])
        if not rider_data:
            continue

        # Use route-projected distance when available, fall back to raw
        def _get_dist(rd):
            return rd.get('route_distance_m', rd['distance_m'])

        max_dist = max(_get_dist(rd)[-1] for rd in rider_data)
        step = 50
        dist_axis = np.arange(0, max_dist, step)

        lead_speed = np.full(len(dist_axis), np.nan)
        avg_draft_efficiency = np.full(len(dist_axis), np.nan)
        elevation = np.full(len(dist_axis), np.nan)
        lead_time = np.full(len(dist_axis), np.nan)
        lead_power = np.full(len(dist_axis), np.nan)
        lead_rider_name = [None] * len(dist_axis)

        # ---- Leader detection ----
        # All riders are already resampled to integer-second grids (step
        # 5d), using the same GPS-projected route_distance_m as the race
        # replay page.  Concatenate, group by second, leader = max
        # distance, then bin into 50 m distance buckets and average.

        all_times = []
        all_dists = []
        all_powers = []
        all_speeds = []
        all_alts = []
        all_drafts = []
        all_names = []

        for rd in rider_data:
            dist_arr = _get_dist(rd)
            t = rd['time_sec']
            n = min(len(t), len(dist_arr))
            all_times.append(t[:n])
            all_dists.append(dist_arr[:n])
            all_powers.append(rd['power'][:n])
            all_speeds.append(rd['speed_mps'][:n])
            all_alts.append(rd['altitude_m'][:n])
            all_drafts.append(rd['draft_watts'][:n])
            all_names.extend([rd['name']] * n)

        cat_times = np.concatenate(all_times).astype(int)
        cat_dists = np.concatenate(all_dists)
        cat_powers = np.concatenate(all_powers)
        cat_speeds = np.concatenate(all_speeds)
        cat_alts = np.concatenate(all_alts)
        cat_drafts = np.concatenate(all_drafts)
        cat_names = np.array(all_names, dtype=object)

        unique_secs = np.unique(cat_times)

        leader_dist_list = []
        leader_power_list = []
        leader_speed_list = []
        leader_alt_list = []
        leader_name_list = []
        leader_time_list = []
        draft_by_bin = {}

        for sec in unique_secs:
            mask = cat_times == sec
            sec_dists = cat_dists[mask]
            sec_powers = cat_powers[mask]
            sec_speeds = cat_speeds[mask]
            sec_alts = cat_alts[mask]
            sec_names = cat_names[mask]
            sec_drafts = cat_drafts[mask]

            lead_idx = np.argmax(sec_dists)
            ld = sec_dists[lead_idx]
            leader_dist_list.append(ld)
            leader_power_list.append(sec_powers[lead_idx])
            leader_speed_list.append(sec_speeds[lead_idx])
            leader_alt_list.append(sec_alts[lead_idx])
            leader_name_list.append(sec_names[lead_idx])
            leader_time_list.append(float(sec))

            bi = int(ld / step)
            if 0 <= bi < len(dist_axis):
                for si in range(len(sec_dists)):
                    p = sec_powers[si]
                    dw = sec_drafts[si]
                    if p > 10:
                        draft_by_bin.setdefault(bi, []).append((dw, p))

        leader_dist_arr = np.array(leader_dist_list)
        leader_power_arr = np.array(leader_power_list)
        leader_speed_arr = np.array(leader_speed_list)
        leader_alt_arr = np.array(leader_alt_list)
        leader_time_arr = np.array(leader_time_list)
        leader_name_arr = np.array(leader_name_list, dtype=object)

        from collections import Counter
        leader_bins = (leader_dist_arr / step).astype(int)

        for di in range(len(dist_axis)):
            in_bin = leader_bins == di
            if not np.any(in_bin):
                continue

            lead_power[di] = float(np.mean(leader_power_arr[in_bin]))
            lead_speed[di] = float(np.mean(leader_speed_arr[in_bin])) * 3.6
            elevation[di] = float(np.median(leader_alt_arr[in_bin]))
            lead_time[di] = float(np.mean(leader_time_arr[in_bin]))

            name_counts = Counter(leader_name_arr[in_bin])
            lead_rider_name[di] = name_counts.most_common(1)[0][0]

            if di in draft_by_bin:
                pairs = draft_by_bin[di]
                total_draft = sum(dw for dw, p in pairs)
                total_power = sum(p for dw, p in pairs)
                if total_power > 0:
                    avg_draft_efficiency[di] = total_draft / total_power

        valid_speed = ~np.isnan(lead_speed)
        valid_draft = ~np.isnan(avg_draft_efficiency)
        valid_power = ~np.isnan(lead_power)

        # Save raw (unsmoothed) values for per-rider breakdown.
        # Smoothing blends power/speed across rider boundaries, which is
        # fine for team-vs-team comparison but wrong for showing individual
        # rider pulls (e.g. a 340 W pull gets diluted to 260 W).
        lead_speed_raw = lead_speed.copy()
        lead_power_raw = lead_power.copy()

        if valid_speed.sum() > 10:
            smooth_w = min(11, int(valid_speed.sum() / 5))
            if smooth_w > 1 and smooth_w % 2 == 0:
                smooth_w += 1
            if smooth_w > 1:
                ls = lead_speed.copy()
                ls[~valid_speed] = 0
                ls = uniform_filter1d(ls, smooth_w)
                ls[~valid_speed] = np.nan
                lead_speed = ls

        if valid_draft.sum() > 10:
            smooth_w = min(11, int(valid_draft.sum() / 5))
            if smooth_w > 1 and smooth_w % 2 == 0:
                smooth_w += 1
            if smooth_w > 1:
                de = avg_draft_efficiency.copy()
                de[~valid_draft] = 0
                de = uniform_filter1d(de, smooth_w)
                de[~valid_draft] = np.nan
                avg_draft_efficiency = de

        if valid_power.sum() > 10:
            smooth_w = min(11, int(valid_power.sum() / 5))
            if smooth_w > 1 and smooth_w % 2 == 0:
                smooth_w += 1
            if smooth_w > 1:
                lp = lead_power.copy()
                lp[~valid_power] = 0
                lp = uniform_filter1d(lp, smooth_w)
                lp[~valid_power] = np.nan
                lead_power = lp

        avg_weight_kg = round(float(np.mean([rd['weight_kg'] for rd in rider_data])), 1)

        # ---- Per-rider pull statistics (from per-second leader data) ----
        # Computed before binning so durations are exact (1 s resolution).
        weight_map = {rd['name']: rd['weight_kg'] for rd in rider_data}
        rider_names_ordered = list(dict.fromkeys(
            n for n in leader_name_list if n))
        pull_stats = {}
        for n in rider_names_ordered:
            pull_stats[n] = {'pulls': 0, 'total_sec': 0,
                             'power_sum': 0.0, 'power_count': 0}

        prev_leader = None
        for i, name in enumerate(leader_name_list):
            if not name or name not in pull_stats:
                prev_leader = None
                continue
            if name != prev_leader:
                pull_stats[name]['pulls'] += 1
            pull_stats[name]['total_sec'] += 1  # 1-second grid
            pw = leader_power_list[i]
            if pw is not None and not np.isnan(pw):
                pull_stats[name]['power_sum'] += float(pw)
                pull_stats[name]['power_count'] += 1
            prev_leader = name

        pull_stats_list = []
        for name in rider_names_ordered:
            s = pull_stats[name]
            if s['pulls'] == 0:
                continue
            avg_dur = s['total_sec'] / s['pulls']
            avg_pow = s['power_sum'] / s['power_count'] if s['power_count'] else 0
            wt = weight_map.get(name, avg_weight_kg)
            pull_stats_list.append({
                'name': name,
                'pulls': s['pulls'],
                'avg_pull_duration_sec': round(avg_dur, 1),
                'total_time_at_front_sec': s['total_sec'],
                'avg_power_w': round(avg_pow, 0),
                'avg_power_wkg': round(avg_pow / wt, 1) if wt else 0,
            })

        team_results.append({
            'label': tag,
            'riders': [{'name': rd['name'], 'weight_kg': rd['weight_kg'], 'activity_id': rd['activity_id']} for rd in rider_data],
            'distance_km': (dist_axis / 1000).tolist(),
            'lead_speed_kph': [None if np.isnan(v) else round(v, 2) for v in lead_speed],
            'avg_draft_efficiency': [None if np.isnan(v) else round(v, 4) for v in avg_draft_efficiency],
            'elevation_m': [None if np.isnan(v) else round(v, 1) for v in elevation],
            'lead_time_sec': [None if np.isnan(v) else round(v, 1) for v in lead_time],
            'lead_power_watts': [None if np.isnan(v) else round(v, 0) for v in lead_power],
            'lead_speed_kph_raw': [None if np.isnan(v) else round(v, 2) for v in lead_speed_raw],
            'lead_power_watts_raw': [None if np.isnan(v) else round(v, 0) for v in lead_power_raw],
            'lead_rider_name': lead_rider_name,
            'avg_weight_kg': avg_weight_kg,
            'pull_stats': pull_stats_list,
        })

    return team_results


@app.route('/api/ttt/reassign', methods=['POST'])
def api_ttt_reassign():
    """Re-aggregate TTT team charts using updated rider-to-team assignments.

    Uses cached processed data from the most recent fetch so no API calls
    are needed.  Expects JSON: ``{subgroup_id, assignments: {activity_id: tag}}``.
    """
    data = request.json or {}
    try:
        subgroup_id = int(data.get('subgroup_id', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid subgroup_id'}), 400

    if subgroup_id not in _ttt_cache:
        return jsonify({'error': 'No cached data. Fetch the race first.'}), 400

    raw_assignments = data.get('assignments', {})
    assignments = {str(k): v for k, v in raw_assignments.items()}

    cached = _ttt_cache[subgroup_id]
    all_processed = cached['all_processed']

    # Preserve original tag order, then append any new tags
    all_tags_in_use = list(dict.fromkeys(assignments.values()))
    ordered_tags = [t for t in cached['team_tags'] if t in all_tags_in_use]
    for t in all_tags_in_use:
        if t not in ordered_tags:
            ordered_tags.append(t)

    team_results = _build_ttt_team_results(all_processed, assignments, ordered_tags)

    assigned_ids = set(assignments.keys())
    unassigned = [
        {'name': rd['name'], 'activity_id': act_id, 'weight_kg': rd['weight_kg']}
        for act_id, rd in all_processed.items()
        if act_id not in assigned_ids
    ]

    # Update cache
    cached['team_assignments'] = dict(assignments)
    cached['team_tags'] = ordered_tags

    return jsonify({
        'event_name': cached['event_name'],
        'teams': team_results,
        'ttt_bike': cached['ttt_bike'],
        'race_distance_km': cached.get('race_distance_km', 0),
        'unassigned': unassigned,
        'team_tags': ordered_tags,
    })


@app.route('/api/ttt/rider', methods=['POST'])
def api_ttt_rider():
    """Return individual rider data for the drilldown chart.

    Expects JSON: ``{subgroup_id, activity_id}``.
    Returns distance_km, altitude_m, power_watts, and draft_watts arrays
    sampled every ~50 m for charting.
    """
    data = request.json or {}
    try:
        subgroup_id = int(data.get('subgroup_id', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid subgroup_id'}), 400

    if subgroup_id not in _ttt_cache:
        return jsonify({'error': 'No cached data. Fetch the race first.'}), 400

    act_id = str(data.get('activity_id', ''))
    cached = _ttt_cache[subgroup_id]
    rd = cached['all_processed'].get(act_id)
    if not rd:
        return jsonify({'error': 'Rider not found in cache'}), 404

    # Downsample to ~50 m intervals for reasonable chart size
    # Use route-projected distance when available
    distance_m = rd.get('route_distance_m', rd['distance_m'])
    step = 50
    dist_axis = np.arange(0, distance_m[-1], step)
    indices = np.searchsorted(distance_m, dist_axis)
    indices = np.clip(indices, 0, len(distance_m) - 1)

    return jsonify({
        'name': rd['name'],
        'activity_id': act_id,
        'weight_kg': rd['weight_kg'],
        'distance_km': (dist_axis / 1000).tolist(),
        'altitude_m': [round(float(rd['altitude_m'][i]), 1) for i in indices],
        'power_watts': [round(float(rd['power'][i]), 0) for i in indices],
        'draft_watts': [round(float(rd['draft_watts'][i]), 0) for i in indices],
    })


@app.route('/api/ttt/debug_bins', methods=['POST'])
def api_ttt_debug_bins():
    """DEBUG: Show per-second leader data for a distance range.

    Expects JSON: ``{subgroup_id, team_tag, dist_start_km, dist_end_km}``.
    """
    data = request.json or {}
    try:
        subgroup_id = int(data.get('subgroup_id', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid subgroup_id'}), 400
    if subgroup_id not in _ttt_cache:
        return jsonify({'error': 'No cached data'}), 400

    cached = _ttt_cache[subgroup_id]
    all_processed = cached['all_processed']
    assignments = cached['team_assignments']
    tag = data.get('team_tag', '')
    dist_start_m = float(data.get('dist_start_km', 22.0)) * 1000
    dist_end_m = float(data.get('dist_end_km', 23.0)) * 1000

    # Gather rider data for this team
    rider_data = [all_processed[aid] for aid, t in assignments.items()
                  if t == tag and aid in all_processed]
    if not rider_data:
        return jsonify({'error': f'No riders for tag {tag}'}), 404

    def _get_dist(rd):
        return rd.get('route_distance_m', rd['distance_m'])

    # Concatenate and find leaders per second (same as _build_ttt_team_results)
    all_times, all_dists, all_speeds, all_powers, all_names = [], [], [], [], []
    for rd in rider_data:
        dist_arr = _get_dist(rd)
        t = rd['time_sec']
        n = min(len(t), len(dist_arr))
        all_times.append(t[:n])
        all_dists.append(dist_arr[:n])
        all_speeds.append(rd['speed_mps'][:n])
        all_powers.append(rd['power'][:n])
        all_names.extend([rd['name']] * n)

    cat_times = np.concatenate(all_times).astype(int)
    cat_dists = np.concatenate(all_dists)
    cat_speeds = np.concatenate(all_speeds)
    cat_powers = np.concatenate(all_powers)
    cat_names = np.array(all_names, dtype=object)

    unique_secs = np.unique(cat_times)
    step = 50

    # Build per-second leader records, filtered to distance range
    records = []
    for sec in unique_secs:
        mask = cat_times == sec
        sec_dists = cat_dists[mask]
        lead_idx = np.argmax(sec_dists)
        ld = sec_dists[lead_idx]
        if ld < dist_start_m or ld > dist_end_m:
            continue
        records.append({
            'time_sec': int(sec),
            'leader_dist_m': round(float(ld), 1),
            'bin': int(ld / step),
            'leader_name': str(cat_names[mask][lead_idx]),
            'leader_speed_kph': round(float(cat_speeds[mask][lead_idx]) * 3.6, 2),
            'leader_power_w': round(float(cat_powers[mask][lead_idx]), 0),
            'num_riders_at_sec': int(mask.sum()),
        })

    # Also show the binned result
    bins = {}
    for r in records:
        bi = r['bin']
        bins.setdefault(bi, []).append(r)

    bin_summary = []
    for bi in sorted(bins.keys()):
        pts = bins[bi]
        speeds = [p['leader_speed_kph'] for p in pts]
        powers = [p['leader_power_w'] for p in pts]
        names = [p['leader_name'] for p in pts]
        bin_summary.append({
            'bin': bi,
            'dist_km': round(bi * step / 1000, 3),
            'num_points': len(pts),
            'avg_speed_kph': round(sum(speeds) / len(speeds), 2),
            'min_speed_kph': round(min(speeds), 2),
            'max_speed_kph': round(max(speeds), 2),
            'avg_power_w': round(sum(powers) / len(powers), 0),
            'leaders': list(set(names)),
            'points': pts,
        })

    # Also dump each rider's raw per-second data in the range
    rider_raw = {}
    rider_dist_source = {}
    for rd in rider_data:
        dist_arr = _get_dist(rd)
        has_route = 'route_distance_m' in rd
        rider_dist_source[rd['name']] = 'route_distance_m' if has_route else 'distance_m'

        # Check monotonicity of full distance array
        diffs = np.diff(dist_arr)
        n_decreasing = int(np.sum(diffs < -10))  # > 10m decrease
        max_decrease = float(np.min(diffs)) if len(diffs) > 0 else 0
        total_pts = len(dist_arr)
        max_dist = float(dist_arr.max()) if len(dist_arr) > 0 else 0

        t = rd['time_sec']
        spd = rd['speed_mps']
        n = min(len(t), len(dist_arr), len(spd))
        raw_pts = []
        for j in range(n):
            if dist_arr[j] >= dist_start_m and dist_arr[j] <= dist_end_m:
                raw_pts.append({
                    'time_sec': int(t[j]),
                    'dist_m': round(float(dist_arr[j]), 1),
                    'speed_kph': round(float(spd[j]) * 3.6, 2),
                    'bin': int(dist_arr[j] / step),
                })
        rider_raw[rd['name']] = {
            'dist_source': 'route_distance_m' if has_route else 'distance_m',
            'total_samples': total_pts,
            'max_dist_m': round(max_dist, 1),
            'n_decreasing': n_decreasing,
            'max_decrease_m': round(max_decrease, 1),
            'points_in_range': raw_pts,
        }

    return jsonify({
        'dist_range_m': [dist_start_m, dist_end_m],
        'total_seconds': len(records),
        'bins': bin_summary,
        'rider_raw': rider_raw,
    })


def _get_race_distance_km(participants, team_results):
    """Determine the official race distance in km.

    Uses ``segment_distance_cm`` from any participant if available,
    otherwise falls back to the maximum charted distance across teams.
    """
    for p in participants:
        sd = p.get('segment_distance_cm')
        if sd:
            return round(sd / 100_000, 2)
    # Fallback: max distance in any team's chart data
    max_d = 0
    for t in team_results:
        dists = t.get('distance_km', [])
        if dists:
            max_d = max(max_d, max(d for d in dists if d is not None))
    return round(max_d, 2) if max_d else 0


def _guess_team_name(names):
    """Try to extract a common team tag from rider names.

    Many Zwift teams put their tag in square brackets like ``[V]`` or ``(RtB)``.
    Returns the most common tag found, or None.
    """
    import re
    tags = []
    for name in names:
        # Match [TAG] or (TAG) patterns
        m = re.findall(r'[\[\(]([A-Za-z0-9!@#$%^&*_+=|~]{1,15})[\]\)]', name)
        tags.extend(m)
    if not tags:
        return None
    from collections import Counter
    most_common = Counter(tags).most_common(1)[0]
    if most_common[1] >= 2:  # At least 2 riders with same tag
        return most_common[0]
    return None


if __name__ == '__main__':
    # Ensure directories exist
    Path('templates').mkdir(exist_ok=True)
    Path('static/css').mkdir(parents=True, exist_ok=True)
    Path('static/js').mkdir(parents=True, exist_ok=True)
    Path('race_data').mkdir(exist_ok=True)
    
    print("Starting Zwift Tools Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, threaded=True,
            use_reloader=True, reloader_type='stat',
            exclude_patterns=['*\\Lib\\*', '*/Lib/*', 'race_data/*'])
