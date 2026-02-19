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
import secrets
import requests
import traceback
import gzip
import json
import queue
import threading
from scipy.ndimage import uniform_filter1d

from bike_comparison.bike_data import get_bike_database, get_bike_stats
from bike_comparison.physics import compare_bike_setups
from shared.utils import calculate_normalized_power
from shared.data_fetcher import (
    fetch_rider_telemetry, convert_telemetry_to_dataframe,
    fetch_race_from_activity, get_activity_details,
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
            print(f"Token expired (expires_at={tokens.expires_at}, now={__import__('time').time()}), refreshing...")
            tokens = refresh_access_token(tokens.refresh_token)
            session['tokens'] = tokens.to_dict()
        
        return tokens
    except Exception as e:
        # Token refresh failed, user needs to re-login
        print(f"get_session_tokens failed: {e}")
        traceback.print_exc()
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
        traceback.print_exc()
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
        traceback.print_exc()
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
        traceback.print_exc()
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
                        print(f"Surface types: {non_default} points on non-tarmac surfaces")
            except Exception as e:
                print(f"Warning: Could not compute surface types: {e}")

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
        traceback.print_exc()
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
                        print(f"FIT surface types: {non_default} points on non-tarmac surfaces")
            except Exception as e:
                print(f"Warning: Could not compute surface types from FIT GPS: {e}")

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
        traceback.print_exc()
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
# Race Replay API Routes
# ---------------------------------------------------------------------------

# In-memory cache of loaded race data (keyed by race_id)
_race_data_cache = {}


@app.route('/api/race/fetch', methods=['POST'])
def api_race_fetch():
    """Fetch race data from Zwift API by activity URL or ID (non-streaming fallback)."""
    headers = get_headers()
    if not headers:
        return jsonify({'error': 'Not authenticated. Please log in first.'}), 401

    data = request.json
    activity_url_or_id = data.get('activity_id')
    if not activity_url_or_id:
        return jsonify({'error': 'activity_id required'}), 400

    force_refresh = data.get('force_refresh', False)

    try:
        race_data_dir = Path('race_data')
        race_data_dir.mkdir(exist_ok=True)

        def log_progress(current, total, name):
            print(f"  [{current}/{total}] Fetching {name}...")

        output_dir, message = fetch_race_from_activity(
            activity_url_or_id, headers,
            output_base_dir=str(race_data_dir),
            progress_callback=log_progress,
            force_refresh=force_refresh
        )

        if not output_dir:
            return jsonify({'error': message}), 400

        race_id = Path(output_dir).name
        return jsonify({'success': True, 'race_id': race_id, 'message': message})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/race/fetch_stream')
def api_race_fetch_stream():
    """SSE endpoint — streams progress while fetching race data."""
    headers = get_headers()
    if not headers:
        print(f"SSE fetch_stream: No auth headers. Session keys: {list(session.keys())}")
        def err_gen():
            yield f"data: {json.dumps({'error': 'Not authenticated. Please log in first.'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    activity_url_or_id = request.args.get('activity_id', '').strip()
    if not activity_url_or_id:
        def err_gen():
            yield f"data: {json.dumps({'error': 'activity_id required'})}\n\n"
        return Response(err_gen(), mimetype='text/event-stream')

    force_refresh = request.args.get('force_refresh', '').lower() in ('1', 'true')

    # Use a queue to bridge the progress_callback (in a thread) to the SSE generator
    q = queue.Queue()

    def run_fetch():
        try:
            race_data_dir = Path('race_data')
            race_data_dir.mkdir(exist_ok=True)

            def on_progress(current, total, name):
                print(f"  [{current}/{total}] Fetching {name}...")
                q.put({'progress': True, 'current': current, 'total': total, 'name': name})

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
                q.put({'success': True, 'race_id': race_id, 'message': message})
        except Exception as e:
            traceback.print_exc()
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


@app.route('/api/race/list')
def api_race_list():
    """List available race data directories (local + blob storage)."""
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
                        summary = pd.read_csv(summary_file)
                        info['rider_count'] = len(summary)
                        info['riders'] = summary[['rank', 'name']].to_dict('records') if 'name' in summary.columns else []
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

    return jsonify({'races': race_dirs})


@app.route('/api/race/load', methods=['POST'])
def api_race_load():
    """Load/clean a race data directory and return metadata."""
    data = request.json
    race_id = data.get('race_id')
    if not race_id:
        return jsonify({'error': 'race_id required'}), 400

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
                    'is_finisher': bool(r.is_finisher),
                    'finish_time_sec': float(r.finish_time_sec) if r.finish_time_sec is not None else None,
                }
                for r in race_data.riders
            ]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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

    # Build rider data — each rider's full time series
    riders = []
    def safe_list(series):
        """Convert pandas Series to list, replacing NaN with None for valid JSON."""
        return [None if pd.isna(v) else v for v in series]

    for r in race_data.riders:
        df = r.data.reset_index()  # time_sec is the index
        rider_json = {
            'rank': int(r.rank),
            'name': r.name,
            'team': r.team,
            'activity_id': str(r.activity_id),
            'player_id': int(r.player_id) if r.player_id else None,
            'weight_kg': float(r.weight_kg),
            'is_finisher': bool(r.is_finisher),
            'finish_time_sec': float(r.finish_time_sec) if r.finish_time_sec is not None else None,
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

    # Load race metadata (start time, event IDs)
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
            print(f"Could not load route latlng: {e}")

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
        traceback.print_exc()
        return jsonify({'streams': [], 'error': str(e)})


if __name__ == '__main__':
    # Ensure directories exist
    Path('templates').mkdir(exist_ok=True)
    Path('static/css').mkdir(parents=True, exist_ok=True)
    Path('static/js').mkdir(parents=True, exist_ok=True)
    Path('race_data').mkdir(exist_ok=True)
    
    print("Starting Zwift Tools Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, threaded=True)
