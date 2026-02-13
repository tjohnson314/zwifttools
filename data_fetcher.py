"""
Data Fetcher
Fetches activity and telemetry data from the Zwift API.
"""

import requests
import pandas as pd

BASE_URL = "https://us-or-rly101.zwift.com/api"


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
    response = requests.get(url, headers=headers, timeout=30)
    
    if response.status_code != 200:
        return None, f"Error fetching activity: {response.status_code}"
    
    return response.json(), None


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
    
    file_url = fitness_data.get('smallDataUrl')
    if not file_url:
        return None, activity_data, "No telemetry file URL"
    
    file_id = file_url.split('/file/')[-1]
    
    # Fetch telemetry
    telem_url = f"{BASE_URL}/activities/{activity_id}/file/{file_id}"
    telem_response = requests.get(telem_url, headers=headers, timeout=30)
    
    if telem_response.status_code != 200:
        return None, activity_data, f"Telemetry error: {telem_response.status_code}"
    
    return telem_response.json(), activity_data, None


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
