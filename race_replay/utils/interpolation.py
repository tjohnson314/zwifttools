"""Interpolation utilities for race telemetry data."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Gaps longer than this in the original telemetry are treated as "no data"
MAX_GAP_SEC = 10

# Minimum jump factor to consider a timestamp offset (relative to normal tick)
OFFSET_JUMP_FACTOR = 10


def fix_timestamp_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and correct timestamp offsets where the server clock jumped
    but the telemetry (distance, speed, power, HR) remained continuous.

    This happens when Zwift's server reassigns a world-time reference mid-ride.
    The actual ride was uninterrupted — only ``timeInSec`` is wrong.

    Returns a new DataFrame with corrected ``time_sec``.
    """
    times = df['time_sec'].values.copy()
    if len(times) < 3:
        return df

    diffs = np.diff(times)
    normal_tick = float(np.median(diffs))
    if normal_tick <= 0:
        return df

    threshold = normal_tick * OFFSET_JUMP_FACTOR  # e.g. 40s for 4-sec data
    jump_indices = np.where(diffs > threshold)[0]

    if len(jump_indices) == 0:
        return df

    cumulative_correction = 0.0
    corrected = times.copy().astype(float)

    for ji in jump_indices:
        actual_jump = diffs[ji]

        # Check telemetry continuity across the jump
        if not _is_continuous_across_jump(df, ji, normal_tick):
            continue  # genuine gap — don't correct

        excess = actual_jump - normal_tick
        cumulative_correction += excess

        # Shift everything after this index back
        corrected[ji + 1:] -= excess

    if cumulative_correction == 0:
        return df

    result = df.copy()
    result['time_sec'] = corrected
    return result


def _is_continuous_across_jump(df: pd.DataFrame, idx: int, normal_tick: float) -> bool:
    """Return True if the telemetry is continuous across a time jump at *idx*.

    Checks that the distance change across the gap is consistent with the
    speed at the boundary points (i.e. the rider never actually stopped).
    """
    if idx + 1 >= len(df):
        return False

    row_before = df.iloc[idx]
    row_after = df.iloc[idx + 1]

    # Distance-based check: expected distance from speed vs actual
    speed_before = row_before.get('speed_kmh', 0)  # km/h
    speed_after = row_after.get('speed_kmh', 0)
    avg_speed_ms = ((speed_before + speed_after) / 2) / 3.6  # m/s

    dist_before = row_before.get('distance_km', 0) * 1000  # m
    dist_after = row_after.get('distance_km', 0) * 1000
    actual_dist_delta = dist_after - dist_before

    expected_dist_delta = avg_speed_ms * normal_tick  # m

    if expected_dist_delta <= 0:
        return False

    # Allow up to 3x tolerance (distance can vary due to gradient changes)
    ratio = actual_dist_delta / expected_dist_delta
    if ratio < 0.2 or ratio > 3.0:
        return False

    # Speed continuity check: speeds should be similar
    if speed_before > 0:
        speed_ratio = speed_after / speed_before
        if speed_ratio < 0.5 or speed_ratio > 2.0:
            return False

    return True


def _build_gap_mask(original_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Return a boolean mask (True = inside a gap) for target_times.

    A gap is any interval between consecutive original_times that exceeds
    MAX_GAP_SEC.  Target times that fall strictly inside such an interval
    are marked True.
    """
    mask = np.zeros(len(target_times), dtype=bool)
    # Find consecutive gaps in the original data
    diffs = np.diff(original_times)
    gap_indices = np.where(diffs > MAX_GAP_SEC)[0]

    for gi in gap_indices:
        gap_start = original_times[gi]
        gap_end = original_times[gi + 1]
        # Mark target times that fall inside this gap (exclusive of endpoints)
        mask |= (target_times > gap_start) & (target_times < gap_end)

    return mask


def interpolate_telemetry(df: pd.DataFrame, target_times: np.ndarray) -> pd.DataFrame:
    """
    Interpolate rider telemetry to have data for each second.
    
    Args:
        df: DataFrame with columns: time_sec, distance_km, altitude_m, speed_kmh, 
            power_watts, hr_bpm, cadence_rpm, lat, lng
        target_times: Array of target timestamps (typically every second)
        
    Returns:
        DataFrame with interpolated values for each target time.
        Includes 'is_interpolated' boolean column.
    """
    original_times = set(df['time_sec'].values)
    
    result = pd.DataFrame({'time_sec': target_times})
    result['is_interpolated'] = ~result['time_sec'].isin(original_times)
    
    # Linear interpolation for: elevation, speed, hr, cadence, lat, lng
    linear_cols = ['altitude_m', 'speed_kmh', 'hr_bpm', 'cadence_rpm', 'lat', 'lng']
    for col in linear_cols:
        if col in df.columns:
            interp = interp1d(
                df['time_sec'].values,
                df[col].values,
                kind='linear',
                bounds_error=False,
                fill_value=(df[col].iloc[0], df[col].iloc[-1])
            )
            result[col] = interp(target_times)
        else:
            result[col] = 0.0
    
    # Step interpolation (fixed) for power
    if 'power_watts' in df.columns:
        power_interp = interp1d(
            df['time_sec'].values,
            df['power_watts'].values,
            kind='previous',  # Step function - hold previous value
            bounds_error=False,
            fill_value=(df['power_watts'].iloc[0], df['power_watts'].iloc[-1])
        )
        result['power_watts'] = power_interp(target_times)
    else:
        result['power_watts'] = 0.0
    
    # Quadratic distance interpolation based on speed
    result['distance_km'] = interpolate_distance_from_speed(
        df['time_sec'].values,
        df['distance_km'].values,
        df['speed_kmh'].values if 'speed_kmh' in df.columns else None,
        target_times
    )

    # Blank out values that fall inside large gaps in the original data
    gap_mask = _build_gap_mask(df['time_sec'].values, target_times)
    if gap_mask.any():
        blank_cols = ['distance_km', 'altitude_m', 'speed_kmh',
                      'power_watts', 'hr_bpm', 'cadence_rpm', 'lat', 'lng']
        for col in blank_cols:
            if col in result.columns:
                result.loc[gap_mask, col] = np.nan

    return result


def interpolate_distance_from_speed(
    original_times: np.ndarray,
    original_distances: np.ndarray,
    original_speeds: np.ndarray,
    target_times: np.ndarray
) -> np.ndarray:
    """
    Interpolate distance using speed to compute proper integral.
    
    If speed changes linearly between two points, distance is the integral
    (area under curve), giving quadratic interpolation.
    
    Args:
        original_times: Original timestamp array
        original_distances: Original distance array (km)
        original_speeds: Original speed array (km/h), or None
        target_times: Target timestamps to interpolate to
        
    Returns:
        Interpolated distance array (km)
    """
    if original_speeds is None:
        # Fall back to linear interpolation
        interp = interp1d(
            original_times, original_distances,
            kind='linear', bounds_error=False,
            fill_value=(original_distances[0], original_distances[-1])
        )
        return interp(target_times)
    
    result = np.zeros(len(target_times))
    
    for i, t in enumerate(target_times):
        # Find the interval containing t
        if t <= original_times[0]:
            result[i] = original_distances[0]
        elif t >= original_times[-1]:
            result[i] = original_distances[-1]
        else:
            # Find index j such that original_times[j] <= t < original_times[j+1]
            j = np.searchsorted(original_times, t, side='right') - 1
            j = max(0, min(j, len(original_times) - 2))
            
            t0, t1 = original_times[j], original_times[j + 1]
            d0, d1 = original_distances[j], original_distances[j + 1]
            v0, v1 = original_speeds[j], original_speeds[j + 1]
            
            # Convert speeds from km/h to km/s
            v0_kms = v0 / 3600.0
            v1_kms = v1 / 3600.0
            
            dt = t1 - t0
            if dt == 0:
                result[i] = d0
                continue
            
            # Linear speed: v(s) = v0 + (v1 - v0) * s / dt, where s = t - t0
            # Distance = integral of v(s) from 0 to s
            # = v0 * s + (v1 - v0) * s^2 / (2 * dt)
            s = t - t0
            delta_d = v0_kms * s + (v1_kms - v0_kms) * s * s / (2 * dt)
            result[i] = d0 + delta_d
    
    return result


def create_time_grid(min_time: float, max_time: float, step: float = 1.0) -> np.ndarray:
    """Create a regular time grid from min to max time."""
    return np.arange(int(min_time), int(max_time) + 1, step)
