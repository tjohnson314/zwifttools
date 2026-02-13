"""
Shared utility functions for Zwift race analysis tools.
"""

import numpy as np
import pandas as pd


def calculate_normalized_power(power_watts, time_sec=None, window_seconds=30):
    """
    Calculate Normalized Power (NP) using the standard algorithm.
    
    NP = 4th root of (average of (30-second rolling average power)^4)
    
    This gives higher weight to power variability, making it a better
    measure of physiological cost than average power.
    
    Algorithm (developed by Dr. Andrew Coggan):
    1. Calculate a rolling 30-second average of power
    2. Raise each averaged value to the 4th power
    3. Take the mean of all the 4th-power values
    4. Take the 4th root of that mean
    
    Args:
        power_watts: Array or Series of power values in watts
        time_sec: Optional array of time values in seconds. If provided,
                  used to calculate the correct window size for variable
                  sample rates. If None, assumes 1-second sampling.
        window_seconds: Rolling average window in seconds (default 30)
        
    Returns:
        Normalized Power in watts as float, or None if insufficient data
    """
    # Convert to numpy array and handle NaN values
    power = np.array(power_watts, dtype=float)
    power = np.nan_to_num(power, nan=0.0)

    if len(power) < window_seconds:
        return float(np.mean(power)) if len(power) > 0 else None

    # Calculate window size based on actual sample rate
    if time_sec is not None:
        dt = np.diff(time_sec)
        avg_sample_rate = np.mean(dt) if len(dt) > 0 else 1.0
        window_size = max(1, int(window_seconds / avg_sample_rate))
    else:
        window_size = window_seconds

    # Apply rolling average
    if window_size >= len(power):
        rolling_avg = np.full_like(power, np.mean(power))
    else:
        rolling_avg = pd.Series(power).rolling(
            window=window_size, min_periods=window_size
        ).mean().dropna().values

    if len(rolling_avg) == 0:
        return None

    # Raise to 4th power, average, then take 4th root
    np_value = (np.mean(rolling_avg ** 4)) ** 0.25

    return float(np_value)
