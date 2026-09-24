"""
Bike comparison tool for Zwift race analysis.

Compares actual race data with hypothetical watts needed for a different bike setup.
Uses the physics model to answer: "What watts would I have needed with bike X to stay
in the exact same position?"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from bike_comparison.bike_data import BikeSetup, get_bike_stats, get_bike_database, BASE_CDA
from shared.utils import calculate_normalized_power


# Physics constants (from Zwift source code)
AIR_DENSITY = 1.226  # kg/m³ at sea level, matching the recovered Zwift comparison
GRAVITY = 9.81  # m/s²
DRIVETRAIN_LOSS = 0.0  # No drivetrain loss


@dataclass
class ComparisonResult:
    """Results of bike comparison analysis."""
    actual_setup: BikeSetup
    alternative_setup: BikeSetup
    
    # Time series data
    time_sec: np.ndarray
    distance_km: np.ndarray
    speed_mps: np.ndarray
    gradient: np.ndarray
    
    actual_watts: np.ndarray
    alternative_watts: np.ndarray
    watts_difference: np.ndarray  # alternative - actual (positive = harder with alt bike)
    draft_watts: np.ndarray  # estimated draft savings (solo_power - recorded_power)
    
    # Summary stats
    total_actual_kj: float
    total_alternative_kj: float
    avg_watts_difference: float
    
    # Normalized Power
    actual_np: float = 0.0
    alternative_np: float = 0.0
    
    def summary(self) -> str:
        """Return a text summary of the comparison."""
        kj_diff = self.total_alternative_kj - self.total_actual_kj
        pct_diff = (kj_diff / self.total_actual_kj) * 100
        
        lines = [
            f"=== Bike Comparison ===",
            f"Actual:      {self.actual_setup}",
            f"Alternative: {self.alternative_setup}",
            f"",
            f"Total Work:",
            f"  Actual:      {self.total_actual_kj:.1f} kJ",
            f"  Alternative: {self.total_alternative_kj:.1f} kJ",
            f"  Difference:  {kj_diff:+.1f} kJ ({pct_diff:+.1f}%)",
            f"",
            f"Average Watts Difference: {self.avg_watts_difference:+.1f} W",
        ]
        return "\n".join(lines)


def calculate_power_for_speed(
    speed_mps: float,
    gradient: float,
    rider_weight_kg: float,
    bike_weight_kg: float,
    cda: float,
    crr: float = 0.004,
    air_density: float = AIR_DENSITY,
    wind_mps: float = 0.0
) -> float:
    """
    Calculate power required to maintain a given speed.
    
    Uses the standard cycling power equation:
    P = (F_gravity + F_rolling + F_aero) * v / (1 - drivetrain_loss)
    
    Args:
        speed_mps: Speed in m/s
        gradient: Road gradient as decimal (e.g., 0.05 for 5%)
        rider_weight_kg: Rider weight in kg
        bike_weight_kg: Bike weight in kg
        cda: Drag coefficient * frontal area (m²)
        crr: Rolling resistance coefficient
        air_density: Air density in kg/m³
        wind_mps: Headwind speed in m/s (positive = headwind)
        
    Returns:
        Power in watts
    """
    total_mass = rider_weight_kg + bike_weight_kg
    
    # Gravity force (positive uphill)
    f_gravity = total_mass * GRAVITY * gradient
    
    # Rolling resistance
    f_rolling = crr * total_mass * GRAVITY * np.cos(np.arctan(gradient))
    
    # Aerodynamic drag (relative to air)
    air_speed = speed_mps + wind_mps
    f_aero = 0.5 * air_density * cda * air_speed * abs(air_speed)
    
    # Total force
    f_total = f_gravity + f_rolling + f_aero
    
    # Power = Force * velocity, adjusted for drivetrain loss
    # NOTE: Can be negative on descents (gravity exceeds resistance).
    # Callers should clip to zero where non-negative power is required,
    # but the raw value is needed for correct draft estimation.
    if speed_mps > 0:
        power = (f_total * speed_mps) / (1 - DRIVETRAIN_LOSS)
    else:
        power = 0.0
    
    return power


def compare_bike_setups(
    telemetry: pd.DataFrame,
    rider_weight_kg: float,
    actual_setup: BikeSetup,
    alternative_setup: BikeSetup,
    crr: float = 0.004,
    frontal_area: float = 0.36,
    alt_rider_weight_kg: float = None,
    alt_frontal_area: float = None,
    alt_crr: float = None
) -> ComparisonResult:
    """
    Compare actual bike setup with an alternative.
    
    Calculates the hypothetical watts needed with the alternative bike
    to maintain the exact same speed/position at each moment.
    
    Args:
        telemetry: DataFrame with columns: time_sec, distance_km, speed_mps, gradient
        rider_weight_kg: Rider weight in kg (for actual bike)
        actual_setup: The bike actually used
        alternative_setup: The alternative bike to compare
        crr: Rolling resistance coefficient for actual bike (can vary by surface)
        frontal_area: Rider frontal area in m² (for actual bike)
        alt_rider_weight_kg: Rider weight for alternative (defaults to rider_weight_kg)
        alt_frontal_area: Frontal area for alternative (defaults to frontal_area)
        alt_crr: Rolling resistance for alternative bike (defaults to crr if None)
        
    Returns:
        ComparisonResult with time series and summary stats
    """
    # Default alternative rider settings to actual if not specified
    if alt_rider_weight_kg is None:
        alt_rider_weight_kg = rider_weight_kg
    if alt_frontal_area is None:
        alt_frontal_area = frontal_area
    
    # Absolute CdA = the rider's own (frontal-area-scaled) CdA plus the bike's
    # CdA bias, which Zwift applies as an additive delta (not scaled by rider
    # size).
    actual_cda = rider_cda_from_area(frontal_area) + actual_setup.cda_bias
    alternative_cda = rider_cda_from_area(alt_frontal_area) + alternative_setup.cda_bias
    
    # Extract required columns
    time_sec = telemetry['time_sec'].values if 'time_sec' in telemetry.columns else telemetry.index.values
    distance_km = telemetry['distance_km'].values
    
    # Speed in m/s
    if 'speed_mps' in telemetry.columns:
        speed_mps = telemetry['speed_mps'].values
    elif 'speed_kph' in telemetry.columns:
        speed_mps = telemetry['speed_kph'].values / 3.6
    else:
        raise ValueError("Telemetry must have speed_mps or speed_kph column")
    
    # Gradient
    if 'gradient' in telemetry.columns:
        gradient = telemetry['gradient'].values
    elif 'grade' in telemetry.columns:
        gradient = telemetry['grade'].values / 100.0  # Convert percent to decimal
    else:
        # Estimate from altitude changes
        if 'altitude_m' in telemetry.columns:
            alt = telemetry['altitude_m'].values
            dist_m = distance_km * 1000
            gradient = np.gradient(alt, dist_m)
            gradient = np.clip(gradient, -0.25, 0.25)  # Reasonable bounds
        else:
            gradient = np.zeros_like(speed_mps)
    
    # Get altitude for energy calculations
    if 'altitude_m' in telemetry.columns:
        altitude_m = telemetry['altitude_m'].values
    else:
        # Integrate gradient to get altitude changes
        dist_m = distance_km * 1000
        altitude_m = np.cumsum(gradient * np.gradient(dist_m))
    
    # Get actual recorded power (if available)
    recorded_power = None
    if 'power_watts' in telemetry.columns:
        recorded_power = telemetry['power_watts'].values
    elif 'power' in telemetry.columns:
        recorded_power = telemetry['power'].values
    
    # Handle CRR - can be array (varying by surface) or scalar
    if isinstance(crr, np.ndarray):
        crr_values = crr
    else:
        crr_values = np.full_like(speed_mps, crr)
    
    # Handle alternative CRR (different bike types have different CRR on same surface)
    if alt_crr is None:
        alt_crr_values = crr_values
    elif isinstance(alt_crr, np.ndarray):
        alt_crr_values = alt_crr
    else:
        alt_crr_values = np.full_like(speed_mps, alt_crr)
    
    # Calculate solo power required with actual bike (no drafting)
    solo_power_actual = np.array([
        calculate_power_for_speed(
            speed_mps[i], gradient[i], rider_weight_kg,
            actual_setup.weight_kg, actual_cda, crr_values[i]
        )
        for i in range(len(speed_mps))
    ])
    
    # Calculate solo power required with alternative bike (no drafting)
    solo_power_alternative = np.array([
        calculate_power_for_speed(
            speed_mps[i], gradient[i], alt_rider_weight_kg,
            alternative_setup.weight_kg, alternative_cda, alt_crr_values[i]
        )
        for i in range(len(speed_mps))
    ])
    
    if recorded_power is not None:
        # Use actual recorded power (includes drafting benefit)
        actual_watts = recorded_power.copy()
        
        # Total masses
        total_mass = rider_weight_kg + actual_setup.weight_kg
        alt_total_mass = alt_rider_weight_kg + alternative_setup.weight_kg
        mass_diff = alt_total_mass - total_mass
        
        # Calculate energy change rate (power going into KE + PE changes)
        # KE = 0.5 * m * v^2, PE = m * g * h
        # dE/dt = power to change mechanical energy state
        dt = np.diff(time_sec, prepend=time_sec[0])
        dt[0] = dt[1] if len(dt) > 1 else 1.0  # Handle first point
        dt = np.maximum(dt, 0.01)  # Avoid division by zero
        
        # Kinetic energy at each point
        ke = 0.5 * total_mass * speed_mps**2
        
        # Potential energy at each point (relative to start)
        pe = total_mass * 9.8067 * altitude_m
        
        # Total mechanical energy
        total_energy = ke + pe
        
        # Rate of energy change (power going to/from energy storage)
        # Use forward difference for causality
        energy_change_rate = np.zeros_like(total_energy)
        energy_change_rate[:-1] = np.diff(total_energy) / dt[1:]
        energy_change_rate[-1] = energy_change_rate[-2] if len(energy_change_rate) > 1 else 0
        
        # Power going to energy change is proportional to mass
        # For alternative bike: energy_power_alt = energy_power * (alt_mass / actual_mass)
        # Energy power difference = energy_power * mass_diff / total_mass
        energy_power_diff = energy_change_rate * mass_diff / total_mass
        
        # Calculate resistance forces (aero + rolling)
        safe_speed = np.maximum(speed_mps, 0.5)
        
        f_rolling_actual = crr_values * total_mass * 9.8067
        f_aero_actual = 0.5 * 1.225 * actual_cda * speed_mps**2
        
        f_rolling_alt = crr_values * alt_total_mass * 9.8067
        f_aero_alt = 0.5 * 1.225 * alternative_cda * speed_mps**2
        
        # Rolling resistance difference (always applies fully - it's ground friction)
        f_rolling_diff = f_rolling_alt - f_rolling_actual
        
        # Aero difference - only applies based on how much aero drag we're actually experiencing
        # Calculate resistance power (what we'd need at steady state, no energy change)
        resistance_power_solo = (f_rolling_actual + f_aero_actual) * safe_speed / (1 - 0.025)
        
        # Estimate what fraction of normal aero drag we're experiencing
        # Actual power minus energy change power = power going to resistance
        resistance_power_actual = actual_watts - energy_change_rate
        resistance_power_actual = np.maximum(resistance_power_actual, 0)
        
        # power_ratio = how much of solo resistance we're actually fighting
        # < 1 means drafting, = 1 means solo, capped at 1 (can't exceed full exposure)
        power_ratio = np.minimum(resistance_power_actual / np.maximum(resistance_power_solo, 1), 1.0)
        
        # Aero difference (only effective portion based on drafting)
        f_aero_diff = (f_aero_alt - f_aero_actual) * power_ratio
        
        # Total resistance force difference
        f_resistance_diff = f_rolling_diff + f_aero_diff
        
        # Power difference from resistance
        resistance_power_diff = f_resistance_diff * safe_speed / (1 - 0.025)
        
        # Total power difference = resistance diff + energy change diff
        power_diff = resistance_power_diff + energy_power_diff
        
        # Alternative power = actual power + difference
        alternative_watts = actual_watts + power_diff
        alternative_watts = np.maximum(alternative_watts, 0)
        
        # Draft savings estimation — gradient-free formulation
        #
        # From energy conservation the drafted rider satisfies:
        #   P_actual*(1-η) = F_roll*v + F_aero_draft*v + d(KE+PE)/dt
        #
        # Draft savings = (F_aero_solo - F_aero_draft)*v / (1-η), which gives:
        #   draft = (F_aero_solo + F_roll)*v/(1-η) - P_actual + d(KE+PE)/dt/(1-η)
        #
        # This avoids estimating gradient entirely.  Gravity enters only
        # through PE = m·g·h (raw altitude, no differentiation for slope).
        #
        # resistance_power_solo and energy_change_rate are already computed
        # above for the bike-comparison logic, so we reuse them directly.
        #
        # Negative values mean unmodeled forces are decelerating the rider
        # beyond what aero+rolling+gravity predict (e.g. the rider applying
        # in-game brakes on a descent).  We leave them visible rather than
        # clamping to zero so the user can see the raw estimate.
        draft_watts = resistance_power_solo - actual_watts + energy_change_rate / (1 - DRIVETRAIN_LOSS)
        
        # Smooth to reduce discrete derivative spikiness.
        draft_window = min(5, len(draft_watts))
        if draft_window > 1:
            dk = np.ones(draft_window) / draft_window
            draft_watts = np.convolve(draft_watts, dk, mode='same')
    else:
        # No recorded power - use physics model only (original behavior)
        # Clip to zero: can't pedal negative watts (freewheeling on descents)
        actual_watts = np.maximum(solo_power_actual, 0)
        alternative_watts = np.maximum(solo_power_alternative, 0)
        draft_watts = np.zeros_like(actual_watts)
    
    # Calculate differences
    watts_difference = alternative_watts - actual_watts
    
    # Calculate total work (kJ)
    dt = np.diff(time_sec, prepend=time_sec[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0  # Handle first point
    
    total_actual_kj = np.sum(actual_watts * dt) / 1000
    total_alternative_kj = np.sum(alternative_watts * dt) / 1000
    
    avg_watts_diff = np.mean(watts_difference)
    
    # Calculate Normalized Power for both
    actual_np = calculate_normalized_power(actual_watts, time_sec)
    alternative_np = calculate_normalized_power(alternative_watts, time_sec)
    
    return ComparisonResult(
        actual_setup=actual_setup,
        alternative_setup=alternative_setup,
        time_sec=time_sec,
        distance_km=distance_km,
        speed_mps=speed_mps,
        gradient=gradient,
        actual_watts=actual_watts,
        alternative_watts=alternative_watts,
        watts_difference=watts_difference,
        draft_watts=draft_watts,
        total_actual_kj=total_actual_kj,
        total_alternative_kj=total_alternative_kj,
        avg_watts_difference=avg_watts_diff,
        actual_np=actual_np,
        alternative_np=alternative_np
    )


def estimate_draft_efficiency(
    telemetry: pd.DataFrame,
    rider_weight_kg: float,
    setup: BikeSetup,
    frontal_area: float,
    crr: float = 0.004,
) -> Optional[dict]:
    """Estimate a rider's drafting efficiency over a ride.

    Uses the same recorded-power draft model as :func:`compare_bike_setups`:
    the effective draft is the aero power saved by sitting in the draft, derived
    from energy conservation (gradient-free; gravity enters only through the
    potential energy of the raw altitude trace).

    Args:
        telemetry: DataFrame with ``time_sec``, a speed column (``speed_mps`` or
            ``speed_kmh``), optional ``altitude_m`` and ``power_watts``.
        rider_weight_kg: Rider mass (kg).
        setup: The bike setup every rider is assumed to ride.
        frontal_area: Rider frontal area (m²), from height and weight.
        crr: Rolling resistance coefficient.

    Returns:
        dict with ``avg_draft_watts`` (time-weighted mean effective draft, W)
        and ``aero_reduction_pct`` (total effective-draft energy divided by
        total solo aero-drag energy, as a percentage), or ``None`` when speed or
        power data is unavailable.
    """
    if 'time_sec' in telemetry.columns:
        time_sec = telemetry['time_sec'].to_numpy(dtype=float)
    else:
        time_sec = np.asarray(telemetry.index, dtype=float)

    if 'speed_mps' in telemetry.columns:
        speed_mps = telemetry['speed_mps'].to_numpy(dtype=float)
    elif 'speed_kmh' in telemetry.columns:
        speed_mps = telemetry['speed_kmh'].to_numpy(dtype=float) / 3.6
    elif 'speed_kph' in telemetry.columns:
        speed_mps = telemetry['speed_kph'].to_numpy(dtype=float) / 3.6
    else:
        return None

    if 'power_watts' in telemetry.columns:
        power = telemetry['power_watts'].to_numpy(dtype=float)
    elif 'power' in telemetry.columns:
        power = telemetry['power'].to_numpy(dtype=float)
    else:
        return None

    if 'altitude_m' in telemetry.columns:
        altitude_m = telemetry['altitude_m'].to_numpy(dtype=float)
    else:
        altitude_m = np.zeros_like(speed_mps)

    # Per-surface rolling resistance when the cleaner provides it, else scalar.
    if 'crr' in telemetry.columns:
        crr_arr = telemetry['crr'].to_numpy(dtype=float)
        crr = np.where(np.isfinite(crr_arr), crr_arr, crr)

    # Drop samples where speed or power is missing so NaNs don't poison the
    # energy-weighted averages.
    valid = np.isfinite(speed_mps) & np.isfinite(power) & np.isfinite(time_sec)
    if valid.sum() < 2:
        return None
    time_sec = time_sec[valid]
    speed_mps = speed_mps[valid]
    power = power[valid]
    altitude_m = np.nan_to_num(altitude_m[valid], nan=0.0)
    if isinstance(crr, np.ndarray):
        crr = crr[valid]

    # Absolute CdA: rider's own (frontal-area-scaled) CdA plus the bike's bias.
    cda = rider_cda_from_area(frontal_area) + setup.cda_bias
    total_mass = rider_weight_kg + setup.weight_kg

    safe_speed = np.maximum(speed_mps, 0.5)

    f_rolling = crr * total_mass * GRAVITY
    f_aero_solo = 0.5 * AIR_DENSITY * cda * speed_mps ** 2

    # Solo power to overcome all resistance, and the aero-only portion of it.
    resistance_power_solo = (f_rolling + f_aero_solo) * safe_speed / (1 - DRIVETRAIN_LOSS)
    solo_aero_power = f_aero_solo * safe_speed / (1 - DRIVETRAIN_LOSS)

    # Rate of mechanical-energy change (KE + PE) from the recorded trace.
    dt = np.diff(time_sec, prepend=time_sec[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.maximum(dt, 0.01)
    total_energy = 0.5 * total_mass * speed_mps ** 2 + total_mass * GRAVITY * altitude_m
    energy_change_rate = np.zeros_like(total_energy)
    energy_change_rate[:-1] = np.diff(total_energy) / dt[1:]
    energy_change_rate[-1] = energy_change_rate[-2] if len(energy_change_rate) > 1 else 0.0

    # Effective draft = aero power saved vs. riding solo (same formula as
    # compare_bike_setups). Left unclamped so real deceleration (braking) shows.
    draft_watts = resistance_power_solo - power + energy_change_rate / (1 - DRIVETRAIN_LOSS)
    draft_window = min(5, len(draft_watts))
    if draft_window > 1:
        dk = np.ones(draft_window) / draft_window
        draft_watts = np.convolve(draft_watts, dk, mode='same')

    total_dt = float(np.sum(dt))
    if total_dt <= 0:
        return None
    avg_draft_watts = float(np.sum(draft_watts * dt) / total_dt)

    total_aero_energy = float(np.sum(solo_aero_power * dt))
    total_draft_energy = float(np.sum(draft_watts * dt))
    aero_reduction_pct = (100.0 * total_draft_energy / total_aero_energy
                          if total_aero_energy > 0 else None)

    return {
        'avg_draft_watts': avg_draft_watts,
        'aero_reduction_pct': aero_reduction_pct,
    }


def speed_from_power(
    power_w: float,
    gradient: float,
    rider_weight_kg: float,
    bike_weight_kg: float,
    cda: float,
    crr: float = 0.004,
    air_density: float = AIR_DENSITY,
) -> float:
    """
    Compute equilibrium speed (m/s) for a rider producing constant power.

    Inverts the power equation P = (F_grav + F_roll + F_aero) * v / (1 - η)
    via binary search on [0, 30] m/s.

    On a steep enough descent the rider freewheels; in that case the returned
    speed is the terminal (freewheel) speed, i.e., the speed where net force
    is zero even with P = 0.  If the caller passes power_w > 0 the rider is
    assumed to still pedal, so the actual speed is the higher-v root where
    P(v) = power_w.

    Args:
        power_w: Rider power output in watts (≥ 0).
        gradient: Road gradient as a decimal (0.05 = 5% uphill, -0.04 = 4% downhill).
        rider_weight_kg: Rider mass in kg.
        bike_weight_kg: Bike mass in kg.
        cda: CdA (drag coefficient × frontal area) in m².
        crr: Rolling resistance coefficient.
        air_density: Air density in kg/m³.

    Returns:
        Speed in m/s (clamped to [0, 30]).
    """
    total_mass = rider_weight_kg + bike_weight_kg

    def p_at_v(v: float) -> float:
        """Power required to ride at speed v (matches calculate_power_for_speed)."""
        f_gravity = total_mass * GRAVITY * gradient
        f_rolling = crr * total_mass * GRAVITY * np.cos(np.arctan(gradient))
        f_aero = 0.5 * air_density * cda * v * v
        return (f_gravity + f_rolling + f_aero) * v / (1.0 - DRIVETRAIN_LOSS)

    # On a descent the net-force function p_at_v(v) has a local minimum
    # at v_min > 0 where dP/dv = 0.  Below that minimum P(v) < 0 (gravity
    # does more work than drag absorbs).  We want the physical root above
    # v_min where P(v) = power_w.
    #
    # Strategy:
    #   1. Find v_term: the positive speed where p_at_v(v) = 0 (terminal on
    #      a freewheel).  This exists only when A = m*g*(grad + crr*cos) < 0.
    #   2. Binary-search in [max(v_term, 0), V_MAX] for p_at_v(v) = power_w.

    V_MAX = 30.0  # m/s ≈ 108 km/h — physically unreachable in Zwift

    # Coefficient of v in the cubic (after factoring out v):
    # p_at_v(v) = [A + B*v²] * v / (1-η)  where A = F_grav + F_roll, B = 0.5*ρ*CdA
    A_coeff = (total_mass * GRAVITY * gradient
               + crr * total_mass * GRAVITY * np.cos(np.arctan(gradient)))
    B_coeff = 0.5 * air_density * cda

    # Terminal speed (freewheel): A + B*v² = 0  →  v = sqrt(-A/B)
    v_lower = 0.0
    if A_coeff < 0.0:
        v_lower = np.sqrt(-A_coeff / B_coeff)

    if v_lower >= V_MAX:
        return V_MAX

    # Verify: at V_MAX, power needed should exceed power_w for any realistic case.
    # If not, cap at V_MAX.
    if p_at_v(V_MAX) < power_w:
        return V_MAX

    # Binary search for p_at_v(v) = power_w in [v_lower, V_MAX]
    lo, hi = v_lower, V_MAX
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if p_at_v(mid) < power_w:
            lo = mid
        else:
            hi = mid

    return (lo + hi) * 0.5


# Recovered from the Zwift game source. H is centimetres, M is kilograms.
_AREA_COEFFICIENT = 0.003014024
_HEIGHT_EXPONENT = 0.655
_WEIGHT_EXPONENT = 0.44
_AREA_OFFSET = 0.1159


def frontal_area_from_rider(height_m: float, weight_kg: float) -> float:
    """
    Estimate cyclist frontal area (m²) from height and weight, matched to Zwift.

        The Zwift source computes rider area as:

                A = 0.003014024 · H^0.655 · M^0.44 - 0.1159

        where H is height in centimetres and M is mass in kilograms. The game then
        applies 1/2·rho when converting this area into aerodynamic drag.

    Args:
        height_m: Rider height in metres.
        weight_kg: Rider mass in kg.

    Returns:
        Estimated frontal area in m².
    """
    height_cm = height_m * 100.0
    return (
        _AREA_COEFFICIENT
        * height_cm ** _HEIGHT_EXPONENT
        * weight_kg ** _WEIGHT_EXPONENT
        - _AREA_OFFSET
    )


# Rider drag coefficient for a zero-bias bike. A bike's CdA bias is an absolute
# delta added on top, matching how Zwift stores per-frame/per-wheel offsets.
def rider_cda_from_area(frontal_area: float, air_density: float = AIR_DENSITY) -> float:
    """Convert rider frontal area to the effective coefficient used by Zwift."""
    return 0.5 * air_density * frontal_area


def rider_cda(height_m: float, weight_kg: float) -> float:
    """Rider-only CdA (m²) on a zero-bias bike, from height and weight.

    Add the bike's ``cda_bias`` to this to get the absolute CdA for a setup.
    """
    return rider_cda_from_area(frontal_area_from_rider(height_m, weight_kg))


if __name__ == "__main__":
    # Demo with synthetic data
    print("=== Bike Comparison Demo ===\n")
    
    # Create sample telemetry (flat then climb then descent)
    n_points = 600  # 10 minutes at 1 second intervals
    time_sec = np.arange(n_points)
    
    # Simulate a route: flat -> climb -> descent
    gradient = np.zeros(n_points)
    gradient[120:300] = 0.06  # 6% climb for 3 minutes
    gradient[300:360] = -0.04  # 4% descent for 1 minute
    
    # Simulate speed (slower on climb, faster on descent)
    base_speed = 10.0  # m/s (36 km/h)
    speed_mps = base_speed - gradient * 50  # Rough approximation
    speed_mps = np.clip(speed_mps, 3, 18)
    
    # Calculate distance
    distance_km = np.cumsum(speed_mps) / 1000
    
    telemetry = pd.DataFrame({
        'time_sec': time_sec,
        'distance_km': distance_km,
        'speed_mps': speed_mps,
        'gradient': gradient
    })
    
    # Compare Tron bike (best aero) vs Specialized Tarmac SL8 (best climbing)
    # F089 = Zwift Concept Z1 (TRON) - has its own wheels built-in
    # F122 = Specialized S-Works Tarmac SL8 - best climbing frame
    # W028 = DT Swiss ARC 62 - good wheels
    
    db = get_bike_database()
    
    # Tron bike (no separate wheels needed - uses its own)
    tron_setup = get_bike_stats('F089', None, upgrade_level=0)
    
    # Tarmac SL8 with DT Swiss wheels at max upgrade
    climb_setup = get_bike_stats('F122', 'W028', upgrade_level=5)
    
    if tron_setup and climb_setup:
        print(f"TRON BIKE: {tron_setup}")
        print(f"  CdA: {tron_setup.cda:.4f} m², Weight: {tron_setup.weight_kg:.3f} kg")
        print(f"\nCLIMB BIKE: {climb_setup}")
        print(f"  CdA: {climb_setup.cda:.4f} m², Weight: {climb_setup.weight_kg:.3f} kg")
        print(f"\nDifference: CdA {(tron_setup.cda - climb_setup.cda):+.4f} m², Weight {(tron_setup.weight_kg - climb_setup.weight_kg)*1000:+.0f}g")
        
        # Compare: What if you used climb bike instead of Tron?
        result = compare_bike_setups(
            telemetry=telemetry,
            rider_weight_kg=75.0,
            actual_setup=tron_setup,
            alternative_setup=climb_setup
        )
        
        print(f"\n{result.summary()}")
    else:
        print("Could not find bike setups")
        if not tron_setup:
            print("  - Tron setup not found")
        if not climb_setup:
            print("  - Climb setup not found")
