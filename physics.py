"""
Bike comparison tool for Zwift race analysis.

Compares actual race data with hypothetical watts needed for a different bike setup.
Uses the physics model to answer: "What watts would I have needed with bike X to stay
in the exact same position?"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from bike_data import BikeSetup, get_bike_stats, get_bike_database
from utils import calculate_normalized_power


# Physics constants (from Zwift/Gribble model)
AIR_DENSITY = 1.225  # kg/m³ at sea level
GRAVITY = 9.8067  # m/s²
DRIVETRAIN_LOSS = 0.025  # 2.5% drivetrain loss


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
    if speed_mps > 0:
        power = (f_total * speed_mps) / (1 - DRIVETRAIN_LOSS)
    else:
        power = 0.0
    
    return max(0, power)  # Can't have negative power requirement


def compare_bike_setups(
    telemetry: pd.DataFrame,
    rider_weight_kg: float,
    actual_setup: BikeSetup,
    alternative_setup: BikeSetup,
    crr: float = 0.004,
    frontal_area: float = 0.36,
    alt_rider_weight_kg: float = None,
    alt_frontal_area: float = None
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
        crr: Rolling resistance coefficient (can vary by surface)
        frontal_area: Rider frontal area in m² (for actual bike)
        alt_rider_weight_kg: Rider weight for alternative (defaults to rider_weight_kg)
        alt_frontal_area: Frontal area for alternative (defaults to frontal_area)
        
    Returns:
        ComparisonResult with time series and summary stats
    """
    # Default alternative rider settings to actual if not specified
    if alt_rider_weight_kg is None:
        alt_rider_weight_kg = rider_weight_kg
    if alt_frontal_area is None:
        alt_frontal_area = frontal_area
    
    # Convert Cd to CdA by multiplying by frontal area
    actual_cda = actual_setup.cd * frontal_area
    alternative_cda = alternative_setup.cd * alt_frontal_area
    
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
            alternative_setup.weight_kg, alternative_cda, crr_values[i]
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
    else:
        # No recorded power - use physics model only (original behavior)
        actual_watts = solo_power_actual
        alternative_watts = solo_power_alternative
    
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
        total_actual_kj=total_actual_kj,
        total_alternative_kj=total_alternative_kj,
        avg_watts_difference=avg_watts_diff,
        actual_np=actual_np,
        alternative_np=alternative_np
    )


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
        print(f"  Cd: {tron_setup.cd:.4f}, Weight: {tron_setup.weight_kg:.3f} kg")
        print(f"\nCLIMB BIKE: {climb_setup}")
        print(f"  Cd: {climb_setup.cd:.4f}, Weight: {climb_setup.weight_kg:.3f} kg")
        print(f"\nDifference: Cd {(tron_setup.cd - climb_setup.cd)*1000:+.1f}mm², Weight {(tron_setup.weight_kg - climb_setup.weight_kg)*1000:+.0f}g")
        
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
