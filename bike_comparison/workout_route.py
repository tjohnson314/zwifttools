"""Time-based workout progress over real route geometry, without drafting."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from bike_comparison.pacing_planner import RouteProfile
from bike_comparison.physics import AIR_DENSITY, DRIVETRAIN_LOSS, GRAVITY
from shared.surface_lookup import surface_types_to_crr
from shared.workouts import MAX_WORKOUT_SEGMENTS

MAX_DURATION_SEC = 24 * 3600


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}.")
    return number


def validate_workout(workout: dict) -> None:
    """Validate uploaded numeric plans again at the simulation trust boundary."""
    if not isinstance(workout, dict):
        raise ValueError("Choose a built-in workout or upload a custom .zwo file.")
    duration = _finite(workout.get("duration_sec"), "Workout duration", positive=True)
    if duration > MAX_DURATION_SEC:
        raise ValueError("Workout route estimates support workouts up to 24 hours.")
    segments = workout.get("segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_WORKOUT_SEGMENTS:
        raise ValueError("Workout must contain 1 to 10,000 timed blocks.")
    if workout.get("ftp_override") is not None:
        _finite(workout["ftp_override"], "Fixed FTP reference", positive=True)
    if not isinstance(workout.get("warnings", []), list) or not all(
        isinstance(warning, str) for warning in workout.get("warnings", [])
    ):
        raise ValueError("Workout warnings must be a list of strings.")
    end = 0.0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Invalid workout block.")
        start = _finite(segment.get("start_sec"), "Block start")
        next_end = _finite(segment.get("end_sec"), "Block end")
        if start != end or next_end <= start:
            raise ValueError("Workout blocks must be contiguous and have positive durations.")
        if not isinstance(segment.get("ramp"), bool):
            raise ValueError("Workout ramp flag must be a boolean.")
        low, high = segment.get("power_low"), segment.get("power_high")
        if (low is None) != (high is None):
            raise ValueError("Workout blocks need both power targets or neither.")
        if low is not None:
            for value in (low, high):
                if _finite(value, "Power target") < 0:
                    raise ValueError("Negative power targets cannot be simulated; they are not clipped to zero.")
        end = next_end
    if end != duration:
        raise ValueError("Workout duration must match its blocks.")


def _scaled_target_power(segment: dict, time_sec: float, reference_ftp: float,
                         rider_ftp: float, untargeted_fraction: float) -> float:
    low, high = segment["power_low"], segment["power_high"]
    if low is None:
        return rider_ftp * untargeted_fraction
    if segment["ramp"]:
        fraction = (time_sec - segment["start_sec"]) / (segment["end_sec"] - segment["start_sec"])
        return (low + (high - low) * fraction) * reference_ftp
    return (low + high) / 2 * reference_ftp


def target_power(segment: dict, time_sec: float, reference_ftp: float,
                 rider_ftp: float, untargeted_fraction: float) -> float:
    watts = _finite(_scaled_target_power(segment, time_sec, reference_ftp,
                                        rider_ftp, untargeted_fraction), "Scaled power target")
    # Match Math.round in Workout Analysis, including halfway values rounding up.
    return math.floor(watts / 5 + 0.5) * 5.0


def _power_steps(workout: dict, reference: float, ftp_w: float,
                 untargeted_fraction: float) -> list[tuple[float, float, float]]:
    steps = []
    for segment in workout["segments"]:
        start, end = segment["start_sec"], segment["end_sec"]
        first = _finite(_scaled_target_power(segment, start, reference, ftp_w, untargeted_fraction),
                        "Scaled power target")
        last = _finite(_scaled_target_power(segment, end, reference, ftp_w, untargeted_fraction),
                       "Scaled power target")
        times = [start, end]
        if first != last:
            first_threshold = math.floor((min(first, last) - 2.5) / 5) + 1
            stop_threshold = math.ceil((max(first, last) - 2.5) / 5)
            if len(steps) + stop_threshold - first_threshold + 1 > MAX_WORKOUT_SEGMENTS:
                raise ValueError("Workout exceeds the 10,000-step simulation limit after rounding ramps.")
            slope = (last - first) / (end - start)
            times.extend(start + (threshold * 5 + 2.5 - first) / slope
                         for threshold in range(first_threshold, stop_threshold))
            times.sort()
        if len(steps) + len(times) - 1 > MAX_WORKOUT_SEGMENTS:
            raise ValueError("Workout exceeds the 10,000-step simulation limit after rounding ramps.")
        for left, right in zip(times, times[1:]):
            steps.append((left, right, target_power(segment, (left + right) / 2,
                                                   reference, ftp_w, untargeted_fraction)))
    return steps


def simulate_workout_route(
    route: RouteProfile, workout: dict, *, rider_weight_kg: float,
    bike_weight_kg: float, cda: float, ftp_w: float,
    untargeted_percent: float = 75.0, bike_type: str = "road_bike",
) -> dict:
    validate_workout(workout)
    for value, label in ((rider_weight_kg, "Rider weight"), (bike_weight_kg, "Bike weight"),
                         (cda, "CdA"), (ftp_w, "FTP")):
        _finite(value, label, positive=True)
    if _finite(untargeted_percent, "Untargeted block percentage") < 0:
        raise ValueError("Untargeted block percentage must be non-negative.")
    distance = np.asarray(route.distance_m, dtype=float)
    altitude = np.asarray(route.altitude_m, dtype=float)
    if (len(distance) < 2 or distance.shape != altitude.shape
            or not np.all(np.isfinite(distance)) or not np.all(np.isfinite(altitude))
            or distance[0] != 0 or np.any(np.diff(distance) < 0) or distance[-1] <= 0):
        raise ValueError("Route needs finite, non-decreasing geometry starting at zero.")
    lengths = np.diff(distance)
    moving = lengths > 0
    segment_starts = distance[:-1][moving]
    grades = np.diff(altitude)[moving] / lengths[moving]
    mass = rider_weight_kg + bike_weight_kg
    if route.surfaces is not None:
        if len(route.surfaces) != len(distance):
            raise ValueError("Route surfaces do not match its geometry.")
        crr = surface_types_to_crr(np.asarray(route.surfaces)[:-1][moving], bike_type)
    else:
        crr = np.full(len(grades), 0.004)
    resistance = mass * GRAVITY * (grades + crr * np.cos(np.arctan(grades)))
    aero = 0.5 * AIR_DENSITY * cda
    reference = workout.get("ftp_override") or ftp_w
    duration = workout["duration_sec"]
    total_distance = float(distance[-1])
    profile = {key: [] for key in ("time_sec", "distance_km", "altitude_m", "power_w", "speed_kph")}
    state = np.array([0.0, 0.0])
    finish_time = None

    def finish(_time, values):
        return values[0] - total_distance

    finish.terminal = True
    finish.direction = 1

    for start, stop, watts in _power_steps(workout, reference, ftp_w, untargeted_percent / 100):
        coasting = watts == 0

        def derivative(time, values):
            position, motion = values
            # Kinetic energy avoids P/v's singularity at rest. Numerical roundoff
            # below zero represents a stop, never an artificial minimum speed.
            speed = max(0.0, motion) if coasting else math.sqrt(2 * max(0.0, motion) / mass)
            index = min(int(np.searchsorted(segment_starts, position, side="right")) - 1, len(grades) - 1)
            force = resistance[index] + aero * speed * speed
            if coasting:
                # With no pedal power, velocity also handles gravity accelerating
                # a stationary rider downhill (energy's zero solution is ambiguous).
                acceleration = -force / mass
                return speed, max(0.0, acceleration) if motion <= 0 else acceleration
            net_power = watts * (1 - DRIVETRAIN_LOSS) - force * speed
            return speed, net_power

        initial = state.copy()
        if coasting:
            initial[1] = math.sqrt(2 * max(0.0, initial[1]) / mass)
        solution = solve_ivp(
            derivative, (start, stop), initial,
            events=finish, dense_output=True, max_step=1.0,
            rtol=1e-6, atol=[1e-5, 1e-4],
        )
        if not solution.success:
            raise ValueError(f"Workout simulation failed: {solution.message}")
        end = float(solution.t[-1])
        samples = np.append(np.arange(start, end, 5.0), end)
        values = solution.sol(samples)
        for time, position, motion in zip(samples, values[0], values[1]):
            profile["time_sec"].append(float(time))
            profile["distance_km"].append(float(position) / 1000)
            profile["altitude_m"].append(float(np.interp(position, distance, altitude)))
            profile["power_w"].append(watts)
            speed = max(0.0, motion) if coasting else math.sqrt(2 * max(0.0, motion) / mass)
            profile["speed_kph"].append(speed * 3.6)
        state = solution.y[:, -1]
        if coasting:
            state[1] = 0.5 * mass * max(0.0, state[1]) ** 2
        if len(solution.t_events[0]):
            finish_time = float(solution.t_events[0][0])
            break
    covered = total_distance if finish_time is not None else float(state[0])
    warnings = list(workout.get("warnings", []))
    if route.surfaces is None:
        warnings.append("Surface geometry unavailable; using road rolling resistance (CRR 0.004).")
    if any(segment["power_low"] is None for segment in workout["segments"]):
        warnings.append(f"Blocks without targets use {untargeted_percent:g}% of rider FTP, including free ride and max effort.")
    return {
        "completed": finish_time is not None,
        "finish_time_sec": finish_time,
        "workout_duration_sec": duration,
        "workout_time_remaining_sec": duration - finish_time if finish_time is not None else 0.0,
        "route_distance_km": total_distance / 1000,
        "distance_covered_km": covered / 1000,
        "distance_remaining_km": (total_distance - covered) / 1000,
        "reference_ftp_w": reference,
        "rider_ftp_w": ftp_w,
        "warnings": warnings,
        "profile": profile,
    }
