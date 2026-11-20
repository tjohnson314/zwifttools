from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bike_comparison.bike_data import BikeSetup
from bike_comparison.physics import DRIVETRAIN_LOSS, GRAVITY, compare_bike_setups


@pytest.mark.parametrize("recorded_power", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_comparison_applies_each_bikes_resistance_reduction_once(recorded_power, reverse):
    stock = BikeSetup("bike", "Bike", "", "", 2, .3, 8., "Standard")
    upgraded = replace(stock, upgrade_level=3, crr_reduction=.0003)
    actual, alternative = (upgraded, stock) if reverse else (stock, upgraded)
    telemetry = pd.DataFrame({
        "time_sec": np.arange(4, dtype=float), "distance_km": np.arange(4) * .01,
        "speed_mps": np.full(4, 10.), "gradient": np.zeros(4),
    })
    if recorded_power:
        telemetry["power_watts"] = 250.
    surfaces = np.array([.004, .016, .016, .004])
    result = compare_bike_setups(telemetry, 75., actual, alternative, crr=surfaces)

    gravity = 9.8067 if recorded_power else GRAVITY
    efficiency = .975 if recorded_power else 1 - DRIVETRAIN_LOSS
    expected = .0003 * 83 * gravity * 10 / efficiency
    np.testing.assert_allclose(result.watts_difference, expected if reverse else -expected)
    np.testing.assert_array_equal(surfaces, [.004, .016, .016, .004])
