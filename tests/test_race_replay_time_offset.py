import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path when run directly as a script
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from race_replay.data_cleaner import (
    RiderData,
    align_riders_to_elevation_profile,
    clean_race_data,
    compute_finish_crossing_time,
)


class TestRaceReplayTimeOffset(unittest.TestCase):
    """Test suite to verify that finish crossing detection and time offsets
    are calculated accurately across race datasets, including routes with
    near-finish road crossovers and long cooldown riding.
    """

    def test_subgroup_distance_aligns_to_shared_profile(self):
        profile_distance = np.arange(0.0, 3.01, 0.01)
        profile_altitude = (
            40.0
            + 5.0 * np.sin(profile_distance * 3.1)
            + 2.0 * np.sin(profile_distance * 8.7)
        )
        elevation_profile = pd.DataFrame({
            'distance_km': profile_distance,
            'altitude_m': profile_altitude,
        })
        expected_shift_km = 0.215
        physical_distance = np.linspace(0.25, 2.9, 300)
        rider_data = pd.DataFrame({
            'distance_km': physical_distance - expected_shift_km,
            'altitude_m': np.interp(
                physical_distance, profile_distance, profile_altitude
            ),
        }, index=pd.Index(np.arange(300, dtype=float), name='time_sec'))
        riders = [RiderData(
            rank=1,
            activity_id='synthetic',
            name='Synthetic Rider',
            team='',
            data=rider_data,
            finish_time_sec=None,
        )]

        shift_km = align_riders_to_elevation_profile(
            riders, elevation_profile
        )

        self.assertAlmostEqual(shift_km, expected_shift_km, delta=0.01)
        rider = riders[0].data
        profile_altitude = np.interp(
            rider['distance_km'],
            elevation_profile['distance_km'],
            elevation_profile['altitude_m'],
        )
        in_course = rider['distance_km'].between(0.1, 2.9)
        residual = rider.loc[in_course, 'altitude_m'] - profile_altitude[in_course]
        self.assertLess(np.std(residual), 0.5)

    def test_custom_finish_crops_nominal_route(self):
        """A custom total distance must crop the nominal route endpoint."""
        race_dir = WORKSPACE_ROOT / 'race_data' / 'race_data_7343298'
        if not race_dir.exists():
            self.skipTest(f"Race directory {race_dir} not found")

        cleaned = clean_race_data(race_dir, cache=False)

        self.assertAlmostEqual(cleaned.finish_line_km, 3.0, places=3)
        self.assertAlmostEqual(
            cleaned.elevation_profile['distance_km'].max(), 3.0, places=6
        )
        self.assertTrue(
            cleaned.elevation_profile['distance_km'].is_monotonic_increasing
        )
        rider = cleaned.riders[0]
        leadin = rider.data[rider.data['distance_km'] <= 0.35]
        profile_altitude = np.interp(
            leadin['distance_km'],
            cleaned.elevation_profile['distance_km'],
            cleaned.elevation_profile['altitude_m'],
        )
        self.assertLess(
            np.max(np.abs(leadin['altitude_m'] - profile_altitude)), 0.25
        )
        finish_sample = rider.data.iloc[
            np.abs(rider.data.index.to_numpy() - rider.finish_time_sec).argmin()
        ]
        self.assertAlmostEqual(finish_sample['distance_km'], 3.0, delta=0.02)

    def test_croissant_race_offset_consistency(self):
        """Test race_data_7329699 (Croissant / activity 2225400252620423200).
        
        The Croissant route features a road section that passes within 50m
        of the finish coordinates ~55 seconds before the actual finish line.
        Verify that all riders (including ranks 1, 4, 6, 7, 8, 9) are correctly
        anchored to their true finish crossing and have consistent time offsets.
        """
        race_dir = Path(__file__).resolve().parent.parent / 'race_data' / 'race_data_7329699'
        if not race_dir.exists():
            self.skipTest(f"Race directory {race_dir} not found")

        cleaned = clean_race_data(race_dir, cache=False)
        self.assertGreater(len(cleaned.riders), 0, "Expected cleaned riders")

        offsets = []
        start_times = []
        rider_offsets_by_rank = {}

        for rider in cleaned.riders:
            self.assertIsNotNone(rider.ttt_time_offset, f"Rider {rider.rank} should have a valid time offset")
            offsets.append(rider.ttt_time_offset)
            start_times.append(rider.data.index.min())
            rider_offsets_by_rank[rider.rank] = rider.ttt_time_offset

        # All offsets should be tightly clustered around ~ -27s
        mean_offset = float(np.mean(offsets))
        std_offset = float(np.std(offsets))

        self.assertAlmostEqual(mean_offset, -27.0, delta=2.5,
                               msg=f"Mean offset {mean_offset:.2f}s is outside expected range")
        self.assertLess(std_offset, 2.0,
                        msg=f"Offset std deviation {std_offset:.2f}s is too high (indicates mismatched crossings)")

        # Verify specific riders that previously had ~ -90s offsets (e.g. ranks 1, 4, 6, 7, 8, 9)
        problem_ranks = [1, 4, 6, 7, 8, 9]
        for rank in problem_ranks:
            if rank in rider_offsets_by_rank:
                rider_offset = rider_offsets_by_rank[rank]
                self.assertAlmostEqual(
                    rider_offset, mean_offset, delta=3.0,
                    msg=f"Rider rank {rank} offset {rider_offset:.2f}s deviates significantly from mean {mean_offset:.2f}s"
                )

        # Start times on the replay clock should all be closely aligned (within ~4s of each other)
        self.assertLess(max(start_times) - min(start_times), 5.0,
                        msg="Rider start times in replay clock are not closely grouped")

    def test_hilly_route_race_offset(self):
        """Test race_data_7253077 (Hilly Route / activity 2210353942104178688).
        
        Verify that race time offsets are properly computed and consistent
        even when segment distance slightly deviates from Strava route distance.
        """
        race_dir = Path(__file__).resolve().parent.parent / 'race_data' / 'race_data_7253077'
        if not race_dir.exists():
            self.skipTest(f"Race directory {race_dir} not found")

        cleaned = clean_race_data(race_dir, cache=False)
        self.assertGreater(len(cleaned.riders), 0, "Expected cleaned riders")

        # Non-DNF riders should all have time offsets
        finishers = [r for r in cleaned.riders if r.finish_time_sec is not None]
        self.assertGreater(len(finishers), 0, "Expected finishers in race")

        offsets = [r.ttt_time_offset for r in finishers if r.ttt_time_offset is not None]
        self.assertEqual(len(offsets), len(finishers), "All finishers should have time offsets")

        mean_offset = float(np.mean(offsets))
        std_offset = float(np.std(offsets))

        self.assertAlmostEqual(mean_offset, -15.3, delta=2.0,
                               msg=f"Mean offset {mean_offset:.2f}s is outside expected range")
        self.assertLess(std_offset, 2.0,
                        msg=f"Offset std deviation {std_offset:.2f}s is too high")

        # Check target activity 2210353942104178688 (Rank 2)
        target_rider = next((r for r in cleaned.riders if str(r.activity_id) == "2210353942104178688"), None)
        self.assertIsNotNone(target_rider, "Activity 2210353942104178688 should be present")
        self.assertAlmostEqual(target_rider.ttt_time_offset, -16.9, delta=2.0)

    def test_synthetic_near_finish_crossover(self):
        """Test compute_finish_crossing_time with synthetic data simulating a road
        that passes within 3m of the finish line 50 seconds before crossing the actual
        finish line (within 4m).
        """
        # Finish coordinates: (0.0, 0.0)
        finish_lat, finish_lng = 0.0, 0.0
        finish_distance_m = 10000.0

        # Construct time series at 1Hz from t=0 to t=1000s
        times = np.arange(1000, dtype=float)
        # 10m per second
        distances = times * 10.0

        # Base lats/lngs far away (1.0 degree away)
        lats = np.full_like(times, 1.0)
        lngs = np.full_like(times, 1.0)

        # Pre-finish crossover at t=890s..896s (closest approach at t=893s, 3m away)
        for t in range(890, 897):
            lats[t] = 0.000027 * (t - 893)
            lngs[t] = 0.000027

        # Actual finish crossing at t=950s..956s (closest approach at t=953s, 4m away)
        for t in range(950, 957):
            lats[t] = 0.000036 * (t - 953)
            lngs[t] = 0.000036

        # Official time is expected around t=953s
        official_time = 953.0

        crossing_time = compute_finish_crossing_time(
            times=times,
            distances_m=distances,
            lats=lats,
            lngs=lngs,
            finish_lat=finish_lat,
            finish_lng=finish_lng,
            finish_distance_m=finish_distance_m,
            expected_time_sec=official_time,
        )

        self.assertIsNotNone(crossing_time)
        self.assertAlmostEqual(crossing_time, 953.0, delta=1.0,
                               msg=f"Crossing time {crossing_time} should match actual finish (953s), not crossover (893s)")


if __name__ == '__main__':
    unittest.main()
