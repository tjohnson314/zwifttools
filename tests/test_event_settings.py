import unittest

from shared.event_settings import _time


class EventTimestampTests(unittest.TestCase):
    def test_supported_offsets_are_converted_to_utc(self):
        for value in [
            "2026-11-18T18:15:00.000+0000",
            "2026-11-18T18:15:00+0000",
            "2026-11-18T18:15:00.000+00:00",
            "2026-11-18T18:15:00Z",
            "2026-11-18T10:15:00.000-0800",
            "2026-11-18T23:45:00.000+0530",
            " 2026-11-18T18:15:00.000+0000 ",
        ]:
            with self.subTest(value=value):
                self.assertEqual(_time(value), "2026-11-18 18:15:00 UTC")

    def test_offset_conversion_can_cross_date_boundary(self):
        self.assertEqual(
            _time("2026-11-18T00:15:00+0530"), "2026-11-17 18:45:00 UTC")

    def test_missing_unknown_and_naive_times_remain_explicit(self):
        self.assertEqual(_time(None), "Not specified")
        self.assertEqual(_time("bad date"), "Unrecognized time: bad date")
        self.assertEqual(_time(123), "Unrecognized time: 123")
        self.assertEqual(
            _time("2026-11-18T18:15:00"),
            "2026-11-18T18:15:00 (time zone not specified)")
