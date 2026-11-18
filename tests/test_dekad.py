"""
Unit test for the dekad rule. Run from the repo root:

    python -m unittest tests.test_dekad -v

Rule: days 1-10 -> 1st dekad of the month, 11-20 -> 2nd, 21-end -> 3rd.
"""

import datetime
import unittest

import pandas as pd

from pipeline.dekad import dekad_of_year

# (day of month, which dekad of the month it must be)
DAY_TO_PART = [(1, 1), (10, 1), (11, 2), (20, 2), (21, 3), (31, 3)]


class TestDekad(unittest.TestCase):
    def test_boundary_days_in_every_month(self):
        # days 1, 10, 11, 20, 21 and 31 (31 only exists in some months)
        for month in range(1, 13):
            for day, part in DAY_TO_PART:
                if day == 31 and month in (2, 4, 6, 9, 11):
                    continue  # no 31st in this month
                with self.subTest(month=month, day=day):
                    expected = (month - 1) * 3 + part
                    self.assertEqual(dekad_of_year(datetime.date(2023, month, day)), expected)

    def test_day_10_is_first_dekad_and_day_20_is_second(self):
        # the old formula (day // 10) got exactly these two wrong
        self.assertEqual(dekad_of_year(datetime.date(2024, 4, 10)), 10)  # not 11
        self.assertEqual(dekad_of_year(datetime.date(2024, 4, 20)), 11)  # not 12

    def test_ends_of_year(self):
        self.assertEqual(dekad_of_year(datetime.date(2024, 1, 1)), 1)
        self.assertEqual(dekad_of_year(datetime.date(2024, 12, 31)), 36)

    def test_february_including_leap_day(self):
        self.assertEqual(dekad_of_year(datetime.date(2024, 2, 29)), 6)  # 3rd dekad of Feb
        self.assertEqual(dekad_of_year(datetime.date(2023, 2, 28)), 6)

    def test_series_matches_single_dates(self):
        dates = pd.Series(pd.date_range("2024-01-01", "2024-12-31", freq="D"))
        got = dekad_of_year(dates)
        self.assertEqual(got.min(), 1)
        self.assertEqual(got.max(), 36)
        for d, value in zip(dates, got):
            self.assertEqual(value, dekad_of_year(d))

    def test_dekad_start_dates_unchanged(self):
        # the data only has the 1st, 11th and 21st: the fix must not change them
        for month in range(1, 13):
            for day, part in ((1, 1), (11, 2), (21, 3)):
                self.assertEqual(dekad_of_year(datetime.date(2010, month, day)),
                                 (month - 1) * 3 + part)


if __name__ == "__main__":
    unittest.main()
