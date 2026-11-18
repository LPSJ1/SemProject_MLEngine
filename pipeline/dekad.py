"""
dekad.py - which of the 36 ten-day periods ("dekads") of the year a date is in.

Each month has 3 dekads:
    1st dekad:  days 1-10
    2nd dekad:  days 11-20
    3rd dekad:  day 21 to the end of the month (8 to 11 days)
So the year runs 1..36 (1 Jan = 1, 31 Dec = 36).

Why this matters: the HDX rainfall data only has dates on the 1st, 11th and
21st, so any rule gives the same answer there. But the live engine and the
forecast mode add up DAILY rain, so every day of the month must land in the
right dekad. In particular day 10 belongs to the 1st dekad and day 20 to the
2nd (an older version, day // 10, wrongly put them in the next dekad).

The same rule is used in pipeline/make_features.py (inline, because that
script has no package imports) and, for the Android app, written out in
backend/HANDOFF.md. Tests: tests/test_dekad.py.
"""


def dekad_of_year(dates):
    """1..36 for one date (datetime / pandas Timestamp) or a pandas Series of dates."""
    if hasattr(dates, "dt"):  # a column of dates
        month, day = dates.dt.month, dates.dt.day
        part = ((day - 1) // 10).clip(upper=2) + 1  # 1, 2 or 3 inside the month
    else:  # one date
        month, day = dates.month, dates.day
        part = min((day - 1) // 10, 2) + 1
    return (month - 1) * 3 + part
