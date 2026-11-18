"""
Tests for the app-side feature rules in pipeline/export_model.py.
Run from the repo root:   python -m unittest tests.test_app_features -v

"Percent of normal" must be the training definition 100 * (x + 5) / (normal + 5),
NOT plain 100 * x / normal.
"""

import math
import unittest

from pipeline.export_model import app_features, pct_of_normal
from pipeline.improve_cv import COUNTY_BASE


def tiny_model():
    """A fake model.json with one county whose normals are easy to check by hand."""
    flat = lambda v: [float(v)] * 36
    return {"normals": {"KE001": {"rfh_avg": flat(20), "r1h_avg": flat(60), "r3h_avg": flat(180)}}}


class TestPercentOfNormal(unittest.TestCase):
    def test_training_definition(self):
        self.assertAlmostEqual(pct_of_normal(25, 20), 120.0)  # 100 * 30 / 25
        self.assertAlmostEqual(pct_of_normal(0, 0), 100.0)  # a dry month in a dry place is "normal"
        self.assertAlmostEqual(pct_of_normal(0, 20), 20.0)  # 100 * 5 / 25, not 0
        self.assertNotAlmostEqual(pct_of_normal(25, 20), 100 * 25 / 20)  # not the plain ratio


class TestAppFeatures(unittest.TestCase):
    def setUp(self):
        self.f = app_features("KE001", 7, rfh=25, rfh_lag1=10, rfh_lag2=5, r3h=120,
                              widespread=0.3, model=tiny_model())

    def test_has_all_17_features(self):
        self.assertEqual(sorted(self.f), sorted(COUNTY_BASE))

    def test_exact_features(self):
        self.assertEqual(self.f["r1h"], 40)  # 25 + 10 + 5
        self.assertEqual(self.f["rfh_avg"], 20)
        self.assertAlmostEqual(self.f["season_sin"], math.sin(2 * math.pi * 7 / 36))

    def test_percent_features_use_plus_5(self):
        self.assertAlmostEqual(self.f["rfq"], 100 * (25 + 5) / (20 + 5))  # 120
        self.assertAlmostEqual(self.f["rfq_lag1"], 100 * (10 + 5) / (20 + 5))  # 60
        self.assertAlmostEqual(self.f["r1q"], 100 * (40 + 5) / (60 + 5))
        self.assertAlmostEqual(self.f["r3q"], 100 * (120 + 5) / (180 + 5))

    def test_max_features_copy_the_county_value(self):
        for plain in ("rfh", "rfq", "r1q", "r3q"):
            self.assertEqual(self.f[plain + "_max"], self.f[plain])
        self.assertEqual(self.f["nat_frac_anom"], 0.3)  # the slider

    def test_previous_dekad_of_dekad_1_is_36(self):
        m = tiny_model()
        m["normals"]["KE001"]["rfh_avg"][35] = 45.0  # dekad 36 (index 35)
        f = app_features("KE001", 1, rfh=0, rfh_lag1=40, rfh_lag2=0, r3h=0, widespread=0, model=m)
        self.assertAlmostEqual(f["rfq_lag1"], 100 * (40 + 5) / (45 + 5))


if __name__ == "__main__":
    unittest.main()
