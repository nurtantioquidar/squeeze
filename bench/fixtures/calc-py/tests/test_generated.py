"""Many small cases, so a verbose test run produces a realistically long log."""

import unittest

from calc import median


class GeneratedMedianTest(unittest.TestCase):
    pass


def _make(n):
    def test(self):
        self.assertEqual(median(list(range(1, 2 * n + 2))), n + 1)

    return test


for _n in range(1, 301):
    setattr(GeneratedMedianTest, f"test_median_of_{2 * _n + 1}_values", _make(_n))
