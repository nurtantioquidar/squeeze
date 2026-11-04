import unittest

from calc import mean, median


class EmptyInputTest(unittest.TestCase):
    def test_mean_empty(self):
        with self.assertRaisesRegex(ValueError, "empty input"):
            mean([])

    def test_median_empty(self):
        with self.assertRaisesRegex(ValueError, "empty input"):
            median([])


if __name__ == "__main__":
    unittest.main()
