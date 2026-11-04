import importlib
import unittest


class LayoutTest(unittest.TestCase):
    def test_stats_module(self):
        stats = importlib.import_module("calc.stats")
        for name in ("mean", "median", "mode"):
            self.assertTrue(callable(getattr(stats, name)))

    def test_package_reexports(self):
        import calc
        import calc.stats

        for name in ("mean", "median", "mode"):
            self.assertIs(getattr(calc, name), getattr(calc.stats, name))


if __name__ == "__main__":
    unittest.main()
