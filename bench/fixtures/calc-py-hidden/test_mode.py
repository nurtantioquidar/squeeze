import unittest

from calc import mode


class ModeTest(unittest.TestCase):
    def test_most_common(self):
        self.assertEqual(mode([1, 2, 2, 3]), 2)

    def test_ties_pick_smallest(self):
        self.assertEqual(mode([3, 1, 3, 1, 2]), 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            mode([])


if __name__ == "__main__":
    unittest.main()
