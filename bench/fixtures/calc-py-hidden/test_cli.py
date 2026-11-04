import subprocess
import sys
import unittest


def run(*args):
    return subprocess.run([sys.executable, "-m", "calc", *args], capture_output=True, text=True)


class CliTest(unittest.TestCase):
    def test_mean(self):
        r = run("mean", "1", "2", "3")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), "2.0")

    def test_median(self):
        self.assertEqual(run("median", "3", "1", "2").stdout.strip(), "2")

    def test_unknown_command(self):
        r = run("nope")
        self.assertEqual(r.returncode, 2)
        self.assertIn("usage", r.stderr.lower())


if __name__ == "__main__":
    unittest.main()
