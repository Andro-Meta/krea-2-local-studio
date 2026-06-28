import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from krea2 import schedulers as S


class TestBetaQuantile(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(S.beta_ppf(0.0), 0.0)
        self.assertEqual(S.beta_ppf(1.0), 1.0)

    def test_symmetric_midpoint(self):
        # Beta(0.6, 0.6) is symmetric -> median is 0.5.
        self.assertAlmostEqual(S.beta_ppf(0.5), 0.5, places=4)

    def test_monotonic(self):
        prev = -1.0
        for i in range(0, 101):
            v = S.beta_ppf(i / 100.0)
            self.assertGreaterEqual(v, prev - 1e-9)
            prev = v

    def test_incomplete_beta_roundtrip(self):
        for x in (0.1, 0.3, 0.5, 0.7, 0.9):
            p = S.regularized_incomplete_beta(0.6, 0.6, x)
            self.assertAlmostEqual(S.beta_ppf(p), x, places=3)


class TestBaseGrid(unittest.TestCase):
    def test_length_and_endpoints(self):
        for sched in S.ALL_SCHEDULERS:
            g = S.base_grid(10, sched)
            self.assertEqual(len(g), 11, sched)
            self.assertAlmostEqual(g[0], 1.0, places=6, msg=sched)
            self.assertEqual(g[-1], 0.0, sched)

    def test_descending(self):
        for sched in S.ALL_SCHEDULERS:
            g = S.base_grid(12, sched)
            for a, b in zip(g[:-1], g[1:]):
                self.assertGreaterEqual(a + 1e-9, b, sched)

    def test_simple_is_uniform(self):
        g = S.base_grid(8, "simple")
        for k, v in enumerate(g):
            self.assertAlmostEqual(v, 1.0 - k / 8.0, places=6)

    def test_normal_matches_simple(self):
        self.assertEqual(S.base_grid(8, "normal"), S.base_grid(8, "simple"))

    def test_beta_is_u_shaped(self):
        # Beta(0.6,0.6) clusters steps near both ends -> the largest gap is in
        # the middle of the trajectory, unlike the uniform schedule.
        g = S.base_grid(16, "beta")
        gaps = [a - b for a, b in zip(g[:-1], g[1:])]
        mid = gaps[len(gaps) // 2]
        ends = max(gaps[0], gaps[-1])
        self.assertGreater(mid, ends)

    def test_sgm_uniform_drops_min_endpoint(self):
        g = S.base_grid(8, "sgm_uniform")
        # second-to-last is 1/steps (the sigma_min point is skipped vs simple).
        self.assertAlmostEqual(g[-2], 1.0 / 8.0, places=6)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            S.base_grid(8, "nope")


if __name__ == "__main__":
    unittest.main()
