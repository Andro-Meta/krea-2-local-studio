import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    import torch
    from krea2 import sampling
except Exception as exc:  # pragma: no cover - torch absent in lightweight CI
    raise unittest.SkipTest(f"torch unavailable: {exc}")


class TestFlowStepFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.img = torch.randn(1, 64, 16)
        self.v = torch.randn(1, 64, 16)
        self.noise = torch.randn(1, 64, 16)

    def test_ancestral_eta0_equals_euler(self):
        # With eta=0 the ancestral step must collapse to the deterministic Euler step.
        euler = sampling.euler_flow_step(self.img, self.v, tcurr=0.8, tprev=0.6)
        anc = sampling.euler_ancestral_flow_step(
            self.img, self.v, tcurr=0.8, tprev=0.6, noise=self.noise, eta=0.0
        )
        self.assertTrue(torch.allclose(euler, anc, atol=1e-5))

    def test_ancestral_final_step_returns_denoised(self):
        out = sampling.euler_ancestral_flow_step(
            self.img, self.v, tcurr=0.1, tprev=0.0, noise=self.noise, eta=1.0
        )
        denoised = self.img - self.v * 0.1
        self.assertTrue(torch.allclose(out, denoised, atol=1e-5))

    def test_ancestral_eta_injects_variation(self):
        a = sampling.euler_ancestral_flow_step(
            self.img, self.v, tcurr=0.8, tprev=0.6, noise=self.noise, eta=1.0
        )
        euler = sampling.euler_flow_step(self.img, self.v, tcurr=0.8, tprev=0.6)
        self.assertFalse(torch.allclose(a, euler, atol=1e-3))
        self.assertTrue(torch.isfinite(a).all())

    def test_cfgpp_finite_and_returns_denoised_at_zero(self):
        v_cond = self.v
        v_uncond = torch.randn_like(self.v)
        out = sampling.euler_cfgpp_flow_step(
            self.img, v_cond, v_uncond, tcurr=0.8, tprev=0.6, noise=self.noise, eta=0.0
        )
        self.assertTrue(torch.isfinite(out).all())
        final = sampling.euler_cfgpp_flow_step(
            self.img, v_cond, v_uncond, tcurr=0.1, tprev=0.0, noise=self.noise, eta=0.0
        )
        self.assertTrue(torch.allclose(final, self.img - v_cond * 0.1, atol=1e-5))

    def test_cfgpp_first_step_t1_is_finite(self):
        # At the pure-noise start t=1.0, alpha_s = 1 - t = 0; the step must stay
        # finite (no division by alpha_s). Regression for the div-by-zero bug.
        for eta in (0.0, 1.0):
            out = sampling.euler_cfgpp_flow_step(
                self.img, self.v, self.v * 0.7, tcurr=1.0, tprev=0.85,
                noise=self.noise, eta=eta, s_noise=1.0,
            )
            self.assertTrue(torch.isfinite(out).all(), f"eta={eta}")

    def test_cfgpp_direction_uses_uncond(self):
        # Changing only the uncond velocity must change the CFG++ step (direction
        # comes from uncond), proving the trick is wired, not ignored.
        base = sampling.euler_cfgpp_flow_step(
            self.img, self.v, self.v, tcurr=0.8, tprev=0.6, noise=self.noise, eta=0.0
        )
        alt = sampling.euler_cfgpp_flow_step(
            self.img, self.v, self.v * 0.5, tcurr=0.8, tprev=0.6, noise=self.noise, eta=0.0
        )
        self.assertFalse(torch.allclose(base, alt, atol=1e-4))


class TestRes2s(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.img = torch.randn(1, 64, 16)
        self.v = torch.randn(1, 64, 16)

    def test_returns_denoised_at_final_step(self):
        out = sampling.res_2s_flow_step(
            self.img, self.v, tcurr=0.1, tprev=0.0,
            velocity_fn=lambda x, t: torch.zeros_like(x),
        )
        self.assertTrue(torch.allclose(out, self.img - self.v * 0.1, atol=1e-5))

    def test_two_calls_and_finite(self):
        calls = []
        def vf(x, t):
            calls.append(t)
            return torch.zeros_like(x)
        out = sampling.res_2s_flow_step(self.img, self.v, tcurr=0.9, tprev=0.5, velocity_fn=vf)
        self.assertEqual(len(calls), 1)  # one intermediate call (plus the caller's first v)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(calls[0], 0.5)
        self.assertLess(calls[0], 0.9)

    def test_first_step_t1_finite(self):
        out = sampling.res_2s_flow_step(
            self.img, self.v, tcurr=1.0, tprev=0.85,
            velocity_fn=lambda x, t: torch.zeros_like(x),
        )
        self.assertTrue(torch.isfinite(out).all())


class TestCfgZeroStar(unittest.TestCase):
    def test_scale_is_projection(self):
        # s* = <c,u>/||u||^2; for u parallel to c (c = k*u), s* = k.
        u = torch.randn(2, 64, 16)
        c = 3.0 * u
        s = sampling.cfg_zero_star_scale(c, u)
        self.assertEqual(s.shape, (2, 1, 1))
        self.assertTrue(torch.allclose(s.flatten(), torch.full((2,), 3.0), atol=1e-3))

    def test_scale_orthogonal_is_zero(self):
        # Orthogonal cond/uncond -> projection ~ 0.
        u = torch.zeros(1, 4, 2); u[0, 0, 0] = 1.0
        c = torch.zeros(1, 4, 2); c[0, 0, 1] = 1.0
        s = sampling.cfg_zero_star_scale(c, u)
        self.assertAlmostEqual(float(s.flatten()[0]), 0.0, places=5)

    def test_scale_finite(self):
        s = sampling.cfg_zero_star_scale(torch.randn(3, 8, 4), torch.randn(3, 8, 4))
        self.assertTrue(torch.isfinite(s).all())


class TestTimestepScheduler(unittest.TestCase):
    def test_scheduler_changes_schedule(self):
        simple = sampling.timesteps(256, 12, 1, 100, mu=1.15, scheduler="simple")
        beta = sampling.timesteps(256, 12, 1, 100, mu=1.15, scheduler="beta")
        self.assertEqual(len(simple), 13)
        self.assertEqual(len(beta), 13)
        self.assertAlmostEqual(simple[0], 1.0, places=4)
        self.assertEqual(simple[-1], 0.0)
        self.assertEqual(beta[-1], 0.0)
        self.assertNotEqual([round(x, 4) for x in simple], [round(x, 4) for x in beta])

    def test_schedule_descending(self):
        for sched in ("simple", "normal", "beta", "sgm_uniform"):
            ts = sampling.timesteps(256, 10, 1, 100, mu=1.15, scheduler=sched)
            for a, b in zip(ts[:-1], ts[1:]):
                self.assertGreaterEqual(a + 1e-6, b, sched)


if __name__ == "__main__":
    unittest.main()
