from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from krea2.sampler_registry import (  # noqa: E402
    KREA_FLOW_SAMPLERS,
    SAMPLER_SPECS,
    normalize_sampler_name,
    recommended_combos,
    recommended_steps,
    sampler_catalog,
    sampler_options,
    scheduler_options,
    validate_sampler_configuration,
    validate_sampler_for_profile,
)


class SamplerRegistryTests(unittest.TestCase):
    def test_comfy_euler_alias_maps_to_existing_euler_flow(self) -> None:
        self.assertEqual(normalize_sampler_name("euler"), "euler_flow")
        self.assertEqual(normalize_sampler_name("euler_flow"), "euler_flow")
        self.assertIn("euler_flow", KREA_FLOW_SAMPLERS)

    def test_krea_flow_sampler_options_include_guarded_comfy_names(self) -> None:
        ids = [option["id"] for option in sampler_options("krea_turbo")]

        self.assertIn("euler", ids)
        self.assertIn("exp_heun_2_x0_sde", ids)
        self.assertIn("lcm", ids)
        self.assertIn("dpmpp_2m", ids)
        self.assertTrue(next(option for option in sampler_options("krea_turbo") if option["id"] == "dpmpp_2m")["disabled"])

    def test_incompatible_sampler_fails_with_clear_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a standard diffusion backend"):
            validate_sampler_for_profile("dpmpp_2m", "krea_turbo")

    def test_lcm_is_guarded_without_compatible_profile(self) -> None:
        spec = SAMPLER_SPECS["lcm"]
        self.assertTrue(spec.requires_lcm_profile)
        with self.assertRaisesRegex(ValueError, "requires an LCM-compatible"):
            validate_sampler_for_profile("lcm", "krea_turbo")

    def test_krea_sampler_validation_includes_scheduler(self) -> None:
        config = validate_sampler_configuration("euler", "simple", "krea_turbo")

        self.assertEqual(config["sampler"], "euler_flow")
        self.assertEqual(config["scheduler"], "simple")

        # beta is now a supported flow scheduler for native samplers.
        beta = validate_sampler_configuration("euler", "beta", "krea_turbo")
        self.assertEqual(beta["scheduler"], "beta")

        # A truly unsupported scheduler still fails with a clear message.
        with self.assertRaisesRegex(ValueError, "scheduler"):
            validate_sampler_configuration("euler", "ddim_uniform", "krea_turbo")

    def test_new_flow_samplers_registered(self) -> None:
        for sid in ("euler_ancestral", "euler_ancestral_cfg_pp", "euler_cfg_pp"):
            self.assertIn(sid, SAMPLER_SPECS)
            self.assertIn(sid, KREA_FLOW_SAMPLERS)
            cfg = validate_sampler_configuration(sid, "beta", "krea_raw")
            self.assertEqual(cfg["scheduler"], "beta")

    def test_scheduler_options_marks_recommended(self) -> None:
        opts = {o["id"]: o for o in scheduler_options()}
        self.assertTrue(opts["beta"]["recommended"])
        self.assertFalse(opts["karras"]["recommended"])

    def test_recommended_steps_turbo_vs_raw(self) -> None:
        self.assertEqual(recommended_steps("euler", "simple", "krea_turbo"), 8)
        self.assertGreaterEqual(recommended_steps("euler", "beta", "krea_raw"), 28)
        self.assertEqual(recommended_steps("lcm", "simple", "krea_turbo"), 4)

    def test_recommended_combos_profile_specific(self) -> None:
        turbo = recommended_combos("krea_turbo")
        raw = recommended_combos("krea_raw")
        self.assertTrue(any(c["label"] == "Turbo default" for c in turbo))
        self.assertTrue(any("CFG++" in c["label"] for c in raw))
        # Turbo combos use few steps; RAW combos use many.
        self.assertTrue(all(c["steps"] <= 12 for c in turbo))
        self.assertTrue(any(c["steps"] >= 28 for c in raw))

    def test_sampler_catalog_shape(self) -> None:
        cat = sampler_catalog("krea_raw")
        self.assertIn("samplers", cat)
        self.assertIn("schedulers", cat)
        self.assertIn("recommended_combos", cat)


if __name__ == "__main__":
    unittest.main()
