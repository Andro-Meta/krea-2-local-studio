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
    sampler_options,
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

        with self.assertRaisesRegex(ValueError, "scheduler"):
            validate_sampler_configuration("euler", "karras", "krea_turbo")


if __name__ == "__main__":
    unittest.main()
