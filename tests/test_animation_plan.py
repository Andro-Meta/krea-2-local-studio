import math
import random

import numpy as np
import pytest
from pydantic import ValidationError

from backend.animation_plan import (
    apply_prompt_strength_boost,
    build_chunk_ranges,
    build_seed_plan,
    evaluate_schedule,
    numeric_chunk_schedule,
    parse_prompt_schedule,
    prompt_chunk_schedule,
)
from backend.schemas import AnimateRequest


def test_animate_request_defaults_and_computed_frames():
    request = AnimateRequest()

    assert request.prompt_schedule == "0: a scenic landscape, cinematic lighting"
    assert request.negative_prompt == ""
    assert request.duration_seconds == 4.0
    assert request.fps == 12
    assert request.render_frames is None
    assert request.total_frames == 48
    assert (request.width, request.height) == (768, 768)
    assert (request.steps, request.sampler_name, request.scheduler, request.seed) == (
        8,
        "er_sde",
        "simple",
        -1,
    )
    assert request.seed_behavior == "iter"
    assert (request.animation_mode, request.border_mode) == ("2D", "replicate")
    assert request.cfg_schedule == "0:(1.0)"
    assert request.strength_schedule == "0:(0.65)"
    assert request.zoom_schedule == "0:(1.0)"
    assert all(
        getattr(request, name) == "0:(0)"
        for name in (
            "angle_schedule",
            "translation_x_schedule",
            "translation_y_schedule",
            "translation_z_schedule",
            "rotation_3d_x_schedule",
            "rotation_3d_y_schedule",
            "rotation_3d_z_schedule",
        )
    )
    assert request.color_coherence == "Match Frame 0 LAB"
    assert request.diffusion_cadence == 1
    assert request.hybrid_strength_schedule == "0:(0.5)"
    assert request.hybrid_mode == "optical_flow"
    assert request.init_image_b64 == ""
    assert request.source_video_upload_id == ""
    assert "source_video_path" not in AnimateRequest.model_fields


def test_animate_request_rejects_source_video_filesystem_path():
    with pytest.raises(ValidationError, match="source_video_path"):
        AnimateRequest(source_video_path="C:/secret.mp4")


@pytest.mark.parametrize("render_frames", [None, 17])
def test_animate_request_snapshot_roundtrips_without_computed_fields(render_frames):
    request = AnimateRequest(render_frames=render_frames)
    payload = request.model_dump()

    assert "total_frames" not in payload
    restored = AnimateRequest(**payload)
    assert restored.model_dump() == payload
    assert restored.total_frames == (17 if render_frames is not None else 48)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_seconds", 0.49),
        ("duration_seconds", 60.01),
        ("fps", 0),
        ("fps", 61),
        ("render_frames", 0),
        ("render_frames", 721),
        ("width", 255),
        ("width", 1537),
        ("height", 255),
        ("height", 1537),
        ("steps", 2),
        ("steps", 53),
        ("seed", -2),
        ("seed", 1 << 64),
        ("diffusion_cadence", 0),
        ("diffusion_cadence", 17),
    ],
)
def test_animate_request_rejects_caps(field, value):
    with pytest.raises(ValidationError):
        AnimateRequest(**{field: value})


@pytest.mark.parametrize(("field", "value"), [("width", 770), ("height", 769)])
def test_animate_request_requires_dimensions_divisible_by_16(field, value):
    with pytest.raises(ValidationError, match=r"(?:divisible by|multiple of) 16"):
        AnimateRequest(**{field: value})


def test_animate_request_prefers_render_frames_and_rejects_computed_overflow():
    assert AnimateRequest(duration_seconds=60, fps=60, render_frames=7).total_frames == 7

    with pytest.raises(ValidationError, match="total_frames"):
        AnimateRequest(duration_seconds=60, fps=60)


def test_animate_request_requires_video_upload_for_video_input():
    with pytest.raises(ValidationError, match="source_video_upload_id"):
        AnimateRequest(animation_mode="Video Input")

    request = AnimateRequest(
        animation_mode="Video Input", source_video_upload_id="upload-123"
    )
    assert request.total_frames == 48


def test_animate_request_literals_are_enforced():
    invalid = {
        "seed_behavior": "other",
        "animation_mode": "video",
        "border_mode": "edge",
        "color_coherence": "RGB",
        "hybrid_mode": "flow",
    }
    for field, value in invalid.items():
        with pytest.raises(ValidationError):
            AnimateRequest(**{field: value})


def test_evaluate_schedule_interpolates_and_holds_edges():
    assert evaluate_schedule("2:(10), 4:(20)", 7) == [
        10.0,
        10.0,
        10.0,
        15.0,
        20.0,
        20.0,
        20.0,
    ]


def test_evaluate_schedule_supports_safe_math_expressions():
    values = evaluate_schedule("0:(sin(pi / 2) + sqrt(9)), 4:(max(8, pow(2, 3)))", 5)
    assert values == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert evaluate_schedule("0:(abs(-5) // 2 + 5 % 2)", 1) == [3.0]


def test_evaluate_schedule_uses_upstream_max_frames_semantics():
    assert evaluate_schedule("0:(max_f)", total_frames=3) == [3.0, 3.0, 3.0]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("cos(0)", 1.0),
        ("tan(0)", 0.0),
        ("min(7, 3)", 3.0),
        ("+5", 5.0),
        ("9 - 4", 5.0),
        ("3 * 4", 12.0),
        ("9 / 4", 2.25),
        ("2 ** 6", 64.0),
    ],
)
def test_evaluate_schedule_explicitly_allows_documented_operations(
    expression, expected
):
    assert evaluate_schedule(f"0:({expression})", 1) == [expected]


@pytest.mark.parametrize("expression", ["max(*(1, 2))", "max(a=1, b=2)"])
def test_evaluate_schedule_rejects_starred_and_keyword_arguments(expression):
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_schedule(f"0:({expression})", 1)


@pytest.mark.parametrize(
    "text",
    [
        "0:(().__class__)",
        "0:(x[0])",
        "0:(__import__('os'))",
        "0:(lambda: 1)",
        "0:([x for x in [1]])",
        "0:([1, 2])",
        "0:(sum(1, 2))",
        "0:(True)",
        "0:(1j)",
        "0:(unknown)",
        "0:(unknown_fn(1))",
        "0:(1 / 0)",
        "0:(10 ** 1000)",
        "0:(1e13)",
        "0:(1e309)",
        "-1:(2)",
        "0:(1), 0:(2)",
        "0:(1",
        "not-a-frame:(1)",
        "0:",
        "0:(1), broken",
    ],
)
def test_evaluate_schedule_rejects_unsafe_or_malformed_input(text):
    with pytest.raises(ValueError, match=r".+"):
        evaluate_schedule(text, 10)


@pytest.mark.parametrize("total_frames", [0, -1, 721])
def test_evaluate_schedule_rejects_frame_count_outside_limit(total_frames):
    with pytest.raises(ValueError, match="total_frames.*1.*720"):
        evaluate_schedule("0:(1)", total_frames)


def test_evaluate_schedule_rejects_out_of_range_keyframe():
    with pytest.raises(ValueError, match="range"):
        evaluate_schedule("10:(1)", 10)


def test_evaluate_schedule_enforces_per_expression_length_limit():
    expression = "+".join(["1"] * 300)
    assert len(expression) > 512
    with pytest.raises(ValueError, match="expression.*too long"):
        evaluate_schedule(f"0:({expression})", 10)


def test_evaluate_schedule_enforces_ast_node_limit_before_expression_limit():
    expression = "+".join(["1"] * 70)
    assert len(expression) < 512
    with pytest.raises(ValueError, match="too complex"):
        evaluate_schedule(f"0:({expression})", 10)


@pytest.mark.parametrize(
    "parser",
    [
        evaluate_schedule,
        parse_prompt_schedule,
    ],
)
def test_schedule_parsers_enforce_total_text_length_before_parsing(parser):
    huge_malformed_text = "(" * (32 * 1024 + 1)
    with pytest.raises(ValueError, match="schedule text.*too long"):
        parser(huge_malformed_text, 10)


def test_numeric_schedule_enforces_keyframe_count_limit():
    schedule = ", ".join(f"{frame % 720}:(1)" for frame in range(721))
    with pytest.raises(ValueError, match="too many keyframes"):
        evaluate_schedule(schedule, 720)


def test_prompt_schedule_enforces_keyframe_count_limit():
    schedule = "\n".join(f"{frame % 720}: prompt" for frame in range(721))
    with pytest.raises(ValueError, match="too many keyframes"):
        parse_prompt_schedule(schedule, 720)


def test_parse_prompt_schedule_holds_prompts_and_preserves_colons():
    assert parse_prompt_schedule("0: dawn: warm\n2: night: blue", 4) == [
        "dawn: warm",
        "dawn: warm",
        "night: blue",
        "night: blue",
    ]
    assert parse_prompt_schedule("2: later", 4) == ["later"] * 4


@pytest.mark.parametrize(
    "text",
    [
        "-1: prompt",
        "4: prompt",
        "0:",
        "missing delimiter",
        "x: prompt",
        "0: first\n0: duplicate",
    ],
)
def test_parse_prompt_schedule_rejects_bad_entries(text):
    with pytest.raises(ValueError, match=r".+"):
        parse_prompt_schedule(text, 4)


@pytest.mark.parametrize("total_frames", [0, -1, 721])
def test_parse_prompt_schedule_rejects_frame_count_outside_limit(total_frames):
    with pytest.raises(ValueError, match="total_frames.*1.*720"):
        parse_prompt_schedule("0: prompt", total_frames)


def test_build_seed_plan_behaviors_are_deterministic_and_wrapped():
    base = (1 << 64) - 2
    assert build_seed_plan(5, "fixed", 4) == [5, 5, 5, 5]
    assert build_seed_plan(base, "iter", 4) == [base, base + 1, 0, 1]
    assert build_seed_plan(5, "ladder", 4) == [5, 1005, 5, 1005]
    assert build_seed_plan(5, "random", 4) == [
        2881021351,
        3457461229,
        97294836,
        3470079268,
    ]


@pytest.mark.parametrize("behavior", ["fixed", "iter", "ladder", "random"])
def test_build_seed_plan_uint64_boundary_never_overflows(behavior):
    uint64_max = (1 << 64) - 1
    plan = build_seed_plan(uint64_max, behavior, 3)

    assert all(0 <= value <= uint64_max for value in plan)
    if behavior == "fixed":
        assert plan == [uint64_max] * 3
    elif behavior == "iter":
        assert plan == [uint64_max, 0, 1]
    elif behavior == "ladder":
        assert plan == [uint64_max, 999, uint64_max]


def test_build_seed_plan_rejects_seed_above_uint64():
    with pytest.raises(ValueError, match="seed"):
        build_seed_plan(1 << 64, "fixed", 1)


def test_build_seed_plan_does_not_mutate_global_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    build_seed_plan(9, "random", 3)

    assert random.getstate() == python_state
    numpy_state_after = np.random.get_state()
    assert numpy_state_after[0] == numpy_state[0]
    assert np.array_equal(numpy_state_after[1], numpy_state[1])
    assert numpy_state_after[2:] == numpy_state[2:]


def test_build_seed_plan_minus_one_chooses_one_persisted_base(monkeypatch):
    calls = []

    def fake_randbits(bits):
        calls.append(bits)
        return 123

    monkeypatch.setattr("backend.animation_plan.secrets.randbits", fake_randbits)
    assert build_seed_plan(-1, "iter", 3) == [123, 124, 125]
    assert calls == [32]


@pytest.mark.parametrize(
    ("seed", "behavior", "count"),
    [(1, "bad", 3), (1, "fixed", 0), (-2, "fixed", 1)],
)
def test_build_seed_plan_rejects_invalid_arguments(seed, behavior, count):
    with pytest.raises(ValueError, match=r".+"):
        build_seed_plan(seed, behavior, count)


def test_build_seed_plan_rejects_total_frames_above_limit():
    with pytest.raises(ValueError, match="total_frames.*1.*720"):
        build_seed_plan(1, "fixed", 721)


@pytest.mark.parametrize(
    ("total", "size", "cadence", "expected"),
    [
        (25, 8, 3, [(0, 9), (9, 18), (18, 25)]),
        (5, 8, 3, [(0, 5)]),
        (10, 2, 100, [(0, 10)]),
    ],
)
def test_build_chunk_ranges(total, size, cadence, expected):
    assert build_chunk_ranges(total, size, cadence) == expected


@pytest.mark.parametrize(
    ("total", "size", "cadence"), [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
)
def test_build_chunk_ranges_rejects_nonpositive_values(total, size, cadence):
    with pytest.raises(ValueError, match=r".+"):
        build_chunk_ranges(total, size, cadence)


def test_build_chunk_ranges_rejects_total_frames_above_limit():
    with pytest.raises(ValueError, match="total_frames.*1.*720"):
        build_chunk_ranges(721, 8, 1)


def test_chunk_serializers_use_local_keys_and_safe_literals():
    assert numeric_chunk_schedule([1, 2.5, math.pi, 4], 1, 3) == (
        "0:(2.5), 1:(3.141592653589793)"
    )
    assert prompt_chunk_schedule(["unused", "line one\nline two", "a: b"], 1, 3) == (
        "0: line one line two\n1: a: b"
    )
    numeric = numeric_chunk_schedule([float("1e12")], 0, 1)
    assert numeric == "0:(1000000000000.0)"
    assert not any(token in numeric for token in ("t", "sin", "__"))


@pytest.mark.parametrize(
    ("values", "start", "end"),
    [([1], -1, 1), ([1], 0, 0), ([1], 0, 2), ([1], 1, 0)],
)
def test_chunk_serializers_validate_bounds(values, start, end):
    with pytest.raises(ValueError, match=r".+"):
        numeric_chunk_schedule(values, start, end)
    with pytest.raises(ValueError, match=r".+"):
        prompt_chunk_schedule([str(value) for value in values], start, end)


def test_chunk_serializers_reject_timelines_above_frame_limit():
    values = [1.0] * 721
    with pytest.raises(ValueError, match="values.*720"):
        numeric_chunk_schedule(values, 0, 1)
    with pytest.raises(ValueError, match="values.*720"):
        prompt_chunk_schedule(["prompt"] * 721, 0, 1)


@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan"), 1e13, True]
)
def test_numeric_chunk_schedule_rejects_nonfinite_and_boolean_values(value):
    with pytest.raises(ValueError, match=r".+"):
        numeric_chunk_schedule([value], 0, 1)

def test_apply_prompt_strength_boost_raises_near_prompt_changes():
    strengths = [0.5] * 10
    prompts = ["a"] * 5 + ["b"] * 5
    out = apply_prompt_strength_boost(strengths, prompts, boost=0.2, window=2)
    # Prompt changes at frame 5 → boost frames 3..7 inclusive.
    assert out[2] == pytest.approx(0.5)
    assert out[3] == pytest.approx(0.7)
    assert out[5] == pytest.approx(0.7)
    assert out[7] == pytest.approx(0.7)
    assert out[8] == pytest.approx(0.5)


def test_apply_prompt_strength_boost_clamps_to_one_and_noop_when_zero():
    strengths = [0.9, 0.9, 0.9]
    prompts = ["a", "b", "b"]
    assert apply_prompt_strength_boost(strengths, prompts, boost=0.0, window=2) == strengths
    out = apply_prompt_strength_boost(strengths, prompts, boost=0.35, window=1)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0)


def test_animate_request_includes_soft_handoff_defaults():
    request = AnimateRequest()
    assert request.prompt_blend_frames == 0
    assert request.prompt_strength_boost == 0.0
    assert request.prompt_strength_boost_frames == 4
