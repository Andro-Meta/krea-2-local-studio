from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "kreadeforum-krea2-chunking.patch"


def test_patch_extends_only_pinned_animator_contract():
    text = PATCH.read_text(encoding="utf-8")

    assert "diff --git a/animator_node.py b/animator_node.py" in text
    assert text.count("diff --git ") == 1
    for name in (
        "frame_offset",
        "init_image_is_previous",
        "reference_image",
        "seed_plan",
        "prompt_blend_frames",
        "KreaDeforumChunkAdapterVersion",
    ):
        assert name in text


def test_patch_validates_exact_seed_plan_and_continues_previous_frame():
    text = PATCH.read_text(encoding="utf-8")

    assert "len(exact_seeds) != max_frames" in text
    assert "0xffffffffffffffff" in text
    assert "exact_seeds[i]" in text
    assert "if i == 0 and not init_image_is_previous:" in text
    assert "reference_frame_np is None" in text
    assert '"krea2-chunking-v2"' in text
    assert "encode_tiled" in text
    assert "_lerp_conditioning" in text
