from __future__ import annotations

import base64
import copy
import io
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from PIL import Image
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import comfy_deforum  # noqa: E402
from animation_plan import build_chunk_ranges, build_seed_plan  # noqa: E402
from animation_state import AnimationProject  # noqa: E402
from schemas import AnimateRequest, GenerationRequest  # noqa: E402


def make_request(frames: int = 10, **updates) -> AnimateRequest:
    values = {"render_frames": frames, "seed": 17, "seed_behavior": "iter"}
    values.update(updates)
    return AnimateRequest(**values)


def make_project(req: AnimateRequest, *, active: int | None = 0) -> AnimationProject:
    ranges = [
        list(bounds)
        for bounds in build_chunk_ranges(
            req.total_frames, 8, req.diffusion_cadence
        )
    ]
    seed_plan = build_seed_plan(
        17 if req.seed == -1 else req.seed, req.seed_behavior, req.total_frames
    )
    return AnimationProject(
        schema_version=2,
        revision=3,
        job_id="abcdef0123456789",
        owner=None,
        role="user",
        status="running" if active is not None else "queued",
        request=req.model_dump(),
        total_frames=req.total_frames,
        chunk_ranges=ranges,
        completed_frames=0,
        completed_chunks=0,
        next_chunk_index=0,
        active_chunk_index=active,
        seed_base=17 if req.seed == -1 else req.seed,
        seed_plan=seed_plan,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def upload(monkeypatch):
    lease = Mock()
    lease.upload.return_value = "leased-init.png"
    lease.metadata.return_value = {
        "input_cleanup": "adapter-owned local upload cleanup attempted"
    }
    monkeypatch.setattr(
        comfy_deforum, "_new_upload_lease", lambda: lease, raising=False
    )
    return lease


def build(req=None, project=None, *, start=0, end=8, **kwargs):
    req = req or make_request()
    project = project or make_project(req)
    return comfy_deforum.build_animation_chunk_graph(
        req, project, start=start, end=end, **kwargs
    )


def animator(graph):
    matches = [
        node for node in graph.values()
        if node["class_type"] == "KreaDeforumAnimator"
    ]
    assert len(matches) == 1
    return matches[0]


def png(width=768, height=768, mode="RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (width, height)).save(out, "PNG")
    return out.getvalue()


def png_header(width: int, height: int) -> bytes:
    payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk_type = b"IHDR"
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def animated_png(width=768, height=768) -> bytes:
    out = io.BytesIO()
    frames = [
        Image.new("RGB", (width, height), color)
        for color in ("red", "blue")
    ]
    frames[0].save(
        out, "PNG", save_all=True, append_images=frames[1:], duration=10, loop=0
    )
    return out.getvalue()


def make_windows_junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junctions are unavailable on this platform")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.fail(
            "Windows junction creation failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    assert os.path.isjunction(link)


def write_video(path: Path, frame_count: int, width=256, height=256) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (width, height),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            writer.write(
                np.full((height, width, 3), index * 20, dtype=np.uint8)
            )
    finally:
        writer.release()


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return frames
            frames.append(frame)
    finally:
        capture.release()


def test_builds_exact_animator_and_websocket_graph():
    graph, metadata = build()
    node = animator(graph)
    inputs = node["inputs"]

    assert set(inputs) == {
        "model", "clip", "vae", "width", "height", "max_frames", "steps",
        "sampler_name", "scheduler", "seed", "seed_behavior", "animation_mode",
        "border_mode", "prompt_schedule", "negative_prompt", "cfg_schedule",
        "strength_schedule", "zoom_schedule", "angle_schedule",
        "translation_x_schedule", "translation_y_schedule",
        "translation_z_schedule", "rotation_3d_x_schedule",
        "rotation_3d_y_schedule", "rotation_3d_z_schedule", "color_coherence",
        "diffusion_cadence", "frame_offset", "init_image_is_previous",
        "seed_plan", "prompt_blend_frames",
    }
    assert inputs["model"][1] == inputs["clip"][1] == inputs["vae"][1] == 0
    assert graph["save_ws"] == {
        "class_type": "SaveImageWebsocket",
        "inputs": {"images": [
            next(k for k, v in graph.items() if v is node), 0
        ]},
    }
    assert not any(
        value["class_type"] == "KreaDeforumSaveVideo" for value in graph.values()
    )
    assert metadata["frame_count"] == 8
    assert metadata["seed_base"] == 17
    assert metadata["external_revision"] == comfy_deforum.KREADEFORUM_REVISION


def test_graph_inputs_match_pinned_patched_api_contract():
    contract = json.loads(
        (
            ROOT
            / "tests"
            / "fixtures"
            / "kreadeforum_animator_contract.json"
        ).read_text(encoding="utf-8")
    )
    graph, _ = build()
    names = set(animator(graph)["inputs"])

    assert names == set(contract["required"]) | {
        "frame_offset",
        "init_image_is_previous",
        "seed_plan",
        "prompt_blend_frames",
    }


def test_package_import_does_not_load_duplicate_top_level_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import backend.comfy_deforum; "
                "assert 'comfy_deforum' not in sys.modules; "
                "assert 'comfy_client' not in sys.modules; "
                "assert 'comfy_workflows' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_mode_managed_root_and_cleanup(tmp_path, monkeypatch):
    import backend.comfy_deforum as packaged
    import backend.settings as packaged_settings

    input_root = tmp_path / "ComfyUI" / "input"
    input_root.mkdir(parents=True)
    monkeypatch.setattr(packaged_settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(
        packaged.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )

    def upload_response(*args, **kwargs):
        name, raw, _media_type = kwargs["files"]["image"]
        (input_root / name).write_bytes(raw)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"name": name}
        return response

    monkeypatch.setattr(packaged.requests, "post", upload_response)
    lease = packaged._new_upload_lease()
    name = lease.upload(png())
    assert lease.managed
    assert (input_root / name).is_file()
    lease.close()
    assert not (input_root / name).exists()


def test_optional_hybrid_and_init_inputs_are_exact(tmp_path, upload, monkeypatch):
    video = tmp_path / "controlled" / "source.mp4"
    video.parent.mkdir()
    write_video(video, 8)
    req = make_request(
        animation_mode="Video Input", source_video_upload_id="opaque"
    )
    project = make_project(req)
    monkeypatch.setattr(
        comfy_deforum.comfy_client, "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )

    graph, _ = build(
        req, project, source_video_path=video,
        controlled_video_root=video.parent, init_image_b64=base64.b64encode(png()).decode(),
    )
    inputs = animator(graph)["inputs"]

    assert inputs["hybrid_video_path"] != str(video.resolve())
    assert inputs["hybrid_strength_schedule"] == "0:(0.5), 1:(0.5), 2:(0.5), 3:(0.5), 4:(0.5), 5:(0.5), 6:(0.5), 7:(0.5)"
    assert inputs["hybrid_mode"] == "optical_flow"
    assert inputs["hybrid_video_has_context"] is False
    assert inputs["init_image"] == ["init_image", 0]
    assert graph["init_image"]["class_type"] == "LoadImage"
    upload.upload.assert_called_once()


def test_schedules_are_global_evaluated_then_localized(upload):
    req = make_request(
        prompt_schedule="0: dawn\n5: dusk",
        zoom_schedule="0:(1+t/max_f), 9:(2)",
        cfg_schedule="0:(sin(t)+2)",
    )
    project = make_project(req, active=1)
    graph, _ = build(
        req, project, start=8, end=10,
        init_image_b64=base64.b64encode(png()).decode(),
        reference_image_b64=base64.b64encode(png()).decode(),
    )
    inputs = animator(graph)["inputs"]

    assert inputs["prompt_schedule"] == "0: dusk\n1: dusk"
    assert inputs["zoom_schedule"] == "0:(1.8888888888888888), 1:(2.0)"
    assert "sin" not in inputs["cfg_schedule"]
    assert "t" not in inputs["cfg_schedule"]


def test_tiny_scientific_schedule_is_accepted_and_localized():
    req = make_request(
        frames=2,
        cfg_schedule="0:(1e-7), 1:(2e-7)",
    )
    graph, _ = build(req, make_project(req), start=0, end=2)

    assert animator(graph)["inputs"]["cfg_schedule"] == (
        "0:(1e-07), 1:(2e-07)"
    )


def test_numeric_chunk_validator_accepts_decimal_and_scientific_literals():
    comfy_deforum._validate_numeric_chunk_schedule(
        "0:(1e-07), 1:(-2.5E+3), 2:(.5), 3:(5.)"
    )


@pytest.mark.parametrize(
    "schedule",
    [
        "0:(nan)",
        "0:(inf)",
        "0:(t)",
        "0:(sin(1))",
        "0:(1+2)",
        "0:(1_000)",
        "0:(1e309)",
    ],
)
def test_numeric_chunk_validator_rejects_nonfinite_or_expression(schedule):
    with pytest.raises(ValueError, match="numeric literal"):
        comfy_deforum._validate_numeric_chunk_schedule(schedule)


def test_malicious_schedule_is_rejected_before_graph_submission():
    req = make_request(zoom_schedule="0:(__import__('os').system('whoami'))")
    with pytest.raises(ValueError, match="unsupported"):
        build(req, make_project(req))


@pytest.mark.parametrize(
    ("loader", "class_type"),
    [
        (GenerationRequest(prompt="x", diffusion_engine="native_gguf", quantization="gguf"), "UnetLoaderGGUF"),
        (GenerationRequest(prompt="x", diffusion_engine="native_int8_convrot", quantization="int8"), "OTUNetLoaderW8A8"),
        (GenerationRequest(prompt="x", diffusion_engine="native_pytorch", quantization="fp8"), "UNETLoader"),
    ],
)
def test_reuses_krea_model_bundle_for_supported_loaders(loader, class_type):
    graph, _ = build(loader_request=loader)
    assert any(node["class_type"] == class_type for node in graph.values())
    assert any(node["class_type"] == "CLIPLoader" for node in graph.values())
    assert any(node["class_type"] == "VAELoader" for node in graph.values())


@pytest.mark.parametrize("behavior", ["fixed", "iter"])
def test_chunk_seed_matches_project_seed_plan(behavior, upload):
    req = make_request(seed=91, seed_behavior=behavior)
    project = make_project(req, active=1)
    graph, metadata = build(
        req, project, start=8, end=10,
        init_image_b64=base64.b64encode(png()).decode(),
        reference_image_b64=base64.b64encode(png()).decode(),
    )
    inputs = animator(graph)["inputs"]

    assert inputs["seed"] == project.seed_plan[8]
    assert [
        (inputs["seed"] + i if inputs["seed_behavior"] == "iter" else inputs["seed"])
        % (1 << 64)
        for i in range(2)
    ] == project.seed_plan[8:10]
    assert metadata["resolved_seed"] == project.seed_plan[8]


def test_ladder_requires_even_global_chunk_start(upload):
    req = make_request(frames=12, seed_behavior="ladder", diffusion_cadence=3)
    project = make_project(req, active=1)
    graph, _ = build(
        req,
        project,
        start=9,
        end=12,
        init_image_b64=base64.b64encode(png()).decode(),
        reference_image_b64=base64.b64encode(png()).decode(),
    )
    assert animator(graph)["inputs"]["seed_plan"] == str(
        project.seed_plan[9:12]
    ).replace(" ", "")


def test_random_rejects_multiple_chunks():
    req = make_request(seed_behavior="random")
    graph, _ = build(req, make_project(req))
    assert animator(graph)["inputs"]["seed_plan"] == str(
        make_project(req).seed_plan[:8]
    ).replace(" ", "")


def test_later_chunk_requires_init_image():
    req = make_request()
    project = make_project(req, active=1)
    with pytest.raises(ValueError, match="init_image_b64"):
        build(req, project, start=8, end=10)


def test_later_lab_chunk_requires_reference_image():
    req = make_request()
    project = make_project(req, active=1)
    with pytest.raises(ValueError, match="reference_image_b64"):
        build(
            req,
            project,
            start=8,
            end=10,
            init_image_b64=base64.b64encode(png()).decode(),
        )


def test_later_chunk_wires_boundary_inputs_and_exact_seed_slice(upload):
    req = make_request()
    project = make_project(req, active=1)
    graph, _ = build(
        req,
        project,
        start=8,
        end=10,
        init_image_b64=base64.b64encode(png()).decode(),
        reference_image_b64=base64.b64encode(png()).decode(),
    )
    inputs = animator(graph)["inputs"]

    assert inputs["frame_offset"] == 8
    assert inputs["init_image_is_previous"] is True
    assert inputs["seed_plan"] == "[25,26]"
    assert inputs["init_image"] == ["init_image", 0]
    assert inputs["reference_image"] == ["reference_image", 0]


def test_whole_seed_plan_is_preflighted_before_chunk_build():
    req = make_request()
    project = make_project(req)
    project.seed_plan[-1] += 100

    with pytest.raises(ValueError, match="whole project seed plan"):
        build(req, project)


def test_request_snapshot_and_declared_range_are_enforced():
    req = make_request()
    project = make_project(req)
    changed = req.model_copy(update={"steps": 9})
    with pytest.raises(ValueError, match="snapshot"):
        build(changed, project)
    with pytest.raises(ValueError, match="active or next chunk"):
        build(req, project, start=0, end=7)


def test_forged_oversized_project_ranges_are_rejected():
    req = make_request(frames=20)
    project = make_project(req)
    project.chunk_ranges = [[0, 20]]

    with pytest.raises(ValueError, match="canonical chunk ranges"):
        build(req, project, start=0, end=20)


def test_chunk_must_match_active_or_next_range():
    req = make_request()
    project = make_project(req, active=0)

    with pytest.raises(ValueError, match="active or next chunk"):
        build(
            req,
            project,
            start=8,
            end=10,
            init_image_b64=base64.b64encode(png()).decode(),
        )


@pytest.mark.parametrize(
    ("behavior", "seed"),
    [
        ("iter", (1 << 64) - 1),
        ("ladder", (1 << 64) - 501),
    ],
)
def test_seed_rollover_is_carried_by_exact_patched_seed_plan(behavior, seed):
    req = make_request(frames=2, seed=seed, seed_behavior=behavior)
    project = make_project(req)

    graph, _ = build(req, project, start=0, end=2)

    assert animator(graph)["inputs"]["seed_plan"] == json.dumps(
        project.seed_plan, separators=(",", ":")
    )


def test_parity_aligned_ladder_preserves_seed_plan():
    req = make_request(frames=8, seed=91, seed_behavior="ladder")
    project = make_project(req)
    graph, _ = build(req, project, start=0, end=8)
    inputs = animator(graph)["inputs"]

    assert inputs["seed"] == 91
    assert [
        inputs["seed"] + (1000 if index % 2 else 0)
        for index in range(8)
    ] == project.seed_plan


@pytest.mark.parametrize("kind", ["traversal", "directory"])
def test_video_path_rejects_invalid_controlled_targets(tmp_path, monkeypatch, kind):
    root = tmp_path / "controlled"
    root.mkdir()
    target = root if kind == "directory" else root / ".." / "escape.mp4"
    if kind == "traversal":
        target.resolve().write_bytes(b"x")
    req = make_request(
        animation_mode="Video Input", source_video_upload_id="opaque"
    )
    monkeypatch.setattr(
        comfy_deforum.comfy_client, "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )
    with pytest.raises(ValueError, match="controlled root|regular file"):
        build(req, make_project(req), source_video_path=target, controlled_video_root=root)


def test_video_path_rejects_symlink_or_junction(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("covered by mandatory Windows junction tests")
    root = tmp_path / "controlled"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    link = root / "link.mp4"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"links unavailable: {exc}")
    req = make_request(
        animation_mode="Video Input", source_video_upload_id="opaque"
    )
    monkeypatch.setattr(
        comfy_deforum.comfy_client, "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )
    with pytest.raises(ValueError, match="link|reparse"):
        build(req, make_project(req), source_video_path=link, controlled_video_root=root)


def test_video_path_rejects_linked_controlled_root(tmp_path, monkeypatch):
    target_root = tmp_path / "real-root"
    target_root.mkdir()
    video = target_root / "source.mp4"
    video.write_bytes(b"x")
    linked_root = tmp_path / "linked-root"
    if os.name == "nt":
        make_windows_junction(linked_root, target_root)
    else:
        linked_root.symlink_to(target_root, target_is_directory=True)
    req = make_request(
        animation_mode="Video Input", source_video_upload_id="opaque"
    )
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )

    with pytest.raises(ValueError, match="controlled root.*link|reparse"):
        build(
            req,
            make_project(req),
            source_video_path=linked_root / video.name,
            controlled_video_root=linked_root,
        )


def test_video_path_rejects_linked_child_escape(tmp_path, monkeypatch):
    root = tmp_path / "controlled"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    video = outside / "source.mp4"
    video.write_bytes(b"x")
    linked_child = root / "linked-child"
    if os.name == "nt":
        make_windows_junction(linked_child, outside)
    else:
        linked_child.symlink_to(outside, target_is_directory=True)
    req = make_request(
        animation_mode="Video Input", source_video_upload_id="opaque"
    )
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )

    with pytest.raises(ValueError, match="link|reparse"):
        build(
            req,
            make_project(req),
            source_video_path=linked_child / video.name,
            controlled_video_root=root,
        )


def test_video_path_rejects_remote_comfy_even_for_local_file(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"x")
    req = make_request(
        animation_mode="Video Input", source_video_upload_id="opaque"
    )
    monkeypatch.setattr(
        comfy_deforum.comfy_client, "comfy_base_url",
        lambda: "https://comfy.example.test",
    )
    with pytest.raises(ValueError, match="remote ComfyUI"):
        build(req, make_project(req), source_video_path=video, controlled_video_root=tmp_path)


def test_render_uses_global_source_video_slice_and_cleans_temp(
    tmp_path, monkeypatch, upload
):
    source = tmp_path / "source.mp4"
    write_video(source, 10)
    req = make_request(
        animation_mode="Video Input",
        source_video_upload_id="opaque",
        width=256,
        height=256,
    )
    project = make_project(req, active=1)
    observed = {}

    class SliceClient:
        def run(self, graph, **kwargs):
            path = Path(animator(graph)["inputs"]["hybrid_video_path"])
            observed["path"] = path
            capture = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            assert len(frames) == 3
            assert 120 < frames[0].mean() < 155
            assert frames[1].mean() > 150
            assert animator(graph)["inputs"]["hybrid_video_has_context"] is True
            assert animator(graph)["inputs"]["max_frames"] == 2
            return [png(256, 256), png(256, 256)]

    monkeypatch.setattr(
        comfy_deforum,
        "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )
    comfy_deforum.render_animation_chunk(
        req,
        project,
        start=8,
        end=10,
        init_image_b64=base64.b64encode(png(256, 256)).decode(),
        reference_image_b64=base64.b64encode(png(256, 256)).decode(),
        source_video_path=source,
        controlled_video_root=tmp_path,
        client=SliceClient(),
    )

    assert not observed["path"].exists()
    assert not observed["path"].parent.exists()


def test_built_graph_resources_live_until_explicit_close(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    write_video(source, 8)
    req = make_request(
        frames=8,
        animation_mode="Video Input",
        source_video_upload_id="opaque",
        width=256,
        height=256,
        hybrid_mode="normal",
    )
    project = make_project(req)
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )

    lease = comfy_deforum.build_animation_chunk_graph(
        req,
        project,
        start=0,
        end=8,
        source_video_path=source,
        controlled_video_root=tmp_path,
    )
    path = Path(animator(lease.graph)["inputs"]["hybrid_video_path"])

    assert path.is_file()
    assert len([frame for frame in _read_video(path)]) == 8
    lease.close()
    assert not path.exists()
    assert not path.parent.exists()


def test_graph_lease_context_closes_on_exception(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    write_video(source, 8)
    req = make_request(
        frames=8,
        animation_mode="Video Input",
        source_video_upload_id="opaque",
        width=256,
        height=256,
    )
    project = make_project(req)
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )
    path = None
    with pytest.raises(RuntimeError, match="boom"):
        with comfy_deforum.build_animation_chunk_graph(
            req,
            project,
            start=0,
            end=8,
            source_video_path=source,
            controlled_video_root=tmp_path,
        ) as lease:
            path = Path(animator(lease.graph)["inputs"]["hybrid_video_path"])
            assert path.is_file()
            raise RuntimeError("boom")
    assert path is not None and not path.exists()


def test_video_source_replacement_after_open_uses_opened_file(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.mp4"
    replacement = tmp_path / "replacement.mp4"
    write_video(source, 10)
    write_video(replacement, 2)
    req = make_request(
        animation_mode="Video Input",
        source_video_upload_id="opaque",
        width=256,
        height=256,
    )
    project = make_project(req)
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )

    real_open = comfy_deforum.os.open

    def racing_open(path, flags):
        if Path(path) == source:
            os.replace(replacement, source)
        return real_open(path, flags)

    monkeypatch.setattr(comfy_deforum.os, "open", racing_open)
    with pytest.raises(ValueError, match="changed while opening"):
        build(
            req,
            project,
            source_video_path=source,
            controlled_video_root=tmp_path,
        )


def test_video_mutation_during_copy_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    write_video(source, 8)
    req = make_request(
        frames=8,
        animation_mode="Video Input",
        source_video_upload_id="opaque",
        width=256,
        height=256,
    )
    project = make_project(req)
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "http://127.0.0.1:8188",
    )

    def mutate(path):
        current = path.stat()
        os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))

    monkeypatch.setattr(comfy_deforum, "_after_video_source_copy", mutate)
    with pytest.raises(ValueError, match="changed while copying"):
        comfy_deforum.build_animation_chunk_graph(
            req,
            project,
            start=0,
            end=8,
            source_video_path=source,
            controlled_video_root=tmp_path,
        )


def test_remote_multi_chunk_chunk_zero_passes_runtime_preflight(monkeypatch):
    req = make_request(frames=10)
    project = make_project(req)
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "https://comfy.example.test",
    )

    loader = comfy_deforum.validate_animation_runtime(project, req)
    assert loader is not None


def test_non_video_rejects_unrelated_source_path(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError, match="Video Input"):
        build(source_video_path=video, controlled_video_root=tmp_path)


def test_metadata_has_3d_requirement_and_no_absolute_paths():
    req = make_request(animation_mode="3D")
    _, metadata = build(req, make_project(req))
    assert metadata["requirements"]["requires_midas"] is True
    assert str(ROOT.resolve()) not in repr(metadata)


def test_render_checks_fresh_node_availability(monkeypatch):
    req = make_request(frames=2)
    project = make_project(req)
    project.chunk_ranges = [[0, 2]]
    monkeypatch.setattr(
        comfy_deforum, "status",
        Mock(return_value={"available": False, "missing_nodes": ["KreaDeforumAnimator"]}),
    )
    with pytest.raises(RuntimeError, match="KreaDeforumAnimator"):
        comfy_deforum.render_animation_chunk(req, project, start=0, end=2, client=Mock())
    comfy_deforum.status.assert_called_once_with(force_refresh=True)


def test_default_loader_rejects_unknown_engine(monkeypatch):
    import settings

    monkeypatch.setattr(settings.settings, "diffusion_engine", "mystery")
    with pytest.raises(ValueError, match="unsupported diffusion engine"):
        comfy_deforum._default_loader_request(make_request())


def test_default_loader_does_not_swallow_runtime_settings_error():
    real_import = __import__

    def failing_import(name, *args, **kwargs):
        if name == "settings":
            raise RuntimeError("settings exploded")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=failing_import):
        with pytest.raises(RuntimeError, match="settings exploded"):
            comfy_deforum._default_loader_request(make_request())


def test_render_forwards_callbacks_timeout_and_output_node(monkeypatch):
    req = make_request(frames=2, width=256, height=256, steps=52)
    project = make_project(req)
    project.chunk_ranges = [[0, 2]]
    client = Mock()
    client.run.return_value = [png(256, 256), png(256, 256)]
    progress, prompt_id = Mock(), Mock()
    monkeypatch.setattr(
        comfy_deforum, "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )

    result = comfy_deforum.render_animation_chunk(
        req, project, start=0, end=2, progress_cb=progress,
        prompt_id_cb=prompt_id, client=client,
    )

    assert result == client.run.return_value
    kwargs = client.run.call_args.kwargs
    assert kwargs["progress_cb"] is progress
    assert kwargs["prompt_id_cb"] is prompt_id
    assert kwargs["image_node_id"] == "save_ws"
    assert 1 <= kwargs["timeout"] <= 1800


@pytest.mark.parametrize(
    ("blobs", "message"),
    [
        ([png(256, 256)], "exactly 2"),
        ([png(256, 256)] * 3, "exactly 2"),
        ([b"not-png", png(256, 256)], "PNG"),
        ([png(512, 256), png(256, 256)], "dimensions"),
        ([png(256, 256, "RGBA"), png(256, 256)], "RGB"),
    ],
)
def test_render_rejects_invalid_output_without_mutating_state(
    monkeypatch, blobs, message
):
    req = make_request(frames=2, width=256, height=256)
    project = make_project(req)
    project.chunk_ranges = [[0, 2]]
    before = copy.deepcopy(project.to_dict())
    client = Mock()
    client.run.return_value = blobs
    monkeypatch.setattr(
        comfy_deforum, "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )

    with pytest.raises(ValueError, match=message):
        comfy_deforum.render_animation_chunk(
            req, project, start=0, end=2, client=client
        )
    assert project.to_dict() == before


def test_render_rejects_oversized_png(monkeypatch):
    req = make_request(frames=1, width=256, height=256)
    project = make_project(req)
    project.chunk_ranges = [[0, 1]]
    client = Mock()
    client.run.return_value = [png(256, 256) + b"x" * (comfy_deforum.MAX_FRAME_BYTES + 1)]
    monkeypatch.setattr(
        comfy_deforum, "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )
    with pytest.raises(ValueError, match="size"):
        comfy_deforum.render_animation_chunk(
            req, project, start=0, end=1, client=client
        )


def test_render_rejects_animated_png_output(monkeypatch):
    req = make_request(frames=1, width=256, height=256)
    project = make_project(req)
    client = Mock()
    client.run.return_value = [animated_png(256, 256)]
    monkeypatch.setattr(
        comfy_deforum,
        "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )
    with pytest.raises(ValueError, match="single-frame"):
        comfy_deforum.render_animation_chunk(
            req, project, start=0, end=1, client=client
        )


def test_render_wraps_execution_error_without_public_details(monkeypatch):
    req = make_request(frames=1, width=256, height=256)
    project = make_project(req)
    client = Mock()
    client.run.side_effect = RuntimeError(
        "https://secret.example E:\\private\\source.mp4 prompt=secret"
    )
    monkeypatch.setattr(
        comfy_deforum,
        "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )
    with pytest.raises(comfy_deforum.ComfyDeforumError) as captured:
        comfy_deforum.render_animation_chunk(
            req, project, start=0, end=1, client=client
        )
    message = str(captured.value)
    assert "secret" not in message
    assert "private" not in message
    assert "http" not in message


@pytest.mark.parametrize(
    ("blob", "message"),
    [
        (png_header(256, 256), "malformed PNG"),
        (png_header(100_000, 100_000), "pixel cap"),
    ],
)
def test_render_rejects_truncated_or_decompression_png(
    monkeypatch, blob, message
):
    req = make_request(frames=1, width=256, height=256)
    project = make_project(req)
    client = Mock()
    client.run.return_value = [blob]
    monkeypatch.setattr(
        comfy_deforum,
        "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )

    with pytest.raises(ValueError, match=message):
        comfy_deforum.render_animation_chunk(
            req, project, start=0, end=1, client=client
        )


def test_render_closes_only_its_upload_lease(monkeypatch):
    req = make_request(frames=1, width=256, height=256)
    project = make_project(req)
    project.chunk_ranges = [[0, 1]]
    lease = Mock()
    lease.upload.return_value = "owned.png"
    lease.metadata.return_value = {}
    monkeypatch.setattr(comfy_deforum, "_new_upload_lease", lambda: lease)
    monkeypatch.setattr(
        comfy_deforum, "status",
        lambda **_: {"available": True, "missing_nodes": []},
    )
    client = Mock()
    client.run.return_value = [png(256, 256)]

    comfy_deforum.render_animation_chunk(
        req, project, start=0, end=1,
        init_image_b64=base64.b64encode(png(256, 256)).decode(),
        client=client,
    )

    lease.close.assert_called_once_with()


def test_owned_upload_cleanup_deletes_only_returned_adapter_file(
    tmp_path, monkeypatch
):
    unrelated = tmp_path / "user.png"
    unrelated.write_bytes(b"user")

    def upload_response(*args, **kwargs):
        name = kwargs["files"]["image"][0]
        (tmp_path / name).write_bytes(kwargs["files"]["image"][1])
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"name": name}
        return response

    monkeypatch.setattr(comfy_deforum.requests, "post", upload_response)
    lease = comfy_deforum._UploadLease(input_root=tmp_path)
    name = lease.upload(png())
    assert (tmp_path / name).is_file()

    lease.close()

    assert not (tmp_path / name).exists()
    assert unrelated.read_bytes() == b"user"


def test_owned_upload_cleanup_failure_is_best_effort(tmp_path, monkeypatch):
    lease = comfy_deforum._UploadLease(input_root=tmp_path)
    owned = tmp_path / "krea_deforum_owned.png"
    owned.write_bytes(b"x")
    lease._owned.append(owned.name)
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=PermissionError("busy")))

    lease.close()


def test_external_comfy_allows_later_init_via_http_upload(monkeypatch, upload):
    req = make_request()
    project = make_project(req, active=1)
    monkeypatch.setattr(
        comfy_deforum.comfy_client,
        "comfy_base_url",
        lambda: "https://comfy.example.test",
    )
    graph, _ = build(
        req,
        project,
        start=8,
        end=10,
        init_image_b64=base64.b64encode(png()).decode(),
        reference_image_b64=base64.b64encode(png()).decode(),
    )
    assert "init_image" in graph
    assert animator(graph)["inputs"]["prompt_blend_frames"] == 0


def test_prompt_strength_boost_is_applied_to_chunk_strength_schedule(upload):
    req = make_request(
        frames=8,
        prompt_schedule="0: dawn\n4: dusk",
        strength_schedule="0:(0.5)",
        prompt_strength_boost=0.2,
        prompt_strength_boost_frames=1,
        prompt_blend_frames=3,
    )
    graph, _ = build(req, make_project(req))
    strength = animator(graph)["inputs"]["strength_schedule"]
    assert "0:(0.5)" in strength
    assert "4:(0.7)" in strength
    assert animator(graph)["inputs"]["prompt_blend_frames"] == 3


def test_init_upload_validates_base64_type_and_dimensions(upload):
    with pytest.raises(ValueError, match="base64"):
        build(init_image_b64="%%%")
    bad = io.BytesIO()
    Image.new("RGB", (768, 768)).save(bad, "JPEG")
    with pytest.raises(ValueError, match="PNG"):
        build(init_image_b64=base64.b64encode(bad.getvalue()).decode())


def test_init_upload_rejects_encoded_length_before_decode(upload, monkeypatch):
    monkeypatch.setattr(comfy_deforum, "MAX_INPUT_BYTES", 3)
    with pytest.raises(ValueError, match="encoded size"):
        build(init_image_b64=base64.b64encode(b"12345678").decode())
    upload.upload.assert_not_called()


def test_init_upload_rejects_decoded_byte_size(upload, monkeypatch):
    monkeypatch.setattr(comfy_deforum, "MAX_INPUT_BYTES", 10)
    with pytest.raises(ValueError, match="decoded byte size"):
        build(init_image_b64=base64.b64encode(b"12345678901").decode())
    upload.upload.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (png_header(768, 768), "malformed PNG"),
        (png_header(100_000, 100_000), "pixel cap"),
        (animated_png(), "single-frame"),
        (png(512, 768), "dimensions"),
    ],
)
def test_init_upload_rejects_unsafe_png_before_network(upload, raw, message):
    with pytest.raises(ValueError, match=message):
        build(init_image_b64=base64.b64encode(raw).decode())
    upload.upload.assert_not_called()


def test_init_upload_normalizes_exact_dimension_image_to_rgb_png(upload):
    palette = Image.new("P", (768, 768))
    out = io.BytesIO()
    palette.save(out, "PNG")

    build(init_image_b64=base64.b64encode(out.getvalue()).decode())

    uploaded = upload.upload.call_args.args[0]
    assert uploaded.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(uploaded)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (768, 768)
