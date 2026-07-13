import json
import os
import hashlib
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import backend.animation_state as animation_state
from backend.animation_state import (
    AnimationProject,
    AnimationStateError,
    AnimationStore,
)
from backend.schemas import AnimateRequest


JOB_A = "123e4567-e89b-12d3-a456-426614174000"
JOB_B = "abcdef0123456789"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [({True: "x"}, {1: "x"}), ({1: "x"}, {True: "x"})],
)
def test_canonical_comparison_rejects_equal_values_with_different_key_types(
    actual, expected
):
    assert animation_state._canonical_mismatch(actual, expected) is not None


@pytest.fixture
def roots(tmp_path):
    return tmp_path / "state", tmp_path / "outputs"


@pytest.fixture
def store(roots):
    return AnimationStore(*roots)


def request(frames=3, seed=7, behavior="iter", cadence=1):
    return AnimateRequest(
        render_frames=frames,
        seed=seed,
        seed_behavior=behavior,
        diffusion_cadence=cadence,
    )


def frames_dir(roots, owner, job_id):
    segment = (
        "_local"
        if owner is None
        else "u-" + hashlib.sha256(owner.encode("utf-8")).hexdigest()
    )
    return roots[1] / segment / "animations" / job_id / "frames"


def write_frames(roots, owner, job_id, names):
    directory = frames_dir(roots, owner, job_id).parent / "staging"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")


def write_staging_frames(roots, owner, job_id, names):
    directory = frames_dir(roots, owner, job_id).parent / "staging"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(f"staged:{name}".encode())


def snapshot(store, job_id):
    return store.load(job_id).to_dict()


def make_directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks unavailable: {symlink_error}")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.skip(
            "neither Windows directory symlinks nor junctions are available: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def make_windows_junction(link, target):
    if os.name != "nt" or not hasattr(os.path, "isjunction"):
        pytest.skip("Windows junctions are unavailable on this platform")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.fail(
            "Windows junction creation is unavailable: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    assert os.path.isjunction(link)


def test_create_persists_versioned_default_shape_and_roundtrips_request(
    store, roots
):
    req = request(frames=50, seed=11, cadence=3)
    project = store.create(req, owner="alice", role="user", job_id=JOB_A)

    assert isinstance(project, AnimationProject)
    assert project.schema_version == 2
    assert project.revision == 0
    assert project.job_id == JOB_A
    assert (project.owner, project.role, project.status) == ("alice", "user", "queued")
    assert AnimateRequest(**project.request).model_dump() == req.model_dump()
    assert project.total_frames == 50
    assert project.chunk_ranges == [
        [0, 9],
        [9, 18],
        [18, 27],
        [27, 36],
        [36, 45],
        [45, 50],
    ]
    assert project.completed_frames == 0
    assert project.completed_chunks == 0
    assert project.next_chunk_index == 0
    assert project.active_chunk_index is None
    assert project.frame_files == []
    assert project.frame_integrity == []
    assert project.seed_base == 11
    assert project.seed_plan == list(range(11, 61))
    assert project.error == ""
    assert project.video_path is None
    assert project.poster_path is None
    assert project.gallery_id is None
    assert project.created_at and project.updated_at
    assert (roots[0] / f"{JOB_A}.json").is_file()
    assert frames_dir(roots, "alice", JOB_A).is_dir()
    assert (frames_dir(roots, "alice", JOB_A).parent / "staging").is_dir()
    assert store.load(JOB_A).to_dict() == project.to_dict()


def test_random_seed_is_resolved_once_and_survives_reload(store, monkeypatch):
    monkeypatch.setattr("backend.animation_state.secrets.randbits", lambda bits: 1234)
    project = store.create(
        request(frames=3, seed=-1, behavior="random"),
        owner=None,
        role="local",
        job_id=JOB_A,
    )

    assert project.request["seed"] == -1
    assert project.seed_base == 1234
    assert store.load(JOB_A).seed_plan == project.seed_plan


def test_atomic_save_leaves_no_temp_files(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    store.mark_status(JOB_A, "blocked")

    assert list(roots[0].glob(f".{JOB_A}.*.tmp")) == []


def test_create_rejects_existing_state_without_creating_output(store, roots):
    state_path = roots[0] / f"{JOB_A}.json"
    state_path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError):
        store.create(request(), owner="alice", role="user", job_id=JOB_A)

    assert state_path.read_text(encoding="utf-8") == "sentinel"
    assert not frames_dir(roots, "alice", JOB_A).parent.exists()


def test_create_rejects_existing_output_directory_and_preserves_it(store, roots):
    project_dir = frames_dir(roots, "alice", JOB_A).parent
    project_dir.mkdir(parents=True)
    sentinel = project_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        store.create(request(), owner="alice", role="user", job_id=JOB_A)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (roots[0] / f"{JOB_A}.json").exists()


def test_create_write_failure_rolls_back_only_new_project_directory(
    store, roots, monkeypatch
):
    owner_parent = frames_dir(roots, "alice", JOB_A).parents[1]
    owner_parent.mkdir(parents=True)
    monkeypatch.setattr(
        "backend.animation_state.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("atomic replace failed")),
    )

    with pytest.raises(OSError, match="atomic replace failed"):
        store.create(request(), owner="alice", role="user", job_id=JOB_A)

    assert owner_parent.is_dir()
    assert list(owner_parent.iterdir()) == []
    assert not (roots[0] / f"{JOB_A}.json").exists()
    assert list(roots[0].glob(f".{JOB_A}.*.tmp")) == []


@pytest.mark.parametrize("seam", ["before_rename", "after_rename"])
def test_create_crash_seams_leave_only_reconcilable_marked_layout(
    store, roots, monkeypatch, seam
):
    method = (
        "_after_create_layout"
        if seam == "before_rename"
        else "_after_create_rename"
    )
    monkeypatch.setattr(
        store,
        method,
        lambda: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with pytest.raises(SystemExit, match="crash"):
        store.create(request(), owner="alice", role="user", job_id=JOB_A)

    animations = frames_dir(roots, "alice", JOB_A).parents[1]
    assert any(path.name.startswith(".creating-") or (path / ".creating").exists() for path in animations.iterdir())
    store.reconcile_staging()
    assert list(animations.iterdir()) == []


def test_create_marker_reconciles_crashed_orphan_but_preserves_unmarked(
    store, roots
):
    orphan = frames_dir(roots, "alice", JOB_A).parent
    orphan.mkdir(parents=True)
    (orphan / ".creating").write_text(JOB_A, encoding="utf-8")
    unmarked = frames_dir(roots, "alice", JOB_B).parent
    unmarked.mkdir(parents=True)
    sentinel = unmarked / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    removed = store.reconcile_staging()

    assert not orphan.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert any(".creating" in item for item in removed)


def test_reconcile_removes_stale_creation_marker_for_valid_state(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    marker = frames_dir(roots, "alice", JOB_A).parent / ".creating"
    marker.write_text(JOB_A, encoding="utf-8")

    store.reconcile_staging()

    assert not marker.exists()
    assert store.load(JOB_A).job_id == JOB_A


def test_load_for_owner_is_exact_and_admin_can_access(store):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)

    assert store.load_for_owner(JOB_A, username="alice").job_id == JOB_A
    assert store.load_for_owner(JOB_A, username="root", is_admin=True).job_id == JOB_A
    with pytest.raises(FileNotFoundError, match="not found"):
        store.load_for_owner(JOB_A, username="bob")
    with pytest.raises(FileNotFoundError, match="not found"):
        store.load_for_owner(JOB_B, username="bob")


@pytest.mark.parametrize(
    "job_id",
    [
        "",
        ".",
        "..",
        "../escape",
        r"..\escape",
        "/absolute",
        r"C:\absolute",
        "has/slash",
        "has space",
        "x" * 129,
    ],
)
def test_job_ids_reject_traversal_and_unsafe_names(store, job_id):
    with pytest.raises(ValueError, match="job"):
        store.create(request(), owner="alice", role="user", job_id=job_id)


@pytest.mark.parametrize("owner", ["Alice", ".alice", "alice-", "Élodie", "../alice"])
def test_owner_paths_are_opaque_and_preserve_exact_identity(store, roots, owner):
    project = store.create(request(), owner=owner, role="user", job_id=JOB_A)
    segment = "u-" + hashlib.sha256(owner.encode("utf-8")).hexdigest()

    assert project.owner == owner
    assert store.load_for_owner(JOB_A, username=owner).owner == owner
    assert (roots[1] / segment / "animations" / JOB_A).is_dir()
    assert owner not in str(store.project_dir(JOB_A))
    output_url = (
        "/api/outputs/"
        + (store.project_dir(JOB_A) / "animation.mp4")
        .relative_to(roots[1])
        .as_posix()
    )
    assert owner not in output_url
    with pytest.raises(FileNotFoundError):
        store.load_for_owner(JOB_A, username=owner.swapcase())


def test_owner_hash_is_case_sensitive_and_filesystem_safe(store, roots):
    store.create(request(), owner="Alice", role="user", job_id=JOB_A)
    store.create(request(), owner="alice", role="user", job_id=JOB_B)

    first = store.project_dir(JOB_A).relative_to(roots[1]).parts[0]
    second = store.project_dir(JOB_B).relative_to(roots[1]).parts[0]
    assert first != second
    assert first == first.lower()
    assert second == second.lower()


def test_corrupt_and_future_state_are_rejected_with_actionable_errors(store, roots):
    roots[0].mkdir(parents=True, exist_ok=True)
    path = roots[0] / f"{JOB_A}.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(AnimationStateError, match="invalid JSON"):
        store.load(JOB_A)

    path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(AnimationStateError, match="future schema"):
        store.load(JOB_A)
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(AnimationStateError, match="older schema"):
        store.load(JOB_A)


@pytest.mark.parametrize(
    "invalid_json",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
        '{"schema_version":Infinity}',
    ],
)
def test_loaded_state_rejects_duplicate_keys_and_nonfinite_json(
    store, roots, invalid_json
):
    path = roots[0] / f"{JOB_A}.json"
    path.write_text(invalid_json, encoding="utf-8")
    with pytest.raises(AnimationStateError, match=r"duplicate|constant|JSON"):
        store.load(JOB_A)


def test_loaded_state_rejects_missing_persisted_field(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["poster_path"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnimationStateError, match="missing.*poster_path"):
        store.load(JOB_A)


def test_loaded_state_rejects_unexpected_persisted_field(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["surprise"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnimationStateError, match="unexpected.*surprise"):
        store.load(JOB_A)


def test_loaded_request_snapshot_rejects_coercible_wrong_types(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["request"]["fps"] = "12"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnimationStateError, match="request snapshot"):
        store.load(JOB_A)


def test_loaded_request_snapshot_rejects_int_for_canonical_float(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["request"]["duration_seconds"] == 4.0
    payload["request"]["duration_seconds"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        AnimationStateError, match=r"request snapshot.*non-canonical"
    ):
        store.load(JOB_A)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "mystery"),
        ("status", []),
        ("completed_frames", -1),
        ("chunk_ranges", [[1, 3]]),
        ("frame_files", ["../escape.png"]),
        ("active_chunk_index", "0"),
    ],
)
def test_loaded_state_validates_types_ranges_and_paths(store, roots, field, value):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnimationStateError, match=r"corrupt|invalid"):
        store.load(JOB_A)


def test_loaded_state_rejects_status_inconsistent_with_progress(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "finalizing"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnimationStateError, match="status"):
        store.load(JOB_A)


def test_loaded_media_path_must_resolve_inside_project(store, roots, tmp_path):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    project_dir = frames_dir(roots, "alice", JOB_A).parent
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project_dir / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["video_path"] = "escape/video.mp4"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnimationStateError, match="video_path"):
        store.load(JOB_A)


def test_begin_and_commit_chunk_advances_in_global_order(store, roots):
    store.create(request(frames=10), owner="alice", role="user", job_id=JOB_A)
    begun = store.begin_chunk(JOB_A, 0)
    names = [f"frame_{index:06d}.png" for index in range(8)]
    write_frames(roots, "alice", JOB_A, names)

    committed = store.commit_chunk(JOB_A, 0, names)

    assert begun.status == "running"
    assert begun.active_chunk_index == 0
    assert committed.frame_files == [f"frames/{name}" for name in names]
    assert committed.completed_frames == 8
    assert committed.completed_chunks == 1
    assert committed.next_chunk_index == 1
    assert committed.active_chunk_index is None
    assert committed.status == "queued"

    store.begin_chunk(JOB_A, 1)
    final_names = ["frame_000008.png", "frame_000009.png"]
    write_frames(roots, "alice", JOB_A, final_names)
    final = store.commit_chunk(JOB_A, 1, final_names)
    assert final.completed_frames == 10
    assert final.status == "finalizing"


@pytest.mark.parametrize("attack", ["replace", "remove", "link"])
def test_commit_staged_frames_survives_source_change_before_publication(
    store, roots, tmp_path, monkeypatch, attack
):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    name = "frame_000000.png"
    write_staging_frames(roots, "alice", JOB_A, [name])
    source = frames_dir(roots, "alice", JOB_A).parent / "staging" / name
    original = source.read_bytes()

    outside = tmp_path / "outside-frame.png"
    outside.write_bytes(b"outside")

    def change_source():
        source.unlink()
        if attack == "replace":
            source.write_bytes(b"attacker replacement")
        elif attack == "link":
            source.symlink_to(outside)

    monkeypatch.setattr(store, "_before_frame_publish", change_source)
    committed = store.commit_chunk(JOB_A, 0, [name])

    assert committed.frame_files == [f"frames/{name}"]
    assert (frames_dir(roots, "alice", JOB_A) / name).read_bytes() == original
    assert store.load(JOB_A).completed_frames == 1
    assert outside.read_bytes() == b"outside"


def test_commit_rejects_existing_canonical_file_as_source(store, roots):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    canonical = frames_dir(roots, "alice", JOB_A) / "frame_000000.png"
    canonical.write_bytes(b"stale")
    with pytest.raises(FileNotFoundError):
        store.commit_chunk(JOB_A, 0, [canonical.name])
    assert store.load(JOB_A).completed_frames == 0


def test_commit_persists_frame_integrity_and_verifies_hash(store, roots):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    name = "frame_000000.png"
    write_staging_frames(roots, "alice", JOB_A, [name])
    expected = (frames_dir(roots, "alice", JOB_A).parent / "staging" / name).read_bytes()
    project = store.commit_chunk(JOB_A, 0, [name])

    assert project.frame_integrity == [{
        "path": f"frames/{name}",
        "sha256": hashlib.sha256(expected).hexdigest(),
        "size": len(expected),
    }]
    assert store.verify_frame_integrity(JOB_A) is True
    (frames_dir(roots, "alice", JOB_A) / name).write_bytes(b"tampered")
    with pytest.raises(AnimationStateError, match="integrity"):
        store.verify_frame_integrity(JOB_A)
    with pytest.raises(AnimationStateError, match="integrity"):
        store.prepare_recovery(JOB_A)


def test_commit_detects_in_place_staging_mutation_during_copy(
    store, roots, monkeypatch
):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    name = "frame_000000.png"
    write_staging_frames(roots, "alice", JOB_A, [name])

    def mutate(source):
        source.write_bytes(b"in-place mutation")

    monkeypatch.setattr(store, "_after_frame_copy", mutate)
    with pytest.raises(ValueError, match="changed while copying"):
        store.commit_chunk(JOB_A, 0, [name])
    assert store.load(JOB_A).completed_frames == 0


def test_commit_cleanup_failure_is_successful_and_reconcile_retries(
    store, roots, monkeypatch
):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    name = "frame_000000.png"
    write_staging_frames(roots, "alice", JOB_A, [name])
    real_unlink = Path.unlink

    def locked(path, *args, **kwargs):
        if path.name.endswith(".journal.json") or path.name == name:
            raise PermissionError("locked cleanup")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    project = store.commit_chunk(JOB_A, 0, [name])
    assert project.completed_frames == 1
    monkeypatch.setattr(Path, "unlink", real_unlink)
    store.reconcile_staging()
    assert not list(roots[0].glob(".frame-*.journal.json"))
    assert not (frames_dir(roots, "alice", JOB_A).parent / "staging" / name).exists()


def test_reconcile_rolls_back_uncommitted_frame_journal(store, roots, monkeypatch):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    name = "frame_000000.png"
    write_staging_frames(roots, "alice", JOB_A, [name])
    monkeypatch.setattr(
        store,
        "_before_state_publish",
        lambda: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with pytest.raises(SystemExit):
        store.commit_chunk(JOB_A, 0, [name])
    canonical = frames_dir(roots, "alice", JOB_A) / name
    assert canonical.exists()
    assert list(roots[0].glob(".frame-*.journal.json"))

    store.reconcile_staging()
    assert not canonical.exists()
    assert (canonical.parent.parent / "staging" / name).exists()
    assert not list(roots[0].glob(".frame-*.journal.json"))
    monkeypatch.setattr(store, "_before_state_publish", lambda: None)
    retried = store.commit_chunk(JOB_A, 0, [name])
    assert retried.completed_frames == 1


@pytest.mark.parametrize("case", ["missing", "wrong_count", "duplicate", "outside"])
def test_commit_rejects_invalid_frames_without_partial_state(store, roots, case):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    names = ["frame_000000.png", "frame_000001.png", "frame_000002.png"]
    write_frames(roots, "alice", JOB_A, names)
    before = snapshot(store, JOB_A)
    if case == "missing":
        (
            frames_dir(roots, "alice", JOB_A).parent / "staging" / names[1]
        ).unlink()
    elif case == "wrong_count":
        names = names[:2]
    elif case == "duplicate":
        names[2] = names[1]
    else:
        names[2] = "../outside.png"

    with pytest.raises((ValueError, FileNotFoundError), match=r".+"):
        store.commit_chunk(JOB_A, 0, names)
    assert snapshot(store, JOB_A) == before


def test_commit_rejects_symlink_frame_without_mutation(store, roots, tmp_path):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    link = (
        frames_dir(roots, "alice", JOB_A).parent
        / "staging"
        / "frame_000000.png"
    )
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    before = snapshot(store, JOB_A)

    with pytest.raises(ValueError, match="symlink|frame"):
        store.commit_chunk(JOB_A, 0, [link.name])
    assert snapshot(store, JOB_A) == before


def test_load_rejects_committed_frame_replaced_by_symlink(
    store, roots, tmp_path
):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    write_frames(roots, "alice", JOB_A, ["frame_000000.png"])
    store.commit_chunk(JOB_A, 0, ["frame_000000.png"])
    frame = frames_dir(roots, "alice", JOB_A) / "frame_000000.png"
    frame.unlink()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    try:
        frame.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(AnimationStateError, match="frame"):
        store.load(JOB_A)


def test_stale_wrong_and_double_chunk_operations_do_not_mutate(store, roots):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    original = snapshot(store, JOB_A)
    with pytest.raises(ValueError, match="next chunk"):
        store.begin_chunk(JOB_A, 1)
    assert snapshot(store, JOB_A) == original

    store.begin_chunk(JOB_A, 0)
    active = snapshot(store, JOB_A)
    with pytest.raises(ValueError, match="active"):
        store.begin_chunk(JOB_A, 0)
    with pytest.raises(ValueError, match="active"):
        store.commit_chunk(JOB_A, 1, [])
    assert snapshot(store, JOB_A) == active

    write_frames(roots, "alice", JOB_A, ["frame_000000.png"])
    store.commit_chunk(JOB_A, 0, ["frame_000000.png"])
    complete = snapshot(store, JOB_A)
    with pytest.raises(ValueError, match=r"active|stale"):
        store.commit_chunk(JOB_A, 0, ["frame_000000.png"])
    assert snapshot(store, JOB_A) == complete


def test_lifecycle_transitions_are_controlled_and_errors_are_bounded(store):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    assert store.mark_status(JOB_A, "running").status == "running"
    errored = store.mark_status(JOB_A, "error", " bad\x00message " + "x" * 5000)
    assert errored.status == "error"
    assert "\x00" not in errored.error
    assert len(errored.error) <= 1024
    assert errored.active_chunk_index is None
    with pytest.raises(ValueError, match=r"transition|terminal"):
        store.mark_status(JOB_A, "done")
    with pytest.raises(ValueError, match="status"):
        store.mark_status(JOB_A, "unknown")

    store.create(request(), owner="bob", role="user", job_id=JOB_B)
    store.begin_chunk(JOB_B, 0)
    cancelled = store.mark_status(JOB_B, "cancelled")
    assert cancelled.active_chunk_index is None
    assert cancelled.completed_frames == 0
    with pytest.raises(ValueError, match="terminal"):
        store.mark_status(JOB_B, "queued")


def test_running_to_queued_clears_active_chunk_atomically(store):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)

    queued = store.mark_status(JOB_A, "queued")

    assert queued.status == "queued"
    assert queued.active_chunk_index is None
    assert store.load(JOB_A).active_chunk_index is None


def test_prepare_recovery_clears_active_or_moves_complete_to_finalizing(store, roots):
    store.create(request(frames=3), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    recovered = store.prepare_recovery(JOB_A)
    assert (recovered.status, recovered.active_chunk_index) == ("queued", None)
    assert recovered.completed_frames == 0
    assert store.prepare_recovery(JOB_A).status == "queued"

    store.create(request(frames=1), owner="bob", role="user", job_id=JOB_B)
    store.begin_chunk(JOB_B, 0)
    write_frames(roots, "bob", JOB_B, ["frame_000000.png"])
    store.commit_chunk(JOB_B, 0, ["frame_000000.png"])
    assert store.prepare_recovery(JOB_B).status == "finalizing"


def test_recoverable_is_sorted_and_excludes_error_and_terminals(store):
    ids = [
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "cccccccc-cccc-cccc-cccc-cccccccccccc",
        "dddddddd-dddd-dddd-dddd-dddddddddddd",
    ]
    for job_id in ids:
        store.create(request(), owner="alice", role="user", job_id=job_id)
    store.mark_status(ids[0], "running")
    store.mark_status(ids[2], "error", "failed")
    store.mark_status(ids[3], "blocked")

    assert [project.job_id for project in store.recoverable()] == sorted(ids[:2])


def test_recoverable_isolates_project_with_corrupt_frame_integrity(store, roots):
    corrupt_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    valid_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    store.create(
        request(frames=1), owner="alice", role="user", job_id=corrupt_id
    )
    store.begin_chunk(corrupt_id, 0)
    write_staging_frames(
        roots, "alice", corrupt_id, ["frame_000000.png"]
    )
    store.commit_chunk(corrupt_id, 0, ["frame_000000.png"])
    (
        frames_dir(roots, "alice", corrupt_id) / "frame_000000.png"
    ).write_bytes(b"corrupt")
    store.create(request(), owner="alice", role="user", job_id=valid_id)

    assert [project.job_id for project in store.recoverable()] == [valid_id]


def test_reconcile_staging_only_removes_temps_and_staging(store, roots):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    write_frames(roots, "alice", JOB_A, ["frame_000000.png"])
    store.begin_chunk(JOB_A, 0)
    store.commit_chunk(JOB_A, 0, ["frame_000000.png"])
    staging = frames_dir(roots, "alice", JOB_A).parent / "staging"
    (staging / "nested").mkdir()
    (staging / "nested" / "partial.png").write_bytes(b"x")
    (roots[0] / ".orphan.123.tmp").write_text("tmp", encoding="utf-8")
    (roots[0] / f"{JOB_A}.json.tmp").write_text("old temp", encoding="utf-8")
    video = frames_dir(roots, "alice", JOB_A).parent / "video.mp4"
    video.write_bytes(b"video")

    removed = store.reconcile_staging()

    assert removed == sorted(removed)
    assert any("partial.png" in item for item in removed)
    assert any(".orphan.123.tmp" in item for item in removed)
    assert any(f"{JOB_A}.json.tmp" in item for item in removed)
    assert (frames_dir(roots, "alice", JOB_A) / "frame_000000.png").is_file()
    assert video.is_file()
    current = store.load(JOB_A)
    assert current.status == "finalizing"
    assert current.completed_frames == current.completed_chunks == 1
    assert current.revision == 2
    assert current.frame_files == ["frames/frame_000000.png"]
    assert len(current.frame_integrity) == 1


def test_reconcile_staging_does_not_follow_external_directory_links(
    store, roots, tmp_path
):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    staging = frames_dir(roots, "alice", JOB_A).parent / "staging"
    outside = tmp_path / "outside-staging"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    link = staging / "external"
    make_directory_link(link, outside)

    removed = store.reconcile_staging()

    assert any(item.endswith("/staging/external") for item in removed)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not link.exists()


def test_reconcile_does_not_follow_outside_windows_junction(
    store, roots, tmp_path
):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    staging = frames_dir(roots, "alice", JOB_A).parent / "staging"
    outside = tmp_path / "junction-outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    link = staging / "external-junction"
    make_windows_junction(link, outside)

    store.reconcile_staging()

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not os.path.lexists(link)


def test_delete_enforces_owner_and_admin_access(store, roots):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    project_dir = frames_dir(roots, "alice", JOB_A).parent

    foreign_deleted = store.delete(JOB_A, username="bob")
    assert foreign_deleted is False
    assert project_dir.is_dir()
    assert (roots[0] / f"{JOB_A}.json").is_file()
    owner_deleted = store.delete(JOB_A, username="alice")
    assert owner_deleted is True
    assert not project_dir.exists()
    assert not (roots[0] / f"{JOB_A}.json").exists()

    store.create(request(), owner="alice", role="user", job_id=JOB_B)
    admin_deleted = store.delete(JOB_B, username="root", is_admin=True)
    assert admin_deleted is True
    assert not frames_dir(roots, "alice", JOB_B).parent.exists()

    missing_owner_deleted = store.delete(JOB_A, username="alice")
    missing_foreign_deleted = store.delete(JOB_A, username="mallory")
    assert missing_owner_deleted is False
    assert missing_foreign_deleted is False


def test_delete_does_not_follow_external_directory_links(store, roots, tmp_path):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    project_dir = frames_dir(roots, "alice", JOB_A).parent
    outside = tmp_path / "outside-delete"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    make_directory_link(project_dir / "staging" / "external", outside)

    deleted = store.delete(JOB_A, username="alice")
    assert deleted is True
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not project_dir.exists()


def test_delete_project_junction_cannot_delete_another_project(
    store, roots
):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    store.create(request(), owner="alice", role="user", job_id=JOB_B)
    project_a = frames_dir(roots, "alice", JOB_A).parent
    project_b = frames_dir(roots, "alice", JOB_B).parent
    animation_state.AnimationStore._remove_without_following(project_a)
    sentinel = project_b / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    make_windows_junction(project_a, project_b)

    deleted = store.delete(JOB_A, username="alice")
    assert deleted is True
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert store.load(JOB_B).job_id == JOB_B


def test_create_rejects_owner_junction_without_touching_target(
    store, roots, tmp_path
):
    outside = tmp_path / "owner-junction-target"
    outside.mkdir()
    owner_link = frames_dir(roots, "alice", JOB_A).parents[2]
    make_windows_junction(owner_link, outside)

    with pytest.raises(ValueError, match=r"link|reparse"):
        store.create(request(), owner="alice", role="user", job_id=JOB_A)

    assert list(outside.iterdir()) == []


def test_delete_tombstones_before_best_effort_cleanup(
    store, roots, monkeypatch
):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    original_remove = store._remove_without_following

    def locked_cleanup(path):
        if ".trash-" in path.name:
            raise PermissionError("locked")
        return original_remove(path)

    monkeypatch.setattr(store, "_remove_without_following", locked_cleanup)
    first_delete = store.delete(JOB_A, username="alice")
    second_delete = store.delete(JOB_A, username="alice")
    assert first_delete is True
    assert second_delete is True
    with pytest.raises(FileNotFoundError):
        store.load(JOB_A)
    assert any(path.name.startswith(".trash-") for path in roots[1].rglob("*"))
    assert any(path.name.startswith(".deleted-") for path in roots[0].iterdir())

    monkeypatch.setattr(store, "_remove_without_following", original_remove)
    store.reconcile_staging()
    assert not any(path.name.startswith(".trash-") for path in roots[1].rglob("*"))
    assert not any(path.name.startswith(".deleted-") for path in roots[0].iterdir())
    assert store.reconcile_staging() == []


def test_deleted_project_journal_is_reconciled_without_touching_other_job(
    store, roots, monkeypatch
):
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    name = "frame_000000.png"
    write_staging_frames(roots, "alice", JOB_A, [name])
    real_unlink = Path.unlink

    def leave_journal(path, *args, **kwargs):
        if path.name.startswith(f".frame-{JOB_A}-"):
            raise PermissionError("journal locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", leave_journal)
    store.commit_chunk(JOB_A, 0, [name])
    journal = next(roots[0].glob(f".frame-{JOB_A}-*.journal.json"))
    unrelated = roots[0] / f".frame-{JOB_B}-unrelated.journal.json"
    unrelated.write_text("not a controlled journal", encoding="utf-8")
    deleted = store.delete(JOB_A, username="alice")
    assert deleted is True
    assert journal.exists()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    store.reconcile_staging()

    assert not journal.exists()
    assert unrelated.read_text(encoding="utf-8") == "not a controlled journal"


def test_delete_matches_embedded_journal_job_not_filename_prefix(
    store, roots, monkeypatch
):
    short_id = "abc"
    prefixed_id = "abc-def"
    real_unlink = Path.unlink

    def leave_journals(path, *args, **kwargs):
        if path.name.endswith(".journal.json"):
            raise PermissionError("journal locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", leave_journals)
    for job_id in (short_id, prefixed_id):
        store.create(
            request(frames=1), owner="alice", role="user", job_id=job_id
        )
        store.begin_chunk(job_id, 0)
        write_staging_frames(
            roots, "alice", job_id, ["frame_000000.png"]
        )
        store.commit_chunk(job_id, 0, ["frame_000000.png"])

    journals = {}
    for path in roots[0].glob(".frame-*.journal.json"):
        journals[json.loads(path.read_text(encoding="utf-8"))["job_id"]] = path
    assert set(journals) == {short_id, prefixed_id}

    monkeypatch.setattr(Path, "unlink", real_unlink)
    deleted = store.delete(short_id, username="alice")
    assert deleted is True

    assert not journals[short_id].exists()
    assert journals[prefixed_id].exists()
    assert store.load(prefixed_id).job_id == prefixed_id
    assert prefixed_id in {
        project.job_id for project in store.recoverable()
    }


def test_delete_failure_before_tombstone_leaves_project_loadable(
    store, monkeypatch
):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    monkeypatch.setattr(
        "backend.animation_state.os.replace",
        lambda *_: (_ for _ in ()).throw(PermissionError("rename denied")),
    )
    with pytest.raises(PermissionError, match="rename denied"):
        store.delete(JOB_A, username="alice")
    assert store.load(JOB_A).job_id == JOB_A


def test_threads_update_distinct_projects_and_serialize_same_project(store):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    store.create(request(), owner="bob", role="user", job_id=JOB_B)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda job_id: store.mark_status(job_id, "running").job_id,
                [JOB_A, JOB_B],
            )
        )
    assert set(results) == {JOB_A, JOB_B}

    store.mark_status(JOB_A, "queued")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(store.begin_chunk, JOB_A, 0) for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result().active_chunk_index)
        except ValueError:
            outcomes.append("rejected")
    assert sorted(outcomes, key=str) == [0, "rejected"]
    assert store.load(JOB_A).active_chunk_index == 0


def test_separate_store_instances_share_root_lock_and_serialize(roots):
    first = AnimationStore(*roots)
    second = AnimationStore(*roots)
    first.create(request(), owner="alice", role="user", job_id=JOB_A)

    assert first._lock is second._lock
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(candidate.begin_chunk, JOB_A, 0)
            for candidate in (first, second)
        ]
    successes = 0
    for future in futures:
        try:
            future.result()
            successes += 1
        except ValueError:
            pass
    assert successes == 1


def test_stale_save_cannot_overwrite_terminal_state(roots):
    first = AnimationStore(*roots)
    second = AnimationStore(*roots)
    created = first.create(request(), owner="alice", role="user", job_id=JOB_A)
    stale = second.load(JOB_A)
    cancelled = first.mark_status(JOB_A, "cancelled")
    assert cancelled.revision == 1
    stale.status = "running"
    with pytest.raises(AnimationStateError, match=r"stale|revision"):
        second.save(stale)
    assert first.load(JOB_A).status == "cancelled"
    assert created.revision == 0


def test_two_store_status_cancel_race_preserves_terminal_state(roots):
    first = AnimationStore(*roots)
    second = AnimationStore(*roots)
    first.create(request(), owner="alice", role="user", job_id=JOB_A)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(candidate.mark_status, JOB_A, status)
            for candidate, status in ((first, "running"), (second, "cancelled"))
        ]
    assert any(future.exception() is None for future in futures)
    result = first.load(JOB_A)
    assert result.status == "cancelled"
    assert result.revision in {1, 2}


def test_separate_stores_serialize_create_commit_and_status(roots):
    first = AnimationStore(*roots)
    second = AnimationStore(*roots)
    with ThreadPoolExecutor(max_workers=2) as pool:
        creates = [
            pool.submit(
                candidate.create,
                request(frames=1),
                owner="alice",
                role="user",
                job_id=JOB_A,
            )
            for candidate in (first, second)
        ]
    assert sum(future.exception() is None for future in creates) == 1

    first.begin_chunk(JOB_A, 0)
    write_staging_frames(roots, "alice", JOB_A, ["frame_000000.png"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        commits = [
            pool.submit(
                candidate.commit_chunk,
                JOB_A,
                0,
                ["frame_000000.png"],
            )
            for candidate in (first, second)
        ]
    assert sum(future.exception() is None for future in commits) == 1
    assert first.load(JOB_A).completed_frames == 1

    first.create(request(), owner="bob", role="user", job_id=JOB_B)
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = [
            pool.submit(candidate.mark_status, JOB_B, status)
            for candidate, status in ((first, "running"), (second, "blocked"))
        ]
    assert any(future.exception() is None for future in statuses)
    assert first.load(JOB_B).status == "blocked"


def test_subprocesses_cannot_double_begin_same_chunk(roots, tmp_path):
    store = AnimationStore(*roots)
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    start = tmp_path / "start"
    script = textwrap.dedent(
        """
        import sys, time
        from pathlib import Path
        from backend.animation_state import AnimationStore
        state, outputs, start, job = map(Path, sys.argv[1:5])
        while not start.exists():
            time.sleep(0.005)
        try:
            AnimationStore(state, outputs).begin_chunk(str(job), 0)
        except ValueError:
            print("rejected")
        else:
            print("ok")
        """
    )
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(roots[0]),
                str(roots[1]),
                str(start),
                JOB_A,
            ],
            cwd=Path(__file__).parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    start.touch()
    outputs = [process.communicate(timeout=15) for process in processes]
    assert [stdout.strip() for stdout, _ in outputs].count("ok") == 1
    assert all(process.returncode == 0 for process in processes)
    assert store.load(JOB_A).revision == 1


def test_subprocess_crash_after_frame_publication_reconciles_for_retry(roots):
    store = AnimationStore(*roots)
    store.create(request(frames=1), owner="alice", role="user", job_id=JOB_A)
    store.begin_chunk(JOB_A, 0)
    name = "frame_000000.png"
    write_staging_frames(roots, "alice", JOB_A, [name])
    script = textwrap.dedent(
        """
        import os, sys
        from pathlib import Path
        from backend.animation_state import AnimationStore
        state, outputs = map(Path, sys.argv[1:3])
        store = AnimationStore(state, outputs)
        store._before_state_publish = lambda: os._exit(77)
        store.commit_chunk(sys.argv[3], 0, ["frame_000000.png"])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(roots[0]), str(roots[1]), JOB_A],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert result.returncode == 77
    canonical = frames_dir(roots, "alice", JOB_A) / name
    assert canonical.exists()

    restarted = AnimationStore(*roots)
    restarted.reconcile_staging()
    assert not canonical.exists()
    assert restarted.commit_chunk(JOB_A, 0, [name]).completed_frames == 1


def test_failed_replace_leaves_previous_state_readable(store, monkeypatch):
    project = store.create(request(), owner="alice", role="user", job_id=JOB_A)
    monkeypatch.setattr("backend.animation_state.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(OSError, match="disk"):
        store.mark_status(JOB_A, "blocked")
    assert store.load(JOB_A).to_dict() == project.to_dict()


def test_failed_save_does_not_mutate_caller_or_disk(store, monkeypatch):
    project = store.create(request(), owner="alice", role="user", job_id=JOB_A)
    original = project.to_dict()
    monkeypatch.setattr(
        "backend.animation_state.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("disk")),
    )

    with pytest.raises(OSError, match="disk"):
        store.save(project)

    assert project.to_dict() == original
    assert store.load(JOB_A).to_dict() == original


def test_temp_cleanup_failure_never_masks_primary_write_error(
    store, roots, monkeypatch
):
    project = store.create(request(), owner="alice", role="user", job_id=JOB_A)
    monkeypatch.setattr(
        "backend.animation_state.json.dump",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("primary")),
    )
    real_unlink = Path.unlink

    def locked_unlink(path, *args, **kwargs):
        if path.name.endswith(".tmp"):
            raise PermissionError("cleanup")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    with pytest.raises(OSError, match="primary"):
        store.save(project)
    assert list(roots[0].glob(f".{JOB_A}.*.tmp"))


def test_job_ids_are_case_canonical_and_collide_consistently(store, roots):
    uppercase = JOB_A.upper()
    project = store.create(
        request(), owner="alice", role="user", job_id=uppercase
    )
    assert project.job_id == JOB_A
    assert store.load(uppercase).job_id == JOB_A
    assert (roots[0] / f"{JOB_A}.json").is_file()
    with pytest.raises((FileExistsError, AnimationStateError)):
        store.create(request(), owner="alice", role="user", job_id=JOB_A)


@pytest.mark.parametrize(
    ("created", "updated"),
    [
        ("2026-01-01T00:00:00", "2026-01-01T00:00:01+00:00"),
        ("2026-01-01T00:00:00+01:00", "2026-01-01T00:00:01+01:00"),
        ("2026-01-01T00:00:02+00:00", "2026-01-01T00:00:01+00:00"),
    ],
)
def test_loaded_state_requires_ordered_utc_timestamps(
    store, roots, created, updated
):
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    path = roots[0] / f"{JOB_A}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["created_at"] = created
    payload["updated_at"] = updated
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnimationStateError, match="timestamp"):
        store.load(JOB_A)


def test_failed_write_leaves_previous_state_and_cleans_temp(
    store, roots, monkeypatch
):
    project = store.create(request(), owner="alice", role="user", job_id=JOB_A)

    def failed_dump(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr("backend.animation_state.json.dump", failed_dump)
    with pytest.raises(OSError, match="write failed"):
        store.mark_status(JOB_A, "blocked")

    assert store.load(JOB_A).to_dict() == project.to_dict()
    assert list(roots[0].glob(f".{JOB_A}.*.tmp")) == []


def test_unique_temp_names_prevent_collision(store, roots, monkeypatch):
    seen = []
    real_replace = os.replace

    def recording_replace(source, destination):
        seen.append(Path(source).name)
        real_replace(source, destination)

    monkeypatch.setattr("backend.animation_state.os.replace", recording_replace)
    store.create(request(), owner="alice", role="user", job_id=JOB_A)
    store.create(request(), owner="bob", role="user", job_id=JOB_B)

    assert len(seen) == len(set(seen)) == 2
    assert all(name.startswith(".") and name.endswith(".tmp") for name in seen)


def test_directory_fsync_is_best_effort_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(animation_state.os, "open", lambda *args: (_ for _ in ()).throw(OSError("unsupported")))
    animation_state._fsync_directory(tmp_path)

    closed = []
    monkeypatch.setattr(animation_state.os, "open", lambda *args: 99)
    monkeypatch.setattr(animation_state.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("unsupported")))
    monkeypatch.setattr(animation_state.os, "close", closed.append)
    animation_state._fsync_directory(tmp_path)
    assert closed == [99]
