from __future__ import annotations

import ast
import math
import operator
import re
import secrets
from collections.abc import Sequence

import numpy as np


_MAX_EXPRESSION_LENGTH = 512
_MAX_AST_NODES = 128
_MAX_ABS_VALUE = 1e12
_MAX_EXPONENT = 64
_MAX_TOTAL_FRAMES = 720
# Bounds cover one keyframe per supported frame while keeping hostile input small.
_MAX_SCHEDULE_TEXT_LENGTH = 32 * 1024
_MAX_KEYFRAMES = 720
_UINT64_LIMIT = 1 << 64
_UINT64_MAX = _UINT64_LIMIT - 1
DEFAULT_ANIMATION_CHUNK_SIZE = 8

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def _checked_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("schedule expressions must produce real numbers")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("schedule expression result is too large") from exc
    if not math.isfinite(result):
        raise ValueError("schedule expression result must be finite")
    if abs(result) > _MAX_ABS_VALUE:
        raise ValueError("schedule expression result is too large")
    return result


def _safe_power(base: float, exponent: float) -> float:
    if abs(exponent) > _MAX_EXPONENT:
        raise ValueError("schedule exponent is too large")
    try:
        return _checked_number(pow(base, exponent))
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("schedule"):
            raise
        raise ValueError("schedule power expression is invalid") from exc


def _evaluate_node(node: ast.AST, names: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return _checked_number(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"unknown schedule name: {node.id}")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _checked_number(
            _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, names))
        )
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, names)
        right = _evaluate_node(node.right, names)
        try:
            if isinstance(node.op, ast.Pow):
                return _safe_power(left, right)
            operation = _BINARY_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("unsupported schedule operator")
            return _checked_number(operation(left, right))
        except (ArithmeticError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("schedule"):
                raise
            raise ValueError("schedule arithmetic is invalid") from exc
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in {*_FUNCTIONS, "pow"}
            or node.keywords
        ):
            raise ValueError("unsupported schedule function call")
        arguments = [_evaluate_node(argument, names) for argument in node.args]
        try:
            if node.func.id == "pow":
                if len(arguments) != 2:
                    raise ValueError("pow requires exactly two arguments")
                return _safe_power(arguments[0], arguments[1])
            return _checked_number(_FUNCTIONS[node.func.id](*arguments))
        except TypeError as exc:
            raise ValueError(
                f"invalid arguments for schedule function {node.func.id}"
            ) from exc
        except (ArithmeticError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(
                ("schedule", "pow requires")
            ):
                raise
            raise ValueError(f"schedule function {node.func.id} failed") from exc
    raise ValueError("unsupported syntax in schedule expression")


def _safe_evaluate(expression: str, frame: int, max_frame: int) -> float:
    if not expression or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("schedule expression is empty or too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("schedule expression is malformed") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("schedule expression is too complex")
    return _evaluate_node(
        tree.body, {"t": float(frame), "max_f": float(max_frame), "pi": math.pi}
    )


def _validate_total_frames(total_frames: int) -> None:
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or not 1 <= total_frames <= _MAX_TOTAL_FRAMES
    ):
        raise ValueError("total_frames must be between 1 and 720")


def _validate_schedule_text(text: str, schedule_kind: str) -> None:
    if not isinstance(text, str):
        raise ValueError(f"{schedule_kind} schedule must be text")
    if len(text) > _MAX_SCHEDULE_TEXT_LENGTH:
        raise ValueError(
            f"{schedule_kind} schedule text is too long "
            f"(maximum {_MAX_SCHEDULE_TEXT_LENGTH} characters)"
        )
    if not text.strip():
        raise ValueError(f"{schedule_kind} schedule must not be empty")


def _split_numeric_entries(text: str) -> list[str]:
    entries: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("numeric schedule has unbalanced parentheses")
        elif character == "," and depth == 0:
            if len(entries) >= _MAX_KEYFRAMES:
                raise ValueError(
                    f"numeric schedule has too many keyframes (maximum {_MAX_KEYFRAMES})"
                )
            entries.append(text[start:index].strip())
            start = index + 1
    if depth:
        raise ValueError("numeric schedule has unbalanced parentheses")
    if len(entries) >= _MAX_KEYFRAMES:
        raise ValueError(
            f"numeric schedule has too many keyframes (maximum {_MAX_KEYFRAMES})"
        )
    entries.append(text[start:].strip())
    if any(not entry for entry in entries):
        raise ValueError("numeric schedule contains an empty entry")
    return entries


def evaluate_schedule(text: str, total_frames: int) -> list[float]:
    """Evaluate a safe Deforum numeric schedule for every global frame."""
    _validate_total_frames(total_frames)
    _validate_schedule_text(text, "numeric")

    keyframes: dict[int, float] = {}
    for entry in _split_numeric_entries(text):
        match = re.fullmatch(r"([+-]?\d+)\s*:\s*\((.*)\)", entry, re.DOTALL)
        if match is None:
            raise ValueError(
                "numeric schedule entries must use frame:(expression)"
            )
        frame = int(match.group(1))
        if frame < 0:
            raise ValueError("numeric schedule frame keys cannot be negative")
        if frame >= total_frames:
            raise ValueError("numeric schedule frame key is outside the frame range")
        if frame in keyframes:
            raise ValueError(f"duplicate numeric schedule frame key: {frame}")
        keyframes[frame] = _safe_evaluate(
            match.group(2).strip(), frame, total_frames
        )

    ordered = sorted(keyframes.items())
    values = [ordered[0][1]] * total_frames
    for (left_frame, left_value), (right_frame, right_value) in zip(
        ordered, ordered[1:]
    ):
        span = right_frame - left_frame
        for frame in range(left_frame, right_frame + 1):
            ratio = (frame - left_frame) / span
            values[frame] = left_value + (right_value - left_value) * ratio
    last_frame, last_value = ordered[-1]
    values[last_frame:] = [last_value] * (total_frames - last_frame)
    return values


def parse_prompt_schedule(text: str, total_frames: int) -> list[str]:
    """Expand line-based prompt keyframes using nearest-preceding semantics."""
    _validate_total_frames(total_frames)
    _validate_schedule_text(text, "prompt")

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > _MAX_KEYFRAMES:
        raise ValueError(
            f"prompt schedule has too many keyframes (maximum {_MAX_KEYFRAMES})"
        )
    keyframes: dict[int, str] = {}
    for line in lines:
        if ":" not in line:
            raise ValueError("prompt schedule entries must use frame: prompt")
        frame_text, prompt = line.split(":", 1)
        try:
            frame = int(frame_text.strip())
        except ValueError as exc:
            raise ValueError("prompt schedule frame key must be an integer") from exc
        if frame < 0 or frame >= total_frames:
            raise ValueError("prompt schedule frame key is outside the frame range")
        if frame in keyframes:
            raise ValueError(f"duplicate prompt schedule frame key: {frame}")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError(f"prompt is missing at frame {frame}")
        keyframes[frame] = prompt

    if not keyframes:
        raise ValueError("prompt schedule must contain at least one prompt")
    ordered = sorted(keyframes.items())
    current = ordered[0][1]
    result: list[str] = []
    for frame in range(total_frames):
        if frame in keyframes:
            current = keyframes[frame]
        result.append(current)
    return result


def build_seed_plan(seed: int, behavior: str, total_frames: int) -> list[int]:
    _validate_total_frames(total_frames)
    if seed == -1:
        base = secrets.randbits(32)
    elif 0 <= seed <= _UINT64_MAX:
        base = seed
    else:
        raise ValueError("seed must be -1 or an unsigned 64-bit integer")

    if behavior == "fixed":
        return [base % _UINT64_LIMIT] * total_frames
    if behavior == "iter":
        return [(base + frame) % _UINT64_LIMIT for frame in range(total_frames)]
    if behavior == "ladder":
        return [
            (base + (1000 if frame % 2 else 0)) % _UINT64_LIMIT
            for frame in range(total_frames)
        ]
    if behavior == "random":
        generator = np.random.default_rng(base)
        values = generator.integers(0, (1 << 32) - 1, size=total_frames)
        return [int(value) % _UINT64_LIMIT for value in values]
    raise ValueError(f"unknown seed behavior: {behavior}")


def build_chunk_ranges(
    total_frames: int, requested_size: int, cadence: int
) -> list[tuple[int, int]]:
    _validate_total_frames(total_frames)
    if requested_size < 1:
        raise ValueError("requested_size must be at least 1")
    if cadence < 1:
        raise ValueError("cadence must be at least 1")
    chunk_size = ((requested_size + cadence - 1) // cadence) * cadence
    return [
        (start, min(start + chunk_size, total_frames))
        for start in range(0, total_frames, chunk_size)
    ]


def _validate_slice(values: Sequence[object], start: int, end: int) -> None:
    if len(values) > _MAX_TOTAL_FRAMES:
        raise ValueError("schedule values cannot exceed 720 total frames")
    if start < 0 or end <= start or end > len(values):
        raise ValueError("chunk bounds must select a non-empty valid slice")


def numeric_chunk_schedule(
    values: Sequence[float], start: int, end: int
) -> str:
    _validate_slice(values, start, end)
    entries = []
    for local_frame, value in enumerate(values[start:end]):
        number = _checked_number(value)
        entries.append(f"{local_frame}:({repr(number)})")
    return ", ".join(entries)


def prompt_chunk_schedule(
    values: Sequence[str], start: int, end: int
) -> str:
    _validate_slice(values, start, end)
    entries = []
    output_length = 0
    for local_frame, value in enumerate(values[start:end]):
        if not isinstance(value, str):
            raise ValueError("prompt schedule values must be strings")
        if len(value) > _MAX_SCHEDULE_TEXT_LENGTH:
            raise ValueError("prompt schedule text is too long")
        prompt = " ".join(value.splitlines()).strip()
        if not prompt:
            raise ValueError("prompt schedule values must not be empty")
        entry = f"{local_frame}: {prompt}"
        output_length += len(entry) + (1 if entries else 0)
        if output_length > _MAX_SCHEDULE_TEXT_LENGTH:
            raise ValueError("prompt schedule text is too long")
        entries.append(entry)
    return "\n".join(entries)
