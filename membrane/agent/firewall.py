"""The object-data firewall.

The brief: "Do not send object names, IDs, colours, shapes, positions, distances
to objects, object counts, segmentation masks, named-target distances or
drop-zone coordinates to Claude. This applies to /state, system prompts, tool
descriptions, filenames and all context." Any leakage scores 0 on BOTH Phase 1
components -- 65 of 100 points.

Discipline is not a control. This module makes leakage *structurally* hard and
then *provably* absent:

  1. STRUCTURAL  every tool result is rebuilt from a per-tool key allowlist, so
     an engine field can never reach Claude just because someone added it.
  2. STATIC      the system prompt and tool schemas are scanned at startup for
     object-describing vocabulary that would amount to a hint.
  3. DYNAMIC     every payload WE construct is scanned for engine-internal
     identifiers and coordinate patterns before it is sent.

What is deliberately NOT banned: colour and shape words in the user's command
(the command must reach Claude) and in Claude's own replies (it inferred those
from pixels, which is exactly the point).
"""
from __future__ import annotations

import re
from typing import Any, Iterable


class LeakageError(RuntimeError):
    """Raised instead of sending a payload that would score 0."""


# -- 1. STRUCTURAL ---------------------------------------------------------
#: Exactly the keys each endpoint may contribute to Claude's context.
TOOL_RESULT_ALLOWLIST: dict[str, set[str]] = {
    "look": set(),                                              # image only
    "depth": {"distance"},
    "move": {"collided", "distance_travelled", "state", "retry_count"},
    "turn": {"heading", "state"},
    "pickup": {"success", "reason", "state"},
    "drop": {"success", "in_zone", "state"},
    "state": {"state", "x", "y", "heading", "holding", "retry_count"},
}

#: Values of `reason` we are willing to forward. Anything else is dropped rather
#: than risk an engine string describing what was in reach.
SAFE_REASONS = {"already_holding", "nothing_in_reach", "not_holding"}


def sanitize(tool: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a tool result from the allowlist. Unknown keys are dropped."""
    allowed = TOOL_RESULT_ALLOWLIST.get(tool)
    if allowed is None:
        raise LeakageError(f"unknown tool '{tool}' has no allowlist")
    out = {k: v for k, v in raw.items() if k in allowed}
    if "reason" in out and out["reason"] not in SAFE_REASONS:
        out.pop("reason")
    return out


# -- 2/3. VOCABULARY AND IDENTIFIER SCANS ----------------------------------
#: Words that would constitute an object hint if they appeared in text WE wrote
#: (system prompt, tool descriptions). The user's command is exempt.
HINT_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black", "grey", "gray",
    "sphere", "spheres", "ball", "cube", "cubes", "box", "boxes", "cylinder", "cylinders",
    "cone", "cones", "round", "square", "tallest", "shortest", "largest", "smallest",
}

#: Engine-internal identifiers that must never appear anywhere in context.
ID_PATTERNS = [
    re.compile(r"\bobj\d+\b", re.I),
    re.compile(r"\bobj\d+_j\b", re.I),
    re.compile(r"\bdropzone\b", re.I),
    re.compile(r"\bdrop_zone\b", re.I),
    re.compile(r"\bgeom\d+\b", re.I),
    re.compile(r"\bbody_?id\b", re.I),
    re.compile(r"\bsegmentation\b", re.I),
    re.compile(r"\bobject_(count|positions?|ids?|names?|colours?|colors?|shapes?)\b", re.I),
]


def _walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, (bytes, bytearray)):
        return          # image bytes are the permitted channel; never text-scanned


def assert_no_engine_identifiers(payload: Any, where: str) -> None:
    """Scan content WE constructed for engine-internal identifiers."""
    for s in _walk_strings(payload):
        for pat in ID_PATTERNS:
            if pat.search(s):
                raise LeakageError(
                    f"engine identifier matching {pat.pattern!r} found in {where}: {s[:120]!r}"
                )


def assert_prompt_is_clean(system_prompt: str, tool_schemas: list[dict[str, Any]]) -> None:
    """Startup gate: our static text must contain no object vocabulary at all."""
    assert_no_engine_identifiers(system_prompt, "system prompt")
    assert_no_engine_identifiers(tool_schemas, "tool schemas")
    for label, text in [("system prompt", system_prompt),
                        ("tool schemas", " ".join(_walk_strings(tool_schemas)))]:
        found = sorted({w for w in HINT_WORDS if re.search(rf"\b{re.escape(w)}\b", text, re.I)})
        if found:
            raise LeakageError(
                f"object-describing words in {label}: {found}. "
                "Nothing we write may hint at what is in the scene."
            )


def assert_frame_name_is_neutral(filename: str) -> None:
    """The brief lists filenames explicitly. Frames are indexed, never described."""
    if not re.fullmatch(r"step_\d{3,4}\.png", filename):
        raise LeakageError(
            f"frame filename {filename!r} must be step_NNN.png -- a descriptive "
            "filename is itself object data"
        )


def coordinate_leak_check(text: str, forbidden_coords: list[tuple[float, float]]) -> None:
    """Optional belt-and-braces: refuse text containing any object's true
    coordinates at 1 or 2 decimal places."""
    for x, y in forbidden_coords:
        for dp in (1, 2):
            if f"{x:.{dp}f}" in text and f"{y:.{dp}f}" in text:
                raise LeakageError(f"object coordinates ({x}, {y}) appear in outbound text")
