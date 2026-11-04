"""The object-data firewall.

Leakage scores 0 on both Phase 1 components -- 65 of 100 points -- so each layer
is tested against a deliberate violation, not just a clean input.
"""
from __future__ import annotations

import pytest

from agent.firewall import (
    LeakageError,
    assert_frame_name_is_neutral,
    assert_no_engine_identifiers,
    assert_prompt_is_clean,
    sanitize,
)
from agent.loop import SYSTEM, TOOLS


class TestStructuralAllowlist:
    def test_engine_fields_are_stripped_from_tool_results(self):
        raw = {"success": True, "state": "Carry", "object_name": "obj0",
               "object_colour": "red", "distance_to_target": 0.4}
        assert sanitize("pickup", raw) == {"success": True, "state": "Carry"}

    def test_state_keeps_only_the_six_permitted_fields(self):
        raw = {"state": "Idle", "x": 0, "y": 0, "heading": 0, "holding": False,
               "retry_count": 0, "object_count": 5, "nearest_object": "obj2"}
        assert set(sanitize("state", raw)) == {
            "state", "x", "y", "heading", "holding", "retry_count"}

    def test_an_unrecognised_reason_string_is_dropped(self):
        """A reason we did not whitelist could describe what was in reach."""
        out = sanitize("pickup", {"success": False, "reason": "red sphere too far", "state": "X"})
        assert "reason" not in out

    def test_a_known_safe_reason_is_kept(self):
        out = sanitize("pickup", {"success": False, "reason": "nothing_in_reach", "state": "X"})
        assert out["reason"] == "nothing_in_reach"

    def test_an_unknown_tool_is_refused_rather_than_passed_through(self):
        with pytest.raises(LeakageError, match="no allowlist"):
            sanitize("peek_at_scene", {"anything": 1})


class TestStaticPromptGate:
    def test_the_shipped_prompt_and_tools_are_clean(self):
        assert_prompt_is_clean(SYSTEM, TOOLS)      # must not raise

    @pytest.mark.parametrize(
        "prompt",
        ["Find the red sphere and bring it back.",
         "The target is a blue cube on the floor.",
         "Fetch the tallest object in the room."],
        ids=["colour-and-shape", "colour", "superlative"],
    )
    def test_a_hinting_prompt_is_rejected(self, prompt):
        with pytest.raises(LeakageError, match="object-describing words"):
            assert_prompt_is_clean(prompt, [])

    def test_a_hinting_tool_description_is_rejected(self):
        tools = [{"name": "grab", "description": "Pick up the red one.",
                  "input_schema": {"type": "object", "properties": {}}}]
        with pytest.raises(LeakageError):
            assert_prompt_is_clean("Neutral text.", tools)


class TestIdentifierScan:
    @pytest.mark.parametrize(
        "payload",
        [{"note": "nearest is obj3"},
         {"d": {"target": "obj0"}},
         ["step", {"geom": "geom7"}],
         {"zone": "dropzone at centre"},
         {"fields": ["object_positions"]}],
        ids=["flat", "nested", "in-list", "geom-name", "data-field"],
    )
    def test_engine_identifiers_are_caught_anywhere_in_the_payload(self, payload):
        with pytest.raises(LeakageError):
            assert_no_engine_identifiers(payload, "test")

    def test_image_bytes_are_not_text_scanned(self):
        """Pixels are the permitted channel; byte sequences must not trip the scan."""
        assert_no_engine_identifiers({"img": b"obj0 \xff\xd8 raw bytes"}, "test")

    def test_claude_describing_what_it_sees_is_allowed(self):
        """Colour and shape words inferred from pixels are the POINT, not a leak."""
        assert_no_engine_identifiers(
            {"claude_reasoning": "I can see a red sphere on the left"}, "test")


class TestFrameNaming:
    def test_step_indexed_names_are_accepted(self):
        assert_frame_name_is_neutral("step_007.png")

    @pytest.mark.parametrize(
        "name", ["red_sphere.png", "step_007_red.png", "target.png", "step7.png"]
    )
    def test_descriptive_or_malformed_names_are_rejected(self, name):
        with pytest.raises(LeakageError):
            assert_frame_name_is_neutral(name)


class TestBudgetGuard:
    """The brief sets a $40 target with 'no hard block', so nothing outside this
    process stops a runaway loop. The guard must be checked before a call, not
    after, because the money is spent the moment the request goes out."""

    def test_spend_file_round_trips(self, tmp_path, monkeypatch):
        import agent.loop as L

        monkeypatch.setattr(L, "SPEND_FILE", tmp_path / "spend.json")
        assert L._read_spend() == 0.0
        assert L._commit_spend(1.25) == 1.25
        assert L._commit_spend(0.75) == 2.0
        assert L._read_spend() == 2.0

    def test_run_count_increments(self, tmp_path, monkeypatch):
        import json

        import agent.loop as L

        monkeypatch.setattr(L, "SPEND_FILE", tmp_path / "spend.json")
        L._commit_spend(0.1)
        L._commit_spend(0.1)
        assert json.loads((tmp_path / "spend.json").read_text())["runs"] == 2

    def test_a_corrupt_spend_file_reads_as_zero_rather_than_crashing(self, tmp_path, monkeypatch):
        import agent.loop as L

        f = tmp_path / "spend.json"
        f.write_text("{not json")
        monkeypatch.setattr(L, "SPEND_FILE", f)
        assert L._read_spend() == 0.0
