"""Provider startup regressions; runnable with unittest without a live engine."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import env, loop
from agent.navigation import ReturnNavigator
from agent.vlm import Decision, FallbackVLM


class ProviderRuntimeTests(unittest.TestCase):
    def test_fallback_keeps_rate_limited_models_for_the_next_step(self):
        class Fake:
            name = "gemini"
            prices = {}
            def __init__(self, model, fail):
                self.model = model
                self.fail = fail
                self.calls = 0
            def decide(self, *args, **kwargs):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("HTTP 429: quota")
                return Decision(tool="turn", args={"degrees": 10}, model=self.model)

        a, b = Fake("flash-a", True), Fake("flash-b", False)
        client = FallbackVLM([a, b])
        first = client.decide("s", [], b"", "t")
        self.assertEqual(first.tool, "turn")
        self.assertEqual(a.calls, 1)
        self.assertNotIn("flash-a", client.dead)
        a.fail = False
        second = client.decide("s", [], b"", "t")
        self.assertEqual(second.model, "flash-a")
        self.assertEqual(a.calls, 2)

    def test_fallback_preserves_last_error_when_every_model_is_gone(self):
        class Fake:
            name = "gemini"
            prices = {}
            def __init__(self, model):
                self.model = model
            def decide(self, *args, **kwargs):
                raise RuntimeError("HTTP 404: no longer available")

        client = FallbackVLM([Fake("gone-a"), Fake("gone-b")])
        first = client.decide("s", [], b"", "t")
        self.assertIn("404", first.error)
        second = client.decide("s", [], b"", "t")
        self.assertIn("no remaining models", second.error)

    def test_carry_survey_acquires_marker_then_navigates(self):
        nav = ReturnNavigator()
        pose = {"holding": True, "x": 0.0, "y": 0.0, "heading": 0.0, "state": "Carry"}
        with patch("agent.navigation.marker_from_image", side_effect=[None, (1.2, 0.0)]):
            first = nav.decide(b"jpg", pose)
            self.assertEqual(first[0], "turn")
            second = nav.decide(b"jpg", pose)
        self.assertEqual(nav.goal, (1.2, 0.0))
        self.assertEqual(second[0], "move")

    def test_windows_env_bom_and_explicit_environment_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text('GEMINI_API_KEY="test-file-key"\n', encoding="utf-8-sig")
            with patch.object(env, "ENV_FILES", [path]), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(env.load(), ["GEMINI_API_KEY"])
                self.assertEqual(os.environ["GEMINI_API_KEY"], "test-file-key")
                os.environ["GEMINI_API_KEY"] = "test-exported-key"
                env.load()
                self.assertEqual(os.environ["GEMINI_API_KEY"], "test-exported-key")

    def test_provider_failure_releases_claimed_engine(self):
        with patch.object(sys, "argv", ["agent", "--command", "test", "--provider", "gemini"]), \
             patch.object(loop, "run", side_effect=RuntimeError("HTTP 403: API key rejected")), \
             patch.object(loop, "_release") as release:
            with self.assertRaisesRegex(SystemExit, "HTTP 403"):
                loop.main()
            release.assert_called_once_with()

    def test_unexpected_failure_also_releases_engine(self):
        with patch.object(sys, "argv", ["agent", "--command", "test"]), \
             patch.object(loop, "run", side_effect=ValueError("bad response")), \
             patch.object(loop, "_release") as release:
            with self.assertRaisesRegex(ValueError, "bad response"):
                loop.main()
            release.assert_called_once_with()

    def test_failed_process_output_survives_repeated_status_polls(self):
        # Only the HTTP handlers are under test; never create a GL context.
        with patch("engine.sim.Sim"):
            server = importlib.import_module("engine.server")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            engine_dir = root / "membrane" / "engine"
            engine_dir.mkdir(parents=True)
            process = Mock()
            process.poll.return_value = 1

            def launch(*args, **kwargs):
                kwargs["stdout"].write("HTTP 403: API key rejected\n")
                kwargs["stdout"].flush()
                return process

            state = {"proc": None, "run_id": None, "started": 0.0}
            with patch.object(server, "__file__", str(engine_dir / "server.py")), \
                 patch.object(server, "_agent_proc", state), \
                 patch.object(server.lock, "active", return_value=False), \
                 patch.object(server.subprocess, "Popen", side_effect=launch):
                result = server.admin_agent_run(server.AgentRunBody(command="test"))
                first = server.admin_agent_status(result["run_id"])
                second = server.admin_agent_status(result["run_id"])
                self.assertFalse(first["running"])
                self.assertEqual(first["returncode"], 1)
                self.assertIn("API key rejected", first["stdout_tail"])
                self.assertEqual(first["stdout_tail"], second["stdout_tail"])


if __name__ == "__main__":
    unittest.main()
