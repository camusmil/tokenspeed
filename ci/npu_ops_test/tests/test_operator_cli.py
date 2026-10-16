# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Exercise mode selection with real callbacks, subprocesses, and operator plans."""

import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from changes import load_manifest
from operator_cli import run_modes


class OperatorCliTests(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.source = Path(__file__).resolve().parents[3]
        self.suite = self.source / "ci/npu_ops_test"

    def invoke(self, mode, accuracy_exit, performance_exit, raise_accuracy):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.txt"
            script = root / "mode_fixture.py"
            script.write_text(
                "import os, sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {str(self.suite)!r})\n"
                "from operator_cli import parse_mode, run_modes\n"
                f"trace = Path({str(trace)!r})\n"
                "def accuracy():\n"
                "    with trace.open('a') as handle: handle.write('accuracy\\n')\n"
                "    if os.environ['RAISE_ACCURACY'] == '1': raise RuntimeError('accuracy failed')\n"
                "    return int(os.environ['ACCURACY_EXIT'])\n"
                "def performance():\n"
                "    with trace.open('a') as handle: handle.write('performance\\n')\n"
                "    return int(os.environ['PERFORMANCE_EXIT'])\n"
                "sys.exit(run_modes(parse_mode(None, 'CPU operator fixture'), accuracy, performance))\n"
            )
            result = subprocess.run(
                [sys.executable, "-B", str(script), "--mode", mode],
                env=dict(
                    os.environ,
                    ACCURACY_EXIT=str(accuracy_exit),
                    PERFORMANCE_EXIT=str(performance_exit),
                    RAISE_ACCURACY=str(int(raise_accuracy)),
                ),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result, trace.read_text().splitlines() if trace.exists() else []

    def test_modes_select_only_requested_callbacks(self):
        for mode, expected in (
            ("accuracy", ["accuracy"]),
            ("performance", ["performance"]),
            ("all", ["accuracy", "performance"]),
        ):
            with self.subTest(mode=mode):
                result, trace = self.invoke(mode, 0, 0, False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(trace, expected)

    def test_all_stops_after_accuracy_nonzero(self):
        result, trace = self.invoke("all", 7, 0, False)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(trace, ["accuracy"])

    def test_all_stops_after_accuracy_exception(self):
        result, trace = self.invoke("all", 0, 0, True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RuntimeError: accuracy failed", result.stderr)
        self.assertEqual(trace, ["accuracy"])

    def test_all_propagates_performance_failure(self):
        result, trace = self.invoke("all", 0, 9, False)
        self.assertEqual(result.returncode, 9)
        self.assertEqual(trace, ["accuracy", "performance"])

    def test_performance_mode_does_not_run_accuracy(self):
        result, trace = self.invoke("performance", 7, 0, True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace, ["performance"])

    def test_invalid_callback_result_stops_all(self):
        completed = []

        def accuracy():
            completed.append("accuracy")
            return None

        def performance():
            completed.append("performance")
            return 0

        with self.assertRaises(TypeError):
            run_modes("all", accuracy, performance)
        self.assertEqual(completed, ["accuracy"])

    def test_unknown_mode_never_invokes_callbacks(self):
        completed = []

        def phase():
            completed.append("unexpected")
            return 0

        with self.assertRaises(ValueError):
            run_modes("unknown", phase, phase)
        self.assertEqual(completed, [])

    def test_registered_scripts_parse_arguments_without_npu_dependencies(self):
        manifest = load_manifest(self.suite / "manifest.json")
        for entry in manifest["operators"]:
            script = self.source / entry["script"]
            for arguments, expected in (
                (["--help"], 0),
                ([], 2),
                (["--mode", "invalid"], 2),
                (["--mode", "accuracy", "--unexpected"], 2),
            ):
                with self.subTest(operator=entry["name"], arguments=arguments):
                    # No site packages: argparse must finish before importing torch.
                    result = subprocess.run(
                        [sys.executable, "-B", "-S", str(script), *arguments],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertIn("--mode", result.stdout + result.stderr)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_registered_cases_configs_and_timing_match_expected_plan(self):
        manifest = load_manifest(self.suite / "manifest.json")
        plan = json.loads((self.suite / "expected-plans.json").read_text())
        for entry in manifest["operators"]:
            with self.subTest(operator=entry["name"]):
                module = runpy.run_path(
                    str(self.source / entry["script"]), run_name="operator_plan"
                )
                expected = plan["operators"][entry["name"]]
                self.assertEqual(
                    module["cases"](), [item["case"] for item in expected["accuracy"]]
                )
                self.assertEqual(
                    module["configs"](),
                    [item["config"] for item in expected["performance"]],
                )
                self.assertEqual(
                    {
                        "warmup": module["WARMUP"],
                        "iterations": module["ITERS"],
                        "sample_count": module["SAMPLES"],
                    },
                    expected["timing"],
                )

    def test_event_emitter_preserves_phase_and_original_fields(self):
        manifest = load_manifest(self.suite / "manifest.json")
        for entry in manifest["operators"]:
            module = runpy.run_path(
                str(self.source / entry["script"]), run_name="operator_events"
            )
            for phase in ("accuracy", "performance"):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    module["emit"](
                        phase,
                        "failed",
                        error="original error \u539f\u59cb",
                        current_case=None,
                    )
                record = json.loads(output.getvalue().removeprefix("NPU_RESULT_JSON="))
                self.assertEqual(
                    record,
                    {
                        "schema_version": 1,
                        "operator": entry["name"],
                        "phase": phase,
                        "event": "failed",
                        "error": "original error \u539f\u59cb",
                        "current_case": None,
                    },
                )


if __name__ == "__main__":
    unittest.main()
