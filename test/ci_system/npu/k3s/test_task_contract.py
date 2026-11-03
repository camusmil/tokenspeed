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


import unittest

from task_contract import stage_plan, task_matrices, validate_chain


class TaskContractTests(unittest.TestCase):
    def test_matrices_select_real_unit_and_model_tasks(self):
        selected = task_matrices(selected=True, runner_label="review-npu-arm64")
        self.assertEqual(selected["unit_matrix"]["include"][0]["profile"], "operators")
        self.assertEqual(selected["model_matrix"]["include"][0]["profile"], "model")
        for stage in ["unit", "model"]:
            self.assertEqual(
                selected[f"{stage}_matrix"]["include"][0]["runner"], "review-npu-arm64"
            )
            self.assertEqual(selected[f"{stage}_has_tasks"], "true")
            skipped = task_matrices(selected=False, runner_label="review-npu-arm64")
            self.assertEqual(skipped[f"{stage}_matrix"], {"include": []})
            self.assertEqual(skipped[f"{stage}_has_tasks"], "false")
        with self.assertRaises(ValueError):
            task_matrices(selected="false", runner_label="review-npu-arm64")

    def test_profiles_execute_their_real_stages(self):
        self.assertEqual(
            stage_plan("operators"), ("operator-accuracy", "operator-performance")
        )
        self.assertEqual(stage_plan("model"), ("model-accuracy", "model-performance"))
        for profile in [
            "normal",
            "operator-failure",
            "model-failure",
            "save-failure",
            "arbitrary-command",
        ]:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                stage_plan(profile)

    def test_runner_label_is_explicit_and_required(self):
        for label in [None, "", "   ", 12]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                task_matrices(selected=True, runner_label=label)

    def test_receipts_require_exact_profile_source_and_stage_order(self):
        chain = {
            "task_id": "task",
            "source_sha": "source",
            "scenario": "operators",
            "ok": True,
            "stages": [
                {"name": n, "status": "passed"} for n in stage_plan("operators")
            ],
        }
        validate_chain(
            chain,
            profile="operators",
            task_id="task",
            source_sha="source",
            require_success=True,
        )
        for changed in [
            dict(chain, source_sha="other"),
            dict(chain, scenario="model"),
            dict(chain, stages=chain["stages"][:1]),
            dict(chain, stages=list(reversed(chain["stages"]))),
            dict(chain, ok=False),
        ]:
            with self.assertRaises(ValueError):
                validate_chain(
                    changed,
                    profile="operators",
                    task_id="task",
                    source_sha="source",
                    require_success=True,
                )
        interrupted = dict(chain, ok=False, stages=chain["stages"][:1])
        validate_chain(
            interrupted,
            profile="operators",
            task_id="task",
            source_sha="source",
            require_success=False,
        )


if __name__ == "__main__":
    unittest.main()
