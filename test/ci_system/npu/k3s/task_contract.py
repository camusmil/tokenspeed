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


"""Registered task stages and the matrices shown by the NPU workflow."""

import argparse
import json
from pathlib import Path

STAGES = (
    "operator-accuracy",
    "operator-performance",
    "model-accuracy",
    "model-performance",
)
PROFILES = ("operators", "model")


def stage_plan(profile):
    if profile not in PROFILES:
        raise ValueError("registered NPU profile required")
    if profile == "operators":
        return STAGES[:2]
    if profile == "model":
        return STAGES[2:]


def validate_chain(chain, *, profile, task_id, source_sha, require_success):
    expected = stage_plan(profile)
    if (
        chain.get("task_id") != task_id
        or chain.get("source_sha") != source_sha
        or chain.get("scenario") != profile
    ):
        raise ValueError("task/source/profile evidence mismatch")
    stages = chain.get("stages")
    if not isinstance(stages, list):
        raise ValueError("stage evidence required")
    names = [stage["name"] for stage in stages]
    if names != list(expected[: len(names)]):
        raise ValueError("stage order differs from the registered profile")
    if require_success and (
        chain.get("ok") is not True
        or names != list(expected)
        or any(stage["status"] != "passed" for stage in stages)
    ):
        raise ValueError("required stages incomplete")


def task_matrices(*, selected, runner_label):
    if type(selected) is not bool:
        raise ValueError("explicit boolean selection required")
    if not isinstance(runner_label, str) or not runner_label.strip():
        raise ValueError("explicit nonempty Runner label required")
    groups = {
        "unit": {
            "name": "npu-operators",
            "runner": runner_label,
            "profile": "operators",
        },
        "model": {
            "name": "qwen3-0.6b-smoke",
            "runner": runner_label,
            "profile": "model",
        },
    }
    result = {}
    for stage, task in groups.items():
        result[f"{stage}_matrix"] = {"include": [task] if selected else []}
        result[f"{stage}_has_tasks"] = "true" if selected else "false"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", choices=["true", "false"], required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--runner-label", required=True)
    args = parser.parse_args()
    with args.github_output.open("a") as output:
        for key, value in task_matrices(
            selected=args.selected == "true", runner_label=args.runner_label
        ).items():
            encoded = (
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, dict)
                else value
            )
            output.write(f"{key}={encoded}\n")


if __name__ == "__main__":
    main()
