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

"""Read the single NPU task declaration without importing test code."""

import json
import re
from pathlib import Path

CONFIG_PATH = "test/ci/npu_test/ut-tokenspeed-kernel-npu.json"


def load_task(path):
    from ci_reporting import decode_record, require

    task = decode_record(Path(path).read_text(encoding="utf-8"))
    require(
        type(task.get("schema_version")) is int and task["schema_version"] == 1,
        "Expected task schema_version=1",
    )
    require(task.get("name") == "ut-tokenspeed-kernel-npu", "Invalid task name")
    require(
        task.get("type") == "operator" and task.get("workflow_stage") == "operator",
        "Invalid task type/stage",
    )
    require(
        task.get("triggers") == ["push", "pull_request", "workflow_dispatch"],
        "Invalid triggers",
    )
    require(
        task.get("execution")
        == {"phases": ["accuracy", "performance"], "fail_fast": True},
        "Invalid phase ordering or fail-fast policy",
    )
    require(
        task.get("performance_policy") == {"verdict": "not_evaluated"},
        "Unsupported performance policy",
    )
    require(
        task.get("resources")
        == {"architecture": "aarch64", "device_ids": [0], "exclusive_host": True},
        "Unsupported VM resource request",
    )
    limits = task.get("timeouts")
    require(
        type(limits) is dict
        and set(limits)
        == {"lock_wait_seconds", "phase_execution_seconds", "job_minutes"},
        "Missing timeout limits",
    )
    require(
        all(type(v) is int and v > 0 for v in limits.values()),
        "Timeout limits must be positive integers",
    )
    require(
        2 * (limits["lock_wait_seconds"] + limits["phase_execution_seconds"])
        < limits["job_minutes"] * 60,
        "Job timeout must leave time for bootstrap and archive",
    )
    patterns = task.get("watch_paths")
    require(
        type(patterns) is list
        and patterns
        and all(type(p) is str and p for p in patterns),
        "Invalid watch_paths",
    )
    operators = task.get("operators")
    require(type(operators) is list and operators, "Empty operator task")
    names, scripts = set(), set()
    for entry in operators:
        require(
            type(entry) is dict and set(entry) == {"name", "script"},
            "Invalid operator registration",
        )
        name, script = entry["name"], entry["script"]
        require(
            type(name) is str and re.fullmatch(r"[a-z][a-z0-9_]*", name),
            "Invalid operator name",
        )
        require(
            type(script) is str
            and re.fullmatch(r"test/ci/npu_test/test_case/test_[a-z0-9_]+\.py", script),
            "Invalid test entry path",
        )
        require(
            name not in names and script not in scripts, "Duplicate operator or script"
        )
        names.add(name)
        scripts.add(script)
    return task
