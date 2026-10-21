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

"""Collect and freeze plans from test definitions before any device execution."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ci_reporting import decode_record, require
from task_config import CONFIG_PATH


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root, task):
    paths = {CONFIG_PATH, *(e["script"] for e in task["operators"])}
    paths.update(
        str(p.relative_to(root))
        for p in (root / "test/ci_system/npu").iterdir()
        if p.suffix in (".py", ".sh", ".json")
    )
    result = {}
    for relative in sorted(paths):
        path = root / relative
        require(
            path.resolve().is_relative_to(root) and not path.is_symlink(),
            f"Invalid source path: {relative}",
        )
        result[relative] = file_digest(path)
    return result


def validate_operator(plan, name):
    require(
        plan.get("schema_version") == 1 and type(plan["schema_version"]) is int,
        "Invalid plan version",
    )
    require(plan.get("operator") == name, "Plan operator identity mismatch")
    for phase, key in (("accuracy", "case"), ("performance", "config")):
        items = plan.get(phase)
        require(type(items) is list and items, f"{name}/{phase}: empty plan")
        names = set()
        for item in items:
            require(
                type(item) is dict and set(item) == {key, "metadata"},
                "Invalid plan item",
            )
            params = item[key]
            require(type(params) is dict and len(params) > 1, "Missing item parameters")
            identifier = params.get("name")
            require(
                type(identifier) is str and re.fullmatch(r"[A-Za-z0-9_]+", identifier),
                "Invalid item ID",
            )
            require(identifier not in names, f"Duplicate plan ID: {identifier}")
            names.add(identifier)
            require(
                type(item["metadata"]) is dict and item["metadata"],
                "Missing item metadata",
            )
            # JSON decoding already rejects NaN and infinity at every nesting level.
            require(
                all(value is not None for value in params.values()),
                "Null test parameter",
            )
    timing = plan.get("timing")
    require(
        type(timing) is dict
        and set(timing) == {"warmup", "iterations", "sample_count"},
        "Invalid timing requirements",
    )
    require(
        all(type(v) is int and v > 0 for v in timing.values()),
        "Timing requirements must be positive integers",
    )


def verify_identity(root, task, plan):
    require(plan.get("schema_version") == 1, "Invalid frozen plan schema")
    require(plan.get("task") == task["name"], "Frozen task identity mismatch")
    require(
        plan.get("source_sha256") == source_identity(root, task),
        "Sources changed after plan collection",
    )
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    require(plan.get("source_sha") == sha, "Frozen source commit mismatch")
    require(
        set(plan["operators"]) == {e["name"] for e in task["operators"]},
        "Frozen operator list mismatch",
    )
    for entry in task["operators"]:
        validate_operator(plan["operators"][entry["name"]], entry["name"])


def freeze_plan(root, task, work, capture, save):
    work = Path(work)
    identity = source_identity(root, task)
    operators = {}
    logs = work / "plan-collection"
    logs.mkdir()
    for entry in task["operators"]:
        log = logs / f"{entry['name']}.json"
        result = capture(
            [
                sys.executable,
                "-B",
                str(root / entry["script"]),
                "--mode",
                "all",
                "--collect-plan",
            ],
            root,
            log,
            30,
        )
        require(
            result["exit_code"] == 0
            and result["log_saved"]
            and result["capture_error"] is None,
            f"Plan collection failed: {entry['name']}: {result}",
        )
        plan = decode_record(log.read_text())
        validate_operator(plan, entry["name"])
        operators[entry["name"]] = plan
    require(
        source_identity(root, task) == identity,
        "Sources changed during plan collection",
    )
    plan = {
        "schema_version": 1,
        "task": task["name"],
        "source_sha": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "task_sha256": file_digest(root / CONFIG_PATH),
        "source_sha256": identity,
        "operators": operators,
    }
    path = work / "frozen-plan.json"
    save(path, json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n")
    path.chmod(0o400)
    verify_identity(root, task, plan)
    return plan, file_digest(path)


def read_frozen(root, task, work, expected_digest):
    path = Path(work) / "frozen-plan.json"
    require(file_digest(path) == expected_digest, "Frozen plan changed after accuracy")
    plan = decode_record(path.read_text())
    verify_identity(root, task, plan)
    return plan
