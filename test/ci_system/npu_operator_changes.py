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

"""Select the NPU operator suite from a complete local Git comparison."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
    ):
        raise ValueError("Expected manifest schema_version=2")
    if manifest["schema_version"] != 2:
        raise ValueError("Expected manifest schema_version=2")
    patterns = manifest.get("watch_paths")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("watch_paths must be a nonempty list")
    if any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError("watch_paths must contain nonempty strings")
    phases = ("accuracy", "performance")
    if manifest.get("expected_plans") != "ci/npu_ops/expected-plans.json":
        raise ValueError("Invalid expected_plans path")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("phases") != list(phases):
        raise ValueError("execution.phases must be accuracy, performance in that order")
    if execution.get("fail_fast") is not True:
        raise ValueError("execution.fail_fast must be true")
    if manifest.get("performance_policy") != {"mode": "record_only"}:
        raise ValueError("Only record_only performance policy is currently supported")
    operators = manifest.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("operators must be a nonempty list")
    names = set()
    for operator in operators:
        if not isinstance(operator, dict):
            raise ValueError("Each operator must be an object")
        name = operator.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid operator name")
        if name in names:
            raise ValueError(f"Duplicate operator: {name}")
        names.add(name)
        if "entry" in operator:
            raise ValueError(f"{name}: legacy entry is not supported")
        scripts = operator.get("scripts")
        if not isinstance(scripts, dict) or set(scripts) != set(phases):
            raise ValueError(f"{name}: scripts must define accuracy and performance")
        for phase in phases:
            if scripts[phase] != f"ci/npu_ops/{name}/{phase}.py":
                raise ValueError(f"{name}: invalid {phase} script path")
        for field in ("accuracy_cases", "performance_configs"):
            value = operator.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}: {field} must be a positive integer")
    return manifest


def select_suite(repo: Path, manifest: dict, event_name: str, event: dict) -> dict:
    changed = None
    matched = []
    if event_name == "workflow_dispatch":
        run, reason = True, "manual"
    elif event_name != "push":
        raise ValueError(f"Unsupported event: {event_name}")
    elif event.get("deleted"):
        run, reason = False, "branch-deleted"
    elif not isinstance(event.get("ref"), str):
        raise ValueError("Push payload must contain a ref string")
    elif not event.get("ref", "").startswith("refs/heads/"):
        run, reason = False, "not-a-branch"
    else:
        before, after = event.get("before", ""), event.get("after", "")
        valid = all(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha)
            for sha in (before, after)
        )
        if not valid or set(before) == {"0"} or set(after) == {"0"}:
            run, reason = True, "comparison-unavailable"
        else:
            try:
                # No rename detection: both the deleted and added paths are checked.
                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--no-ext-diff",
                        "--no-renames",
                        "--name-only",
                        "-z",
                        before,
                        after,
                        "--",
                    ],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                paths = [p for p in os.fsdecode(result.stdout).split("\0") if p]
                changed = len(paths)
                matched = sorted(
                    p
                    for p in paths
                    if any(
                        fnmatchcase(p, pattern) for pattern in manifest["watch_paths"]
                    )
                )
                run = bool(matched)
                reason = "watched-path-changed" if run else "no-watched-changes"
            except (OSError, subprocess.SubprocessError):
                run, reason = True, "git-diff-unavailable"
    return {
        "should_run": run,
        "reason": reason,
        "changed_files": changed,
        "matched_files": len(matched),
        "matched_paths_sample": matched[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--event-name", choices=("push", "workflow_dispatch"), required=True
    )
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.repo_root / args.manifest)
        event = json.loads(args.event_path.read_text(encoding="utf-8"))
        if not isinstance(event, dict):
            raise ValueError("Event payload must be an object")
        result = select_suite(args.repo_root, manifest, args.event_name, event)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as output:
                output.write(f"should_run={str(result['should_run']).lower()}\n")
                output.write(f"reason={result['reason']}\n")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(1, f"NPU change scan failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
