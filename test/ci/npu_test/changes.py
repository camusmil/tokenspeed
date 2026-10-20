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

"""Select the NPU operator suite from GitHub API changed paths."""

from __future__ import annotations

import argparse
import json
import re
from fnmatch import fnmatchcase
from pathlib import Path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
    ):
        raise ValueError("Expected manifest schema_version=3")
    if manifest["schema_version"] != 3:
        raise ValueError("Expected manifest schema_version=3")
    patterns = manifest.get("watch_paths")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("watch_paths must be a nonempty list")
    if any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError("watch_paths must contain nonempty strings")
    phases = ("accuracy", "performance")
    if manifest.get("expected_plans") != "test/ci/npu_test/expected-plans.json":
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
        if "entry" in operator or "scripts" in operator:
            raise ValueError(f"{name}: legacy entry/scripts are not supported")
        if operator.get("script") != f"test/ci/npu_test/test_case/{name}-test.py":
            raise ValueError(f"{name}: invalid script path")
        for field in ("accuracy_cases", "performance_configs"):
            value = operator.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}: {field} must be a positive integer")
    return manifest


def select_suite(manifest: dict, event_name: str, paths: set[str]) -> dict:
    if event_name not in ("push", "pull_request", "workflow_dispatch"):
        raise ValueError(f"Unsupported event: {event_name}")
    matched = sorted(
        p
        for p in paths
        if not p.lower().endswith((".md", ".rst"))
        and any(fnmatchcase(p, pattern) for pattern in manifest["watch_paths"])
    )
    manual = event_name == "workflow_dispatch"
    run = manual or bool(matched)
    return {
        "should_run": run,
        "reason": (
            "manual"
            if manual
            else ("watched-path-changed" if run else "no-watched-changes")
        ),
        "changed_files": len(paths),
        "matched_files": len(matched),
        "matched_paths_sample": matched[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--event-name",
        choices=("push", "pull_request", "workflow_dispatch"),
        required=True,
    )
    parser.add_argument("--changed-files", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.repo_root / args.manifest)
        paths = {
            line.strip()
            for line in args.changed_files.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        result = select_suite(manifest, args.event_name, paths)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as output:
                output.write(f"should_run={str(result['should_run']).lower()}\n")
                output.write(f"reason={result['reason']}\n")
        print(json.dumps(result, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        parser.exit(1, f"NPU change scan failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
