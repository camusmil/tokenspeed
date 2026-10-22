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

"""Select NPU operator and model tasks from one complete GitHub API change list."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

from lifecycle import CONFIG_PATH, MODEL_CONFIG_PATH, config_path, load_task


def select_suite(task: dict, event_name: str, paths: set[str]) -> dict:
    if event_name not in ("push", "pull_request", "workflow_dispatch"):
        raise ValueError(f"Unsupported event: {event_name}")
    matched = sorted(
        p
        for p in paths
        if not p.lower().endswith((".md", ".rst"))
        and any(fnmatchcase(p, pattern) for pattern in task["watch_paths"])
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


def github_json(endpoint, *, paginate):
    command = ["gh", "api", endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120
    )
    return json.loads(result.stdout)


def changed_paths(repository, event_name, event, api=github_json):
    """Reject partial API responses rather than treating missing paths as unrelated."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    if event_name == "workflow_dispatch":
        return set()
    if event_name == "pull_request":
        pr = event["pull_request"]
        number = event["number"]
        if type(number) is not int or number <= 0:
            raise ValueError("Invalid pull request number")
        current = api(f"/repos/{repository}/pulls/{number}", paginate=False)
        if any(current[ref]["sha"] != pr[ref]["sha"] for ref in ("head", "base")):
            raise ValueError("Pull request identity changed since this event")
        count = current["changed_files"]
        if type(count) is not int or not 0 <= count <= 3000:
            raise ValueError(
                "Pull request file list exceeds the API completeness limit"
            )
        pages = api(
            f"/repos/{repository}/pulls/{number}/files?per_page=100", paginate=True
        )
        if not isinstance(pages, list) or not all(
            isinstance(page, list) for page in pages
        ):
            raise ValueError("Invalid paginated pull request files")
        files = [item for page in pages for item in page]
        after = api(f"/repos/{repository}/pulls/{number}", paginate=False)
        if (
            any(after[ref]["sha"] != current[ref]["sha"] for ref in ("head", "base"))
            or after["changed_files"] != count
        ):
            raise ValueError("Pull request changed while reading files")
        if len(files) != count:
            raise ValueError("Incomplete pull request file list")
    elif event_name == "push":
        before, after = event["before"], event["after"]
        if any(
            not isinstance(sha, str)
            or not re.fullmatch(r"[0-9a-f]{40}", sha)
            or sha == "0" * 40
            for sha in (before, after)
        ):
            raise ValueError("Push requires nonzero before and after SHAs")
        comparison = api(
            f"/repos/{repository}/compare/{before}...{after}", paginate=False
        )
        files = comparison["files"]
        if len(files) >= 300:
            raise ValueError("Comparison may be truncated at 300 files")
    else:
        raise ValueError(f"Unsupported event: {event_name}")
    if not isinstance(files, list):
        raise ValueError("Invalid changed file list")
    paths, seen = set(), set()
    for item in files:
        filename = item["filename"]
        if filename in seen:
            raise ValueError("Duplicate changed file")
        seen.add(filename)
        for path in (filename, item.get("previous_filename")):
            if path is None:
                continue
            if (
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or ".." in Path(path).parts
            ):
                raise ValueError("Invalid changed path")
            paths.add(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task", type=Path, action="append", required=True)
    parser.add_argument(
        "--event-name",
        choices=("push", "pull_request", "workflow_dispatch"),
        required=True,
    )
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        tasks = [load_task(args.repo_root / path) for path in args.task]
        if len({task["workflow_stage"] for task in tasks}) != len(tasks):
            raise ValueError("Duplicate task stage")
        for path, task in zip(args.task, tasks):
            if str(path) != config_path(task):
                raise ValueError("Task path and declared identity differ")
        paths = changed_paths(
            args.repository, args.event_name, json.loads(args.event_path.read_text())
        )
        stages = {}
        for task in tasks:
            selection = select_suite(task, args.event_name, paths)
            # The public scanner is the single source of task matrix metadata.
            import sys

            command = [
                sys.executable,
                "-B",
                str(args.repo_root / "test/ci_system/pipeline.py"),
                "scan",
                "--repo-root",
                str(args.repo_root),
                "--runner-group",
                "npu",
                "--workflow-stage",
                "unit-test",
                "--trigger",
                {
                    "pull_request": "per-commit",
                    "push": "post-submit",
                    "workflow_dispatch": "manual",
                }[args.event_name],
            ]
            matrix = json.loads(subprocess.check_output(command, text=True, timeout=60))
            entries = [
                entry
                for entry in matrix["include"]
                if entry["config"] == config_path(task)
            ]
            if len(entries) != 1:
                raise ValueError(
                    "Public scanner did not produce exactly one selected NPU task"
                )
            selection["matrix"] = {
                "include": entries if selection["should_run"] else []
            }
            selection["timeout_minutes"] = task["timeouts"]["job_minutes"]
            stages[task["workflow_stage"]] = selection
        result = dict(
            stages["operator"] if "operator" in stages else stages["model"],
            stages=stages,
        )
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as output:
                for stage, selection in stages.items():
                    for key in ("should_run", "reason", "matrix", "timeout_minutes"):
                        value = selection[key]
                        rendered = (
                            json.dumps(value)
                            if isinstance(value, (dict, bool))
                            else str(value)
                        )
                        output.write(f"{stage}_{key}={rendered}\n")
                # Preserve the single-suite CLI outputs for direct callers.
                for key in ("should_run", "reason", "matrix", "timeout_minutes"):
                    value = result[key]
                    rendered = (
                        json.dumps(value)
                        if isinstance(value, (dict, bool))
                        else str(value)
                    )
                    output.write(f"{key}={rendered}\n")
        print(json.dumps(result, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"NPU change scan failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
