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


"""Single-VM NPU task configuration, pytest execution, reporting and lifecycle."""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import select
import selectors
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Result validation and reporting

PREFIX = "NPU_RESULT_JSON="


class ReportError(ValueError):
    """Evidence is malformed, incomplete, or inconsistent."""


def require(condition, message):
    if not condition:
        raise ReportError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ReportError(f"Invalid JSON number: {value}")


def _finite_float(value):
    number = float(value)
    require(math.isfinite(number), f"Non-finite JSON number: {value}")
    return number


def decode_record(text):
    try:
        record = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ValueError as exc:
        raise ReportError(f"Invalid structured JSON: {exc}") from exc
    require(type(record) is dict, "Record must be a JSON object")
    return record


def read_records(log_path, operator, phase):
    require(
        type(operator) is str and re.fullmatch(r"[a-z][a-z0-9_]*", operator),
        "Invalid operator name",
    )
    events = {
        "accuracy": (
            "started",
            "case_started",
            "case_completed",
            "completed",
            "failed",
        ),
        "performance": (
            "started",
            "config_started",
            "sample",
            "config_completed",
            "completed",
            "failed",
        ),
    }
    require(phase in ("accuracy", "performance"), "Unknown phase")
    path = Path(log_path)
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if PREFIX not in line:
                    continue
                try:
                    require(line.startswith(PREFIX), "Misplaced JSON marker")
                    record = decode_record(line[len(PREFIX) :])
                    version = record.get("schema_version")
                    require(
                        type(version) is int and version == 1,
                        "Unsupported schema_version",
                    )
                    require(record.get("operator") == operator, "Operator mismatch")
                    require(record.get("phase") == phase, "Phase mismatch")
                    require(record.get("event") in events[phase], "Unknown event")
                except ReportError as exc:
                    raise ReportError(f"{path}:{line_number}: {exc}") from exc
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise ReportError(f"Cannot read log {path}: {exc}") from exc
    require(records, f"No structured records in {path}")
    return records


def check_count(record, key, expected):
    value = record.get(key)
    require(type(value) is int and value >= 0, f"Invalid count: {key}")
    require(value == expected, f"{key}: expected {expected}, got {value}")


def latency_value(value):
    require(type(value) in (int, float), "Latency must be a number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReportError("Invalid latency value") from exc
    require(math.isfinite(number) and number >= 0, "Invalid latency value")
    return number


class RecordCursor:
    """Consume records in order and reject missing or extra events."""

    def __init__(self, records):
        self.records = records
        self.index = 0

    def take(self, event):
        position = self.index + 1
        require(
            self.index < len(self.records),
            f"Record {position}: missing {event}",
        )
        record = self.records[self.index]
        actual = record.get("event")
        require(
            actual == event,
            f"Record {position}: expected {event}, got {actual!r}",
        )
        self.index += 1
        return record

    def finish(self):
        require(
            self.index == len(self.records),
            f"Unexpected records after record {self.index}",
        )


def same_json(actual, expected):
    options = {"sort_keys": True, "allow_nan": False}
    return json.dumps(actual, **options) == json.dumps(expected, **options)


def validate_accuracy(records, planned, exit_code):
    require(type(exit_code) is int, "Missing or invalid process exit code")
    require(type(planned) is list and planned, "Empty accuracy plan")
    names = []
    for item in planned:
        require(
            type(item) is dict and set(item) == {"case", "metadata"},
            "Invalid accuracy plan item",
        )
        case, metadata = item["case"], item["metadata"]
        require(
            type(case) is dict and type(case.get("name")) is str and case["name"],
            "Invalid planned case name",
        )
        require(
            type(metadata) is dict
            and type(metadata.get("dtype")) is str
            and metadata["dtype"],
            "Missing planned accuracy dtype",
        )
        names.append(case["name"])
    require(len(set(names)) == len(names), "Duplicate planned case names")
    require(
        type(records) is list and all(type(r) is dict for r in records),
        "Invalid accuracy records",
    )
    cursor = RecordCursor(records)

    def peek():
        return (
            records[cursor.index].get("event") if cursor.index < len(records) else None
        )

    def match(record, item):
        require(same_json(record.get("case"), item["case"]), "Case parameters mismatch")
        for key, expected in item["metadata"].items():
            require(
                key in record and same_json(record[key], expected),
                f"{item['case']['name']}: {key} mismatch",
            )

    start = cursor.take("started")
    check_count(start, "expected_cases", len(planned))
    require(start.get("case_names") == names, "Accuracy case names mismatch")
    results, current, started = [], None, 0
    for item in planned:
        if peek() == "failed":
            break
        match(cursor.take("case_started"), item)
        current = item["case"]["name"]
        started += 1
        if peek() == "failed":
            break
        record = cursor.take("case_completed")
        match(record, item)
        status = record.get("status")
        require(status in ("passed", "failed"), "Invalid accuracy case status")
        results.append(
            {
                "case": item["case"],
                "metadata": item["metadata"],
                "dtype": item["metadata"]["dtype"],
                "status": status,
            }
        )
        if status == "failed":
            break
    terminal = "failed" if peek() == "failed" else "completed"
    end = cursor.take(terminal)
    cursor.finish()
    passed = sum(item["status"] == "passed" for item in results)
    failed = len(results) - passed
    for key, expected in (
        ("expected_cases", len(planned)),
        ("completed_cases", len(results)),
        ("passed_cases", passed),
        ("failed_cases", failed),
    ):
        check_count(end, key, expected)
    result = {
        "status": "failed",
        "exit_code": exit_code,
        "expected_cases": len(planned),
        "started_cases": started,
        "completed_cases": len(results),
        "passed_cases": passed,
        "failed_cases": failed,
        "incomplete_cases": started - len(results),
        "not_run_cases": len(planned) - started,
        "cases": results,
    }
    if terminal == "failed":
        require(exit_code != 0, "Failed accuracy log has zero exit code")
        require(
            "current_case" in end and end["current_case"] == current,
            "Failed accuracy current_case mismatch",
        )
        require(
            type(end.get("error_type")) is str and end["error_type"],
            "Missing accuracy error type",
        )
        require(type(end.get("error")) is str, "Missing accuracy error message")
        result.update(
            current_case=current, error_type=end["error_type"], error=end["error"]
        )
    else:
        require(
            len(results) == len(planned) and failed == 0,
            "Accuracy completed without all cases passing",
        )
        result["status"] = "passed" if exit_code == 0 else "failed"
    return result


def validate_config(cursor, config, metadata, warmup, iterations, sample_count):
    require(
        type(config) is dict and type(config.get("name")) is str and config["name"],
        "Invalid planned config",
    )
    require(
        type(metadata) is dict
        and type(metadata.get("dtype")) is str
        and metadata["dtype"],
        "Missing planned performance dtype",
    )
    reserved = {
        "config",
        "metadata",
        "status",
        "warmup",
        "iterations",
        "expected_samples",
        "sample_count",
        "samples_ms",
        "median_latency_ms",
        "threshold_ms",
        "performance_verdict",
        "schema_version",
        "operator",
        "phase",
        "event",
    }
    require(not reserved.intersection(metadata), "Reserved key in config metadata")
    for key, value, minimum in (
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
        ("sample_count", sample_count, 1),
    ):
        require(type(value) is int and value >= minimum, f"Invalid plan: {key}")

    def pending_failure():
        return (
            cursor.index < len(cursor.records)
            and cursor.records[cursor.index].get("event") == "failed"
        )

    def match(record):
        require(same_json(record.get("config"), config), "Config parameters mismatch")
        check_count(record, "warmup", warmup)
        check_count(record, "iterations", iterations)
        for key, expected in metadata.items():
            require(
                key in record and same_json(record[key], expected),
                f"{config['name']}: {key} mismatch",
            )

    start = cursor.take("config_started")
    match(start)
    check_count(start, "expected_samples", sample_count)
    samples = []
    for index in range(sample_count):
        if pending_failure():
            break
        record = cursor.take("sample")
        require(record.get("config") == config["name"], "Sample config mismatch")
        check_count(record, "sample_index", index)
        samples.append(latency_value(record.get("latency_ms")))
    result = {
        "config": config,
        "metadata": metadata,
        **metadata,
        "status": "incomplete",
        "warmup": warmup,
        "iterations": iterations,
        "expected_samples": sample_count,
        "sample_count": len(samples),
        "samples_ms": samples,
        "median_latency_ms": None,
        "threshold_ms": None,
        "performance_verdict": "not_evaluated",
    }
    if pending_failure():
        return result
    end = cursor.take("config_completed")
    match(end)
    check_count(end, "sample_count", sample_count)
    reported = end.get("samples_ms")
    require(
        type(reported) is list and len(reported) == sample_count,
        "Missing or incomplete sample list",
    )
    require(
        [latency_value(value) for value in reported] == samples,
        "Sample list differs from individual records",
    )
    median = statistics.median(samples)
    require(
        latency_value(end.get("median_latency_ms")) == median,
        "Median differs from recorded samples",
    )
    require(
        "threshold_ms" in end and end["threshold_ms"] is None,
        "Unexpected or missing performance threshold",
    )
    require(
        end.get("performance_verdict") == "not_evaluated",
        "Unexpected performance verdict",
    )
    result.update(status="completed", median_latency_ms=median)
    return result


def validate_performance(
    records, planned, exit_code, operator, warmup, iterations, sample_count
):
    require(type(exit_code) is int, "Missing or invalid process exit code")
    require(
        type(operator) is str and re.fullmatch(r"[a-z][a-z0-9_]*", operator),
        "Invalid operator name",
    )
    require(type(planned) is list and planned, "Empty performance plan")
    for key, value, minimum in (
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
        ("sample_count", sample_count, 1),
    ):
        require(type(value) is int and value >= minimum, f"Invalid plan: {key}")
    names = []
    for item in planned:
        require(
            type(item) is dict and set(item) == {"config", "metadata"},
            "Invalid performance plan item",
        )
        config, metadata = item["config"], item["metadata"]
        require(
            type(config) is dict and type(config.get("name")) is str and config["name"],
            "Invalid planned config name",
        )
        require(
            type(metadata) is dict
            and type(metadata.get("dtype")) is str
            and metadata["dtype"],
            "Missing planned performance dtype",
        )
        names.append(config["name"])
    require(len(set(names)) == len(names), "Duplicate planned config names")
    require(
        type(records) is list and all(type(r) is dict for r in records),
        "Invalid performance records",
    )
    cursor = RecordCursor(records)

    def peek():
        return (
            records[cursor.index].get("event") if cursor.index < len(records) else None
        )

    start = cursor.take("started")
    check_count(start, "expected_configs", len(planned))
    require(start.get("config_names") == names, "Performance config names mismatch")
    results, current = [], None
    for item in planned:
        if peek() == "failed":
            break
        current = item["config"]["name"]
        result = validate_config(
            cursor, item["config"], item["metadata"], warmup, iterations, sample_count
        )
        results.append(result)
        if result["status"] != "completed":
            break
    terminal = "failed" if peek() == "failed" else "completed"
    end = cursor.take(terminal)
    cursor.finish()
    completed = sum(item["status"] == "completed" for item in results)
    check_count(end, "expected_configs", len(planned))
    check_count(end, "completed_configs", completed)
    attempted = len(results)
    error = {}
    if terminal == "failed":
        require(exit_code != 0, "Failed performance log has zero exit code")
        require("current_config" in end, "Missing failed current_config")
        observed = end["current_config"]
        if observed != current:
            require(
                completed == len(results)
                and completed < len(planned)
                and observed == names[completed],
                "Failed current_config mismatch",
            )
            attempted += 1
        require(
            type(end.get("error_type")) is str and end["error_type"],
            "Missing performance error type",
        )
        require(type(end.get("error")) is str, "Missing performance error message")
        error = dict(
            current_config=observed, error_type=end["error_type"], error=end["error"]
        )
    else:
        require(
            completed == len(planned), "Performance completed with unfinished configs"
        )
    return {
        "status": (
            "completed" if terminal == "completed" and exit_code == 0 else "failed"
        ),
        "exit_code": exit_code,
        "expected_configs": len(planned),
        "attempted_configs": attempted,
        "started_configs": len(results),
        "completed_configs": completed,
        "incomplete_configs": attempted - completed,
        "not_run_configs": len(planned) - attempted,
        "expected_samples": len(planned) * sample_count,
        "completed_samples": sum(item["sample_count"] for item in results),
        "performance_verdict": "not_evaluated",
        "configs": results,
        **error,
    }


def summarize_phase(
    log_path,
    operator,
    phase,
    planned,
    exit_code,
    *,
    warmup=None,
    iterations=None,
    sample_count=None,
):
    result = {
        "operator": operator,
        "phase": phase,
        "status": "failed",
        "log_path": str(log_path),
        "exit_code": exit_code if type(exit_code) is int else None,
        "structured_log_valid": False,
        "parsed_records": None,
    }
    if phase == "performance":
        result["performance_verdict"] = "not_evaluated"
    try:
        require(type(exit_code) is int, "Missing or invalid process exit code")
        records = read_records(log_path, operator, phase)
        result["parsed_records"] = len(records)
        if phase == "accuracy":
            details = validate_accuracy(records, planned, exit_code)
        else:
            details = validate_performance(
                records,
                planned,
                exit_code,
                operator,
                warmup,
                iterations,
                sample_count,
            )
        result.update(details)
        result["structured_log_valid"] = True
        if exit_code != 0 and "error" not in result:
            result["error_type"] = "ProcessExitError"
            result["error"] = f"Process exited with code {exit_code}"
        elif result["status"] == "failed" and "error" not in result:
            result["error_type"] = "AccuracyFailure"
            result["error"] = "One or more accuracy cases failed"
    except Exception as exc:
        # Keep reporting possible even when the log or validation fails.
        result["status"] = "failed"
        result["structured_log_valid"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def phase_succeeded(result, phase):
    require(phase in ("accuracy", "performance"), "Unknown report phase")
    wanted = "passed" if phase == "accuracy" else "completed"
    return (
        type(result) is dict
        and result.get("status") == wanted
        and type(result.get("exit_code")) is int
        and result["exit_code"] == 0
        and result.get("structured_log_valid") is True
        and result.get("log_saved") is True
    )


def make_report(results, metadata, task, phase):
    require(phase in ("accuracy", "performance"), "Unknown report phase")
    require(
        type(results) is list and all(type(r) is dict for r in results),
        "Operator results must be a list of objects",
    )
    require(type(metadata) is dict, "Report metadata must be an object")
    require(
        type(task) is dict
        and type(task.get("operators")) is list
        and task["operators"],
        "Invalid report task",
    )
    names = []
    for entry in task["operators"]:
        require(
            type(entry) is dict
            and type(entry.get("name")) is str
            and re.fullmatch(r"[a-z][a-z0-9_]*", entry["name"]),
            "Invalid registered operator name",
        )
        names.append(entry["name"])
    require(len(set(names)) == len(names), "Duplicate registered operators")
    expected = [(name, phase) for name in names]
    actual = [(item.get("operator"), item.get("phase")) for item in results]
    errors = []
    if actual != expected[: len(actual)]:
        errors.append("Results must follow registry order in the selected phase")
    unexpected = [
        item
        for item in results
        if item.get("operator") not in names or item.get("phase") != phase
    ]
    if unexpected:
        errors.append("Unexpected operator or phase results")
    ordered, first_failure = [], None
    for name in names:
        matches = [
            item
            for item in results
            if item.get("operator") == name and item.get("phase") == phase
        ]
        if first_failure is not None:
            if matches:
                message = (
                    f"{name}/{phase}: result received after failure of {first_failure}"
                )
                errors.append(message)
                ordered.append(
                    {
                        "operator": name,
                        "phase": phase,
                        "status": "failed",
                        "error_type": "FailFastViolation",
                        "error": message,
                        "reported_results": matches,
                    }
                )
            else:
                ordered.append(
                    {
                        "operator": name,
                        "phase": phase,
                        "status": "not_run",
                        "exit_code": None,
                        "structured_log_valid": False,
                        "log_saved": False,
                        "reason": "not_run_after_failure",
                        "blocked_by": first_failure,
                    }
                )
            continue
        if len(matches) != 1:
            message = f"{name}/{phase}: expected one result, got {len(matches)}"
            errors.append(message)
            ordered.append(
                {
                    "operator": name,
                    "phase": phase,
                    "status": "failed",
                    "exit_code": None,
                    "structured_log_valid": False,
                    "log_saved": False,
                    "error_type": "EvidenceError",
                    "error": message,
                    "conflicting_results": matches,
                }
            )
            first_failure = name
            continue
        item = dict(matches[0])
        if not phase_succeeded(item, phase):
            item["test_status"] = item.get("status")
            item["status"] = "failed"
            item.setdefault("error_type", "PhaseOrEvidenceFailure")
            item.setdefault("error", "Test or required evidence collection failed")
            errors.append(f"{name}/{phase}: {item['error']}")
            first_failure = name
        ordered.append(item)
    report = {
        "schema_version": 2,
        "phase": phase,
        "status": "failed" if errors else "passed",
        "exit_code": 1 if errors else 0,
        "expected_operators": names,
        "first_failure": first_failure,
        "metadata": metadata,
        "operators": ordered,
        "errors": errors,
        "unexpected_results": unexpected,
    }
    if phase == "performance":
        report["performance_verdict"] = "not_evaluated"
    return report


def _md_cell(value):
    import html

    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    text = html.escape(str(value), quote=False)
    for character in "\\`*_[]|":
        text = text.replace(character, "\\" + character)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _md_row(values):
    return "| " + " | ".join(_md_cell(value) for value in values) + " |"


def render_summary(report):
    phase = report["phase"]
    require(phase in ("accuracy", "performance"), "Unknown summary phase")
    phase_name = "accuracy tests" if phase == "accuracy" else "performance tests"
    lines = [
        f"## Ascend NPU operator {phase_name}",
        "",
        "Phase status: **" + _md_cell(report["status"]) + "**",
        "",
        "Phase exit code: " + _md_cell(report["exit_code"]),
        "",
    ]
    if phase == "performance":
        lines += [
            "Performance verdict: **not_evaluated** (no latency thresholds are configured).",
            "",
        ]
    lines += [
        "| Operator | Result | Progress | Exit code | Structured log valid | Log saved |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["operators"]:
        progress = "No data available"
        if item["status"] == "not_run":
            progress = "Not run after failure of " + str(
                item.get("blocked_by", "a previous test")
            )
        elif phase == "accuracy" and "completed_cases" in item:
            progress = (
                f"Completed {item['completed_cases']}/{item['expected_cases']}; "
                f"passed {item['passed_cases']}; failed {item['failed_cases']}; "
                f"incomplete {item['incomplete_cases']}; not run {item['not_run_cases']}"
            )
        elif phase == "performance" and "completed_configs" in item:
            progress = (
                f"Completed configs {item['completed_configs']}/{item['expected_configs']}; "
                f"Collected samples {item['completed_samples']}/{item['expected_samples']}; "
                f"incomplete {item['incomplete_configs']}; not run {item['not_run_configs']}"
            )
        lines.append(
            _md_row(
                [
                    item["operator"],
                    item["status"],
                    progress,
                    item.get("exit_code"),
                    item.get("structured_log_valid"),
                    item.get("log_saved"),
                ]
            )
        )
    if phase == "performance":
        lines += [
            "",
            "### Configuration sampling results",
            "",
            "| Operator | Configuration | Result | Samples | Median latency (ms/execution) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in report["operators"]:
            if not item.get("structured_log_valid"):
                continue
            for config in item.get("configs", []):
                median = config.get("median_latency_ms")
                lines.append(
                    _md_row(
                        [
                            item["operator"],
                            config["config"]["name"],
                            config["status"],
                            f"{config['sample_count']}/{config['expected_samples']}",
                            format(median, ".9g") if median is not None else None,
                        ]
                    )
                )
    details = list(report["errors"])
    for item in report["operators"]:
        if item.get("error") is not None:
            details.append(
                f"{item['operator']}: {item.get('error_type', 'Error')}: {item['error']}"
            )
        elif item["status"] == "not_run":
            details.append(
                f"{item['operator']}: not run after failure of {item['blocked_by']}"
            )
    if details:
        lines += ["", "### Failure and execution details", ""]
        lines += ["- " + _md_cell(reason) for reason in dict.fromkeys(details)]
    lines += ["", "### Evidence", "", "| Operator | Log path |", "| --- | --- |"]
    for item in report["operators"]:
        lines.append(_md_row([item["operator"], item.get("log_path")]))
    lines += ["", "### Run information", "", "| Field | Value |", "| --- | --- |"]
    for key, value in report["metadata"].items():
        lines.append(_md_row([key, value]))
    return "\n".join(lines) + "\n"


def save_report(report, output_dir, github_summary=None):
    """Save both report formats; propagate any write failure to the caller."""
    require(
        type(report) is dict and report.get("status") in ("passed", "failed"),
        "Invalid report status",
    )
    require(
        type(report.get("exit_code")) is int
        and report["exit_code"] == (0 if report["status"] == "passed" else 1),
        "Report status and exit code differ",
    )
    json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = render_summary(report)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    for key, filename, text in (
        ("json", "report.json", json_text),
        ("markdown", "summary.md", markdown),
    ):
        target = directory / filename
        expected = text.encode("utf-8")
        with target.open("xb") as handle:
            require(
                handle.write(expected) == len(expected), f"Incomplete write: {target}"
            )
        require(target.read_bytes() == expected, f"Saved file differs: {target}")
        saved[key] = {
            "path": str(target),
            "sha256": hashlib.sha256(expected).hexdigest(),
        }
    saved["github_summary"] = None
    if github_summary is not None:
        target = Path(github_summary)
        content = "\n\n" + markdown
        with target.open("a", encoding="utf-8") as handle:
            require(
                handle.write(content) == len(content),
                "Incomplete Actions summary write",
            )
        saved["github_summary"] = {"path": str(target), "appended": True}
    return saved


# Task declaration

CONFIG_PATH = "test/ci/npu_test/ut-tokenspeed-kernel-npu.json"


def load_task(path):

    task = decode_record(Path(path).read_text(encoding="utf-8"))
    require(
        type(task.get("schema_version")) is int and task["schema_version"] == 1,
        "Expected task schema_version=1",
    )
    require(task.get("name") == "ut-tokenspeed-kernel-npu", "Invalid task name")
    require(
        task.get("type") == "ut" and task.get("workflow_stage") == "operator",
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
            and re.fullmatch(
                r"tokenspeed-kernel/test/npu/ops/test_[a-z0-9_]+\.py", script
            ),
            "Invalid test entry path",
        )
        require(
            name not in names and script not in scripts, "Duplicate operator or script"
        )
        names.add(name)
        scripts.add(script)
    require(task.get("runner") == "ascend-npu-ops-test", "Unsupported runner")
    report = task.get("report", {})
    require(
        set(report)
        == {"artifact_retention_days", "artifact_max_bytes", "upload_attempts"},
        "Invalid report policy",
    )
    require(
        all(type(v) is int and v > 0 for v in report.values()), "Invalid report values"
    )
    require(
        report["artifact_retention_days"] <= 90 and report["upload_attempts"] == 3,
        "Unsupported archive policy",
    )
    cleanup = task.get("cleanup", {})
    require(
        set(cleanup) == {"delete_unarchived", "pending_warning_seconds"}
        and cleanup["delete_unarchived"] is False,
        "Invalid cleanup policy",
    )
    require(
        type(cleanup["pending_warning_seconds"]) is int
        and cleanup["pending_warning_seconds"] > 0,
        "Invalid warning interval",
    )
    return task


# Frozen test plans


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


# Trusted host lifecycle and process supervision

SCHEMA = 2


RUN_NAME = re.compile(r"actions-([0-9]+)-([0-9]+)\.[A-Za-z0-9_]+")


PHASE_FILES = {"context.json", "report.json", "summary.md", "executor-error.log"}


OPERATOR_FILES = {"process.json", "result.json"}


def read_json(path):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), f"Not a regular file: {path}")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "r") as handle:
        text = handle.read(1024 * 1024 + 1)
    require(len(text) <= 1024 * 1024, f"Metadata too large: {path}")
    return decode_record(text)


def write_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), f"Refusing metadata symlink: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def utc(timestamp):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def load_policy(path):
    deployment = read_json(path)
    task = load_task(deployment["task_path"])
    policy = dict(deployment, **task["report"], **task["cleanup"])
    require(
        policy.get("schema_version") == 1 and policy.get("delete_unarchived") is False,
        "Unarchived deletion is not permitted",
    )
    for key in ("jobs_root", "evidence_root", "state_root"):
        directory = Path(policy[key])
        require(
            directory.is_absolute() and directory == directory.resolve(),
            f"Unsafe policy path: {key}",
        )
        require(directory.is_dir(), f"Deployment directory missing: {directory}")
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", policy["repository"]),
        "Invalid repository",
    )
    for key in (
        "artifact_retention_days",
        "artifact_max_bytes",
        "upload_attempts",
        "minimum_free_bytes",
        "pending_warning_seconds",
    ):
        require(type(policy[key]) is int and policy[key] >= 0, f"Invalid policy: {key}")
    require(1 <= policy["artifact_retention_days"] <= 90, "Invalid artifact retention")
    require(1 <= policy["upload_attempts"] <= 10, "Invalid upload attempts")
    require(policy["artifact_max_bytes"] > 0, "Invalid artifact size limit")
    return policy


def process_identity(pid):
    directory = Path(f"/proc/{pid}")
    # Missing /proc on a non-Linux test host is treated conservatively.
    if not Path("/proc").is_dir():
        return {"pid": pid, "boot_id": "unknown", "start_ticks": "unknown"}
    text = (directory / "stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_ticks": fields[19],
    }


def worker_identity():
    pid = os.getpid()
    for _ in range(30):
        directory = Path(f"/proc/{pid}")
        if (directory / "comm").read_text().strip() == "Runner.Worker":
            return process_identity(pid)
        text = (directory / "stat").read_text()
        pid = int(text[text.rfind(")") + 2 :].split()[1])
        if pid <= 1:
            break
    raise RuntimeError(
        "Runner.Worker ancestor not found; refusing an unowned CI directory"
    )


def owner_status(owner):
    try:
        os.kill(int(owner["pid"]), 0)
    except ProcessLookupError:
        return "gone"
    except (PermissionError, ValueError, KeyError):
        return "unknown"
    try:
        actual = process_identity(int(owner["pid"]))
        if Path("/proc").is_dir():
            text = Path(f"/proc/{owner['pid']}/stat").read_text()
            if text[text.rfind(")") + 2 :].split()[0] == "Z":
                return "gone"
    except FileNotFoundError:
        return "gone"
    except (OSError, ValueError):
        return "unknown"
    if "unknown" in (actual["boot_id"], owner.get("boot_id")):
        return "unknown"
    return "alive" if actual == owner else "gone"


def checked_work(policy, path):
    work = Path(path)
    require(
        work.parent == Path(policy["jobs_root"]),
        "Work directory is outside the configured slot",
    )
    match = RUN_NAME.fullmatch(work.name)
    require(
        match is not None and work == work.resolve() and work.is_dir(),
        "Invalid work directory",
    )
    state = read_json(work / "lifecycle.json")
    require(
        state["schema"] == SCHEMA and state["repository"] == policy["repository"],
        "Run policy mismatch",
    )
    require(
        [state["run_id"], state["attempt"]] == list(match.groups()),
        "Run identity mismatch",
    )
    require(state["work"] == str(work), "Work identity mismatch")
    return work, state


def initialize(policy, identity, owner, now):
    require(
        identity["repository"] == policy["repository"],
        "Repository does not match host policy",
    )
    for name in ("run_id", "attempt"):
        require(re.fullmatch(r"[0-9]+", identity[name]), f"Invalid {name}")
    require(re.fullmatch(r"[0-9a-f]{40}", identity["sha"]), "Invalid event SHA")
    require(
        shutil.disk_usage(policy["jobs_root"]).free >= policy["minimum_free_bytes"],
        "Insufficient free disk space for NPU CI",
    )
    work = Path(
        tempfile.mkdtemp(
            prefix=f"actions-{identity['run_id']}-{identity['attempt']}.",
            dir=policy["jobs_root"],
        )
    )
    state = dict(
        identity, schema=SCHEMA, work=str(work), owner=owner, started_at=now, phases={}
    )
    write_json(work / "lifecycle.json", state)
    return work


def phase_state(policy, work, phase, stage, rc, now):
    require(phase in ("accuracy", "performance"), "Unknown phase")
    work, state = checked_work(policy, work)
    state["phases"][phase] = {"stage": stage, "exit": rc, "at": now}
    write_json(work / "lifecycle.json", state)


def regular_files(directory):
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            require(
                not (Path(parent) / name).is_symlink(),
                "Evidence must not contain symlinks",
            )
        for name in files:
            path = Path(parent) / name
            require(
                stat.S_ISREG(path.lstat().st_mode),
                "Evidence must contain regular files only",
            )
            yield path


def digest(path):
    value = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def evidence_directory(policy, state):
    return Path(policy["evidence_root"]) / f"{state['run_id']}-{state['attempt']}"


def collect(policy, work, now, outcomes):
    work, state = checked_work(policy, work)
    for directory in (
        work / "accuracy",
        work / "performance",
        work / "plan-collection",
        work,
    ):
        if directory.exists():
            for record in directory.rglob("*.owner.json"):
                for identity in read_json(record).values():
                    require(
                        owner_status(identity) == "gone",
                        "Evidence producer is still active; refusing collection",
                    )
    logger = work / "bootstrap.owner.json"
    if logger.exists():
        require(
            owner_status(read_json(logger)) == "gone",
            "Bootstrap logger is still active",
        )
    output = evidence_directory(policy, state)
    require(not output.is_symlink(), "Invalid evidence directory")
    output.mkdir(exist_ok=True)
    bundle = output / "bundle"
    if bundle.exists():
        verify_bundle(bundle)
        return bundle
    temporary = Path(tempfile.mkdtemp(prefix=".bundle-", dir=output))
    try:
        selected = []
        require(set(outcomes) == {"accuracy", "performance"}, "Missing step outcomes")
        succeeded = all(value == "success" for value in outcomes.values()) and all(
            state["phases"].get(phase, {}).get("exit") == 0
            for phase in ("accuracy", "performance")
        )
        for name in (
            ()
            if succeeded
            else (
                "bootstrap.log",
                "fetch.tsv",
                "source.txt",
                "source.json",
                "init.log",
                "checkout.log",
                "frozen-plan.json",
                *(f"fetch-{i}.log" for i in range(1, 5)),
                *(f"fetch-{i}.json" for i in range(1, 5)),
            )
        ):
            if (work / name).exists():
                require(
                    not (work / name).is_symlink(), "Evidence must not contain symlinks"
                )
                selected.append(work / name)
        for phase in ("accuracy", "performance"):
            directory = work / phase
            if state["phases"].get(phase, {}).get("exit") == 0:
                for name in ("context.json", "report.json", "summary.md"):
                    require(
                        (directory / name).is_file(),
                        f"Required phase evidence missing: {phase}/{name}",
                    )
            if not directory.exists():
                continue
            require(directory == directory.resolve(), "Invalid phase directory")
            for path in regular_files(directory):
                if succeeded and path.name not in ("context.json", "summary.md"):
                    continue
                relative = path.relative_to(directory)
                if len(relative.parts) == 1 and path.name in PHASE_FILES:
                    selected.append(path)
                elif len(relative.parts) == 2 and re.fullmatch(
                    r"[A-Za-z0-9_-]+", relative.parts[0]
                ):
                    if path.name in OPERATOR_FILES | {f"{phase}.log", f"{phase}.rc"}:
                        selected.append(path)
        require(selected, "No evidence available for upload")
        total = sum(source.stat().st_size for source in selected)
        require(
            total <= policy["artifact_max_bytes"],
            "Evidence exceeds artifact size limit; retained locally",
        )
        entries = []
        for source in sorted(selected):
            require(stat.S_ISREG(source.lstat().st_mode), "Invalid evidence file")
            target = temporary / source.relative_to(work)
            target.parent.mkdir(parents=True, exist_ok=True)
            # These roots must share a filesystem; one inode holds each raw log.
            os.link(source, target, follow_symlinks=False)
            entries.append(
                {
                    "path": str(target.relative_to(temporary)),
                    "bytes": target.stat().st_size,
                    "sha256": digest(target),
                }
            )
        write_json(
            temporary / "run.json",
            {
                key: state[key]
                for key in (
                    "schema",
                    "repository",
                    "run_id",
                    "attempt",
                    "sha",
                    "ref",
                    "runner",
                    "started_at",
                    "phases",
                )
            },
        )
        entries.append(
            {
                "path": "run.json",
                "bytes": (temporary / "run.json").stat().st_size,
                "sha256": digest(temporary / "run.json"),
            }
        )
        write_json(
            temporary / "manifest.json",
            {"schema": SCHEMA, "collected_at": utc(now), "files": entries},
        )
        require(
            sum(path.stat().st_size for path in regular_files(temporary))
            <= policy["artifact_max_bytes"],
            "Evidence metadata exceeds artifact size limit",
        )
        os.rename(temporary, bundle)
        write_json(output / "lifecycle.json", state)
        return bundle
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_bundle(bundle):
    require(bundle == bundle.resolve(), "Invalid evidence bundle path")
    manifest = read_json(bundle / "manifest.json")
    require(
        manifest["schema"] == SCHEMA and manifest["files"], "Invalid evidence manifest"
    )
    expected_files = {item["path"] for item in manifest["files"]}
    require(len(expected_files) == len(manifest["files"]), "Duplicate evidence file")
    require(
        {str(path.relative_to(bundle)) for path in regular_files(bundle)}
        == expected_files | {"manifest.json"},
        "Unexpected evidence files",
    )
    for item in manifest["files"]:
        relative = Path(item["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Invalid manifest path",
        )
        path = bundle / relative
        require(
            path == path.resolve() and stat.S_ISREG(path.lstat().st_mode),
            "Invalid manifest file",
        )
        require(
            path.stat().st_size == item["bytes"] and digest(path) == item["sha256"],
            f"Evidence integrity failure: {relative}",
        )


def remove_directory(path, parent):
    path, parent = Path(path), Path(parent)
    require(
        path.parent == parent and path == path.resolve() and path.is_dir(),
        "Refusing unsafe deletion",
    )
    require(
        shutil.rmtree.avoids_symlink_attacks, "Platform lacks safe recursive deletion"
    )
    shutil.rmtree(path)


def acknowledge(
    policy, work, outcome, artifact_id, artifact_digest, accuracy, performance, now
):
    work, state = checked_work(policy, work)
    output = evidence_directory(policy, state)
    require(
        output == output.resolve() and output.is_dir(),
        "Evidence collection did not finish",
    )
    verify_bundle(output / "bundle")
    uploaded = (
        outcome == "success"
        and bool(re.fullmatch(r"[0-9]+", artifact_id))
        and bool(re.fullmatch(r"[0-9a-f]{64}", artifact_digest))
    )
    receipt = {
        "schema": SCHEMA,
        "uploaded": uploaded,
        "at": now,
        "accuracy": accuracy,
        "performance": performance,
        "artifact_id": artifact_id,
        "artifact_digest": artifact_digest,
        "work_removed": False,
    }
    write_json(output / "receipt.json", receipt)
    if not uploaded:
        return receipt
    # Only this job's finalization explicitly deletes its own active work area.
    # The timer always waits for the owning Runner.Worker to disappear.
    remove_directory(work, policy["jobs_root"])
    receipt["work_removed"] = True
    write_json(output / "receipt.json", receipt)
    # Preserve the minimal receipt outside the disposable evidence directory.
    write_json(
        Path(policy["state_root"]) / f"{state['run_id']}-{state['attempt']}.json",
        dict(
            receipt,
            repository=state["repository"],
            sha=state["sha"],
            run_id=state["run_id"],
            attempt=state["attempt"],
        ),
    )
    remove_directory(output, policy["evidence_root"])
    receipt["local_evidence_removed"] = True
    return receipt


def garbage_collect(policy, mode, now):
    """Report abandoned unarchived runs; never destroy their only evidence."""
    result = {
        "at": utc(now),
        "mode": mode,
        "pending": [],
        "unmanaged": [],
        "errors": [],
    }
    for work in Path(policy["jobs_root"]).iterdir():
        if not RUN_NAME.fullmatch(work.name):
            result["unmanaged"].append(work.name)
            continue
        try:
            _, state = checked_work(policy, work)
            status = owner_status(state["owner"])
            if (
                status == "gone"
                and now - state["started_at"] >= policy["pending_warning_seconds"]
            ):
                result["pending"].append(
                    {
                        "work": work.name,
                        "reason": "unarchived-evidence-requires-operator-action",
                    }
                )
        except Exception as exc:
            result["errors"].append({"work": work.name, "error": str(exc)})
    if mode == "apply":
        write_json(Path(policy["state_root"]) / "last-gc.json", result)
    return result


def stream_bootstrap(work, source, console):
    path = Path(work) / "bootstrap.log"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab") as log:
        for line in source:
            stamp = (
                datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            log.write(stamp.encode() + b" " + line)
            log.flush()
            console.write(line)
            console.flush()
        os.fsync(log.fileno())


def append_summary(text):
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as handle:
        handle.write(text)


def supervise(command, parent_fd):
    """Keep a test group owned after controller/Runner death, including SIGKILL."""
    require(sys.platform == "linux", "Process supervision requires Linux")
    interrupted = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: interrupted.append(number))
    owner = worker_identity() if os.environ.get("GITHUB_ACTIONS") == "true" else None
    lock_fds = ()
    try:
        os.fstat(9)
        lock_fds = (9,)
    except OSError:
        pass
    process = subprocess.Popen(command, start_new_session=True, pass_fds=lock_fds)
    reason = None
    try:
        record = os.environ.get("NPU_PROCESS_RECORD")
        if record:
            write_json(
                record,
                {
                    "supervisor": process_identity(os.getpid()),
                    "test": process_identity(process.pid),
                },
            )
        while True:
            # WNOWAIT keeps the group leader's PID reserved until group cleanup.
            if (
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                is not None
            ):
                break
            if interrupted:
                reason = f"supervisor interrupted: signal={interrupted[0]}"
                break
            ready, _, _ = select.select([parent_fd], [], [], 0.1)
            if ready and not os.read(parent_fd, 1):
                reason = "phase controller exited; reclaiming its test group"
                break
            if owner is not None and owner_status(owner) != "alive":
                reason = "Runner worker exited; reclaiming its test group"
                break
    finally:
        if reason:
            try:
                print(reason, file=sys.stderr, flush=True)
            except BrokenPipeError:
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(0.2)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        os.close(parent_fd)
        exit_record = os.environ.get("NPU_PROCESS_EXIT_RECORD")
        if exit_record:
            write_json(
                exit_record,
                {"process_exit_code": process.returncode, "supervisor_reason": reason},
            )
    if reason:
        return 128 + (interrupted[0] if interrupted else signal.SIGTERM)
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def lifecycle_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    phase = commands.add_parser("phase")
    phase.add_argument("--work", required=True)
    phase.add_argument("--phase", choices=("accuracy", "performance"), required=True)
    phase.add_argument("--stage", required=True)
    phase.add_argument("--exit", type=int, required=True)
    logger = commands.add_parser("log")
    logger.add_argument("--work", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--work", required=True)
    collector.add_argument("--accuracy", required=True)
    collector.add_argument("--performance", required=True)
    ack = commands.add_parser("ack")
    for name in (
        "work",
        "outcome",
        "artifact-id",
        "artifact-digest",
        "accuracy",
        "performance",
    ):
        ack.add_argument(f"--{name}", required=True)
    gc = commands.add_parser("gc")
    gc.add_argument("--mode", choices=("dry-run", "apply"), required=True)
    args = parser.parse_args(argv)
    policy = load_policy(args.policy)
    now = time.time()
    if args.command in ("phase", "collect", "ack"):
        _, owned = checked_work(policy, args.work)
        require(owned["owner"] == worker_identity(), "Another worker owns this run")
        require(
            owned["run_id"] == os.environ["GITHUB_RUN_ID"]
            and owned["attempt"] == os.environ["GITHUB_RUN_ATTEMPT"],
            "Another job owns this run",
        )
    if args.command == "init":
        identity = {
            name: os.environ[variable]
            for name, variable in (
                ("repository", "GITHUB_REPOSITORY"),
                ("run_id", "GITHUB_RUN_ID"),
                ("attempt", "GITHUB_RUN_ATTEMPT"),
                ("sha", "NPU_CHECKOUT_REF"),
                ("ref", "GITHUB_REF"),
                ("runner", "RUNNER_NAME"),
            )
        }
        work = initialize(policy, identity, worker_identity(), now)
        with Path(os.environ["GITHUB_ENV"]).open("a") as handle:
            handle.write(f"NPU_CI_WORK={work}\n")
        print(work)
    elif args.command == "phase":
        phase_state(policy, args.work, args.phase, args.stage, args.exit, now)
        if args.phase == "accuracy" and (Path(args.work) / "fetch.tsv").exists():
            with (Path(args.work) / "fetch.tsv").open() as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            append_summary(
                "\n### Source download\n\n| Attempt | Started (UTC) | Finished (UTC) | Seconds | Exit |\n| --- | --- | --- | --- | --- |\n"
                + "".join("| " + " | ".join(row) + " |\n" for row in rows)
            )
    elif args.command == "log":
        checked_work(policy, args.work)
        write_json(
            Path(args.work) / "bootstrap.owner.json", process_identity(os.getpid())
        )
        stream_bootstrap(args.work, sys.stdin.buffer, sys.stdout.buffer)
    elif args.command == "collect":
        bundle = collect(
            policy,
            args.work,
            now,
            {"accuracy": args.accuracy, "performance": args.performance},
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
            handle.write(
                f"path={bundle}\nretention_days={policy['artifact_retention_days']}\nupload_attempts={policy['upload_attempts']}\n"
            )
        print(
            json.dumps(
                {
                    "path": str(bundle),
                    "files": [str(p) for p in regular_files(bundle)],
                    "retention_days": policy["artifact_retention_days"],
                    "upload_attempts": policy["upload_attempts"],
                }
            )
        )
    elif args.command == "ack":
        receipt = acknowledge(
            policy,
            args.work,
            args.outcome,
            args.artifact_id,
            args.artifact_digest,
            args.accuracy,
            args.performance,
            now,
        )
        link = f"https://github.com/{policy['repository']}/actions/runs/{os.environ['GITHUB_RUN_ID']}/artifacts/{args.artifact_id}"
        append_summary(
            f"\n### Evidence archive\n\nAccuracy: `{args.accuracy}`; performance: `{args.performance}`.\n\nUpload confirmed: `{receipt['uploaded']}`; work directory removed: `{receipt['work_removed']}`.\n"
            + (
                f"\n[Download evidence]({link})\n\nSHA-256: `{args.artifact_digest}`\n"
                if receipt["uploaded"]
                else "\nUpload failed. All local evidence and work files have been retained.\n"
            )
        )
        return 0 if receipt["uploaded"] and receipt["work_removed"] else 1
    else:
        import fcntl

        with (Path(policy["state_root"]) / "gc.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another cleanup is active; skipped")
                return 0
            result = garbage_collect(policy, args.mode, now)
        print(json.dumps(result, indent=2))
        if result["pending"]:
            print(
                "::warning::Unarchived NPU evidence is older than 24 hours; manual handling required"
            )
        return 1 if result["errors"] or result["pending"] else 0
    return 0


# Phase execution


def environment_packages():
    versions = {}
    for name in ("torch", "torch-npu", "triton-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def positive_seconds(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("accuracy", "performance"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=positive_seconds, required=True)
    parser.add_argument(
        "--github-summary",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY") or None,
    )
    return parser.parse_args(argv)


def load_inputs(repo_root, phase, work):
    root = Path(repo_root).resolve(strict=True)
    task = load_task(root / CONFIG_PATH)
    reporting = globals()
    if phase == "accuracy":
        plan, plan_digest = freeze_plan(root, task, work, capture_test, save_text)
    else:
        report = decode_record((Path(work) / "accuracy/report.json").read_text())
        require(report["exit_code"] == 0, "Accuracy did not complete successfully")
        plan_digest = report["metadata"]["plan_sha256"]
        plan = read_frozen(root, task, work, plan_digest)
    return root, task, plan, plan_digest, reporting


def stop_process_group(process):
    """Stop the private test process group, including surviving children."""
    if sys.platform == "linux" and process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def capture_test(command, repo_root, log_path, timeout_seconds):
    """Stream merged output to disk and console; preserve the child exit code."""
    process = None
    parent_write = None
    parent_read = None
    saved = False
    capture_error = None
    path = Path(log_path)
    try:
        with path.open("xb") as log:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            started = time.monotonic()
            deadline = started + timeout_seconds
            next_heartbeat = started + 30
            try:
                passed_fds = ()
                if sys.platform == "linux":
                    parent_read, parent_write = os.pipe()
                    passed_fds = (parent_read,)
                    try:
                        os.fstat(9)
                        passed_fds += (9,)
                    except OSError:
                        pass
                    command = [
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "supervise",
                        str(parent_read),
                        *command,
                    ]
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    bufsize=0,
                    pass_fds=passed_fds,
                    env=dict(
                        os.environ,
                        NPU_PROCESS_RECORD=str(path.with_suffix(".owner.json")),
                        NPU_PROCESS_EXIT_RECORD=str(path.with_suffix(".exit.json")),
                    ),
                )
                if parent_read is not None:
                    os.close(parent_read)
                    parent_read = None
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        now = time.monotonic()
                        if now >= next_heartbeat:
                            print(
                                f"Test is still running: elapsed={now-started:.1f}s, log={path}",
                                flush=True,
                            )
                            next_heartbeat = now + 30
                        remaining = deadline - now
                        if remaining <= 0:
                            raise TimeoutError(
                                f"Test exceeded {timeout_seconds} seconds"
                            )
                        for key, _ in selector.select(min(remaining, 0.2)):
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            if log.write(chunk) != len(chunk):
                                raise OSError("Incomplete test log write")
                            log.flush()
                            sys.stdout.write(decoder.decode(chunk))
                            sys.stdout.flush()
                sys.stdout.write(decoder.decode(b"", final=True))
                sys.stdout.flush()
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process is not None:
                    try:
                        stop_process_group(process)
                    finally:
                        process.stdout.close()
                for fd in (parent_read, parent_write):
                    if fd is not None:
                        os.close(fd)
            log.flush()
            os.fsync(log.fileno())
        saved = True
    except Exception as exc:
        capture_error = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, TimeoutError):
            try:
                with path.open("r+b") as retained:
                    retained.flush()
                    os.fsync(retained.fileno())
                saved = True
            except Exception as save_error:
                capture_error = {
                    "type": type(save_error).__name__,
                    "message": str(save_error),
                }
    child = {}
    if sys.platform == "linux" and process is not None:
        try:
            child = read_json(path.with_suffix(".exit.json"))
        except Exception as exc:
            capture_error = {
                "type": type(exc).__name__,
                "message": f"Process exit evidence missing: {exc}",
            }
            saved = False
    return {
        **child,
        "exit_code": None if process is None else process.returncode,
        "log_path": str(path),
        "log_saved": saved,
        "capture_error": capture_error,
    }


def save_text(path, text):
    raw = text.encode("utf-8")
    with Path(path).open("xb") as handle:
        if handle.write(raw) != len(raw):
            raise OSError(f"Incomplete evidence write: {path}")
        handle.flush()
        os.fsync(handle.fileno())
    if Path(path).read_bytes() != raw:
        raise OSError(f"Saved evidence differs: {path}")


def run_operator(root, entry, phase, plan, reporting, output_dir, timeout):
    name = entry["name"]
    directory = Path(output_dir) / name
    log = directory / (phase + ".log")
    result = {
        "operator": name,
        "phase": phase,
        "status": "failed",
        "exit_code": None,
        "log_path": str(log),
        "structured_log_valid": False,
        "log_saved": False,
    }
    try:
        directory.mkdir()
        command = [
            sys.executable,
            "-B",
            "-u",
            str(root / entry["script"]),
            "--mode",
            phase,
        ]
        captured = capture_test(command, root, log, timeout)
        result.update(exit_code=captured["exit_code"], log_saved=captured["log_saved"])
        timing = plan["timing"] if phase == "performance" else {}
        result = reporting["summarize_phase"](
            log,
            name,
            phase,
            plan[phase],
            captured["exit_code"],
            **timing,
        )
        result["log_saved"] = captured["log_saved"]
        result["capture_error"] = captured["capture_error"]
        if captured["capture_error"] is not None:
            result["status"] = "failed"
            result["validation_error"] = result.get("error")
            result["error_type"] = captured["capture_error"]["type"]
            result["error"] = captured["capture_error"]["message"]
        save_text(
            directory / "process.json",
            json.dumps(captured, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        if captured["exit_code"] is not None:
            save_text(directory / (phase + ".rc"), str(captured["exit_code"]) + "\n")
        save_text(
            directory / "result.json",
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
    except Exception as exc:
        result["status"] = "failed"
        result["log_saved"] = False
        result["evidence_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result.setdefault("error_type", type(exc).__name__)
        previous = result.get("error")
        message = f"Operator execution/evidence error: {type(exc).__name__}: {exc}"
        result["error"] = f"{previous}; {message}" if previous else message
    return result


def execute_phase(
    root,
    task,
    plan,
    plan_digest,
    reporting,
    phase,
    output_dir,
    timeout,
    github_summary=None,
):
    reporting["require"](phase in task["execution"]["phases"], "Unknown phase")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "repo_root": str(root),
        "python": sys.executable,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_sha256": plan["task_sha256"],
        "plan_sha256": plan_digest,
        "source_sha": plan["source_sha"],
        "environment": {
            "python_version": sys.version,
            "packages": environment_packages(),
            "device_ids": task["resources"]["device_ids"],
            "environment_entry": os.environ.get("NPU_ENVIRONMENT_ENTRY"),
            "environment_sha256": os.environ.get("NPU_ENVIRONMENT_SHA256"),
        },
        "timeout_seconds": timeout,
        "lock_wait_seconds": os.environ.get("NPU_LOCK_WAIT_SECONDS"),
        "evidence_directory": str(output),
    }
    save_text(
        output / "context.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    results = []
    actions = os.environ.get("GITHUB_ACTIONS") == "true"
    deadline = time.monotonic() + timeout
    for entry in task["operators"]:
        verify_identity(root, task, plan)
        require(time.monotonic() < deadline, "Phase execution timeout")
        name = entry["name"]
        started = time.monotonic()
        print(f"Execution started: {name}/{phase}", flush=True)
        if actions:
            print(f"::group::{name}/{phase}", flush=True)
        try:
            result = run_operator(
                root,
                entry,
                phase,
                plan["operators"][name],
                reporting,
                output,
                max(0.001, deadline - time.monotonic()),
            )
        finally:
            if actions:
                print("::endgroup::", flush=True)
        print(
            f"Execution finished: {name}/{phase}, exit={result['exit_code']}, "
            f"elapsed={time.monotonic()-started:.2f}s, status={result['status']}",
            flush=True,
        )
        verify_identity(root, task, plan)
        require(
            file_digest(output.parent / "frozen-plan.json") == plan_digest,
            "Frozen plan changed during execution",
        )
        results.append(result)
        if not reporting["phase_succeeded"](result, phase):
            detail = result.get("error") or f"process exit={result['exit_code']}"
            report_error(f"{name}/{phase}: {detail}")
            print(
                f"{name}/{phase} failed; remaining operators will not run.", flush=True
            )
            break
    metadata["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = reporting["make_report"](results, metadata, task, phase)
    print(reporting["render_summary"](report), flush=True)
    saved = reporting["save_report"](report, output, github_summary)
    print("Structured report:", saved["json"]["path"], flush=True)
    print("Markdown summary:", saved["markdown"]["path"], flush=True)
    return report["exit_code"]


def report_error(message):
    print(message, file=sys.stderr, flush=True)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{escaped}", flush=True)


def save_failure_summary(path, phase, message):
    if path is not None:
        # Escape raw errors rather than interpreting them as Markdown/HTML.
        import html

        text = (
            f"\n### ARM NPU phase execution failed\n\nPhase: `{phase}`. Remaining tests have been stopped.\n\n"
            f"<pre>{html.escape(message)}</pre>\n"
        )
        with Path(path).open("a", encoding="utf-8") as handle:
            if handle.write(text) != len(text):
                raise OSError("Incomplete failure summary write")


class PhaseInterrupted(BaseException):
    def __init__(self, signum):
        super().__init__(f"Interrupted by signal {signum}")
        self.signum = signum


def handle_interrupt(signum, frame):
    # Allow the active capture's finally block to finish process cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise PhaseInterrupted(signum)


def phase_main(argv=None):
    args = parse_args(argv)
    phase_started = time.monotonic()
    previous_handlers = {}
    previous_pythonpath = os.environ.get("PYTHONPATH")
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_interrupt)
        root, task, plan, plan_digest, reporting = load_inputs(
            args.repo_root, args.phase, args.output_dir.parent
        )
        paths = [
            root / "tokenspeed-kernel/python",
            root / "tokenspeed-kernel-npu/python",
        ]
        for path in paths:
            if not path.is_dir():
                raise FileNotFoundError(f"Checkout source directory missing: {path}")
        pythonpath = os.pathsep.join(str(path) for path in paths)
        if previous_pythonpath:
            pythonpath += os.pathsep + previous_pythonpath
        os.environ["PYTHONPATH"] = pythonpath
        if os.environ.get("GITHUB_ACTIONS") == "true" and args.github_summary is None:
            raise RuntimeError("GITHUB_STEP_SUMMARY is required inside Actions")
        print("Phase:", args.phase, flush=True)
        print("Python:", sys.executable, flush=True)
        print("Source directory:", root, flush=True)
        remaining = args.timeout_seconds - (time.monotonic() - phase_started)
        require(remaining > 0, "Phase timeout during plan preparation")
        return execute_phase(
            root,
            task,
            plan,
            plan_digest,
            reporting,
            args.phase,
            args.output_dir,
            remaining,
            args.github_summary,
        )
    except PhaseInterrupted as exc:
        message = f"Phase interrupted: signal={exc.signum}; remaining tests have been stopped."
        report_error(message)
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            save_text(Path(args.output_dir) / "executor-error.log", message + "\n")
            save_failure_summary(args.github_summary, args.phase, message)
        except Exception as summary_error:
            report_error(f"Failed to save summary: {summary_error}")
        return 128 + exc.signum
    except Exception as exc:
        message = f"Phase execution failed: {type(exc).__name__}: {exc}"
        detail = traceback.format_exc()
        sys.stderr.write(detail)
        report_error(message)
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            save_text(Path(args.output_dir) / "executor-error.log", detail)
            save_failure_summary(args.github_summary, args.phase, message)
        except Exception as summary_error:
            report_error(f"Failed to save summary: {summary_error}")
        return 1
    finally:
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


# Pytest collection and standalone test adapter


class NpuPytestPlugin:
    """Collect the actual parametrized items and reject incomplete NPU sessions."""

    def __init__(self, phase, collect_only=False):
        self.phase = phase
        self.collect_only = collect_only
        self.plan = None
        self.module = None
        self.expected = 0
        self.completed = 0
        self.passed = 0
        self.failures = []
        self.current = None

    def pytest_configure(self, config):
        # Pytest 8 assertion formatting reads this even without its terminal plugin.
        if not hasattr(config.option, "verbose"):
            config.option.verbose = 0

    def pytest_collectreport(self, report):
        if report.failed:
            self.failures.append(str(report.longrepr))
            print(str(report.longrepr), file=sys.stderr, flush=True)

    def pytest_collection_modifyitems(self, session, config, items):
        if not items:
            if not self.failures:
                self.failures.append("NPU suite collected zero tests")
                print(self.failures[-1], file=sys.stderr, flush=True)
            return
        modules = {item.module for item in items}
        require(len(modules) == 1, "Expected exactly one registered operator module")
        module = self.module = modules.pop()
        plan = {
            "schema_version": 1,
            "operator": module.OPERATOR,
            "accuracy": [],
            "performance": [],
            "timing": {
                "warmup": module.WARMUP,
                "iterations": module.ITERS,
                "sample_count": module.SAMPLES,
            },
        }
        selected, deselected = [], []
        for item in items:
            kind = item.originalname
            require(
                kind in ("test_accuracy", "test_performance"),
                "Unregistered test function",
            )
            phase = kind.removeprefix("test_")
            key = "case" if phase == "accuracy" else "config"
            params = item.callspec.params[key]
            metadata = getattr(module, phase + "_metadata")(params)
            plan[phase].append({key: params, "metadata": metadata})
            (
                selected if self.collect_only or phase == self.phase else deselected
            ).append(item)
        validate_operator(plan, module.OPERATOR)
        self.plan = plan
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected
        self.expected = len(selected)
        require(self.expected > 0, "NPU phase selected zero tests")

    def pytest_collection_finish(self, session):
        if self.collect_only or self.module is None:
            return
        phase = self.phase
        key = "case" if phase == "accuracy" else "config"
        fields = {
            (
                "expected_cases" if phase == "accuracy" else "expected_configs"
            ): self.expected,
            "case_names" if phase == "accuracy" else "config_names": [
                item[key]["name"] for item in self.plan[phase]
            ],
        }
        self.module.emit(phase, "started", **fields)

    def pytest_runtest_logstart(self, nodeid, location):
        self.current = nodeid.split("[")[-1].rstrip("]")

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.completed += 1
            self.passed += int(report.passed)
        if report.when == "setup" and (report.failed or report.skipped):
            self.current = None
        if report.failed or report.skipped:
            self.failures.append(str(report.longrepr))
            print(str(report.longrepr), file=sys.stderr, flush=True)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.collect_only:
            if not self.plan or not self.expected:
                session.exitstatus = 1
            return
        ok = (
            exitstatus == 0
            and not self.failures
            and self.completed == self.expected
            and self.expected > 0
        )
        if not ok:
            session.exitstatus = 1
        if self.module is None:
            return
        if self.phase == "accuracy":
            fields = {
                "expected_cases": self.expected,
                "completed_cases": self.completed,
                "passed_cases": self.passed,
                "failed_cases": self.completed - self.passed,
            }
        else:
            fields = {
                "expected_configs": self.expected,
                "completed_configs": self.passed,
            }
        if not ok:
            fields.update(
                error_type="PytestFailure",
                error="\n".join(self.failures)
                or f"Incomplete pytest session: exit={exitstatus}",
            )
            fields["current_case" if self.phase == "accuracy" else "current_config"] = (
                self.current
            )
        self.module.emit(self.phase, "completed" if ok else "failed", **fields)


def test_entry(script, argv=None):
    import contextlib
    import io

    import pytest

    parser = argparse.ArgumentParser(
        description="Run a registered NPU operator with pytest"
    )
    parser.add_argument(
        "--mode", choices=("accuracy", "performance", "all"), required=True
    )
    parser.add_argument("--collect-plan", action="store_true")
    args = parser.parse_args(argv)
    # Only registered files are collected. Dedicated execution intentionally does
    # not load GPU-oriented parent conftest fixtures or third-party plugins.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    command = [
        str(script.resolve()),
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "-x",
        "-s",
        "-p",
        "no:terminal",
        "-o",
        "addopts=",
    ]
    if args.collect_plan:
        require(args.mode == "all", "Plan collection requires --mode all")
        plugin = NpuPytestPlugin("all", collect_only=True)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            rc = pytest.main(command + ["--collect-only"], plugins=[plugin])
        require(
            rc == 0 and plugin.plan is not None,
            "Pytest plan collection failed: " + captured.getvalue(),
        )
        print(json.dumps(plugin.plan, sort_keys=True, allow_nan=False))
        return 0
    phases = ("accuracy", "performance") if args.mode == "all" else (args.mode,)
    for phase in phases:
        rc = pytest.main(command, plugins=[NpuPytestPlugin(phase)])
        if rc != 0:
            return int(rc)
    return 0


# Environment and bounded host admission

NPU_ENVIRONMENT = Path("/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh")


def run_task(work, mode):
    import fcntl
    import platform
    import pwd

    work = Path(work).resolve(strict=True)
    root = work / "checkout"
    require(
        platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"),
        "Linux ARM64 is required",
    )
    require(
        pwd.getpwuid(os.geteuid()).pw_name == "ci-npu", "The ci-npu account is required"
    )
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    require(
        sha == os.environ.get("NPU_CHECKOUT_REF", os.environ.get("GITHUB_SHA")),
        "Checkout SHA mismatch",
    )
    require(
        not subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip(),
        "Checkout contains modified or untracked files",
    )
    task = load_task(root / CONFIG_PATH)
    os.environ["NPU_ENVIRONMENT_ENTRY"] = str(NPU_ENVIRONMENT)
    os.environ["NPU_ENVIRONMENT_SHA256"] = file_digest(NPU_ENVIRONMENT)
    environment = subprocess.check_output(
        [
            "bash",
            "-c",
            'set -e; source "$1" >/dev/null; env -0',
            "npu-env",
            str(NPU_ENVIRONMENT),
        ]
    )
    for item in environment.split(b"\0"):
        if b"=" in item:
            key, value = os.fsdecode(item).split("=", 1)
            os.environ[key] = value
    for name, path in {
        "TMPDIR": work / "tmp",
        "XDG_CACHE_HOME": work / "cache",
        "TRITON_CACHE_DIR": work / "cache/triton",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    phases = ("accuracy", "performance") if mode == "all" else (mode,)
    for phase in phases:
        started = time.monotonic()
        fd = os.open("/srv/npu-ci/locks/host-all.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            os.dup2(fd, 9, inheritable=True)
            if fd != 9:
                os.close(fd)
            print(
                f"Waiting for host lock: limit={task['timeouts']['lock_wait_seconds']}s",
                flush=True,
            )
            while True:
                try:
                    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    require(
                        time.monotonic() - started
                        < task["timeouts"]["lock_wait_seconds"],
                        "Host lock wait timed out; tests were not started",
                    )
                    time.sleep(0.1)
            os.environ["NPU_LOCK_WAIT_SECONDS"] = str(
                round(time.monotonic() - started, 3)
            )
            print("Host lock acquired; starting " + phase, flush=True)
            rc = phase_main(
                [
                    "--repo-root",
                    str(root),
                    "--phase",
                    phase,
                    "--output-dir",
                    str(work / phase),
                    "--timeout-seconds",
                    str(task["timeouts"]["phase_execution_seconds"]),
                ]
            )
            if rc:
                return rc
        finally:
            os.close(9)
    return 0


def fetch_source(work, repository, sha):
    checkout = work / "checkout"
    for attempt in range(1, 5):
        started = time.time()
        print(
            f"Fetching source: attempt {attempt}/4, SHA={sha}",
            flush=True,
        )
        result = capture_test(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "credential.helper=",
                "fetch",
                "--no-tags",
                "--depth=1",
                f"https://github.com/{repository}.git",
                sha,
            ],
            work,
            work / f"fetch-{attempt}.log",
            60,
        )
        write_json(work / f"fetch-{attempt}.json", result)
        print(f"Source fetch attempt {attempt}/4 finished: {result}", flush=True)
        with (work / "fetch.tsv").open("a") as output:
            output.write(
                f"{attempt}\t{utc(started)}\t{utc(time.time())}\t{time.time()-started:.1f}\t{result['exit_code']}\n"
            )
        timed_out = (
            result["capture_error"] is not None
            and result["capture_error"]["type"] == "TimeoutError"
        )
        require(
            (result["log_saved"] and not result["capture_error"])
            or (timed_out and result["log_saved"]),
            f"Source fetch evidence failed: {result}",
        )
        if result["exit_code"] == 0 and not timed_out:
            break
        if (
            not timed_out and result["exit_code"] in (130, 143, -2, -15)
        ) or attempt == 4:
            raise RuntimeError(f"Source fetch failed after attempt {attempt}: {result}")
        print("Retrying source fetch in 30 seconds", flush=True)
        time.sleep(30)


def prepare_task(argv):
    """Bootstrap exact sources from the installed, root-owned runtime."""
    import platform
    import pwd

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--timeout-minutes", type=int, required=True)
    args = parser.parse_args(argv)
    policy = load_policy(args.policy)
    require(
        platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64"),
        "Linux ARM64 is required",
    )
    require(
        pwd.getpwuid(os.geteuid()).pw_name == "ci-npu", "The ci-npu account is required"
    )
    require(
        os.environ["RUNNER_ENVIRONMENT"] == "self-hosted", "Expected self-hosted runner"
    )
    require(
        os.environ["GITHUB_SERVER_URL"] == "https://github.com",
        "Unsupported GitHub server",
    )
    require(
        os.environ["RUNNER_NAME"] == os.environ["TOKENSPEED_NPU_ARM_RUNNER_NAME"],
        "Runner name mismatch",
    )
    repository = os.environ["GITHUB_REPOSITORY"]
    require(
        repository == "lightseekorg/tokenspeed"
        or repository == os.environ.get("TOKENSPEED_NPU_CI_REPOSITORY"),
        "Repository not allowed",
    )
    identity = {
        name: os.environ[key]
        for name, key in (
            ("repository", "GITHUB_REPOSITORY"),
            ("run_id", "GITHUB_RUN_ID"),
            ("attempt", "GITHUB_RUN_ATTEMPT"),
            ("sha", "NPU_CHECKOUT_REF"),
            ("ref", "GITHUB_REF"),
            ("runner", "RUNNER_NAME"),
        )
    }
    work = initialize(policy, identity, worker_identity(), time.time())
    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        output.write(f"NPU_CI_WORK={work}\n")
    checkout = work / "checkout"
    try:
        require(args.config == CONFIG_PATH, "Unsupported task config")
        prepared = capture_test(
            ["git", "init", "-q", str(checkout)], work, work / "init.log", 30
        )
        require(
            prepared["exit_code"] == 0
            and prepared["log_saved"]
            and not prepared["capture_error"],
            "Source init failed: " + str(prepared),
        )
        fetch_source(work, repository, identity["sha"])
        prepared = capture_test(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "core.hooksPath=/dev/null",
                "checkout",
                "-q",
                "--detach",
                "FETCH_HEAD",
            ],
            work,
            work / "checkout.log",
            30,
        )
        require(
            prepared["exit_code"] == 0
            and prepared["log_saved"]
            and not prepared["capture_error"],
            "Source checkout failed: " + str(prepared),
        )
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        require(actual == identity["sha"], "Checkout SHA mismatch")
        require(
            Path(__file__).read_bytes()
            == (checkout / "test/ci_system/npu/pipeline.py").read_bytes(),
            "Host pipeline runtime does not match selected source",
        )
        require(
            Path(policy["task_path"]).read_bytes()
            == (checkout / CONFIG_PATH).read_bytes(),
            "Host task policy does not match selected source",
        )
        task = load_task(checkout / CONFIG_PATH)
        require(
            task["name"] == args.name and task["runner"] == args.runner,
            "Matrix identity does not match task",
        )
        require(
            task["timeouts"]["job_minutes"] == args.timeout_minutes,
            "Job timeout does not match task",
        )
        write_json(work / "source.json", identity)
        print(f"Prepared exact source: {actual}", flush=True)
        return 0
    except BaseException:
        save_text(work / "bootstrap.log", traceback.format_exc())
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "supervise":
        return supervise(argv[2:], int(argv[1]))
    if argv and argv[0] == "prepare":
        return prepare_task(argv[1:])
    if argv and argv[0] == "run":
        parser = argparse.ArgumentParser()
        parser.add_argument("--work", required=True)
        parser.add_argument(
            "--mode", choices=("accuracy", "performance", "all"), required=True
        )
        args = parser.parse_args(argv[1:])
        return run_task(args.work, args.mode)
    if argv and argv[0] == "lifecycle":
        return lifecycle_cli(argv[1:])
    return phase_main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        report_error(f"NPU task failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
