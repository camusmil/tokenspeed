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

"""Explicit mode selection shared by standalone operator test scripts."""

import argparse
import json
from collections.abc import Callable


def parse_mode(argv, description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode", choices=("accuracy", "performance", "all"), required=True
    )
    parser.add_argument(
        "--collect-plan",
        action="store_true",
        help="Print the plan without loading device libraries",
    )
    return parser.parse_args(argv)


def run_modes(
    mode: str, accuracy: Callable[[], int], performance: Callable[[], int]
) -> int:
    """Run the selected phases, stopping on the first nonzero exit or exception."""
    if mode not in ("accuracy", "performance", "all"):
        raise ValueError(f"Unknown operator test mode: {mode}")
    phases = (
        (accuracy, performance)
        if mode == "all"
        else (accuracy if mode == "accuracy" else performance,)
    )
    for phase in phases:
        result = phase()
        if type(result) is not int:
            raise TypeError("Operator phase must return an integer exit code")
        if result != 0:
            return result
    return 0


def emit_plan(
    operator, cases, configs, timing, accuracy_metadata, performance_metadata
):
    """Serialize the same definitions used by execution, without device work."""
    print(
        json.dumps(
            {
                "schema_version": 1,
                "operator": operator,
                "accuracy": [
                    {"case": c, "metadata": accuracy_metadata(c)} for c in cases
                ],
                "performance": [
                    {"config": c, "metadata": performance_metadata(c)} for c in configs
                ],
                "timing": timing,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
