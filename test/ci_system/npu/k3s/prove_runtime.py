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


import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECTS = {
    "tokenspeed": "python",
    "tokenspeed-scheduler": "tokenspeed-scheduler",
    "tokenspeed-engine": "tokenspeed-engine",
    "tokenspeed-kernel": "tokenspeed-kernel/python",
    "tokenspeed-kernel-npu": "tokenspeed-kernel-npu",
}


def write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def record_install():
    root = Path(os.environ["NPU_CI_SOURCE_ROOT"])
    work = Path(os.environ["NPU_CI_WORK"])
    if Path(sys.prefix).resolve() != (work / "model-venv").resolve():
        raise ValueError("Installation proof requires this task's virtualenv")
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    (work / "runtime/pip-check.log").write_text(check.stdout + check.stderr)
    check.check_returncode()
    installed = {}
    for name, relative in PROJECTS.items():
        distribution = importlib.metadata.distribution(name)
        direct = json.loads(distribution.read_text("direct_url.json"))
        if direct["url"] != (root / relative).as_uri():
            raise ValueError(
                f"Runtime component is not installed from this checkout: {name}"
            )
        item = {"version": distribution.version, "direct_url": direct}
        if (
            name == "tokenspeed-kernel"
            and f"git{os.environ['TOKENSPEED_CI_SOURCE_SHA'][:8]}"
            not in distribution.version
        ):
            raise ValueError("Installed kernel version has a different source SHA")
        if name == "tokenspeed-scheduler":
            import tokenspeed_scheduler.tokenspeed_scheduler_ext as extension

            binary = Path(extension.__file__).resolve(strict=True)
            if not binary.is_relative_to(Path(sys.prefix).resolve()):
                raise ValueError(
                    "Scheduler extension is outside this task's virtualenv"
                )
            libraries = subprocess.check_output(["ldd", str(binary)], text=True)
            if "not found" in libraries:
                raise ValueError("Scheduler extension has missing libraries")
            item["binary"] = {
                "path": str(binary),
                "sha256": digest(binary),
                "ldd": libraries,
            }
        if name == "tokenspeed-engine":
            import tokenspeed_engine
            import tokenspeed_engine._native as native

            item["binaries"] = {}
            for label, binary in {
                "native": Path(native.__file__),
                "host": Path(tokenspeed_engine.host_binary()),
            }.items():
                binary = binary.resolve(strict=True)
                if not binary.is_relative_to(Path(sys.prefix).resolve()):
                    raise ValueError("Engine binary is outside this task's virtualenv")
                with binary.open("rb") as handle:
                    header = handle.read(20)
                if (
                    header[:6] != b"\x7fELF\x02\x01"
                    or int.from_bytes(header[18:20], "little") != 183
                ):
                    raise ValueError("Engine requires native Linux ARM64 ELF binaries")
                libraries = subprocess.check_output(["ldd", str(binary)], text=True)
                if "not found" in libraries:
                    raise ValueError("Engine binary has missing libraries")
                item["binaries"][label] = {
                    "path": str(binary),
                    "sha256": digest(binary),
                    "machine": "aarch64",
                    "ldd": libraries,
                }
        installed[name] = item
    write_json(
        work / "runtime/install-evidence.json",
        {
            "schema_version": 1,
            "components": "full-runtime",
            "source_sha": os.environ["TOKENSPEED_CI_SOURCE_SHA"],
            "source_root": str(root),
            "pip_check_passed": True,
            "installed": installed,
            "pip_report_sha256": digest(work / "runtime/pip-report.json"),
        },
    )


record_install()
