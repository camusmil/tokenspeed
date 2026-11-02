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
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path

PROJECTS = {
    "tokenspeed": "python",
    "tokenspeed-scheduler": "tokenspeed-scheduler",
    "tokenspeed-engine": "tokenspeed-engine",
    "tokenspeed-kernel": "tokenspeed-kernel/python",
    "tokenspeed-kernel-npu": "tokenspeed-kernel-npu",
}
BASE_REQUIREMENTS = [
    "torch==2.9.0+cpu",
    "torch-npu==2.9.0.post2",
    "triton-ascend==3.2.1",
    "numpy==1.26.4",
    "transformers==5.12.0",
    "apache-tvm-ffi==0.1.13.post3",
]


def verify_wheelhouse(directory, *, owner_uid):
    def protected(path, *, directory):
        info = path.lstat()
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected_type(info.st_mode)
            or info.st_uid != owner_uid
            or info.st_mode & 0o022
        ):
            raise ValueError(f"NPU wheelhouse requires protected owned files: {path}")

    protected(directory, directory=True)
    manifest_path = directory / "manifest.json"
    protected(manifest_path, directory=False)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    files = manifest["files"]
    if manifest["schema_version"] != 1 or not isinstance(files, dict) or not files:
        raise ValueError("Invalid NPU wheelhouse manifest")
    if {path.name for path in directory.iterdir()} != set(files) | {"manifest.json"}:
        raise ValueError("NPU wheelhouse contents differ from its manifest")
    for name, expected in files.items():
        if (
            Path(name).name != name
            or not name.endswith(".whl")
            or not isinstance(expected, str)
            or re.fullmatch("[0-9a-f]{64}", expected) is None
        ):
            raise ValueError(f"Invalid NPU wheelhouse entry: {name}")
        path = directory / name
        protected(path, directory=False)
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"NPU wheelhouse digest mismatch: {name}")
    return {
        "path": str(directory),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "wheel_count": len(files),
        "files": files,
        "pip_index_disabled": True,
    }


def verify_engine_dependencies(directory, source, *, owner_uid):
    def protected(path):
        info = path.lstat()
        if (
            not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
            or info.st_uid != owner_uid
            or info.st_mode & 0o022
        ):
            raise ValueError(f"Unprotected NPU engine build dependency: {path}")

    protected(directory)
    manifest_path = directory / "manifest.json"
    protected(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    files = manifest["files"]
    if manifest["schema_version"] != 1 or not isinstance(files, dict) or not files:
        raise ValueError("Invalid NPU engine dependency manifest")
    actual = set()
    for path in directory.rglob("*"):
        protected(path)
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(files):
        raise ValueError("NPU engine dependencies differ from their manifest")
    for name, expected in files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(expected, str)
            or re.fullmatch("[0-9a-f]{64}", expected) is None
        ):
            raise ValueError(f"Invalid NPU engine dependency entry: {name}")
        with (directory / relative).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"NPU engine dependency digest mismatch: {name}")
    for name, relative in {
        "cargo_lock_sha256": "Cargo.lock",
        "xgrammar_build_sha256": "crates/ts-grammar/CMakeLists.txt",
    }.items():
        with (source / "tokenspeed-engine" / relative).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != manifest[name]:
                raise ValueError(
                    f"Engine source needs refreshed dependencies: {relative}"
                )
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "file_count": len(files),
        "rust_version": manifest["rust_version"],
        "cargo_lock_sha256": manifest["cargo_lock_sha256"],
        "xgrammar_build_sha256": manifest["xgrammar_build_sha256"],
        "cargo_offline": True,
    }


def prepare_engine_build(root, work, directory):
    evidence = verify_engine_dependencies(directory, root, owner_uid=0)
    rust = directory / "rust"
    version = subprocess.check_output(
        [str(rust / "bin/rustc"), "--version"], text=True
    ).strip()
    if version != f"rustc {evidence['rust_version']}":
        raise ValueError("NPU Rust compiler differs from the protected manifest")
    cargo_home = work / "cache/cargo-home"
    cargo_home.mkdir(parents=True, exist_ok=True)
    (cargo_home / "config.toml").write_text(
        '[source.crates-io]\nreplace-with = "vendored-sources"\n'
        "[source.vendored-sources]\ndirectory = "
        + json.dumps(str(directory / "vendor"))
        + "\n[net]\noffline = true\n"
    )
    python_lib = Path(os.environ["NPU_CI_PYTHON_LIBDIR"])
    library = sysconfig.get_config_var("LDLIBRARY")
    if (
        sys.implementation.name != "cpython"
        or not sysconfig.get_config_var("Py_ENABLE_SHARED")
        or sysconfig.get_config_var("Py_DEBUG")
        or sysconfig.get_config_var("Py_GIL_DISABLED")
        or not library
        or not (python_lib / library).is_file()
    ):
        raise ValueError(
            "Engine requires release CPython and its deployed shared library"
        )
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    pyo3 = work / "runtime/pyo3-config.txt"
    pyo3.write_text(
        "implementation=CPython\n"
        f"version={python_version}\n"
        "shared=true\n"
        f"target_abi=CPython-gil_enabled-{python_version}\n"
        f"lib_name=python{python_version}\n"
        f"lib_dir={python_lib}\n"
        f"executable={work / 'model-venv/bin/python'}\n"
        "pointer_width=64\n"
        "build_flags=\n"
        "suppress_build_script_link_lines=false\n"
    )
    openssl = Path(os.environ["NPU_CI_OPENSSL_ROOT"])
    crypto = next(
        (
            p
            for p in (openssl / "lib", openssl / "lib64")
            if (p / "libcrypto.so").is_file()
        ),
        None,
    )
    if crypto is None:
        raise ValueError("Engine requires the deployed OpenSSL link library")
    evidence["python_library_dir"] = str(python_lib)
    evidence["openssl_library_dir"] = str(crypto)
    os.environ.update(
        PYO3_CONFIG_FILE=str(pyo3),
        LIBRARY_PATH=os.pathsep.join(
            filter(
                None, [str(python_lib), str(crypto), os.environ.get("LIBRARY_PATH", "")]
            )
        ),
        PATH=f"{rust / 'bin'}{os.pathsep}{os.environ['PATH']}",
        CARGO_HOME=str(cargo_home),
        CARGO_TARGET_DIR=str(work / "cache/cargo-target"),
        CARGO_NET_OFFLINE="true",
        CARGO_BUILD_JOBS="4",
        FETCHCONTENT_SOURCE_DIR_XGRAMMAR=str(directory / "xgrammar"),
        OPENSSL_DIR=os.environ["NPU_CI_OPENSSL_ROOT"],
        CMAKE_PREFIX_PATH=os.pathsep.join(
            filter(
                None,
                [
                    os.environ["NPU_CI_OPENSSL_ROOT"],
                    os.environ.get("CMAKE_PREFIX_PATH", ""),
                ],
            )
        ),
    )
    (work / "runtime/engine-build-dependencies.json").write_text(
        json.dumps(evidence) + "\n"
    )
    print(f"Verified engine build dependencies: {evidence}", flush=True)


def install_from_wheelhouse(python, wheels, requirements, *, options):
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            *options,
            *requirements,
        ],
        check=True,
    )


def install_runtime():
    root = Path(os.environ["NPU_CI_SOURCE_ROOT"])
    work = Path(os.environ["NPU_CI_WORK"])
    venv = work / "model-venv"
    python = str(venv / "bin/python")
    wheels = Path(os.environ["NPU_CI_WHEELS"])
    evidence = verify_wheelhouse(wheels, owner_uid=0)
    (work / "runtime/dependency-wheelhouse.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(
        f"Verified NPU dependency wheelhouse: {evidence['wheel_count']} wheels; "
        f"manifest sha256={evidence['manifest_sha256']}. "
        "PyPI access is disabled; missing dependencies require a wheelhouse refresh.",
        flush=True,
    )
    subprocess.run(
        [os.environ["NPU_CI_BOOTSTRAP_PYTHON"], "-m", "venv", str(venv)], check=True
    )
    constraints = work / "runtime/constraints.txt"
    constraints.write_text("\n".join(BASE_REQUIREMENTS + ["triton==3.5.0"]) + "\n")
    os.environ["PIP_CONSTRAINT"] = str(constraints)
    os.environ["PIP_NO_INDEX"] = "1"
    os.environ["PIP_FIND_LINKS"] = str(wheels)
    install_from_wheelhouse(
        python, wheels, BASE_REQUIREMENTS, options=["--only-binary=:all:"]
    )
    requirements = set()
    for relative in PROJECTS.values():
        with (root / relative / "pyproject.toml").open("rb") as handle:
            requirements.update(tomllib.load(handle)["build-system"]["requires"])
    install_from_wheelhouse(
        python, wheels, sorted(requirements), options=["--only-binary=:all:"]
    )
    prepare_engine_build(root, work, Path(os.environ["NPU_CI_ENGINE_DEPS"]))
    install_from_wheelhouse(
        python,
        wheels,
        [str(root / p) for p in PROJECTS.values()],
        options=[
            "--no-build-isolation",
            "--report",
            str(work / "runtime/pip-report.json"),
        ],
    )


if __name__ == "__main__":
    install_runtime()
