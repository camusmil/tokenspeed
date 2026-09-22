#!/usr/bin/env bash
set -euo pipefail

# 设定默认artifacts和cache路径
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_root="${TS_CI_ARTIFACT_ROOT:-/mnt/nfs01/${USER}/tokenspeed-slurm}"
cache_dir="${TS_CI_CACHE_DIR:-/mnt/lustre01/${USER}/tokenspeed-cache}"
# 设定默认镜像
container_image="${TS_CI_CONTAINER_IMAGE:-ghcr.io/lightseekorg/tokenspeed-runner:cu130-torch-2.13.0-flashinfer-0.6.18@sha256:d6067daeeb1fafecc531d45e282797076e1cd2e2c16eaa90712634dd76a709ca}"

if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    config="$1"
    shift
    set -- --config "${config}" "$@"
fi

# 将设置参数传入slurm_submit.py
exec python3 "${repo_root}/test/ci_system/slurm_submit.py" \
    --repo-root "${repo_root}" \
    --artifact-root "${artifact_root}" \
    --cache-dir "${cache_dir}" \
    --container-image "${container_image}" \
    "$@"
