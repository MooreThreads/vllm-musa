#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
BASE_IMAGE="${BASE_IMAGE:-ubuntu:22.04}"

MUSA_APT_SOURCE="${MUSA_APT_SOURCE:-https://dl.mthreads.com/repo/repository/ubuntu2204/}"
INSTALL_MUSA_STACK="${INSTALL_MUSA_STACK:-auto}"
MUSA_RUNTIME_VERSION="${MUSA_RUNTIME_VERSION:-5.2}"
MCCL_VERSION="${MCCL_VERSION:-2.4.0}"

# Python package indexes. MUSA/MT wheels (torch, torch_musa, mate, ...) resolve
# from the internal Moore Threads index only; ordinary third-party wheels resolve
# from public PyPI. See docker/musa.Dockerfile for why the two are not merged.
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
MUSA_PIP_INDEX_URL="${MUSA_PIP_INDEX_URL:-https://dl.mthreads.com/repo/api/pypi/pypi/simple}"

PYTORCH_RELEASE="$(
    awk -F'==' '/^torch==/ {gsub(/\.\*/, "", $2); print $2; exit}' \
        requirements/musa_private.txt
)"
if [[ -z "${PYTORCH_RELEASE}" ]]; then
    echo "Failed to derive PyTorch release from requirements/musa_private.txt" >&2
    exit 1
fi
PYTORCH_RELEASE_TAG="$(printf '%s' "${PYTORCH_RELEASE}" | sed 's/[^A-Za-z0-9_.-]/_/g')"

VLLM_MUSA_COMMIT="$(git rev-parse HEAD)"
VLLM_MUSA_REF="$(git describe --tags --exact-match 2>/dev/null || git branch --show-current)"
VLLM_TAG="$(awk -F= '$1 == "VLLM_TAG" {print $2; exit}' third_party/PINS)"

MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.13}"
BUILD_VLLM_RS="${BUILD_VLLM_RS:-1}"

IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-vllm-musa}"
IMAGE_FLAVOR="${IMAGE_FLAVOR:-ubuntu22.04_py${PYTHON_VERSION}_musa_runtime_${MUSA_RUNTIME_VERSION}_pytorch_release_${PYTORCH_RELEASE_TAG}}"
IMAGE_TAG="${IMAGE_TAG:-${IMAGE_REPOSITORY}:${IMAGE_FLAVOR}}"

docker build \
    --network host \
    -f docker/musa.Dockerfile \
    -t "${IMAGE_TAG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg MUSA_APT_SOURCE="${MUSA_APT_SOURCE}" \
    --build-arg PYPI_INDEX_URL="${PYPI_INDEX_URL}" \
    --build-arg MUSA_PIP_INDEX_URL="${MUSA_PIP_INDEX_URL}" \
    --build-arg INSTALL_MUSA_STACK="${INSTALL_MUSA_STACK}" \
    --build-arg MUSA_RUNTIME_VERSION="${MUSA_RUNTIME_VERSION}" \
    --build-arg MCCL_VERSION="${MCCL_VERSION}" \
    --build-arg MOONCAKE_VERSION="${MOONCAKE_VERSION}" \
    --build-arg BUILD_VLLM_RS="${BUILD_VLLM_RS}" \
    --build-arg VLLM_MUSA_COMMIT="${VLLM_MUSA_COMMIT}" \
    --build-arg VLLM_MUSA_REF="${VLLM_MUSA_REF}" \
    --build-arg VLLM_TAG="${VLLM_TAG}" \
    "$@" \
    .
