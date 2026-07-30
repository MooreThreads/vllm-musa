#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_OMNI_SOURCE="${VLLM_OMNI_SOURCE:-${REPO_ROOT}/../vllm-omni}"
if [[ ! -f "${VLLM_OMNI_SOURCE}/pyproject.toml" || \
      ! -d "${VLLM_OMNI_SOURCE}/vllm_omni" ]]; then
    echo "VLLM_OMNI_SOURCE is not a vllm-omni source checkout: ${VLLM_OMNI_SOURCE}" >&2
    exit 1
fi
VLLM_OMNI_SOURCE="$(cd "${VLLM_OMNI_SOURCE}" && pwd)"

if [[ -z "${VLLM_MUSA_IMAGE:-}" ]]; then
    echo "VLLM_MUSA_IMAGE must name the vllm-musa image to extend" >&2
    exit 1
fi
IMAGE_TAG="${IMAGE_TAG:-vllm-omni-musa:latest}"
COMMON_WORKDIR="${COMMON_WORKDIR:-/app}"

derived_commit=unknown
derived_ref=unknown
derived_version=
if git -C "${VLLM_OMNI_SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    derived_commit="$(git -C "${VLLM_OMNI_SOURCE}" rev-parse HEAD)"
    derived_ref="$({
        git -C "${VLLM_OMNI_SOURCE}" describe --tags --exact-match 2>/dev/null ||
            git -C "${VLLM_OMNI_SOURCE}" branch --show-current
    } || true)"
    derived_ref="${derived_ref:-detached}"
    if [[ "${derived_ref}" =~ ^v[0-9] ]]; then
        derived_version="${derived_ref#v}+musa"
    fi
fi
VLLM_OMNI_COMMIT="${VLLM_OMNI_COMMIT:-${derived_commit}}"
VLLM_OMNI_REF="${VLLM_OMNI_REF:-${derived_ref}}"
VLLM_OMNI_VERSION_OVERRIDE="${VLLM_OMNI_VERSION_OVERRIDE:-${derived_version}}"

docker build \
    --network host \
    -f "${REPO_ROOT}/docker/vllm-omni.Dockerfile" \
    -t "${IMAGE_TAG}" \
    --build-arg BASE_IMAGE="${VLLM_MUSA_IMAGE}" \
    --build-arg COMMON_WORKDIR="${COMMON_WORKDIR}" \
    --build-arg VLLM_OMNI_COMMIT="${VLLM_OMNI_COMMIT}" \
    --build-arg VLLM_OMNI_REF="${VLLM_OMNI_REF}" \
    --build-arg VLLM_OMNI_VERSION_OVERRIDE="${VLLM_OMNI_VERSION_OVERRIDE}" \
    "$@" \
    "${VLLM_OMNI_SOURCE}"
