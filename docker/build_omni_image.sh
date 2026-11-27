#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build vLLM-Omni on a pinned vLLM-MUSA base.
#
#   bash docker/build_omni_image.sh --base <image>
#   bash docker/build_omni_image.sh --base <image> --omni-ref <branch-or-tag>
#   bash docker/build_omni_image.sh --base <image> --omni-clone ../vllm-omni --target test
#   bash docker/build_omni_image.sh --base <image> --dry-run
#
# The Omni source is exported with `git archive` from a local clone when one is
# available. Without a local clone, the build fetches the selected remote ref.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE_IMAGE="${VLLM_MUSA_IMAGE:-}"
OMNI_REF="origin/main"
OMNI_REPO="https://github.com/vllm-project/vllm-omni.git"
OMNI_CLONE="${VLLM_OMNI_CLONE:-}"
TAG=""
TARGET="omni"
DRY_RUN=0
PYPI_INDEX_URL="${PYPI_INDEX_URL:-}"

usage() {
    sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --base IMAGE        pinned vLLM-MUSA base image (or $VLLM_MUSA_IMAGE)
  --omni-ref REF      vLLM-Omni branch, tag or commit (default: origin/main)
  --omni-clone DIR    local vllm-omni clone to export from (or $VLLM_OMNI_CLONE)
  --omni-repo URL     fetch from this repo inside the build when no local clone is given
  --target TARGET     omni (default) or test for an editable install
  --tag TAG           image tag (default: vllm-omni-musa:<ref-or-local-sha>)
  --dry-run           print the resolved plan and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base) BASE_IMAGE="$2"; shift 2 ;;
        --omni-ref) OMNI_REF="$2"; shift 2 ;;
        --omni-clone) OMNI_CLONE="$2"; shift 2 ;;
        --omni-repo) OMNI_REPO="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${BASE_IMAGE}" ]]; then
    echo "error: no base image. Pass --base or set VLLM_MUSA_IMAGE." >&2
    echo "       Resolve one with scripts/prebuilt_image_catalog.py in the workspace." >&2
    exit 2
fi

case "${TARGET}" in
    omni|test) ;;
    *) echo "error: target must be omni or test" >&2; exit 2 ;;
esac
VLLM_MUSA_COMMIT="$(git rev-parse HEAD)"
VLLM_MUSA_REF="$(git describe --tags --exact-match 2>/dev/null || git branch --show-current)"

OMNI_COMMIT=""
if [[ -n "${OMNI_CLONE}" ]]; then
    # `git rev-parse` rather than a .git directory test: a worktree's .git is a
    # file, and worktrees are how this workspace keeps concurrent branches.
    if ! git -C "${OMNI_CLONE}" rev-parse --git-dir >/dev/null 2>&1; then
        echo "error: ${OMNI_CLONE} is not a git checkout" >&2
        exit 2
    fi
    if ! OMNI_COMMIT="$(git -C "${OMNI_CLONE}" rev-parse --verify --quiet --end-of-options "${OMNI_REF}^{commit}")"; then
        OMNI_COMMIT="$(git -C "${OMNI_CLONE}" rev-parse --verify --end-of-options "refs/remotes/origin/${OMNI_REF}^{commit}")"
    fi
fi

if [[ -z "${TAG}" ]]; then
    if [[ -n "${OMNI_COMMIT}" ]]; then
        TAG="vllm-omni-musa:${OMNI_COMMIT:0:9}"
    else
        TAG="vllm-omni-musa:${OMNI_REF//\//-}"
    fi
    if [[ "${TARGET}" == "test" ]]; then TAG+="-test"; fi
fi

cat <<EOF
base image        ${BASE_IMAGE}
vllm-musa         ${VLLM_MUSA_REF} (${VLLM_MUSA_COMMIT:0:9})
vllm-musa pin     $(awk -F= '$1 == "VLLM_TAG" {print $2}' third_party/PINS) @ $(awk -F= '$1 == "VLLM_COMMIT" {print substr($2, 1, 9)}' third_party/PINS)
vllm-omni ref     ${OMNI_REF}
vllm-omni commit  ${OMNI_COMMIT:-(resolved in the build)}
vllm-omni source  ${OMNI_CLONE:-fetch from ${OMNI_REPO} in-build}
target            ${TARGET}
tag               ${TAG}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "(dry run, not building)"
    exit 0
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/omni-src"

OMNI_SOURCE_REFRESH=""
if [[ -n "${OMNI_CLONE}" ]]; then
    # An immutable export: no dirty tree, no moving branch, no untracked files.
    git -C "${OMNI_CLONE}" archive --format=tar "${OMNI_COMMIT}" > "${STAGE}/omni-src/vllm-omni.tar"
    echo "${OMNI_COMMIT}" > "${STAGE}/omni-src/omni-commit.txt"
elif [[ ! "${OMNI_REF}" =~ ^[0-9a-f]{40}$ ]]; then
    # Refresh movable refs on each invocation. A full SHA remains cacheable.
    OMNI_SOURCE_REFRESH="${STAGE##*/}"
fi

# Keep concurrent builds isolated and send only the required source files.
mkdir -p "${STAGE}/docker" "${STAGE}/tools"
cp docker/omni.Dockerfile "${STAGE}/docker/"
cp -r docker/omni "${STAGE}/docker/"
cp tools/check_omni_compat.py "${STAGE}/tools/"

DOCKER_BUILDKIT=1 docker build \
    --network host \
    --file "${STAGE}/docker/omni.Dockerfile" \
    --target "${TARGET}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg OMNI_REF="${OMNI_REF}" \
    --build-arg OMNI_REPO="${OMNI_REPO}" \
    --build-arg OMNI_SOURCE_REFRESH="${OMNI_SOURCE_REFRESH}" \
    --build-arg PYPI_INDEX_URL="${PYPI_INDEX_URL}" \
    --build-arg VLLM_MUSA_COMMIT="${VLLM_MUSA_COMMIT}" \
    --build-arg VLLM_MUSA_REF="${VLLM_MUSA_REF}" \
    --tag "${TAG}" \
    "${STAGE}"

echo
echo "built ${TAG}"
echo "  docker run --rm --entrypoint cat ${TAG} /opt/vllm-omni-shim-verify.json"
echo "  docker run --rm --entrypoint python ${TAG} -m vllm_musa.omni_compat status"
