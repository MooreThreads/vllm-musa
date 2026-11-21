#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build vLLM-Omni on a pinned vLLM-MUSA base.
#
#   bash docker/build_omni_image.sh --omni-ref origin/main
#   bash docker/build_omni_image.sh --omni-ref v0.33.0rc1 --base <image> --tag my:tag
#   bash docker/build_omni_image.sh --omni-ref origin/main --dry-run
#
# The Omni source is exported with `git archive` from a local clone when one is
# available, so the build is reproducible and never picks up a dirty tree; set
# --omni-repo to clone inside the build instead.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE_IMAGE="${VLLM_MUSA_IMAGE:-}"
OMNI_REF="origin/main"
OMNI_REPO="https://github.com/vllm-project/vllm-omni.git"
OMNI_CLONE="${VLLM_OMNI_CLONE:-}"
TAG=""
DRY_RUN=0
PYPI_INDEX_URL="${PYPI_INDEX_URL:-}"

usage() {
    sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --base IMAGE        pinned vLLM-MUSA base image (or $VLLM_MUSA_IMAGE)
  --omni-ref REF      vLLM-Omni ref to build (default: origin/main)
  --omni-clone DIR    local vllm-omni clone to export from (or $VLLM_OMNI_CLONE)
  --omni-repo URL     clone this inside the build when no local clone is given
  --tag TAG           image tag (default: vllm-omni-musa:<omni-sha>)
  --dry-run           print the resolved plan and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base) BASE_IMAGE="$2"; shift 2 ;;
        --omni-ref) OMNI_REF="$2"; shift 2 ;;
        --omni-clone) OMNI_CLONE="$2"; shift 2 ;;
        --omni-repo) OMNI_REPO="$2"; shift 2 ;;
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

VLLM_MUSA_COMMIT="$(git rev-parse HEAD)"
VLLM_MUSA_REF="$(git describe --tags --exact-match 2>/dev/null || git branch --show-current)"

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/omni-src"

OMNI_COMMIT="(resolved in the build)"
if [[ -n "${OMNI_CLONE}" ]]; then
    # `git rev-parse` rather than a .git directory test: a worktree's .git is a
    # file, and worktrees are how this workspace keeps concurrent branches.
    if ! git -C "${OMNI_CLONE}" rev-parse --git-dir >/dev/null 2>&1; then
        echo "error: ${OMNI_CLONE} is not a git checkout" >&2
        exit 2
    fi
    OMNI_COMMIT="$(git -C "${OMNI_CLONE}" rev-parse "${OMNI_REF}")"
    # An immutable export: no dirty tree, no moving branch, no untracked files.
    git -C "${OMNI_CLONE}" archive --format=tar "${OMNI_COMMIT}" > "${STAGE}/omni-src/vllm-omni.tar"
    echo "${OMNI_COMMIT}" > "${STAGE}/omni-src/omni-commit.txt"
fi

if [[ -z "${TAG}" ]]; then
    if [[ "${OMNI_COMMIT}" == "(resolved in the build)" ]]; then
        TAG="vllm-omni-musa:${OMNI_REF//\//-}"
    else
        TAG="vllm-omni-musa:${OMNI_COMMIT:0:9}"
    fi
fi

cat <<EOF
base image        ${BASE_IMAGE}
vllm-musa         ${VLLM_MUSA_REF} (${VLLM_MUSA_COMMIT:0:9})
vllm-musa pin     $(awk -F= '$1 == "VLLM_TAG" {print $2}' third_party/PINS) @ $(awk -F= '$1 == "VLLM_COMMIT" {print substr($2, 1, 9)}' third_party/PINS)
vllm-omni ref     ${OMNI_REF}
vllm-omni commit  ${OMNI_COMMIT}
vllm-omni source  ${OMNI_CLONE:-clone ${OMNI_REPO} in-build}
tag               ${TAG}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "(dry run, not building)"
    exit 0
fi

# The build context is this repo plus the staged Omni export. Clear the target
# first: `cp -r src dst` nests under an existing dst, which would hide
# vllm-omni.tar one level down and make the Dockerfile fall back to cloning the
# default branch -- an image whose contents do not match the ref in its tag.
rm -rf "${REPO_ROOT}/.omni-build-context"
cp -r "${STAGE}/omni-src" "${REPO_ROOT}/.omni-build-context"
trap 'rm -rf "${STAGE}" "${REPO_ROOT}/.omni-build-context"' EXIT

docker build \
    --network host \
    --file docker/omni.Dockerfile \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg OMNI_REF="${OMNI_REF}" \
    --build-arg OMNI_REPO="${OMNI_REPO}" \
    --build-arg OMNI_CONTEXT=".omni-build-context" \
    --build-arg PYPI_INDEX_URL="${PYPI_INDEX_URL}" \
    --build-arg VLLM_MUSA_COMMIT="${VLLM_MUSA_COMMIT}" \
    --build-arg VLLM_MUSA_REF="${VLLM_MUSA_REF}" \
    --tag "${TAG}" \
    "${REPO_ROOT}"

echo
echo "built ${TAG}"
echo "  docker run --rm --entrypoint cat ${TAG} /opt/vllm-omni-compat.json"
echo "  docker run --rm --entrypoint python ${TAG} -m vllm_musa.omni_compat status"
