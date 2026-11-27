# SPDX-License-Identifier: Apache-2.0
# Build with docker/build_omni_image.sh and a bundled vllm-openai base.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS omni-src

ARG OMNI_REPO=https://github.com/vllm-project/vllm-omni.git
ARG OMNI_REF=origin/main
ARG OMNI_SOURCE_REFRESH=

WORKDIR /src

# A local export takes precedence over cloning and excludes working tree edits.
COPY omni-src/ /staged/
RUN set -eux; \
    : "${OMNI_SOURCE_REFRESH}"; \
    mkdir /src/vllm-omni; \
    if [ -f /staged/vllm-omni.tar ]; then \
        cp /staged/omni-commit.txt /src/omni-commit.txt; \
        tar -xf /staged/vllm-omni.tar -C /src/vllm-omni; \
    else \
        git clone --filter=blob:none --no-checkout -- "${OMNI_REPO}" /tmp/omni-clone; \
        if ! git -C /tmp/omni-clone rev-parse --verify --quiet --end-of-options "${OMNI_REF}^{commit}" > /src/omni-commit.txt; then \
            git -C /tmp/omni-clone rev-parse --verify --end-of-options "refs/remotes/origin/${OMNI_REF}^{commit}" > /src/omni-commit.txt; \
        fi; \
        git -C /tmp/omni-clone checkout --detach "$(cat /src/omni-commit.txt)"; \
        git -C /tmp/omni-clone archive HEAD | tar -xf - -C /src/vllm-omni; \
        rm -rf /tmp/omni-clone; \
    fi

FROM ${BASE_IMAGE} AS omni-base
ARG PYPI_INDEX_URL=
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /vllm-workspace

# Preserve the base's installed Python stack, including private MUSA wheels.
COPY docker/omni/verify_install.py /opt/omni-compat/verify_install.py
RUN python /opt/omni-compat/verify_install.py --snapshot /opt/vllm-omni-foundation.txt

COPY --from=omni-src /src/omni-commit.txt /opt/vllm-omni-commit.txt
RUN python /opt/omni-compat/verify_install.py --write-version

FROM omni-base AS headless-wheels
COPY docker/omni/build_headless_wheels.py docker/omni/headless-upstream.txt /opt/omni-compat/
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"; \
    python -m pip install wheel; \
    python -m pip download --require-hashes --only-binary=:all: --no-deps \
        --dest /tmp/upstream-wheels -r /opt/omni-compat/headless-upstream.txt; \
    python /opt/omni-compat/build_headless_wheels.py \
        --input-dir /tmp/upstream-wheels --output-dir /wheels

FROM omni-base AS omni-wheel
COPY --from=omni-src /src/vllm-omni /src/vllm-omni
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"; \
    VLLM_OMNI_TARGET_DEVICE=musa \
    VLLM_OMNI_VERSION_OVERRIDE="$(cat /opt/vllm-omni-version.txt)" \
    python -m pip wheel --no-deps --wheel-dir /wheels /src/vllm-omni

FROM omni-base AS omni-install
COPY docker/omni/verify_shim.py tools/check_omni_compat.py /opt/omni-compat/
RUN python -m vllm_musa.omni_compat install-pth

ARG OMNI_REF=origin/main
ARG VLLM_MUSA_COMMIT
ARG VLLM_MUSA_REF
LABEL org.opencontainers.image.source="https://github.com/MooreThreads/vllm-musa" \
      org.opencontainers.image.revision="${VLLM_MUSA_COMMIT}" \
      com.mthreads.vllm-musa.ref="${VLLM_MUSA_REF}" \
      com.mthreads.vllm-omni.ref="${OMNI_REF}"
ENTRYPOINT []
CMD ["/bin/bash"]

FROM omni-install AS test
COPY --from=omni-src /src/vllm-omni /vllm-workspace/vllm-omni
RUN --mount=type=bind,from=headless-wheels,source=/wheels,target=/headless-wheels \
    --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"; \
    export VLLM_OMNI_TARGET_DEVICE=musa; \
    export VLLM_OMNI_VERSION_OVERRIDE="$(cat /opt/vllm-omni-version.txt)"; \
    python -m pip install -c /opt/vllm-omni-foundation.txt \
        /headless-wheels/*.whl -e /vllm-workspace/vllm-omni; \
    python -m pip check; \
    python /opt/omni-compat/verify_install.py --mode editable; \
    omni_root="$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm_omni").origin).parent.parent)')"; \
    python /opt/omni-compat/check_omni_compat.py \
        --omni-root "$omni_root" --report /opt/vllm-omni-compat.json; \
    python /opt/omni-compat/verify_shim.py --driverless --report /opt/vllm-omni-shim-verify.json
FROM omni-install AS omni
RUN --mount=type=bind,from=headless-wheels,source=/wheels,target=/headless-wheels \
    --mount=type=bind,from=omni-wheel,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"; \
    python -m pip install -c /opt/vllm-omni-foundation.txt /headless-wheels/*.whl /wheels/*.whl; \
    python -m pip check; \
    python /opt/omni-compat/verify_install.py --mode wheel; \
    omni_root="$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm_omni").origin).parent.parent)')"; \
    python /opt/omni-compat/check_omni_compat.py \
        --omni-root "$omni_root" --report /opt/vllm-omni-compat.json; \
    python /opt/omni-compat/verify_shim.py --driverless --report /opt/vllm-omni-shim-verify.json
# Keep Omni assets separate from the base vLLM-MUSA examples and benchmarks.
COPY --from=omni-src /src/vllm-omni/examples /vllm-workspace/omni/examples
COPY --from=omni-src /src/vllm-omni/benchmarks /vllm-workspace/omni/benchmarks
