# SPDX-License-Identifier: Apache-2.0
# Build with docker/build_omni_image.sh and a bundled vllm-openai base.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS omni-src

ARG OMNI_REPO=https://github.com/vllm-project/vllm-omni.git
ARG OMNI_REF=e937f1c9e3032d34ce08b3e8b54c6028b8e85274


WORKDIR /src

# A pre-staged archive wins over a clone so a build can be reproduced offline
# and from a dirty working tree; either way the result is an immutable export.
COPY omni-src/ /staged/
RUN set -eux; \
    if [ -f /staged/vllm-omni.tar ]; then \
        test "$(cat /staged/omni-commit.txt)" = "${OMNI_REF}"; \
        mkdir /src/vllm-omni; \
        tar -xf /staged/vllm-omni.tar -C /src/vllm-omni; \
    else \
        git init /src/vllm-omni; \
        git -C /src/vllm-omni fetch --depth=1 "${OMNI_REPO}" "${OMNI_REF}"; \
        git -C /src/vllm-omni checkout --detach FETCH_HEAD; \
        test "$(git -C /src/vllm-omni rev-parse HEAD)" = "${OMNI_REF}"; \
    fi; \
    printf '%s\n' "${OMNI_REF}" > /src/omni-commit.txt


FROM ${BASE_IMAGE} AS omni-base
ARG PYPI_INDEX_URL=
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /vllm-workspace

# Preserve the base's installed Python stack, including private MUSA wheels.
COPY docker/omni/verify_install.py /opt/omni-compat/verify_install.py
RUN python /opt/omni-compat/verify_install.py --snapshot /opt/vllm-omni-foundation.txt

COPY --from=omni-src /src/omni-commit.txt /opt/vllm-omni-commit.txt
RUN python /opt/omni-compat/verify_install.py --write-version

FROM omni-base AS omni-wheels
# Also build openai-whisper here because PyPI distributes it as an sdist.
COPY docker/omni/build_headless_wheels.py docker/omni/headless-upstream.txt /opt/omni-compat/
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"; \
    python -m pip install "setuptools>=77.0.3,<81.0.0" wheel "setuptools-scm>=8.0"; \
    python -m pip download --require-hashes --only-binary=:all: --no-deps \
        --dest /tmp/upstream-wheels -r /opt/omni-compat/headless-upstream.txt; \
    python /opt/omni-compat/build_headless_wheels.py \
        --input-dir /tmp/upstream-wheels --output-dir /wheels; \
    python -m pip wheel --no-deps --wheel-dir /wheels openai-whisper==20250625

COPY --from=omni-src /src/vllm-omni /src/vllm-omni
COPY tools/check_omni_compat.py /opt/omni-compat/check_omni_compat.py
RUN python /opt/omni-compat/check_omni_compat.py \
        --omni-root /src/vllm-omni --report /opt/vllm-omni-compat-preflight.json
RUN --mount=type=cache,target=/root/.cache/pip \
    VLLM_OMNI_TARGET_DEVICE=musa \
    VLLM_OMNI_VERSION_OVERRIDE="$(cat /opt/vllm-omni-version.txt)" \
    python -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheels /src/vllm-omni

FROM omni-base AS omni-install
COPY --from=omni-wheels /opt/vllm-omni-compat-preflight.json /opt/vllm-omni-compat-preflight.json
COPY docker/omni/verify_shim.py tools/check_omni_compat.py /opt/omni-compat/
RUN python -m vllm_musa.omni_compat install-pth

ARG OMNI_REF=e937f1c9e3032d34ce08b3e8b54c6028b8e85274
ARG VLLM_MUSA_COMMIT
ARG VLLM_MUSA_REF
LABEL org.opencontainers.image.source="https://github.com/MooreThreads/vllm-musa" \
      org.opencontainers.image.revision="${VLLM_MUSA_COMMIT}" \
      com.mthreads.vllm-musa.ref="${VLLM_MUSA_REF}" \
      com.mthreads.vllm-omni.ref="${OMNI_REF}"
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

FROM omni-install AS test
COPY --from=omni-src /src/vllm-omni /vllm-workspace/vllm-omni
RUN --mount=type=bind,from=omni-wheels,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"; \
    export VLLM_OMNI_TARGET_DEVICE=musa; \
    export VLLM_OMNI_VERSION_OVERRIDE="$(cat /opt/vllm-omni-version.txt)"; \
    python -m pip install -c /opt/vllm-omni-foundation.txt "setuptools>=77.0.3,<81.0.0" wheel "setuptools-scm>=8.0"; \
    python -m pip install --only-binary=:all: --no-build-isolation -c /opt/vllm-omni-foundation.txt \
        /wheels/cosmos_guardrail-*.whl /wheels/retinaface_py-*.whl \
        /wheels/openai_whisper-*.whl -e /vllm-workspace/vllm-omni; \
    python -m pip check; \
    python /opt/omni-compat/verify_install.py --mode editable; \
    omni_root="$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm_omni").origin).parent.parent)')"; \
    python /opt/omni-compat/check_omni_compat.py \
        --omni-root "$omni_root" --report /opt/vllm-omni-compat.json; \
    python /opt/omni-compat/verify_shim.py --driverless --report /opt/vllm-omni-shim-verify.json
ENTRYPOINT []
CMD ["/bin/bash"]

FROM omni-install AS omni
RUN --mount=type=bind,from=omni-wheels,source=/wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"; \
    python -m pip install --only-binary=:all: -c /opt/vllm-omni-foundation.txt /wheels/*.whl; \
    python -m pip check; \
    python /opt/omni-compat/verify_install.py --mode wheel; \
    omni_root="$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm_omni").origin).parent.parent)')"; \
    python /opt/omni-compat/check_omni_compat.py \
        --omni-root "$omni_root" --report /opt/vllm-omni-compat.json; \
    python /opt/omni-compat/verify_shim.py --driverless --report /opt/vllm-omni-shim-verify.json
# Keep Omni assets separate from the base vLLM-MUSA examples and benchmarks.
COPY --from=omni-src /src/vllm-omni/examples /vllm-workspace/omni/examples
COPY --from=omni-src /src/vllm-omni/benchmarks /vllm-workspace/omni/benchmarks
ENTRYPOINT ["vllm-omni", "serve", "--omni"]
CMD []
