# SPDX-License-Identifier: Apache-2.0
# vLLM-Omni on a pinned vLLM-MUSA base.
#
# The base image is a released vLLM-MUSA, which pins one upstream vLLM commit
# (third_party/PINS). vLLM-Omni declares no vllm requirement and tracks vLLM
# main, so the Omni ref built here is usually newer than that pin. The drift is
# absorbed by vllm_musa.omni_compat and verified by tools/check_omni_compat.py,
# both of which run as build steps: an Omni commit whose drift the shim does not
# cover fails the build instead of a model smoke.
#
#   bash docker/build_omni_image.sh --omni-ref origin/main
#
# Build context is the vllm-musa repo; the Omni source arrives as a build arg
# repository or a pre-staged archive at ${OMNI_CONTEXT}.

# Required, with no default: a dated tag written here goes stale silently, and
# Omni declares requires-python >=3.11, so a base from the Python 3.10 line
# cannot build it at all. Resolve a base and pin its digest first --
# scripts/prebuilt_image_catalog.py in the workspace does this.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS omni-src

ARG OMNI_REPO=https://github.com/vllm-project/vllm-omni.git
ARG OMNI_REF=main
ARG OMNI_CONTEXT=omni-src

WORKDIR /src

# A pre-staged archive wins over a clone so a build can be reproduced offline
# and from a dirty working tree; either way the result is an immutable export.
COPY ${OMNI_CONTEXT}/ /staged/
RUN set -eux; \
    if [ -f /staged/vllm-omni.tar ]; then \
        mkdir -p /src/vllm-omni && tar xf /staged/vllm-omni.tar -C /src/vllm-omni; \
    else \
        git clone --filter=blob:none "${OMNI_REPO}" /src/vllm-omni && \
        git -C /src/vllm-omni checkout --detach "${OMNI_REF}"; \
    fi; \
    git -C /src/vllm-omni rev-parse HEAD > /src/omni-commit.txt 2>/dev/null \
        || cat /staged/omni-commit.txt > /src/omni-commit.txt; \
    cat /src/omni-commit.txt

# --------------------------------------------------------------------------- #
FROM ${BASE_IMAGE} AS omni

ARG OMNI_REF=main
# Empty by default: the base image already configures a working PIP_INDEX_URL
# and overriding it with a guess breaks packages that mirror carries and the
# guess does not. Pass one only to redirect deliberately.
ARG PYPI_INDEX_URL=

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /vllm-workspace

COPY --from=omni-src /src/vllm-omni /vllm-workspace/vllm-omni
COPY --from=omni-src /src/omni-commit.txt /opt/vllm-omni-commit.txt

# The contract the rest of the build is checked against: read from the base and
# recorded, so every later step and every report names the stack it ran on. This
# step records, it does not gate -- the gates are the audit passes, the
# foundation re-verify and verify_shim below, and they are what a base bump
# fails on. The Python version is part of the record: the editable install
# enforces Omni's own requires-python, so a base whose interpreter Omni refuses
# fails at that step with pip's own message rather than being worked around.
RUN set -eux; \
    python -c "import vllm, torch, sys; \
print('vllm', vllm.__version__); print('torch', torch.__version__); \
print('python', '.'.join(map(str, sys.version_info[:3])))" | tee /opt/vllm-omni-base.txt

# The shim itself. A base built from a vllm-musa that already ships
# omni_compat needs nothing here and the step is a no-op; an older base, or one
# whose shim predates the Omni ref being built, gets this checkout's pure-Python
# shim copied into its installed vllm_musa. Nothing is rebuilt.
COPY tools/check_omni_compat.py docker/omni/stage_shim.py \
     docker/omni/verify_shim.py /opt/omni-compat/
COPY vllm_musa/omni_compat /opt/omni-compat/src/vllm_musa/omni_compat
COPY vllm_musa_omni_compat_pth.py /opt/omni-compat/src/vllm_musa_omni_compat_pth.py
RUN set -eux; \
    python /opt/omni-compat/stage_shim.py --source /opt/omni-compat/src

# Gate 1, before anything is installed: can the pinned vLLM serve this Omni
# tree's imports and calls, with the shim? Source only, no GPU. The signature
# diff against the vLLM Omni targets is not run here -- it needs a second vLLM
# source tree this build deliberately does not fetch -- so the verdict line
# says so rather than printing a bare PASS.
RUN set -eux; \
    python /opt/omni-compat/check_omni_compat.py \
        --omni-root /vllm-workspace/vllm-omni \
        --report /opt/vllm-omni-compat-preflight.json

# vLLM-Omni's media and guardrail dependencies import cv2. Some MUSA base
# images contain both OpenCV distributions, whose wheels write the same cv2/
# directory. Restore the base-pinned headless wheel last, while leaving the GUI
# distribution installed for packages that declare it, so cv2 never needs libGL.
# The MUSA foundation is a contract, not a suggestion: torch, torch_musa, vllm,
# vllm-musa, MATE, numpy and the rest were built and validated together. Freezing
# everything the base installed as a pip constraint lets Omni *add* packages
# (diffusers, media libraries, pure-Python helpers) while making any attempt to
# *move* one of them a ResolutionImpossible at build time rather than a strange
# runtime failure. No hand-maintained package list to keep in sync.
#
# Scoped to the paths this interpreter owns, which is not the same as everything
# it can import. The image runs a deadsnakes Python 3.12 as the *system*
# interpreter, so Ubuntu 22.04's /usr/lib/python3/dist-packages -- about twenty
# apt packages built for python3.10 -- is still on sys.path and still shows up in
# a bare `pip freeze`. Pinning those is wrong twice over: they are not part of
# the MUSA foundation, and some have no distribution for this interpreter at all,
# so a transitive requirement on one is an unsatisfiable `==`. cryptography==3.4.8
# is the worked example: apt-provided, no cp312 wheel, and enough to fail the
# whole Omni install with ResolutionImpossible.
RUN set -eux; \
    SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    PLAT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"; \
    set -- --path "${SITE}"; \
    if [ "${PLAT}" != "${SITE}" ]; then set -- "$@" --path "${PLAT}"; fi; \
    python -m pip list "$@" --exclude-editable --format=freeze \
        | grep -vE '^(-e|#)' > /opt/vllm-omni-foundation.txt; \
    python -m pip install --no-deps \
        ${PYPI_INDEX_URL:+--index-url ${PYPI_INDEX_URL}} \
        'numpy>=1.26.4,<2'; \
    python -m pip list "$@" --exclude-editable --format=freeze \
        | grep -vE '^(-e|#)' > /opt/vllm-omni-foundation.txt; \
    # Keep Omni's opencv-python requirement deterministic. The unconstrained
    # package currently offers several large ABI-tagged wheels and makes the
    # resolver spend several minutes backtracking on the image build.
    printf '%s\n' 'opencv-python==4.10.0.84' >> /opt/vllm-omni-foundation.txt; \
    OPENCV_HEADLESS="$(awk -F== '$1 == "opencv-python-headless" {print $2}' \
        /opt/vllm-omni-foundation.txt)"; \
    if [ -z "${OPENCV_HEADLESS}" ]; then \
        OPENCV_HEADLESS="$(awk -F== '$1 == "opencv-python" {print $2}' \
            /opt/vllm-omni-foundation.txt)"; \
    fi; \
    if [ -n "${OPENCV_HEADLESS}" ]; then \
        python -m pip install --no-deps --force-reinstall \
            ${PYPI_INDEX_URL:+--index-url ${PYPI_INDEX_URL}} \
            "opencv-python-headless==${OPENCV_HEADLESS}"; \
    else \
        echo "opencv-python is absent from the pinned MUSA foundation" >&2; \
        exit 1; \
    fi; \
    python -c "import cv2; info=cv2.getBuildInformation(); gui=next((line.strip() for line in info.splitlines() if line.strip().startswith('GUI:')), ''); print('cv2', cv2.__version__, gui); assert gui.endswith('NONE'), gui"; \
    wc -l /opt/vllm-omni-foundation.txt

# The Omni source arrives as a git export with no .git, so setuptools_scm
# cannot derive a version and falls back to "dev", which its own device suffix
# turns into the invalid "dev+cpu". Both inputs are therefore set explicitly:
# the device, which is musa and not the cpu a driverless build detects, and a
# version whose major.minor equals vLLM's so Omni's
# warn_if_misaligned_vllm_version() stays quiet, with the Omni commit recorded
# in the local segment.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    VLLM_MM="$(awk '$1 == "vllm" {split($2, v, "."); print v[1] "." v[2]}' /opt/vllm-omni-base.txt)"; \
    test -n "${VLLM_MM}"; \
    OMNI_SHA="$(cut -c1-9 /opt/vllm-omni-commit.txt)"; \
    export VLLM_OMNI_TARGET_DEVICE=musa; \
    export VLLM_OMNI_VERSION_OVERRIDE="${VLLM_MM}.0+musa.g${OMNI_SHA}"; \
    echo "installing vllm-omni ${VLLM_OMNI_VERSION_OVERRIDE}"; \
    python -m pip install --no-build-isolation \
        ${PYPI_INDEX_URL:+--index-url ${PYPI_INDEX_URL}} \
        --constraint /opt/vllm-omni-foundation.txt \
        -e /vllm-workspace/vllm-omni

# Prove the contract held rather than trusting the constraint file. Omni is
# expected to have *added* packages, so only the base's own pins are compared.
RUN set -eux; \
    SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    PLAT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"; \
    set -- --path "${SITE}"; \
    if [ "${PLAT}" != "${SITE}" ]; then set -- "$@" --path "${PLAT}"; fi; \
    python -m pip list "$@" --exclude-editable --format=freeze \
        | grep -vE '^(-e|#)' > /opt/vllm-omni-foundation-after.txt; \
    sort /opt/vllm-omni-foundation.txt > /tmp/before.sorted; \
    sort /opt/vllm-omni-foundation-after.txt > /tmp/after.sorted; \
    comm -23 /tmp/before.sorted /tmp/after.sorted > /opt/vllm-omni-foundation.moved; \
    if [ -s /opt/vllm-omni-foundation.moved ]; then \
        echo "the Omni install moved packages the base had pinned:" >&2; \
        cat /opt/vllm-omni-foundation.moved >&2; \
        exit 1; \
    fi; \
    echo "foundation unchanged: $(wc -l < /opt/vllm-omni-foundation.txt) pins held"

# The startup hook, which is what makes the shim work for vllm-omni's CLI:
# Omni binds moved vLLM symbols at import time, before any plugin entry point
# runs, so the hook has to be installed by the interpreter itself.
RUN set -eux; \
    python -m vllm_musa.omni_compat install-pth; \
    python -m vllm_musa.omni_compat status | tee /opt/vllm-omni-compat-status.json

# Gate 2, against the installed trees: imports resolve, the shim is active, and
# every shim entry is still needed on this pin.
RUN set -eux; \
    python /opt/omni-compat/check_omni_compat.py \
        --omni-root /vllm-workspace/vllm-omni \
        --report /opt/vllm-omni-compat.json
# Gate 3: exercise the shim against the installed vLLM. Driverless by design --
# `import vllm_omni` activates the MUSA platform plugin, which queries device
# capability and raises without a driver, so that import is a hardware check:
#
#   python /opt/omni-compat/verify_shim.py --require-omni
#
# on a leased GPU, which is also where a model smoke belongs.
RUN set -eux; \
    python /opt/omni-compat/verify_shim.py --driverless --report /opt/vllm-omni-shim-verify.json

ARG VLLM_MUSA_COMMIT
ARG VLLM_MUSA_REF
LABEL org.opencontainers.image.source="https://github.com/MooreThreads/vllm-musa" \
      org.opencontainers.image.revision="${VLLM_MUSA_COMMIT}" \
      com.mthreads.vllm-omni.ref="${OMNI_REF}"

# The base sets ENTRYPOINT ["vllm", "serve"]; Omni has its own CLI and the
# compat status is worth seeing in the log of anything that starts here.
ENTRYPOINT []
CMD ["/bin/bash"]
