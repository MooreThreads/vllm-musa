# vllm-musa image for Ubuntu 22.04 and the MUSA apt stack. Release flow only --
# do not add validation-only wheel/tar download-and-extract logic here.

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION

FROM base AS apt_base

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        g++-12 \
        gcc-12 \
        git \
        git-lfs \
        gnupg \
        infiniband-diags \
        libcurl4-openssl-dev \
        libdrm2 \
        libibverbs-dev \
        libmkl-core \
        libmkl-gnu-thread \
        libmkl-intel-lp64 \
        libnuma1 \
        libomp-dev \
        libopenblas-base \
        libopenmpi-dev \
        librdmacm-dev \
        libssl-dev \
        libstdc++-12-dev \
        libtool \
        libyaml-dev \
        lsb-release \
        lsof \
        make \
        ninja-build \
        numactl \
        openmpi-bin \
        openssh-client \
        patchelf \
        pkg-config \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        python-is-python3 \
        rdma-core \
        unzip \
        xz-utils \
        zip && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# The MUSA PyTorch wheel links MKL with .so.2 sonames. Ubuntu 22.04 apt
# ships the same logical MKL components without that suffix.
RUN ln -sf /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so \
        /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so.2 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so \
        /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so.2 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libmkl_core.so \
        /usr/lib/x86_64-linux-gnu/libmkl_core.so.2 && \
    ldconfig

FROM apt_base AS devel

ARG MUSA_APT_SOURCE=https://dl.mthreads.com/repo/repository/ubuntu2204/
ARG INSTALL_MUSA_STACK=auto
ARG MUSA_RUNTIME_VERSION=5.2
ARG MCCL_VERSION=2.4.0
# mthreads-mtml lacks a runtime-suffixed package, so pin it by version.
ARG MUSA_MTML_VERSION=2.4.1

ENV MUSA_HOME=/usr/local/musa
ENV PATH=/usr/local/mtshmem/bin:${MUSA_HOME}/bin:${MUSA_HOME}/mudnn/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/mtshmem/lib:${MUSA_HOME}/lib:${MUSA_HOME}/mudnn/lib:/usr/local/lib

# Install the MUSA stack from the apt source. MUSA_RUNTIME_VERSION (major.minor,
# e.g. 5.2) is the single version selector and derives the "-5-2" package suffix.
RUN printf 'deb [trusted=true] %s jammy main\n' "${MUSA_APT_SOURCE}" \
        > /etc/apt/sources.list.d/musa.list && \
    if [[ "${INSTALL_MUSA_STACK}" == "0" ]]; then \
        echo "Skipping MUSA apt stack install because INSTALL_MUSA_STACK=0"; \
        exit 0; \
    fi && \
    if [[ "${INSTALL_MUSA_STACK}" == "auto" ]] && command -v mcc >/dev/null 2>&1; then \
        echo "Keeping MUSA stack from BASE_IMAGE"; \
        mcc --version || true; \
        exit 0; \
    fi && \
    apt-get update && \
    if [[ "${MUSA_RUNTIME_VERSION}" =~ ^([0-9]+)\.([0-9]+)(\.|$) ]]; then \
        runtime_major="${BASH_REMATCH[1]}"; \
        runtime_minor="${BASH_REMATCH[2]}"; \
        runtime_suffix="${runtime_major}-${runtime_minor}"; \
    else \
        echo "MUSA_RUNTIME_VERSION must start with <major>.<minor>, got ${MUSA_RUNTIME_VERSION}" >&2; \
        exit 1; \
    fi && \
    resolve_apt_package() { \
        local logical="$1"; \
        local versions="$2"; \
        local allow_unversioned="$3"; \
        shift 3; \
        local spec=""; \
        local version pkg found_version; \
        for version in ${versions}; do \
            for pkg in "$@"; do \
                if ! apt-cache show "${pkg}" >/dev/null 2>&1; then \
                    continue; \
                fi; \
                found_version="$(apt-cache madison "${pkg}" | awk -v w="${version}" '$3 ~ "^" w "([-+~:]|$)" && ex=="" {ex=$3} $3 ~ "^" w "[.]" && pf=="" {pf=$3} END {print (ex!="" ? ex : pf)}')"; \
                if [[ -n "${found_version}" ]]; then \
                    spec="${pkg}=${found_version}"; \
                    break 2; \
                fi; \
            done; \
        done; \
        if [[ -z "${spec}" && "${allow_unversioned}" == "1" ]]; then \
            for pkg in "$@"; do \
                if apt-cache show "${pkg}" >/dev/null 2>&1; then \
                    spec="${pkg}"; \
                    break; \
                fi; \
            done; \
        fi; \
        if [[ -z "${spec}" ]]; then \
            echo "No apt package found for ${logical} with versions [${versions}]; checked: $*" >&2; \
            return 1; \
        fi; \
        echo "${spec}"; \
    } && \
    # Pin the whole MUSA stack to the runtime line and install it in ONE apt
    # transaction: the source now mixes 5.1.0/5.2.0 builds, so an unversioned or
    # per-package install can split it across /usr/local/musa-5.1 and -5.2. The
    # runtime suffix pins most packages; the rest are pinned by version. Entry
    # format: "logical|version-prefixes|candidate,packages" (empty prefixes take
    # the name as-is, else matched against apt-cache madison -- an exact version
    # match wins over a longer one, while a line prefix like 5.1 matches 5.1.0).
    # TODO: replace mccl-s5000 with generic mccl once MCCL ships a unified package.
    musa_pkg_defs=( \
        "musa-toolkit||musa-toolkit-${runtime_suffix}" \
        "musa-toolkit-config||musa-toolkit-${runtime_suffix}-config-common" \
        "mtcc||mtcc-${runtime_suffix}" \
        "musa-musart||musa-musart-${runtime_suffix}" \
        "musa-mupti||musa-mupti-${runtime_suffix}" \
        "musa-mualg||musa-mualg-${runtime_suffix}" \
        "musa-muthrust||musa-muthrust-${runtime_suffix}" \
        "libmublas||libmublas-${runtime_suffix}" \
        "libmufft||libmufft-${runtime_suffix}" \
        "libmupp||libmupp-${runtime_suffix}" \
        "libmurand||libmurand-${runtime_suffix}" \
        "libmusparse||libmusparse-${runtime_suffix}" \
        "libmusolver||libmusolver-${runtime_suffix}" \
        "libmublaslt||libmublaslt-${runtime_suffix}" \
        "libmthreads-compute|${MUSA_RUNTIME_VERSION}|libmthreads-compute" \
        "libmudnn3||libmudnn3-musa-${runtime_suffix},libmudnn3-musa-${runtime_major}" \
        "libmudnn3-dev||libmudnn3-dev-musa-${runtime_suffix},libmudnn3-musa-${runtime_major}-dev" \
        "libmthreads-mtml|${MUSA_MTML_VERSION}|libmthreads-mtml" \
        "mccl-s5000|${MCCL_VERSION}|mccl-s5000" \
    ) && \
    musa_specs=() && \
    for musa_def in "${musa_pkg_defs[@]}"; do \
        IFS='|' read -r musa_logical musa_versions musa_pkgs <<< "${musa_def}"; \
        IFS=',' read -r -a musa_pkg_arr <<< "${musa_pkgs}"; \
        if [[ -n "${musa_versions}" ]]; then musa_allow_unv=0; else musa_allow_unv=1; fi; \
        musa_spec="$(resolve_apt_package "${musa_logical}" "${musa_versions}" "${musa_allow_unv}" "${musa_pkg_arr[@]}")" || exit 1; \
        echo "Pinning ${musa_logical}: ${musa_spec}"; \
        musa_specs+=("${musa_spec}"); \
    done && \
    apt-get install -y --allow-downgrades --no-install-recommends "${musa_specs[@]}" && \
    printf '%s\n' \
        "${MUSA_HOME}/lib" \
        "${MUSA_HOME}/mudnn/lib" \
        "/usr/local/mtshmem/lib" \
        "/usr/lib/x86_64-linux-gnu" \
        > /etc/ld.so.conf.d/musa-runtime.conf && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

# Point /usr/local/musa at whichever /usr/local/musa-* dir actually holds the
# runtime library (libmusart.so.5.*) and register the MUSA lib dirs, in case a
# base image aimed the symlink at a toolkit-less dir. This is a no-op on a
# correctly set-up base.
RUN real_lib="$(ls /usr/local/musa-*/lib/libmusart.so.5.* 2>/dev/null | sort -V | tail -1)"; \
    if [ -n "${real_lib}" ]; then \
        musa_dir="$(readlink -f "$(dirname "${real_lib}")/..")"; \
        if [ "$(readlink -f /usr/local/musa 2>/dev/null)" != "${musa_dir}" ]; then \
            ln -sfn "${musa_dir}" /usr/local/musa; \
            echo "musa-path: repointed /usr/local/musa -> ${musa_dir}"; \
        fi; \
    fi; \
    { echo "${MUSA_HOME}/lib"; \
      echo "${MUSA_HOME}/mudnn/lib"; \
      for d in /usr/local/musa-*/lib /usr/local/musa-*/mudnn/lib; do \
          [ -d "$d" ] && echo "$d"; \
      done; \
      echo /usr/local/mtshmem/lib; \
      echo /usr/lib/x86_64-linux-gnu; } \
      > /etc/ld.so.conf.d/musa-runtime.conf; \
    ldconfig

FROM devel AS vllm_musa_deps

# These architecture selectors are consumed only after the MUSA Python stack
# is installed. Keep them out of base/apt/devel, but retain them in the final
# image because the vllm-musa JIT uses MATE_MUSA_ARCH_LIST at runtime.
# torch_musa uses "31"; MATE parses dotted arch tokens such as "3.1".
ENV MTGPU_TARGET=mp_31 \
    TORCH_MUSA_ARCH_LIST=31 \
    MATE_MUSA_ARCH_LIST=3.1

# vllm-musa Python deps, installed before the source copy so the layers cache.
# Split across two indexes (per the MUSA release_5.2.0 wheel guide):
#   * PYPI_INDEX_URL (public PyPI): build tools, common.txt runtime deps,
#     transformers, and the MUSA wheels' ordinary deps.
#   * MUSA_PIP_INDEX_URL (internal MT index): the MUSA/MT wheels pinned in
#     requirements/musa_private.txt (torch, torch_musa, MATE, flash_attn_3, ...).
# Some MUSA names (torch, mate, apache-tvm-ffi) also exist on public PyPI, so the
# two indexes must never be merged into one resolve or pip may pick the wrong
# (CUDA/CPU) wheel. URLs live only in docker/build_image.sh and come in as build
# args; a bare `docker build` must supply --build-arg PYPI_INDEX_URL and
# MUSA_PIP_INDEX_URL.
ARG PYPI_INDEX_URL
ARG MUSA_PIP_INDEX_URL

ENV PIP_CACHE_DIR=/root/.cache/pip \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements/ /vllm-workspace/requirements/
WORKDIR /vllm-workspace

# 1. MUSA/MT wheels from the internal index only, FIRST and --no-deps (their
#    ordinary deps come in steps 2-3). MUSA torch must land before the public
#    phase: torchada/transformers declare an unpinned `torch`, so a public-first
#    resolve would pull public CUDA torch and the multi-GB nvidia-cuda-* stack.
RUN python -m pip install \
        --no-deps \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        -r requirements/musa_private.txt

# 2. Ordinary third-party wheels from public PyPI: build tools, common.txt, and
#    musa.txt's direct pins (transformers). Step 1 already satisfies `torch`.
RUN musa_public_extras="$(grep -vE '^[[:space:]]*(#|-r[[:space:]]|--|$)' requirements/musa.txt || true)" && \
    python -m pip install \
        --index-url "${PYPI_INDEX_URL}" \
        -r requirements/build.txt \
        -r requirements/common.txt \
        -r requirements/test.txt \
        ${musa_public_extras}

# 3. Fill in the MUSA wheels' ordinary deps (sympy, networkx, ...) from public
#    PyPI. Step 1 already pinned every MUSA wheel, so none are re-resolved here.
RUN python -m pip install \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        -r requirements/musa_private.txt

# MATE <0.2.6 emits invalid FX code for boolean augmented assignment.
COPY docker/patches/apply_mate_dynamo_bool_patch.py \
     docker/patches/mate-dynamo-bool-augassign.patch \
     /tmp/vllm-musa-mate-patch/
RUN python /tmp/vllm-musa-mate-patch/apply_mate_dynamo_bool_patch.py && \
    rm -rf /tmp/vllm-musa-mate-patch

FROM vllm_musa_deps AS vllm_musa_installed

ARG SKIP_THIRD_PARTY=0

# setup.py's develop_dynamic_library() installs the vendored vLLM with
# --no-deps to avoid replacing the MUSA torch stack. Route the explicit vLLM
# runtime dependency install and the numpy re-pin below through the same public
# index as the deps stage. PYPI_INDEX_URL comes from docker/build_image.sh.
ARG PYPI_INDEX_URL
ENV PIP_INDEX_URL=${PYPI_INDEX_URL}

COPY . /vllm-workspace
RUN SKIP_THIRD_PARTY="${SKIP_THIRD_PARTY}" python -m pip install \
        -e . --no-build-isolation -v && \
    python -m pip install numpy==1.26

RUN python -m pip install \
        --no-cache-dir \
        --no-deps \
        -r third_party/vllm/requirements/common.txt && \
    python -m pip install \
        --no-cache-dir \
        -r requirements/vllm_runtime_transitive.txt

RUN printf '%s\n' \
        'import importlib' \
        'import re' \
        'from pathlib import Path' \
        'from importlib.metadata import version' \
        '' \
        'def requirement_prefix(dist_name):' \
        '    pattern = re.compile(rf"^{re.escape(dist_name)}==(.+)$")' \
        '    requirement_files = (' \
        '        "requirements/musa_private.txt",' \
        '        "requirements/common.txt",' \
        '    )' \
        '    for requirement_file in requirement_files:' \
        '        for line in Path(requirement_file).read_text().splitlines():' \
        '            match = pattern.match(line.strip())' \
        '            if match:' \
        '                return match.group(1).split("*", 1)[0]' \
        '    raise RuntimeError(f"missing {dist_name} pin in {requirement_files}")' \
        '' \
        'exact_version_dists = frozenset({"torchada", "torch", "torch_musa", "torchvision", "torchaudio", "deep_ep"})' \
        'exact_version_dists |= frozenset({"mate", "mate-mubin", "flash_attn_3", "flash_mla", "deep-gemm", "flashinfer-python", "sageattention", "tilelang_musa", "apache-tvm-ffi", "torch_c_dlpack_ext"})' \
        'expected = (' \
        '    ("torchada", "torchada", requirement_prefix("torchada")),' \
        '    ("numpy", "numpy", "1.26."),' \
        '    ("torch", "torch", requirement_prefix("torch")),' \
        '    ("torch_musa", "torch_musa", requirement_prefix("torch_musa")),' \
        '    ("torchvision", "torchvision", requirement_prefix("torchvision")),' \
        '    ("torchaudio", "torchaudio", requirement_prefix("torchaudio")),' \
        '    ("mate", "mate", requirement_prefix("mate")),' \
        '    ("mate-mubin", "mate_mubin", requirement_prefix("mate-mubin")),' \
        '    ("flash_attn_3", "flash_attn_3", requirement_prefix("flash_attn_3")),' \
        '    ("flash_mla", "flash_mla", requirement_prefix("flash_mla")),' \
        '    ("deep-gemm", "deep_gemm", requirement_prefix("deep-gemm")),' \
        '    ("flashinfer-python", "flashinfer", requirement_prefix("flashinfer-python")),' \
        '    ("sageattention", "sageattention", requirement_prefix("sageattention")),' \
        '    ("deep_ep", "deep_ep", requirement_prefix("deep_ep")),' \
        '    ("tilelang_musa", "tilelang", requirement_prefix("tilelang_musa")),' \
        '    ("triton", "triton", requirement_prefix("triton")),' \
        '    ("uvloop", "uvloop", ""),' \
        '    ("pycountry", "pycountry", ""),' \
        '    ("pytest", "pytest", ""),' \
        '    ("apache-tvm-ffi", "tvm_ffi", requirement_prefix("apache-tvm-ffi")),' \
        '    ("torch_c_dlpack_ext", "torch_c_dlpack_ext", ""),' \
        ')' \
        '' \
        'for dist_name, module_name, prefix in expected:' \
        '    installed = version(dist_name)' \
        '    skip_import = (dist_name == "tilelang_musa" and installed == "0.1.12+musa.2")' \
        '    if not skip_import: importlib.import_module(module_name)' \
        '    if dist_name in exact_version_dists and installed != prefix:' \
        '        raise RuntimeError(f"{dist_name} expected exactly {prefix}, got {installed}")' \
        '    if dist_name not in exact_version_dists and prefix and not installed.startswith(prefix):' \
        '        raise RuntimeError(f"{dist_name} expected {prefix}, got {installed}")' \
        '    action = "skip import" if skip_import else "import"' \
        '    print(f"PASS {action} {module_name} version={installed}")' \
        '' \
        'for module_name in ("vllm", "vllm_musa"):' \
        '    module = importlib.import_module(module_name)' \
        '    print("PASS import %s version=%s" % (module_name, getattr(module, "__version__", "unknown")))' \
        > /tmp/vllm_musa_import_check.py && \
    python /tmp/vllm_musa_import_check.py && \
    rm /tmp/vllm_musa_import_check.py

# Build vllm-rs from the vLLM tree cloned and patched by vllm-musa setup.py,
# while keeping Rust/protoc tooling out of the runtime image.
FROM vllm_musa_installed AS vllm_rs_build

ARG BUILD_VLLM_RS=1

RUN if [[ "${BUILD_VLLM_RS}" == "1" ]]; then \
        /vllm-workspace/third_party/vllm/tools/install_protoc.sh && \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
            sh -s -- -y --profile minimal --default-toolchain none; \
    elif [[ "${BUILD_VLLM_RS}" == "0" ]]; then \
        echo "Skipping Rust toolchain install because BUILD_VLLM_RS=0"; \
    else \
        echo "Unsupported BUILD_VLLM_RS=${BUILD_VLLM_RS}" >&2; \
        exit 1; \
    fi

ENV PATH=/root/.cargo/bin:${PATH}
ENV CARGO_BUILD_JOBS=4
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true \
    CARGO_NET_RETRY=10

COPY docker/cargo-config.toml /root/.cargo/config.toml

RUN mkdir -p /tmp/vllm-rs-artifacts && \
    printf '%s\n' "${BUILD_VLLM_RS}" > /tmp/vllm-rs-artifacts/build-mode && \
    if [[ "${BUILD_VLLM_RS}" == "1" ]]; then \
        cd /vllm-workspace/third_party/vllm && \
        bash build_rust.sh && \
        install -Dm755 vllm/vllm-rs /tmp/vllm-rs-artifacts/vllm-rs && \
        install -Dm755 vllm/_rust_tool_parser.abi3.so \
            /tmp/vllm-rs-artifacts/_rust_tool_parser.abi3.so; \
    else \
        echo "Skipping vllm-rs build because BUILD_VLLM_RS=0"; \
    fi

# Install the prebuilt MUSA wheel after torch/vllm-musa so Mooncake integrations
# can import the installed accelerator stack. Device visibility remains unset;
# the final image enables it below.
FROM vllm_musa_installed AS mooncake

ARG MOONCAKE_VERSION=0.3.12.post1
ARG PYPI_INDEX_URL

RUN python -m pip install \
        --no-cache-dir \
        --only-binary=:all: \
        --index-url "${PYPI_INDEX_URL}" \
        "mooncake-transfer-engine-musa==${MOONCAKE_VERSION}" && \
    MOONCAKE_VERSION="${MOONCAKE_VERSION}" python -c 'import os; from importlib.metadata import version; from mooncake.engine import TransferEngine; actual = version("mooncake-transfer-engine-musa"); expected = os.environ["MOONCAKE_VERSION"]; assert actual == expected, (actual, expected); print(f"PASS mooncake-transfer-engine-musa version={actual}")'

FROM mooncake AS final

ARG BUILD_VLLM_RS=1

ENV MTHREADS_VISIBLE_DEVICES=all

# Match the upstream vLLM runtime-image workspace contract.
WORKDIR /vllm-workspace

COPY --from=vllm_rs_build /tmp/vllm-rs-artifacts/ /tmp/vllm-rs-artifacts/

RUN test "$(cat /tmp/vllm-rs-artifacts/build-mode)" = "${BUILD_VLLM_RS}" && \
    if [[ "${BUILD_VLLM_RS}" == "1" ]]; then \
        install -Dm755 /tmp/vllm-rs-artifacts/vllm-rs \
            /vllm-workspace/third_party/vllm/vllm/vllm-rs && \
        install -Dm755 /tmp/vllm-rs-artifacts/_rust_tool_parser.abi3.so \
            /vllm-workspace/third_party/vllm/vllm/_rust_tool_parser.abi3.so && \
        VLLM_USE_RUST_FRONTEND=1 python -c 'from pathlib import Path; from vllm import envs; path = Path(envs.VLLM_RUST_FRONTEND_PATH); assert path.is_file(), f"missing vllm-rs binary: {path}"; print(f"PASS vllm-rs path={path}")'; \
    elif [[ "${BUILD_VLLM_RS}" == "0" ]]; then \
        rm -f \
            /vllm-workspace/third_party/vllm/vllm/vllm-rs \
            /vllm-workspace/third_party/vllm/vllm/_rust_tool_parser.abi3.so && \
        echo "Skipping vllm-rs artifact install because BUILD_VLLM_RS=0"; \
    else \
        echo "Unsupported BUILD_VLLM_RS=${BUILD_VLLM_RS}" >&2; \
        exit 1; \
    fi && \
    rm -rf /tmp/vllm-rs-artifacts

RUN rm -rf \
        /root/.cache/pip \
        /root/.cache/vllm-musa \
        /tmp/pip-*

ARG VLLM_MUSA_COMMIT
ARG VLLM_MUSA_REF
ARG VLLM_TAG

LABEL org.opencontainers.image.source="https://github.com/MooreThreads/vllm-musa" \
      org.opencontainers.image.revision="${VLLM_MUSA_COMMIT}" \
      org.opencontainers.image.version="${VLLM_MUSA_REF}" \
      com.mthreads.vllm.version="${VLLM_TAG}"

CMD ["/bin/bash"]

FROM final AS vllm-openai

ENTRYPOINT ["vllm", "serve"]

FROM vllm-openai AS vllm-omni

ARG VLLM_OMNI_REPO=https://github.com/vllm-project/vllm-omni.git
ARG VLLM_OMNI_REF=v0.28.0
ARG VLLM_OMNI_COMMIT=eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c
ARG MUSA_PIP_INDEX_URL
ARG PYPI_INDEX_URL
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONUNBUFFERED=1 \
    MTHREADS_VISIBLE_DEVICES=

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
        jq \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libsndfile1 \
        libxcb1 \
        libxext6 \
        libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Keep Omni's ordinary dependencies on the public index. The MUSA-specific
# entries in requirements/musa.txt are already installed by the parent image
# and are checked below; resolving that file against a merged index can replace
# them with unrelated public CUDA packages.
# OpenCV 5 currently requires NumPy 2 on Python 3.10+, so keep the Omni child
# on the NumPy 1.x ABI supported by the MUSA parent image.
RUN git clone --depth 1 --branch "${VLLM_OMNI_REF}" "${VLLM_OMNI_REPO}" /vllm-workspace/vllm-omni && \
    cd /vllm-workspace/vllm-omni && \
    test "$(git rev-parse HEAD)" = "${VLLM_OMNI_COMMIT}" && \
    printf '%s\n' "${VLLM_OMNI_COMMIT}" > /vllm-workspace/vllm-omni/VLLM_OMNI_COMMIT && \
    python -c 'from pathlib import Path; p=Path("vllm_omni/diffusion/data.py"); old="from enum import Enum, StrEnum"; new="from enum import Enum\ntry:\n    from enum import StrEnum\nexcept ImportError:\n    from strenum import StrEnum"; source=p.read_text(); assert source.count(old) == 1; p.write_text(source.replace(old, new))' && \
    python -m pip install --no-deps --index-url "${MUSA_PIP_INDEX_URL}" \
        "tilelang-musa==0.1.8+musa.3" && \
    for distribution in apache-tvm-ffi deep-gemm deep_ep flash_attn_3 flash_mla flashinfer-python mate mate-mubin sageattention tilelang_musa torch torch-c-dlpack-ext torchaudio torch_musa torchada torchvision triton vllm vllm-musa; do \
        installed_version="$(python -c 'from importlib.metadata import version; import sys; print(version(sys.argv[1]))' "${distribution}")"; \
        printf '%s==%s\n' "${distribution}" "${installed_version}"; \
    done > /tmp/vllm-musa-constraints.txt && \
    VLLM_OMNI_TARGET_DEVICE=musa VLLM_OMNI_VERSION_OVERRIDE=0.28.0+musa \
        python -m pip install --no-deps -e . --no-build-isolation -v && \
    if python -m pip show opencv-python-headless >/dev/null 2>&1; then \
        python -m pip uninstall -y opencv-python-headless; \
    fi && \
    python -m pip install --constraint /tmp/vllm-musa-constraints.txt \
        --index-url "${PYPI_INDEX_URL}" \
        -r requirements/common.txt \
        "numpy>=1.26.4,<2" \
        "opencv-python==4.11.0.86" \
        "onnxruntime>=1.23.2" \
        "mcp-types==2.1.1" \
        "strenum==0.4.15; python_version < '3.11'" && \
    printf '%s\n' \
        'from importlib.metadata import version' \
        'from pathlib import Path' \
        'from packaging.requirements import Requirement' \
        'from packaging.version import Version' \
        '' \
        'checks = (("torchada", ">=0.1.52"), ("mate", ">=0.2.0"), ("flash_attn_3", ">=0.1.4"))' \
        'for name, spec in checks:' \
        '    installed = Version(version(name))' \
        '    assert installed in Requirement(f"{name}{spec}").specifier, (name, installed, spec)' \
        '    print(f"PASS preserve {name} version={installed}")' \
        '' \
        'for line in Path("/tmp/vllm-musa-constraints.txt").read_text().splitlines():' \
        '    name, expected = line.split("==", 1)' \
        '    actual = version(name)' \
        '    assert actual == expected, (name, expected, actual)' \
        '    print(f"PASS preserve {name} version={actual}")' \
        '' \
        'numpy_version = Version(version("numpy"))' \
        'assert numpy_version >= Version("1.26.4") and numpy_version < Version("2"), numpy_version' \
        'print(f"PASS omni numpy override version={numpy_version}")' \
        > /tmp/vllm_omni_preserve_check.py && \
    python /tmp/vllm_omni_preserve_check.py && \
    python -c 'import av, cv2, soundfile; assert cv2.__version__.startswith("4.11."), cv2.__version__; print("PASS media imports av=" + av.__version__ + " cv2=" + cv2.__version__ + " soundfile=" + soundfile.__version__)' && \
    python -c 'import re, subprocess; result=subprocess.run(["python", "-m", "pip", "check"], text=True, capture_output=True); lines=tuple(line for line in result.stdout.splitlines() if line.strip()); allowed=(r"vllm 0\.28\.0 requires opencv-python-headless(?:>=4\.13\.0)?, which is not installed\.", r"vllm-musa .+ has requirement click==8\.2\.0, but you have click .+\.", r"vllm-musa .+ has requirement numpy==1\.26, but you have numpy 1\.26\.4\.", r"vllm-musa .+ has requirement tilelang_musa==0\.1\.12\+musa\.2, but you have tilelang-musa .+\.", r"vllm-musa .+ has requirement transformers==5\.5\.3, but you have transformers 5\.14\.1\.", r"xgrammar .+ has requirement transformers<5,>=4\.38\.0, but you have transformers 5\.14\.1\."); unexpected=tuple(line for line in lines if not any(re.fullmatch(pattern, line) for pattern in allowed)); assert not unexpected, "unexpected pip check output: " + repr(unexpected); print("pip check allowlist entries:"); print("\\n".join(lines) if lines else "(clean)"); print("PASS pip check allowlist lines=" + str(len(lines)))' && \
    rm -f /tmp/vllm_omni_preserve_check.py /tmp/vllm-musa-constraints.txt && \
    rm -rf /root/.cache/pip /vllm-workspace/vllm-omni/.git

RUN ffmpeg -version >/dev/null && \
    ffprobe -version >/dev/null && \
    python -c 'import importlib.util; from importlib.metadata import version; assert importlib.util.find_spec("vllm_omni") is not None; assert version("vllm-omni") == "0.28.0+musa", version("vllm-omni"); print("PASS spec vllm_omni version=" + version("vllm-omni"))' && \
    python -m compileall -q /vllm-workspace/vllm-omni/vllm_omni

LABEL com.mthreads.vllm-omni.repository="${VLLM_OMNI_REPO}" \
      com.mthreads.vllm-omni.ref="${VLLM_OMNI_REF}" \
      com.mthreads.vllm-omni.revision="${VLLM_OMNI_COMMIT}" \
      com.mthreads.vllm-omni.python310-strenum-backport="strenum==0.4.15"

ENV MTHREADS_VISIBLE_DEVICES=all

ENTRYPOINT ["vllm", "serve"]

# Keep Docker's implicit (no --target) build behavior on the regular image.
FROM vllm-openai AS default
