# vLLM-Omni image for Moore Threads MUSA GPUs. This mirrors the source-image
# layout of vllm-omni's Dockerfile.cuda while inheriting the MUSA runtime,
# vLLM, and vllm-musa installation from the base image.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG BASE_IMAGE
ARG COMMON_WORKDIR=/app
ARG VLLM_OMNI_VERSION_OVERRIDE

WORKDIR ${COMMON_WORKDIR}

# Match the system utilities available through the upstream CUDA image. The
# media libraries normally come from vllm/vllm-openai, but the MUSA base image
# starts directly from Ubuntu and therefore installs them here.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        jq \
        libgl1 \
        libsm6 \
        libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${COMMON_WORKDIR}/vllm-omni

# The Docker build context is a vllm-omni checkout. Select MUSA explicitly so
# setup.py never falls back to CPU merely because no GPU is visible while the
# Docker layer is being built.
ENV VLLM_OMNI_TARGET_DEVICE=musa

COPY . ${COMMON_WORKDIR}/vllm-omni
RUN for distribution in \
        flash_attn_3 \
        flash_mla \
        mate \
        torch \
        torch_musa \
        torchada \
        triton \
        vllm \
        vllm-musa; do \
        installed_version="$(python -c 'from importlib.metadata import version; import sys; print(version(sys.argv[1]))' "${distribution}")"; \
        printf '%s==%s\n' "${distribution}" "${installed_version}"; \
    done > /tmp/vllm-musa-constraints.txt
RUN cd ${COMMON_WORKDIR}/vllm-omni && \
    if [[ -n "${VLLM_OMNI_VERSION_OVERRIDE}" ]]; then \
        export VLLM_OMNI_VERSION_OVERRIDE; \
    fi && \
    python -m pip install \
        --constraint /tmp/vllm-musa-constraints.txt \
        --no-build-isolation \
        --no-cache-dir \
        "." && \
    while IFS='=' read -r distribution _ expected_version; do \
        installed_version="$(python -c 'from importlib.metadata import version; import sys; print(version(sys.argv[1]))' "${distribution}")"; \
        test "${installed_version}" = "${expected_version}" || { \
            echo "MUSA base package changed: ${distribution} ${expected_version} -> ${installed_version}" >&2; \
            exit 1; \
        }; \
    done < /tmp/vllm-musa-constraints.txt && \
    python -m pytest --version && \
    python -c 'import torchada; import torch; import torch_musa; import vllm; import vllm_musa; import vllm_omni; assert hasattr(torch.version, "musa") and torch.version.musa is not None; print("PASS vllm-omni MUSA image imports")' && \
    rm -f /tmp/vllm-musa-constraints.txt

RUN ln -sf /usr/bin/python3 /usr/bin/python

ARG VLLM_OMNI_REF
ARG VLLM_OMNI_COMMIT

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm-omni" \
      org.opencontainers.image.revision="${VLLM_OMNI_COMMIT}" \
      org.opencontainers.image.version="${VLLM_OMNI_REF}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENTRYPOINT []
