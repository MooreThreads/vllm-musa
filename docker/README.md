# Building the vLLM-MUSA image

`docker/build_image.sh` builds the vLLM plugin for Moore Threads (MUSA) GPUs into
a runnable Docker image from `docker/musa.Dockerfile`. It is the supported entry
point — it defines every setting in one place and passes them to the build as
`--build-arg`s, so the Dockerfile itself carries no hardcoded URLs or versions.

The resulting image contains:

- the MUSA runtime SDK (installed from apt),
- the MUSA/MT Python wheels (`torch`, `torch_musa`, `mate`, `flash_attn_3`,
  `flash_mla`, `deep-gemm`, `tilelang_musa`, `apache-tvm-ffi`,
  `torch_c_dlpack_ext`),
- `torchada` and the other common runtime dependencies from the public index,
- `vllm-musa` and the vendored upstream vLLM, built from source,
- the `vllm-rs` executable when `BUILD_VLLM_RS=1`,
- `mooncake-transfer-engine-musa`,
- `pytest`, so the repository's pytest-based validation can run directly in the
  image.

The default `vllm-openai` target starts `vllm serve` in `/vllm-workspace`.
This directory contains `examples/`, `benchmarks/`, `collect_env.py`, and
`torch_lib_versions.txt`, matching the upstream vLLM runtime layout. The
`final` target opens a shell with the same runtime layout. Use the `test`
target for an editable checkout under `/vllm-workspace`.

## Prerequisites

- **Docker** on the build host.
- **Network access** from the build to:
  - the Moore Threads pip index (`MUSA_PIP_INDEX_URL`) — hosts the MUSA/MT
    wheels,
  - a public PyPI index/mirror (`PYPI_INDEX_URL`) — ordinary third-party wheels
    and the vendored vLLM's dependencies,
  - the MUSA apt source (`MUSA_APT_SOURCE`) — the runtime SDK,
  - GitHub — the vendored vLLM/flashinfer clones, the Rust frontend's
    `llm-multimodal` dependency when the Rust frontend is enabled.
- A MUSA GPU is required for serving and accelerator smoke tests.

## Quick start

For v0.32.0 deployments, use the release `vllm-openai` image:

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.32.0
docker pull "${VLLM_MUSA_IMAGE}"
```

The registry tag is published independently of this branch and may carry a
different dependency revision. Verify package versions before benchmarking; use
the source-built image below for exact branch parity.

To build the same serving target from source, run from the repository root:

```bash
bash docker/build_image.sh
```

## Build a vLLM-Omni image

Use [`docs/omni_compat.md`](../docs/omni_compat.md) for the compatibility
contract and use `docker/build_omni_image.sh` for the actual derived-image
build. `docker/build_image.sh` builds the vLLM-MUSA base image only.

The Omni build must start from a vLLM-MUSA image whose Python, torch, MUSA,
and pinned vLLM versions are known. Resolve and record the image digest before
building. Export Omni from a clean local checkout when reproducibility matters.

```bash
VLLM_MUSA_IMAGE=sh-harbor.mthreads.com/mcctest/vllm:<musa-tag>
VLLM_OMNI_CLONE=/path/to/vllm-omni
OMNI_REF=<omni-commit-or-ref>

bash docker/build_omni_image.sh \
  --base "${VLLM_MUSA_IMAGE}" \
  --omni-clone "${VLLM_OMNI_CLONE}" \
  --omni-ref "${OMNI_REF}" \
  --tag "vllm-omni-musa:${OMNI_REF//\//-}"
```

The script records the vLLM-MUSA and Omni revisions, stages an immutable Omni
archive, copies the compatibility shim into the base image, and runs the import
audit plus installed-tree shim verification during the build. The build uses
driverless verification. A MUSA-enabled Docker host is required for the final
hardware import check and model smoke. The resulting image inherits
`VLLM_WORKER_MULTIPROC_METHOD=spawn` from the vLLM-MUSA base.

After the build, record the image digest and run the model-specific smoke from
the corresponding Omni documentation. For the current Qwen-Image diffusion
path, the image-generation request is the required semantic check. A successful
Docker build or server startup alone is not a model smoke.

With the defaults this produces:

```
vllm-musa:ubuntu22.04_py3.12_musa_runtime_5.2_pytorch_release_2.13.0.post1_musa5.2.0
```

The release image accepts the model and engine arguments directly. This block
is self-contained; set the visible-device list explicitly through
`MUSA_VISIBLE_DEVICES`, which is authoritative for vLLM-MUSA. The Docker host
must provide the MUSA container runtime; this generic example relies on the
host's runtime configuration rather than selecting a runtime by name.

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.32.0
MODEL_PATH=/path/to/model
VISIBLE_DEVICES=0
docker run --rm --privileged --ipc=host --network=host \
  --env MUSA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  --env MTHREADS_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
  "${VLLM_MUSA_IMAGE}" "${MODEL_PATH}" --host 0.0.0.0
```

If the host does not configure a MUSA-compatible default runtime, follow the
host platform's container-runtime setup before starting the image. Keep the
MUSA visibility setting scoped to the GPUs assigned to this container.
Because these examples use `--ipc=host`, shared-memory capacity comes from the
host IPC namespace; a separate `--shm-size` setting is not applied.
Treat this as an isolated-host/test shape. The `--privileged` flag is shown for
broad host compatibility and grants wide device access. For production,
replace it with the platform's least-privilege MUSA device policy (and explicit
RDMA devices when using Mooncake); the visibility variables scope vLLM's
logical devices but do not reduce Docker privileges.

To build and run a local serving image instead of using the release image:

```bash
MODEL_PATH=/path/to/model
VISIBLE_DEVICES=0
export VLLM_MUSA_SOURCE_IMAGE=vllm-musa:v0.28.0-local
IMAGE_TAG="${VLLM_MUSA_SOURCE_IMAGE}" bash docker/build_image.sh
docker run --rm --privileged --ipc=host --network=host \
  --env MUSA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  --env MTHREADS_VISIBLE_DEVICES="${VISIBLE_DEVICES}" \
  -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
  "${VLLM_MUSA_SOURCE_IMAGE}" "${MODEL_PATH}" --host 0.0.0.0
```

Every setting is an environment variable — override by exporting it or prefixing
the command, e.g.:

```bash
MUSA_RUNTIME_VERSION=5.2 IMAGE_TAG=vllm-musa:dev bash docker/build_image.sh
```

Any extra arguments are forwarded verbatim to `docker build`, so you can also pass
`--build-arg`, `--target`, `--no-cache`, etc.:

```bash
bash docker/build_image.sh --no-cache --build-arg http_proxy=http://proxy:8118
```

Build the shell/test target under a separate tag when arbitrary container
commands should run without overriding an entrypoint:

```bash
IMAGE_TAG=vllm-musa:test bash docker/build_image.sh --target final
```

Verify the workspace and test-runner contract with:

```bash
docker run --rm --entrypoint /bin/bash vllm-musa:test \
  -lc 'test "$PWD" = /vllm-workspace && python -m pytest --version'
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BASE_IMAGE` | `ubuntu:22.04` | Base image. Point at a local/mirror image if Docker Hub is unreachable, or an `mthreads/musa:*-devel` image to reuse its runtime. |
| `PYTHON_VERSION` | `3.12` | System Python version. Only `3.12` is accepted: the torch wheels pinned in `requirements/musa_private.txt` are published for cp312 on x86_64 only. |
| `DEADSNAKES_MIRROR_URL` | empty | Optional apt mirror for the Python 3.12 packages; empty uses `ppa:deadsnakes/ppa`. |
| `DEADSNAKES_GPGKEY_URL` | empty | Signing key URL; required with `DEADSNAKES_MIRROR_URL`. |
| `GET_PIP_URL` | `https://bootstrap.pypa.io/get-pip.py` | Location of `get-pip.py` used to bootstrap pip. |
| `PIP_BOOTSTRAP_INDEX_URL` | value of `PYPI_INDEX_URL` | Index used by the pip bootstrap. |
| `MUSA_APT_SOURCE` | `https://dl.mthreads.com/repo/repository/ubuntu2204/` | apt repo for the MUSA runtime SDK. |
| `INSTALL_MUSA_STACK` | `auto` | `auto`: install the MUSA apt stack unless the base already provides `mcc`; `0`: skip (base image supplies the runtime). |
| `MUSA_RUNTIME_VERSION` | `5.2` | MUSA runtime line as `major.minor`; derives apt package names (e.g. `musa-toolkit-5-2`). |
| `MCCL_VERSION` | `2.4.0` | MCCL (collective communication library) version. |
| `MTSHMEM_VERSION` | `3.2.5.1` | mtshmem version: ships the NVSHMEM bootstrap and transport plugins (uid, ibrc, ibgda) that DeepEP dlopens from `/usr/local/mtshmem/lib`. |
| `PYPI_INDEX_URL` | `https://mirrors.aliyun.com/pypi/simple` | Public index for ordinary third-party wheels **and** the vendored vLLM's dependencies. |
| `MUSA_PIP_INDEX_URL` | `https://dl.mthreads.com/repo/api/pypi/pypi/simple` | Moore Threads index for the MUSA/MT wheels. |
| `VLLM_MUSA_GIT_PROXY` | `https://gh-proxy.org/` | Mirror for the vLLM/FlashInfer fetch that `setup.py` performs while building the image. Empty keeps the default; `0` fetches from github.com directly. |
| `MOONCAKE_VERSION` | `0.3.14` | Exact `mooncake-transfer-engine-musa` version installed from `PYPI_INDEX_URL`. The wheel provides both the Transfer Engine and the Store bindings. |
| `BUILD_VLLM_RS` | `1` | `1`: build and install the `vllm-rs` executable; `0`: omit it and skip the Rust toolchain setup. |
| `IMAGE_REPOSITORY` | `vllm-musa` | Image repository name. |
| `IMAGE_FLAVOR` | `ubuntu22.04_py<py>_musa_runtime_<ver>_pytorch_release_<torch>` | Tag flavor; `<torch>` is derived from `requirements/musa_private.txt` and sanitized for Docker tags. |
| `IMAGE_TAG` | `${IMAGE_REPOSITORY}:${IMAGE_FLAVOR}` | Full image tag. |

## Common scenarios

**Use a specific PyPI mirror** (ordinary wheels + vendored vLLM deps):

```bash
PYPI_INDEX_URL=https://<mirror>/simple bash docker/build_image.sh
```

**Build behind an HTTP proxy** (covers apt, git, and every pip step, including the
nested vLLM install):

```bash
bash docker/build_image.sh \
  --build-arg http_proxy=http://<proxy>:<port> \
  --build-arg https_proxy=http://<proxy>:<port> \
  --build-arg no_proxy=.mthreads.com
```

Keep `no_proxy=.mthreads.com` so the MUSA wheel index and apt source stay on a
direct connection.

**Docker Hub not reachable** — build from a locally-present base:

```bash
BASE_IMAGE=<local-ubuntu-22.04-image> bash docker/build_image.sh
```

**Override the MUSA runtime line**:

```bash
MUSA_RUNTIME_VERSION=5.2 MUSA_APT_SOURCE=<5.2-apt-repo> bash docker/build_image.sh
```

The prebuilt MUSA Mooncake wheel is installed after the torch and vLLM stacks.
It provides both the Transfer Engine (`mooncake.engine`) and the Store bindings
(`mooncake.store`); the image stage checks both imports. Pin a different
published build with `MOONCAKE_VERSION`; the image does not clone or compile
Mooncake source.

**Skip the Rust frontend:**

```bash
BUILD_VLLM_RS=0 bash docker/build_image.sh
```

**Build only up to the dependency layer** (installs the wheels, skips the vLLM
compile and the import verify — handy for verifying the pip install offline of a
GPU):

```bash
bash docker/build_image.sh --target vllm_musa_deps
```

## Building on a MUSA host

The `final` stage's verify step imports every MUSA package, including `tilelang`
and `flash_mla`, which require `torch.musa.is_available()` to be `True` at import
time. That is only satisfied when the build step can see the GPU — i.e. when the
**MUSA container runtime is the host's default docker runtime**, so `docker build`
`RUN` steps get the device. On a CPU-only builder the build otherwise completes
and then fails at the verify step with
`ImportError: cannot import name 'GPUEvent' from 'tilelang.utils.device'`.

Two build-time details make this work on such a host:

- **Build stages disable device injection.** The final image enables device
  visibility after its build-time checks. An `apt` step can fail with
  `Invalid cross-device link` if the MUSA runtime bind-mounts driver libraries
  into a build container.
- **Runtime compatibility shims.** When intentionally overriding the runtime to
  an older 5.1 line, the devel stage can add a `libmupti.so.1` soname link and
  a `libmusolver` OpenBLAS dependency so `import torch` works. Both are guarded
  and no-op on the default 5.2 runtime.

## Verify the built image

```bash
docker run --rm <MUSA GPU flags> \
  --entrypoint python \
  vllm-musa:ubuntu22.04_py3.12_musa_runtime_5.2_pytorch_release_2.13.0.post1_musa5.2.0 \
  -c "import torch, torch_musa; print('musa available:', torch.musa.is_available())"
```

On a MUSA GPU you should see `musa available: True`.

Mooncake over RoCE also requires host networking and explicit RDMA device
access. See the
[container and RDMA prerequisites](../examples/disaggregated/disaggregated_serving/README.md#container-and-rdma-prerequisites)
for the documented container shape and host-runtime prerequisite.

## How it works (build stages)

`docker/musa.Dockerfile` is multi-stage:

1. **base** — base image and shell behavior only.
2. **apt_base** — generic build environment, toolchain, and Python from apt.
3. **devel** — the MUSA SDK from apt (`INSTALL_MUSA_STACK`), MUSA library paths,
   and guarded compatibility shims.
4. **vllm_musa_deps** — MUSA architecture selectors and Python dependencies,
   installed in three passes:
   1. MUSA/MT wheels from `MUSA_PIP_INDEX_URL` only (`--no-deps`),
   2. ordinary third-party wheels from `PYPI_INDEX_URL`,
   3. the MUSA wheels' remaining ordinary deps from `PYPI_INDEX_URL`.

   The split keeps names like `torch`/`mate`/`apache-tvm-ffi` resolving from the
   internal index only, so pip never pulls the unrelated public (CUDA) builds.
5. **vllm_musa_installed** — copies the source, builds `vllm-musa` + vendored
   vLLM in editable mode, and verifies the installed dependencies.
6. **vllm_rs_build** — optionally builds the `vllm-rs` executable
   (`BUILD_VLLM_RS`) without carrying the Rust toolchain into the final image.
7. **vllm_wheel** — packages the patched vLLM source and prepared native
   extensions into the `vllm-musa` wheel.
8. **vllm_runtime** — installs the wheel with its public dependencies and the
   private stack constraints, then copies the runtime workspace files.
9. **test** — exposes the source-bearing editable installation.
10. **mooncake** — installs the pinned `mooncake-transfer-engine-musa` wheel on
   top of the torch/vLLM stack.
11. **final** — verifies wheel ownership and dependencies, enables MUSA device
    visibility, and opens a shell by default.
12. **vllm-openai** — the default serving target, with `vllm serve` as its
   entrypoint.

The Triton `3.2.0` pin is intentional for the MUSA 5.2 stack and is maintained
independently of the PyTorch release. Validate image imports, Inductor, and
compiled serving before changing it. The wheel and editable install use the
same public runtime requirements, including NumPy below 2 and a compatible
OpenCV release.

For the reasoning behind the pip-index split and the runtime shims, see the
comments in `docker/musa.Dockerfile`.

## Installing the wheel outside the image

The `vllm-musa` wheel bundles both Python import packages. Install the pinned
hardware dependencies from the MUSA index first, then resolve the wheel's
public dependencies with the same pins as constraints:

```bash
python -m pip install --no-deps --index-url "${MUSA_PIP_INDEX_URL}" \
  -r requirements/musa_private.txt
python -m pip install --index-url "${PYPI_INDEX_URL}" \
  --extra-index-url "${MUSA_PIP_INDEX_URL}" -r requirements/musa_private.txt
python -m pip install --index-url "${PYPI_INDEX_URL}" \
  -c requirements/musa_private.txt /path/to/vllm_musa-*.whl
```
