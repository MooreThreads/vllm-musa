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
- `vllm-musa` and the vendored upstream vLLM, built from source,
- `vllm-rs` and its Python tool-parser extension when `BUILD_VLLM_RS=1`,
- `mooncake-transfer-engine-musa`.

## Prerequisites

- **Docker** on the build host.
- **Network access** from the build to:
  - the Moore Threads pip index (`MUSA_PIP_INDEX_URL`) — hosts the MUSA/MT
    wheels,
  - a public PyPI index/mirror (`PYPI_INDEX_URL`) — ordinary third-party wheels
    and the vendored vLLM's dependencies,
  - the MUSA apt source (`MUSA_APT_SOURCE`) — the runtime SDK,
  - the Deadsnakes PPA (or configured mirror) — system CPython 3.12 for jammy,
  - `GET_PIP_URL` — bootstraps pip into that system interpreter,
  - GitHub — the vendored vLLM/flashinfer clones, the Rust frontend's
    `llm-multimodal` dependency when the Rust frontend is enabled.
- **A MUSA GPU visible to the build** if you want the final-stage import verify to
  pass — see [Building on a MUSA host](#building-on-a-musa-host).

## Quick start

From the repository root:

```bash
bash docker/build_image.sh
```

With the defaults this produces:

```
vllm-musa:ubuntu22.04_py3.12_musa_runtime_5.2_pytorch_release_2.9.1.post1_musa5.2.0
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

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BASE_IMAGE` | `ubuntu:22.04` | Base image. Point at a local/mirror image if Docker Hub is unreachable, or an `mthreads/musa:*-devel` image to reuse its runtime. |
| `PYTHON_VERSION` | `3.12` | Supported system Python version. The build rejects other values because the pinned MUSA wheel set targets cp312. |
| `DEADSNAKES_MIRROR_URL` | empty | Optional apt mirror for the Deadsnakes jammy repository. When set, `DEADSNAKES_GPGKEY_URL` is required. |
| `DEADSNAKES_GPGKEY_URL` | empty | GPG key URL for a configured Deadsnakes mirror. |
| `GET_PIP_URL` | `https://bootstrap.pypa.io/get-pip.py` | PyPA bootstrap script; override this URL when the default host is unavailable from the build network. |
| `PIP_BOOTSTRAP_INDEX_URL` | `${PYPI_INDEX_URL}` | Simple index used by the pip bootstrap; keep its host in `no_proxy` when the mirror is reached directly. |
| `RUSTUP_DIST_SERVER` | empty | Optional rustup distribution mirror for the vllm-rs stage (for example `https://rsproxy.cn`). |
| `RUSTUP_UPDATE_ROOT` | empty | Optional rustup update-root mirror paired with `RUSTUP_DIST_SERVER` (for example `https://rsproxy.cn/rustup`). |
| `MUSA_APT_SOURCE` | `https://dl.mthreads.com/repo/repository/ubuntu2204/` | apt repo for the MUSA runtime SDK. |
| `INSTALL_MUSA_STACK` | `auto` | `auto`: install the MUSA apt stack unless the base already provides `mcc`; `0`: skip (base image supplies the runtime). |
| `MUSA_RUNTIME_VERSION` | `5.2` | MUSA runtime line as `major.minor`; derives apt package names (e.g. `musa-toolkit-5-2`). |
| `MCCL_VERSION` | `2.4.0` | MCCL (collective communication library) version. |
| `PYPI_INDEX_URL` | `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` | Public index for ordinary third-party wheels **and** the vendored vLLM's dependencies. |
| `MUSA_PIP_INDEX_URL` | `https://dl.mthreads.com/repo/api/pypi/pypi/simple` | Moore Threads index for the MUSA/MT wheels. |
| `MOONCAKE_VERSION` | `0.3.12.post1` | Exact `mooncake-transfer-engine-musa` version installed from `PYPI_INDEX_URL`. |
| `BUILD_VLLM_RS` | `1` | `1`: build and install `vllm-rs` plus `_rust_tool_parser`; `0`: omit both and skip Rust/protoc setup. |
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
Pin a different published build with `MOONCAKE_VERSION`; the image does not
clone or compile Mooncake source.

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

- **Device visibility is scoped to the import verify and final stage.**
  `MTHREADS_VISIBLE_DEVICES` is deliberately *not* inherited by earlier stages,
  including Mooncake: otherwise the MUSA runtime can bind-mount host driver
  libraries into an `apt` step and break package installs
  (`Invalid cross-device link`).
- **Runtime compatibility shims.** When intentionally overriding the runtime to
  an older 5.1 line, the devel stage can add a `libmupti.so.1` soname link and
  a `libmusolver` OpenBLAS dependency so `import torch` works. Both are guarded
  and no-op on the default 5.2 runtime.

## Verify the built image

```bash
docker run --rm <MUSA GPU flags> \
  vllm-musa:ubuntu22.04_py3.12_musa_runtime_5.2_pytorch_release_2.9.1.post1_musa5.2.0 \
  python -c "import sys, torch, torch_musa; assert sys.prefix == sys.base_prefix; print(sys.version, 'musa available:', torch.musa.is_available())"
```

On a MUSA GPU you should see `musa available: True`.

Mooncake over RoCE also requires host networking and explicit RDMA device
access. See the
[container and RDMA prerequisites](../docs/example/README.md#container-and-rdma-prerequisites)
for the validated runtime flags.

## How it works (build stages)

`docker/musa.Dockerfile` is multi-stage:

1. **base** — base image and shell behavior only.
2. **apt_base** — generic build environment and a system Python 3.12 installed
   from the Deadsnakes PPA. It does not install uv or create a virtualenv.
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
   vLLM, re-pins numpy, installs runtime dependencies, and runs import checks.
6. **vllm_rs_build** — optionally builds Rust artifacts (`BUILD_VLLM_RS`) without
   carrying Rust/protoc into the final image.
7. **mooncake** — installs the pinned `mooncake-transfer-engine-musa` wheel on
   top of the torch/vLLM stack.
8. **final** — installs optional Rust artifacts, enables MUSA device visibility,
   and removes build caches.

The Triton `3.2.0` pin is intentional for the torch `2.9.1` MUSA stack. Torch
Inductor reads `KernelMetadata.cluster_dims`; the MUSA backend in Triton `3.6.0`
does not expose that field, while the `3.2.0` `mtgpu` backend does. The explicit
vLLM runtime dependency pass also restores the `fastapi[standard]` extras and
`pycountry` that are skipped when the vendored requirements are installed with
`--no-deps` to protect the MUSA torch pins.

For the reasoning behind the pip-index split and the runtime shims, see the
comments in `docker/musa.Dockerfile`.
