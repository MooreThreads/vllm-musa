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
- `vllm-rs` and its Python tool-parser extension when `BUILD_VLLM_RS=1`,
- `mooncake-transfer-engine-musa`,
- `pytest`, so the repository's pytest-based validation can run directly in the
  image.

The source tree and default working directory are `/vllm-workspace`, matching
the upstream vLLM runtime-image contract. A default build produces the
`vllm-openai` target with `ENTRYPOINT ["vllm", "serve"]`; the `final` target
retains `CMD ["/bin/bash"]` for tests and interactive use.

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
- **A MUSA GPU visible to the build** if you want the final-stage import verify to
  pass — see [Building on a MUSA host](#building-on-a-musa-host).

## Quick start

For v0.28.0 deployments, use the release `vllm-openai` image:

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0
docker pull "${VLLM_MUSA_IMAGE}"
```

The registry tag is published independently of this branch and may carry a
different dependency revision. Verify package versions before benchmarking; use
the source-built image below for exact branch parity.

To build the same serving target from source, run from the repository root:

```bash
bash docker/build_image.sh
```

With the defaults this produces:

```
vllm-musa:ubuntu22.04_py3.10_musa_runtime_5.2_pytorch_release_2.11.0.post1_musa5.2.0
```

The release image accepts the model and engine arguments directly. This block
is self-contained; set the visible-device list explicitly through
`MUSA_VISIBLE_DEVICES`, which is authoritative for vLLM-MUSA. The Docker host
must provide the MUSA container runtime; this generic example relies on the
host's runtime configuration rather than selecting a runtime by name.

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0
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

To build the matching vLLM-Omni MUSA image on top of the complete runtime
target:

```bash
IMAGE_TAG=vllm-omni-musa:0.28.0 bash docker/build_image.sh --target vllm-omni
```

The Omni source is pinned by `VLLM_OMNI_REPO`, `VLLM_OMNI_REF`, and
`VLLM_OMNI_COMMIT` (defaults to the vLLM-Omni repository, `v0.28.0`, and the
release commit). The target extends `vllm-openai` and keeps the `vllm serve`
entrypoint; pass the model first and then `--omni`, as required by the v0.28
CLI:

```bash
docker run --runtime=mthreads vllm-omni-musa:0.28.0 \
  /path/to/omni-model --omni
```

The final extension also installs the audio/video and media runtime packages
used by Omni serving paths (`ffmpeg`, `espeak-ng`, `libsndfile1`, and the
GL/X11 runtime needed by `opencv-python`). Python dependencies are resolved
from the public index, while the preinstalled MUSA packages remain constrained
and verified. Omni's resolver requires child-only compatibility pins of
NumPy `>=1.26.4,<2` and `opencv-python==4.11.0.86`; the parent vllm-musa
target remains unchanged at its existing NumPy/OpenCV pins. Native
`vllm_omni` import is intentionally deferred to the runtime smoke so CPU-only
Docker builders do not require a MUSA driver.
The child fixes Omni's platform-aware version override to report `0.28.0+musa` while retaining the
immutable Omni commit in a separate provenance label.
For the default Python 3.10 image, the Omni stage also applies the narrow
`StrEnum` compatibility fallback used by its diffusion CLI imports and pins
the `strenum==0.4.15` backport; Python 3.11+ continues to use the standard
library implementation.

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
| `PYTHON_VERSION` | `3.10` | Python version (apt `python3.X`). The wheels pinned in `requirements/musa_private.txt` are published for 3.10 on x86_64 only, so other values fail the dependency install. |
| `MUSA_APT_SOURCE` | `https://dl.mthreads.com/repo/repository/ubuntu2204/` | apt repo for the MUSA runtime SDK. |
| `INSTALL_MUSA_STACK` | `auto` | `auto`: install the MUSA apt stack unless the base already provides `mcc`; `0`: skip (base image supplies the runtime). |
| `MUSA_RUNTIME_VERSION` | `5.2` | MUSA runtime line as `major.minor`; derives apt package names (e.g. `musa-toolkit-5-2`). |
| `MCCL_VERSION` | `2.4.0` | MCCL (collective communication library) version. |
| `PYPI_INDEX_URL` | `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` | Public index for ordinary third-party wheels **and** the vendored vLLM's dependencies. |
| `MUSA_PIP_INDEX_URL` | `https://dl.mthreads.com/repo/api/pypi/pypi/simple` | Moore Threads index for the MUSA/MT wheels. |
| `MOONCAKE_VERSION` | `0.3.12.post1` | Exact `mooncake-transfer-engine-musa` version installed from `PYPI_INDEX_URL`. |
| `BUILD_VLLM_RS` | `1` | `1`: build and install `vllm-rs` plus `_rust_tool_parser`; `0`: omit both and skip Rust/protoc setup. |
| `SKIP_THIRD_PARTY` | `0` | `1`: use pre-staged pinned `third_party` source caches; `0`: clone pinned sources during the build. |
| `VLLM_OMNI_REPO` | `https://github.com/vllm-project/vllm-omni.git` | Source repository used by the optional `vllm-omni` target. |
| `VLLM_OMNI_REF` | `v0.28.0` | Human-readable tag/ref checked out by the optional target. |
| `VLLM_OMNI_COMMIT` | `eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c` | Immutable commit verification for the optional target. |
| `VLLM_MUSA_COMMIT` | current Git `HEAD` | Source revision label; set explicitly when building from a Git archive without `.git`. |
| `VLLM_MUSA_REF` | exact tag or current branch | Source ref label; set explicitly when building from a Git archive without `.git`. |
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
  --entrypoint python \
  vllm-musa:ubuntu22.04_py3.10_musa_runtime_5.2_pytorch_release_2.11.0.post1_musa5.2.0 \
  -c "import torch, torch_musa; print('musa available:', torch.musa.is_available())"
```

On a MUSA GPU you should see `musa available: True`.

Mooncake over RoCE also requires host networking and explicit RDMA device
access. See the
[container and RDMA prerequisites](../docs/example/README.md#container-and-rdma-prerequisites)
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
   vLLM, re-pins numpy, installs runtime dependencies, and validates imports.
   Driver-dependent native packages use metadata plus import-spec checks during
   image construction; real imports remain a hardware smoke requirement.
6. **vllm_rs_build** — optionally builds Rust artifacts (`BUILD_VLLM_RS`) without
   carrying Rust/protoc into the final image.
7. **mooncake** — installs the pinned `mooncake-transfer-engine-musa` wheel on
   top of the torch/vLLM stack.
8. **final** — installs optional Rust artifacts, enables MUSA device visibility,
   removes build caches, and retains a shell command for test/debug use.
9. **vllm-openai** — the default serving target, with `vllm serve` as its
   entrypoint.
10. **vllm-omni** — optional final extension target that adds the pinned
    vLLM-Omni release and its public/MUSA-compatible requirements on top of
    `vllm-openai`.
11. **default** — implicit terminal alias of `vllm-openai`, preserving the
    existing no-`--target` build behavior.

The Triton `3.2.0` pin is intentional for the MUSA 5.2 stack and is maintained
independently of the PyTorch release. Validate image imports, Inductor, and
compiled serving before changing it. The explicit vLLM runtime dependency pass
also restores the `fastapi[standard]` extras and `pycountry` that are skipped
when the vendored requirements are installed with `--no-deps` to protect the
MUSA torch pins.

For the reasoning behind the pip-index split and the runtime shims, see the
comments in `docker/musa.Dockerfile`.
