# Installation

This page contains the detailed installation paths for vLLM-MUSA. For a
model-specific launch command, start with the [serving cookbook](cookbook/README.md).

## Requirements

- x86_64 Linux and CPython 3.10 (the pinned MUSA wheels target this
  interpreter/platform pair).
- A Moore Threads GPU with a compatible MUSA driver and toolkit.
- vLLM-MUSA v0.28.0-dev and its matching PyTorch/MUSA wheels.

The v0.28.0-dev dependency stack uses PyTorch 2.11.x. Keep the driver, toolkit,
and wheel versions from the same release family.

## Package indexes

The install uses two package indexes:

| Index | Provides |
|---|---|
| Moore Threads | `torch`, `torch_musa`, `mate`, `flash_attn_3`, `flash_mla`, `deep-gemm`, `tilelang_musa`, `triton`, and the other MUSA wheels pinned in `requirements/musa_private.txt` |
| Public PyPI | `torchada`, ordinary build/runtime dependencies, and vendored vLLM dependencies |

Install the MUSA wheels from the Moore Threads index first with `--no-deps`.
Otherwise an unpinned package such as `torch` can resolve to a public CUDA
build instead of the MUSA build.

## Container

The quickest path for v0.28.0-dev-compatible serving is the v0.28.0 release image:

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0
docker pull "${VLLM_MUSA_IMAGE}"
```

The registry tag is published independently of this branch and can carry a
different dependency revision. Verify the installed package versions against
the branch pins; build and use the local image below when exact parity is
required.
At the time of this backport, the published tag reports MATE/Flash 0.2.4 and
tilelang_musa 0.1.8, while this branch pins 0.2.6 and 0.1.12 respectively.

The image uses `vllm serve` as its entrypoint, so pass the model path followed
by the engine arguments from the [serving cookbook](cookbook/README.md). Apply
your deployment's standard MUSA runtime and model-volume flags to
`docker run`.

To build an image from source instead, run `bash docker/build_image.sh` from
the repository root. Use `--target final` for a shell/test image. Build
arguments and image compatibility notes are maintained in
[docker/README.md](../docker/README.md).

## Source install

MUSA wheels are hosted on the Moore Threads package index. For the v0.28.0
release line, clone the matching source branch (or replace it with the exact
release tag or commit used by your deployment), then install the MUSA wheels
before the public dependencies:

```bash
git clone --branch v0.28.0-dev --single-branch \
  https://github.com/MooreThreads/vllm-musa.git
cd vllm-musa

export MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
export PYPI_INDEX_URL=https://pypi.org/simple

pip install --no-deps --index-url "${MUSA_PIP_INDEX_URL}" \
  -r requirements/musa_private.txt
pip install --index-url "${PYPI_INDEX_URL}" \
  -r requirements/build.txt -r requirements/common.txt
pip install --index-url "${MUSA_PIP_INDEX_URL}" \
  --extra-index-url "${PYPI_INDEX_URL}" \
  -r requirements/musa_private.txt

export PIP_INDEX_URL="${PYPI_INDEX_URL}"
pip install . --no-build-isolation -v
```

For an editable install:

```bash
pip install -e . --no-build-isolation -v
```

Verify that the plugin can be imported:

```bash
python -c "from vllm_musa import musa_platform_plugin; print('MUSA plugin loaded')"
python -c "from vllm_musa.platform import mtml_available; print(mtml_available)"
```

`requirements/musa_private.txt` is the source of truth for the MUSA stack; do
not substitute a public CUDA wheel with the same package name.

## Device visibility

Set `MUSA_VISIBLE_DEVICES` before starting a server. It is the authoritative
device list used by the plugin:

```bash
export MUSA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=musa,musa_custom_ops
```

For ccache, native compiler selection, and build troubleshooting, see
[Configuration](configuration.md).
