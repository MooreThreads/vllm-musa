<p align="center">
  <img src="assets/logo.png" alt="vLLM MUSA" width="60%">
</p>

<h2 align="center">
vLLM Hardware Plugin for Moore Threads MUSA
</h2>

<p align="center">
  English | <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"></a>
</p>

---

## About

The vLLM Hardware Plugin for Moore Threads MUSA integrates [Moore Threads](https://www.mthreads.com/) (MUSA) GPUs with [vLLM](https://docs.vllm.ai/en/latest/) to enable high-performance large language model inference. It follows the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162) and [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161) principles, providing a modular interface for Moore Threads MUSA hardware.

The plugin leverages the following key components:

- **[torchada](https://github.com/MooreThreads/torchada)**: CUDA→MUSA compatibility layer for PyTorch — run CUDA code on MUSA with zero code changes
- **[mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py)**: Moore Threads Management Library (MTML) Python bindings for device management and queries
- **[MATE](https://github.com/MooreThreads/mate)**: MUSA AI Tensor Engine — high-performance computing library optimized for LLM inference on MUSA architecture
- **[torch_musa](https://github.com/MooreThreads/torch_musa)**: PyTorch backend for Moore Threads (MUSA) GPUs — extends PyTorch with native MUSA device support

## Requirements

- **Python**: 3.10 — the pinned MUSA wheels are published for CPython 3.10 on
  x86_64, so other versions cannot resolve them
- **Hardware**: Moore Threads (MUSA) GPU with MUSA toolkit installed
- **Dependencies**:
  - [torchada](https://github.com/MooreThreads/torchada) — CUDA→MUSA compatibility layer
  - [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py) — MTML Python bindings (pymtml)
  - [MATE](https://github.com/MooreThreads/mate) — MUSA AI Tensor Engine
  - [torch_musa](https://github.com/MooreThreads/torch_musa) — PyTorch backend for MUSA GPUs

## Getting Started

### Supported Versions

| vLLM Version | PyTorch Version | Engine  | Status       |
|--------------|-----------------|---------|--------------|
| v0.24.0       | 2.11.x          | V1 only | ✅ Supported |

> **Note**: This plugin uses vLLM's V1 engine architecture (the V0 engine is not supported). Within the V1 engine, vLLM v0.24.0 auto-selects its **Model Runner V2** for certain architectures (e.g. Qwen3, DeepSeek-V2, Llama) and the V1 model runner for others; both are supported on MUSA. Set `VLLM_USE_V2_MODEL_RUNNER=1` or `0` to force one.

### Docker image

The Docker flow installs the MUSA SDK, the MUSA wheels, `vllm-musa`, the
vendored vLLM, and `pytest` into one image, and is the least error-prone way to
get a working environment. Its working directory is `/vllm-workspace`, matching
the upstream vLLM runtime image. The default target also matches
`vllm-openai` and starts `vllm serve`:

```bash
bash docker/build_image.sh
```

Use `--target final` to build the shell/test image without the serving
entrypoint.

See [docker/README.md](docker/README.md) for version compatibility and build
options. To install onto a host that already has the MUSA SDK, use the source
install below.

### Package indexes

`vllm-musa` resolves its dependencies from **two** indexes:

| Index | URL | Provides |
|---|---|---|
| Moore Threads | `https://dl.mthreads.com/repo/api/pypi/pypi/simple` | the MUSA wheels pinned in `requirements/musa_private.txt` — `torch`, `torch_musa`, `mate`, `flash_attn_3`, `flash_mla`, `deep-gemm`, `tilelang_musa`, `triton`, `apache-tvm-ffi`, … |
| Public PyPI | `https://pypi.org/simple`, or a mirror | the ordinary third-party wheels in `requirements/build.txt` and `requirements/common.txt` |

Most of the MUSA wheels are not published on public PyPI, so installing
`requirements/musa.txt` with pip's default index fails with
`No matching distribution found for torch==2.11.0.post1+musa5.2.0`.

The MUSA wheels must be **resolved** with the Moore Threads index as the sole
`--index-url`, which is why the install below starts with a pass that installs
only them.

Pass 3 below does pass both indexes. That is safe only because pass 1 has already
installed every MUSA wheel at its pinned version, so nothing MUSA is re-resolved
there and the merged index only supplies the ordinary dependencies.

### Install from Source

1. Clone the repository:

    ```bash
    git clone https://github.com/MooreThreads/vllm-musa.git
    cd vllm-musa
    ```

2. Select the two indexes:

    ```bash
    export MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
    export PYPI_INDEX_URL=https://pypi.org/simple
    ```

3. Install the Python dependencies in three passes:

    ```bash
    # 1. The MUSA wheels, from the Moore Threads index only. This pass runs
    #    first and with --no-deps because torchada and transformers declare an
    #    unpinned `torch`: resolving them first would pull the public CUDA torch
    #    and the multi-GB nvidia-cuda-* stack over the MUSA one.
    pip install --no-deps --index-url "${MUSA_PIP_INDEX_URL}" \
        -r requirements/musa_private.txt

    # 2. The ordinary third-party wheels, from public PyPI. Pass 1 already
    #    satisfies `torch`.
    pip install --index-url "${PYPI_INDEX_URL}" \
        -r requirements/build.txt -r requirements/common.txt

    # 3. The MUSA wheels' own ordinary dependencies (sympy, networkx, ...).
    #    Pass 1 pinned every MUSA wheel, so none are re-resolved here.
    pip install --index-url "${MUSA_PIP_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        -r requirements/musa_private.txt
    ```

4. Install vLLM Hardware Plugin for Moore Threads MUSA. The vendored vLLM takes
   its dependencies from public PyPI, so keep that index selected here:

    ```bash
    export PIP_INDEX_URL="${PYPI_INDEX_URL}"

    # Standard installation (installs vLLM MUSA plugin and vLLM)
    pip install . --no-build-isolation -v

    # Or editable installation for development
    pip install -e . --no-build-isolation -v
    ```

    A standard installation creates one `vllm-musa` distribution that owns
    both the `vllm` and `vllm_musa` Python packages. Do not install the official
    `vllm` wheel in the same environment because both distributions would own
    the same `vllm/` files. Editable installs intentionally keep the existing
    two-distribution development layout: `vllm` points at
    `third_party/vllm`, and `vllm-musa` points at this repository.

    Build and inspect a release wheel with:

    ```bash
    python -m build --wheel --no-isolation
    python tests/packaging/verify_bundled_wheel.py dist/vllm_musa-*.whl
    ```

5. Verify the installation:

    ```bash
    # Check plugin registration
    python -c "from vllm_musa import musa_platform_plugin; print('Plugin loaded successfully')"

    # Check MTML device management
    python -c "from vllm_musa.platform import mtml_available; print(f'MTML available: {mtml_available}')"
    ```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MUSA_VISIBLE_DEVICES` | Control which MUSA devices are visible (similar to `CUDA_VISIBLE_DEVICES`) |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Recommended for multi-process workers |
| `VLLM_MUSA_CUSTOM_OP_USE_NATIVE` | Use vLLM custom ops native implementation (default: `False`) |
| `VLLM_MUSA_WORKER_TERMINATION_TIMEOUT_S` | Control vLLM v1 worker shutdown timeout (default: `4s`) |
| `VLLM_MUSA_USE_CCACHE` | Enable ccache for native extension builds when `ccache` is installed (default: `1`) |
| `VLLM_MUSA_CCACHE` | Override the ccache executable used by `setup.py` (default: first `ccache` in `PATH`) |
| `VLLM_MUSA_CCACHE_DIR` | Override the ccache directory used by `setup.py` (default: `<repo>/.ccache`) |
| `VLLM_MUSA_CCACHE_MAXSIZE` | Optional ccache max-size value passed through as `CCACHE_MAXSIZE` |
| `VLLM_MUSA_REAL_MCC` | Override the real MUSA compiler wrapped by ccache (default: detected `mcc`) |

### ccache for native rebuilds

When `ccache` is available in `PATH`, source installs automatically route the
host C++ compiler and MUSA `mcc` through ccache. The generated `mcc` wrapper
normalizes MUSA-only inputs such as `.mu` sources to cacheable `.cu` copies and
hides `-x musa` from ccache while still passing it to `mcc`. The default cache
lives in `<repo>/.ccache`, so a second
`pip install -e . --no-build-isolation -v` from the same checkout can reuse
cached `.cu`, `.mu`, and C++ object compilation.

Useful commands:

```bash
ccache --zero-stats
pip install -e . --no-build-isolation -v
ccache --show-stats
```

## Usage

Once installed, the plugin is **automatically detected** by vLLM. Simply run vLLM as usual:

```python
from vllm import LLM, SamplingParams

# vLLM will automatically use the MUSA platform
llm = LLM(model="your-model-path", trust_remote_code=True)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate(["Hello, how are you?"], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI-Compatible Server

```bash
# Start the server
vllm serve /path/to/model/

# Test completions API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/model/", "prompt": "Hello!", "max_tokens": 50}'

# Test chat completions API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/model/", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50}'
```

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_musa.py -v
pytest tests/test_patches.py -v

# Run with coverage
make test-cov
```

## Project Structure

```
vllm-musa/
├── pyproject.toml              # Project configuration
├── README.md                   # Documentation (English)
├── README_CN.md                # Documentation (中文)
├── LICENSE                     # Apache 2.0 License
├── requirements/               # Dependency pins (build, common, musa_private)
├── docker/                     # Image build flow (musa.Dockerfile, build_image.sh)
├── third_party/                # PINS + the upstream vLLM cloned at build time
├── build_utils/                # Build helpers (ccache wrapper)
├── tools/                      # Sync, verify, and patch-validation utilities
├── example/                    # Usage examples
├── csrc/                       # C/C++ source files
├── docs/                       # Additional documentation
├── vllm_musa/                  # Main package
│   ├── __init__.py             # Plugin entry point
│   ├── platform.py             # MUSA platform implementation
│   └── patches/                # Patches against upstream vLLM
│       ├── __init__.py         # Patch application logic
│       ├── series/             # Build-time source patch series
│       └── *.patch.py          # Import-time object patches
└── tests/                      # Test suite
    ├── conftest.py             # Pytest fixtures
    ├── test_musa.py            # Platform tests
    └── test_patches.py         # Patch system tests
```

## Patches

The plugin carries two kinds of change to upstream vLLM. Source edits — the vast
majority — are applied to the pinned vLLM clone **at build time** as a
`git format-patch` series under `vllm_musa/patches/series/`, so the installed
vLLM is already patched. A small set of live-object monkey-patches that have no
source-diff form run **at import time** (`vllm_musa/patches/*.patch.py`); these
are the only runtime patches. For details on both mechanisms, see
[patches/README.md](vllm_musa/patches/README.md).

## Contributing

We welcome and value any contributions and collaborations. Please set up pre-commit hooks to ensure code quality before submitting:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

Once installed, the hooks will automatically run on every commit, checking for:
- Trailing whitespace and file formatting
- Import sorting (isort)
- Code formatting (black)
- Linting (ruff)
- Spelling errors (codespell)
- Common issues (merge conflicts, debug statements, large files, etc.)

You can also run checks manually:

```bash
make pre-commit    # Run pre-commit hooks on all files
make test          # Run tests
make test-cov      # Run tests with coverage
```

## Contact Us

- For technical questions and feature requests, please use GitHub [Issues](https://github.com/MooreThreads/vllm-musa/issues).
- When reporting a bug, please include your environment information by running `vllm_collect_env` (or `python -m vllm_musa.collect_env`) and pasting the output in your issue.

## Related Projects

| Project | Description |
|---------|-------------|
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput LLM serving engine |
| [torchada](https://github.com/MooreThreads/torchada) | CUDA→MUSA compatibility layer for PyTorch |
| [torch_musa](https://github.com/MooreThreads/torch_musa) | PyTorch support for Moore Threads GPUs |
| [MATE](https://github.com/MooreThreads/mate) | MUSA AI Tensor Engine for LLM acceleration |
| [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py) | MTML Python bindings |

## License

This project is licensed under the [Apache License 2.0](LICENSE).

```
Copyright (c) 2026 Moore Threads Technology Co., Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
