<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" alt="vLLM" width="30%">
  </picture>
</p>

<h2 align="center">
vLLM Hardware Plugin for Moore Threads MUSA
</h2>

<p align="center">
  English | <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
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

- **Python**: 3.9 or higher
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
| 0.17.0       | 2.7.1           | V1 only | ✅ Supported |

> **Note**: This plugin uses vLLM's V1 engine architecture. V0 engine is not supported.

### Install from Source

1. Clone the repository:

    ```bash
    git clone https://github.com/MooreThreads/vllm-musa.git
    cd vllm-musa
    ```

2. Install vLLM Hardware Plugin for Moore Threads MUSA:

    ```bash
    # Standard installation (installs vLLM MUSA plugin and vLLM)
    pip install . --no-build-isolation -v

    # Or editable installation for development
    pip install -e . --no-build-isolation -v
    ```

3. Verify the installation:

    ```bash
    # Check plugin registration
    python -c "from vllm_musa import musa_platform_plugin; print('Plugin loaded successfully')"

    # Check MTML device management
    python -c "from vllm_musa.musa import mtml_available; print(f'MTML available: {mtml_available}')"
    ```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MUSA_VISIBLE_DEVICES` | Control which MUSA devices are visible (similar to `CUDA_VISIBLE_DEVICES`) |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Recommended for multi-process workers |
| `VLLM_MUSA_CUSTOM_OP_USE_NATIVE` | Use vLLM custom ops native implementation (default: `False`) |

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
├── example/                    # Usage examples
├── csrc/                       # C/C++ source files
├── docs/                       # Additional documentation
├── vllm_musa/                  # Main package
│   ├── __init__.py             # Plugin entry point
│   ├── musa.py                 # MUSA platform implementation
│   └── patches/                # Runtime compatibility patches
│       ├── __init__.py         # Patch application logic
│       └── *.patch.py          # Individual patch files
└── tests/                      # Test suite
    ├── conftest.py             # Pytest fixtures
    ├── test_musa.py            # Platform tests
    └── test_patches.py         # Patch system tests
```

## Runtime Patches

The plugin includes runtime patches to ensure compatibility with upstream vLLM. For details on the patching mechanism, see [patches/README.md](vllm_musa/patches/README.md).

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
