# Development

## Repository layout

| Path | Role |
|---|---|
| `vllm_musa/` | MUSA plugin and platform implementation |
| `vllm_musa/patches/series/` | Build-time vLLM patch series |
| `vllm_musa/patches/*.patch.py` | Import-time runtime patches |
| `requirements/` | Dependency pins |
| `docker/` | Image build flow |
| `tools/` | Validation utilities |
| `csrc/` | Native MUSA sources |
| `tests/` | Plugin and patch tests |
| `docs/cookbook/` | Copy-ready serving recipes |
| `pyproject.toml` | Package metadata and build configuration |
| `third_party/` | Pinned upstream source and dependency revisions |
| `build_utils/` | Build helpers and patch application utilities |
| `example/` | Small runnable examples |

## Project background

vLLM-MUSA is maintained by [Moore Threads](https://www.mthreads.com/) as a
hardware backend for [vLLM](https://github.com/vllm-project/vllm). The upstream
design discussions for a pluggable hardware backend and MUSA enablement are
tracked in [vLLM RFC #11162](https://github.com/vllm-project/vllm/issues/11162)
and [RFC #19161](https://github.com/vllm-project/vllm/issues/19161).

The runtime stack is split across these companion projects:

| Project | Role |
|---|---|
| [torch_musa](https://github.com/MooreThreads/torch_musa) | PyTorch MUSA device/runtime implementation |
| [torchada](https://github.com/MooreThreads/torchada) | CUDA-to-MUSA compatibility layer |
| [MATE](https://github.com/MooreThreads/mate) | MUSA attention and tensor kernels |
| [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py) | MUSA ML runtime bindings |

## Editable development

```bash
pip install -e . --no-build-isolation -v
```

## Tests and checks

```bash
make test
make test-cov
pytest tests/test_musa.py -v
pytest tests/test_patches.py -v
make pre-commit
```

Install the hooks once:

```bash
pip install pre-commit
pre-commit install
```

The standard pre-commit suite covers formatting, linting, import hygiene, and
repository policy checks. Run the focused pytest commands above when changing
the plugin or patch series.

## Patch workflow

Most upstream changes are applied to the pinned vLLM source at build time as a
`git format-patch` series. A small set of changes without a source-diff form
is applied to live Python objects at import time. See
[vllm_musa/patches/README.md](../vllm_musa/patches/README.md) for ordering and
details.

## Issue reports

Open a [GitHub issue](https://github.com/MooreThreads/vllm-musa/issues) with the
model, serving command, MUSA/driver versions, relevant logs, and output from
`vllm_collect_env` or `python -m vllm_musa.collect_env`.

See [mdm-developer-guide.md](mdm-developer-guide.md) for the broader developer
workflow.
