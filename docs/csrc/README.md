# csrc

Native C++/MU kernels for vLLM-MUSA. The code is organized into MUSA-specific custom operators and MTGPU-adapted ports of upstream vLLM kernels.

## Directory Structure

### `musa/`

Custom operators written for MTGPU, registered via `torch::library` under `_C_musa_ops`:

| Operator | Description |
|---|---|
| `musa_fused_gemv` | Fused GEMV |
| `musa_fused_gemv_moe` | Fused GEMV for Mixture-of-Experts routing |

Supporting headers: `common.muh`, `dtype.muh`, FP8 kernels (`fp8/`), and utility includes.

### `quantization/`

Adapted quantization kernels from upstream vLLM:

- `activation_kernels.cu` – FP8 activation quantization
- `gptq/q_gemm.cu` – GPTQ dequantization GEMM (adapted from ExLlamaV2 / GPTQ-for-LLaMa)

### `mamba/`

- `mamba_ssm/selective_scan_fwd.cu` – Selective scan forward kernel for Mamba SSM models

### `custom_all_reduce.cu`

Custom all-reduce implementation for multi-GPU communication via IPC shared memory.
