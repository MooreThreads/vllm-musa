# Generation examples

Model-specific offline and online generation examples for S5000.

## Qwen3.5-35B-A3B-FP8

Both examples use the
[cookbook profile](../../docs/cookbook/qwen/qwen3.5-35b-a3b-fp8.md) — TP4 on
four S5000 cards with three-token MTP speculative decoding — sized for at most
four concurrent requests. With MTP3 every request verifies four tokens per
decode step, so the CUDA graphs cover 4, 8, 12 and 16 tokens.

Thinking is disabled, and sampling follows the model card for general tasks:
`temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.

The checkpoint path is `/home/dist/models/Qwen3.5-35B-A3B-FP8`; edit `MODEL`
in the offline script or the `vllm serve` argument if yours differs. With
fewer cards, lower the tensor parallelism to match (the FP8 weights fit on
two).

### Offline

`qwen3_5_offline.py` runs four chat prompts in one batch through `vllm.LLM`
and prints the answers.

```bash
MUSA_VISIBLE_DEVICES=0,1,2,3 python example/generate/qwen3_5_offline.py
```

### Online

Start the server:

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1
MUSA_VISIBLE_DEVICES=0,1,2,3 vllm serve /home/dist/models/Qwen3.5-35B-A3B-FP8 \
    --served-model-name Qwen3.5-35B-A3B-FP8 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --mamba-ssm-cache-dtype float32 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --attention-config '{"backend":"FLASH_ATTN"}' \
    --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16]}' \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

Then send a regular and a streaming chat request with `qwen3_5_online.py`
(`VLLM_BASE_URL` and `VLLM_MODEL` override the default
`http://localhost:8000` and `Qwen3.5-35B-A3B-FP8`):

```bash
python example/generate/qwen3_5_online.py
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 256,
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Notes

- Some TileLang kernels compile on first use, so the first requests after
  startup are slow. Measure performance with `vllm bench serve` against a
  warmed-up server.
- To see the model's reasoning, set `enable_thinking` to `true` and raise
  `max_tokens`; the model card recommends `temperature=1.0, top_p=0.95` for
  thinking mode.
