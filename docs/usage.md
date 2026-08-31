# Usage

After installation, vLLM discovers the MUSA plugin automatically.

## Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model-path", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate(["Hello, how are you?"], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## OpenAI-compatible server

Start the server:

```bash
vllm serve /path/to/model \
  --trust-remote-code \
  --served-model-name my-model
```

Test the completions API:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","prompt":"Hello!","max_tokens":50}'
```

Test the chat-completions API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
```

For model-specific TP, scheduler, graph, and speculative-decoding settings,
use the [serving cookbook](cookbook/README.md).
