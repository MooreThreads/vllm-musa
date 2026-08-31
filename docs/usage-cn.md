# 使用方式

安装完成后，vLLM 会自动发现 MUSA 插件。

## Python 接口

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model-path", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate(["Hello, how are you?"], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## OpenAI 兼容服务

启动服务：

```bash
vllm serve /path/to/model \
  --trust-remote-code \
  --served-model-name my-model
```

测试补全接口：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","prompt":"Hello!","max_tokens":50}'
```

测试聊天补全接口：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
```

模型专属的张量并行、调度器、图捕获和推测解码参数，请参阅
[服务配置示例（英文）](cookbook/README.md)。
