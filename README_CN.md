<p align="center">
  <img src="assets/logo.png" alt="vLLM-MUSA" width="60%">
</p>

<h2 align="center">面向摩尔线程（Moore Threads）MUSA 的高性能大语言模型服务</h2>

<p align="center">
  <a href="README.md">英文版</a> ·
  <a href="https://github.com/MooreThreads/vllm-musa/issues">问题反馈</a> ·
  <a href="docs/cookbook/README.md">服务配置示例（英文）</a> ·
  <a href="docs/installation-cn.md">文档</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="许可证"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"></a>
</p>

---

*新模型支持镜像* 🔥

- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:qwen38-flash-next`
- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:glm53-flash`
- [Hy4-preview](https://huggingface.co/tencent/Hy4-preview) — `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:hy4-preview`

这些模型专用标签独立于当前分支维护；投入生产前请核对确切的镜像和模型版本。

---

## 项目简介

vLLM-MUSA 是摩尔线程为 [vLLM](https://github.com/vllm-project/vllm) 提供的后端，
为 MUSA GPU 提供 OpenAI 兼容接口的推理服务引擎。

- v0.28.0-dev 系列开发版本，采用 vLLM V1 引擎。
- 使用固定的 MUSA 软件栈：PyTorch/torch_musa 2.11.0.post2（MUSA 5.2.0）、
  MATE 0.2.7 和 torchada 0.1.95；当前分支仍属于升级候选版本。
- 提供 MUSA 原生注意力机制、通信、自定义算子和编译支持。
- 针对 S5000 上 Qwen、DeepSeek-V4-Flash 和 Nemotron-3.5 的各个模型检查点提供配置示例。

## 快速开始

### 安装

使用已发布的 v0.28.0 正式镜像：

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0
docker pull "${VLLM_MUSA_IMAGE}"
```

该镜像独立于当前分支构建。请将镜像内已安装的软件包版本与 `third_party/PINS`
核对；需要与当前分支完全一致时，请按照[安装指南](docs/installation-cn.md)构建本地镜像。

#### 预构建集成镜像

`registry.mthreads.com/mcconline/inference/vllm-omni:flux3` 打包了 vLLM
`0.28.0`、`vllm-musa` `0.1.28`，以及带 FLUX.3 Image Klein 的 vllm-omni
`0.28.0`（vllm-omni PR
[#9007](https://github.com/vllm-project/vllm-omni/pull/9007) 回移到 `v0.28.0`
tag，并包含两处面向 MUSA 的调整：按平台门控的 RoPE dtype，以及 VAE 融合
GroupNorm+SiLU 路径的设备判断）：

```bash
docker run -it --rm --entrypoint bash \
  --runtime=mthreads --shm-size 32g \
  -e MTHREADS_VISIBLE_DEVICES=0 \
  -v /ipfs:/ipfs \
  registry.mthreads.com/mcconline/inference/vllm-omni:flux3
```

它需要 MUSA `5.2.0-server` 驱动，以及一张空闲显存约 25 GB 的 S5000。FLUX.3 权重未
打包进镜像，请挂载或下载后通过 `--model <path>` 指定。镜像内
`/opt/flux3/TESTING.md` 记录了六项验证步骤，从免权重的正确性 smoke 到 1024x1024 的
官方文生图示例。已在单张 S5000 上验证：官方示例在约 36 秒权重加载后约 2.5 秒出图，
FLUX.3 单元测试通过（140 passed、39 skipped）。上面的 tag 是可变指针，本次验证的
manifest 为
`sha256:edb4ae4a04f497c36f0cc282b1a188ad29acbe428457b6fe33b8306ade5ced43`，
需要可复现时可用 `@sha256:` 固定。

- [安装与源码构建指南](docs/installation-cn.md)
- [Docker 构建指南（英文）](docker/README.md)
- [配置参考（英文）](docs/configuration.md)

### 快速启动

```bash
vllm serve /path/to/model \
  --trust-remote-code \
  --served-model-name my-model
```

上面的命令假设 `vllm` 已安装并位于当前环境的 `PATH` 中；使用正式镜像时，请参阅
[Docker 构建指南（英文）](docker/README.md)中的容器启动方式。

OpenAI 兼容接口地址为 `http://localhost:8000/v1`。
推荐的张量并行、调度器和推测解码参数请参阅
[服务配置示例（英文）](docs/cookbook/README.md)。

## 文档

- [服务配置示例（英文）](docs/cookbook/README.md)
- [安装](docs/installation-cn.md)
- [Python 与 OpenAI 兼容接口用法](docs/usage-cn.md)
- [配置（英文）](docs/configuration.md)
- [开发指南（英文）](docs/development.md)
- [MUSA 开发者指南（英文）](docs/mdm-developer-guide.md)

## 支持版本

| vLLM-MUSA | PyTorch/MUSA | 引擎 | 状态 |
|---|---|---|---|
| v0.28.0-dev | 2.11.x | V1 引擎；支持时使用运行器 V2 | 升级候选 |

V1 引擎会在支持的模型架构上自动选择模型运行器 V2；MUSA 同时支持两种运行器。
当前分支跟踪 `third_party/PINS` 中固定的上游 v0.28.0 提交，并保留 MUSA
PyTorch 2.11.x 软件栈；只有完成构建和模型冒烟测试后，才能将其提升为正式支持版本。

## 贡献

可编辑安装、测试、补丁流程和问题反馈方式请参阅[开发指南（英文）](docs/development.md)。
欢迎通过 [GitHub 问题反馈](https://github.com/MooreThreads/vllm-musa/issues)
提交问题或参与贡献。

## 相关项目

- [Moore Threads](https://www.mthreads.com/)
- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM 文档](https://docs.vllm.ai/en/latest/)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [torchada](https://github.com/MooreThreads/torchada)
- [torch_musa](https://github.com/MooreThreads/torch_musa)
- [MATE](https://github.com/MooreThreads/mate)
- [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py)

## 许可证

vLLM-MUSA 采用 [Apache 2.0 许可证](LICENSE)。
