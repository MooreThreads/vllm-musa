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

## 项目简介

vLLM-MUSA 是摩尔线程为 [vLLM](https://github.com/vllm-project/vllm) 提供的后端，
为 MUSA GPU 提供 OpenAI 兼容接口的推理服务引擎。

- v0.24.0 系列版本，采用 vLLM V1 引擎。
- 已在 PyTorch 2.11.x MUSA 软件栈上完成验证。
- 提供 MUSA 原生注意力机制、通信、自定义算子和编译支持。
- 针对 S5000 上 Qwen 和 DeepSeek-V4-Flash 的各个模型检查点提供配置示例。

## 快速开始

### 安装

使用 v0.24.0 正式镜像：

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.24.0
docker pull "${VLLM_MUSA_IMAGE}"
```

- [安装与源码构建指南](docs/installation-cn.md)
- [Docker 构建指南（英文）](docker/README.md)
- [配置参考（英文）](docs/configuration.md)

### 快速启动

```bash
vllm serve /path/to/model \
  --trust-remote-code \
  --served-model-name my-model
```

上面的命令假设 `vllm` 已安装并位于当前环境的 `PATH` 中；使用正式镜像
时，请参阅 [Docker 构建指南（英文）](docker/README.md) 中的容器启动方式。

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

| vLLM-MUSA | PyTorch/MUSA | 引擎 |
|---|---|---|
| v0.24.0 | 2.11.x | V1 |

V1 引擎会在支持的模型架构上自动选择模型运行器 V2；MUSA 同时支持两种
运行器。

## 贡献

可编辑安装、测试、补丁流程和问题反馈方式请参阅 [开发指南（英文）](docs/development.md)。
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
