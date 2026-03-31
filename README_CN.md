<div align="center">

# vLLM MUSA 平台插件

**在摩尔线程 GPU 上无缝运行 vLLM**

[English](README.md) | 中文

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

---

一个 vLLM 平台插件，支持在[摩尔线程](https://www.mthreads.com/) (MUSA) GPU 上运行 [vLLM](https://github.com/vllm-project/vllm)，提供完整的功能支持。

## 概述

本插件通过以下组件为 vLLM 提供 MUSA（元计算统一系统架构）支持：

- **[torchada](https://github.com/MooreThreads/torchada)**：CUDA→MUSA 兼容层 — 无需修改代码即可在 MUSA 上运行 CUDA 代码
- **[mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py)**：摩尔线程管理库 (MTML) Python 绑定，用于设备查询
- **[MATE](https://github.com/MooreThreads/mate)**：MUSA AI 张量引擎 — 针对 MUSA 架构优化的高性能 LLM 推理库

## 环境要求

- **Python**：3.9 或更高版本
- **vLLM**：作为依赖项安装
- **硬件**：安装了 MUSA 工具包的摩尔线程 GPU
- **依赖项**：
  - [torchada](https://github.com/MooreThreads/torchada) — CUDA→MUSA 兼容层
  - [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py) — MTML Python 绑定 (pymtml)
  - [MATE](https://github.com/MooreThreads/mate) — MUSA AI 张量引擎

## 安装

### 支持的版本

| vLLM 版本 | PyTorch 版本 | 引擎    | 状态         |
|-----------|--------------|---------|--------------|
| 0.17.0    | 2.7.1        | 仅 V1   | ✅ 已支持    |

> **注意**：本插件使用 vLLM 的 V1 引擎架构，不支持 V0 引擎。

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/MooreThreads/vllm-musa.git
cd vllm-musa

# 标准安装（安装 vLLM MUSA 和 vLLM）
pip install . --no-build-isolation -v

# 或可编辑安装（用于开发）
pip install -e . --no-build-isolation -v
```

## 验证安装

安装后，验证插件是否正常工作：

```bash
# 检查插件注册
python -c "from vllm_musa import musa_platform_plugin; print('插件加载成功')"

# 检查 MTML 设备管理
python -c "from vllm_musa.platform import mtml_available; print(f'MTML 可用: {mtml_available}')"
```

## 环境变量

| 变量 | 描述 |
|------|------|
| `MUSA_VISIBLE_DEVICES` | 控制可见的 MUSA 设备（类似于 `CUDA_VISIBLE_DEVICES`） |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 多进程 worker 推荐设置 |
| `VLLM_MUSA_CUSTOM_OP_USE_NATIVE` | 使用 vLLM 自定义算子的原生实现（默认：False） |

## 快速开始

安装后，插件会被 vLLM **自动检测**。直接像往常一样运行 vLLM 即可：

```python
from vllm import LLM, SamplingParams

# vLLM 会自动使用 MUSA 平台
llm = LLM(model="your-model-path", trust_remote_code=True)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate(["你好，最近怎么样？"], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI 兼容服务器

```bash
# 启动服务器
vllm serve /path/to/model/

# 测试 completions API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/model/", "prompt": "你好！", "max_tokens": 50}'

# 测试 chat completions API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/model/", "messages": [{"role": "user", "content": "1+1等于几？"}], "max_tokens": 50}'
```

## 测试

运行测试套件：

```bash
# 运行所有测试
make test

# 运行特定测试文件
pytest tests/test_musa.py -v
pytest tests/test_patches.py -v

# 运行覆盖率测试
make test-cov
```

## 项目结构

```
vllm-musa/
├── pyproject.toml              # 项目配置
├── README.md                   # 文档（英文）
├── README_CN.md                # 文档（中文）
├── LICENSE                     # Apache 2.0 许可证
├── example/                    # 使用示例
├── csrc/                       # C/C++ 源文件
├── docs/                       # 附加文档
├── vllm_musa/                  # 主包
│   ├── __init__.py             # 插件入口
│   ├── musa.py                 # MUSA 平台实现
│   └── patches/                # 运行时兼容性补丁
│       ├── __init__.py         # 补丁应用逻辑
│       └── *.patch.py          # 单独的补丁文件
└── tests/                      # 测试套件
    ├── conftest.py             # Pytest fixtures
    ├── test_musa.py            # 平台测试
    └── test_patches.py         # 补丁系统测试
```

## 补丁

本插件包含用于 vLLM 兼容性的运行时补丁。详情请参阅 [patches/README.md](vllm_musa/patches/README.md)。

## 贡献

欢迎贡献！提交前请设置 pre-commit hooks 以确保代码质量：

### 设置 Pre-commit Hooks

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 git hooks
pre-commit install

# （可选）对所有文件运行检查
pre-commit run --all-files
```

安装后，hooks 会在每次提交时自动运行，检查以下内容：
- 尾部空白和文件格式
- 导入排序（isort）
- 代码格式化（black）
- 代码检查（ruff）
- 拼写检查（codespell）
- 常见问题（合并冲突、调试语句、大文件等）

### 手动检查

```bash
# 对所有文件运行 pre-commit hooks
make pre-commit

# 运行测试
make test

# 运行覆盖率测试
make test-cov
```

## 问题反馈

提交 Bug 时，请附上环境信息以帮助我们定位问题。运行以下命令并将输出粘贴到 Issue 中：

```bash
vllm_collect_env
```

也可以作为 Python 模块运行：

```bash
python -m vllm_musa.collect_env
```

该命令会输出系统信息、MUSA 驱动/SDK 版本、GPU 详情、PyTorch 和 vLLM 版本以及相关环境变量。

## 相关项目

| 项目 | 描述 |
|------|------|
| [vLLM](https://github.com/vllm-project/vllm) | 高吞吐量 LLM 推理引擎 |
| [torchada](https://github.com/MooreThreads/torchada) | PyTorch 的 CUDA→MUSA 兼容层 |
| [torch_musa](https://github.com/MooreThreads/torch_musa) | 摩尔线程 GPU 的 PyTorch 支持 |
| [MATE](https://github.com/MooreThreads/mate) | 用于 LLM 加速的 MUSA AI 张量引擎 |
| [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py) | MTML Python 绑定 |

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

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
