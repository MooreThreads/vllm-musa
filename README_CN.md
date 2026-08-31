<p align="center">
  <img src="assets/logo.png" alt="vLLM MUSA" width="60%">
</p>

<h2 align="center">
摩尔线程 MUSA 的 vLLM 硬件插件
</h2>

<p align="center">
  <a href="README.md">English</a> | 中文
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"></a>
</p>

---

## 关于

摩尔线程 MUSA 的 vLLM 硬件插件将[摩尔线程](https://www.mthreads.com/) (MUSA) GPU 与 [vLLM](https://docs.vllm.ai/en/latest/) 集成，以实现高性能大语言模型推理。该插件遵循 [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162) 和 [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161) 设计原则，为摩尔线程 MUSA 硬件提供模块化接口。

本插件基于以下核心组件：

- **[torchada](https://github.com/MooreThreads/torchada)**：CUDA→MUSA 兼容层 — 无需修改代码即可在 MUSA 上运行 CUDA 代码
- **[mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py)**：摩尔线程管理库 (MTML) Python 绑定，用于设备管理和查询
- **[MATE](https://github.com/MooreThreads/mate)**：MUSA AI 张量引擎 — 针对 MUSA 架构优化的高性能 LLM 推理库
- **[torch_musa](https://github.com/MooreThreads/torch_musa)**：摩尔线程 (MUSA) GPU 的 PyTorch 后端 — 为 PyTorch 提供原生 MUSA 设备支持

## 环境要求

- **Python**：3.10 — 本套件所固定的 MUSA wheel 仅发布了 x86_64 上的 CPython 3.10
  版本，其他 Python 版本无法解析这些依赖
- **硬件**：安装了 MUSA 工具包的摩尔线程 (MUSA) GPU
- **依赖项**：
  - [torchada](https://github.com/MooreThreads/torchada) — CUDA→MUSA 兼容层
  - [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py) — MTML Python 绑定 (pymtml)
  - [MATE](https://github.com/MooreThreads/mate) — MUSA AI 张量引擎
  - [torch_musa](https://github.com/MooreThreads/torch_musa) — MUSA GPU 的 PyTorch 后端

## 快速上手

### 支持的版本

| vLLM 版本 | PyTorch 版本 | 引擎    | 状态         |
|-----------|--------------|---------|--------------|
| v0.28.0    | 2.11.x（MUSA）| 仅 V1   | 升级候选     |

> **注意**：当前分支以 `third_party/PINS` 中固定的 vLLM v0.28.0 release
> commit 为基线。依赖栈有意保留 MUSA PyTorch 2.11 wheel，而不是上游 vLLM
> CUDA 使用的 PyTorch 2.13；升级候选必须通过构建与模型 smoke 后才能标记为已支持。

### Docker 镜像

Docker 流程会把 MUSA SDK、MUSA wheel、`vllm-musa`、内置的 vLLM 以及 `pytest`
一次性装入同一镜像，是最不容易出错的方式。镜像工作目录为 `/vllm-workspace`，
与上游 vLLM runtime 镜像保持一致。默认 target 也与 `vllm-openai` 一致，启动
`vllm serve`：

```bash
bash docker/build_image.sh
```

如需不带 serving entrypoint 的 shell/test 镜像，请增加 `--target final`。

版本兼容性和构建选项参见 [docker/README.md](docker/README.md)。若要安装到已具备
MUSA SDK 的主机上，请使用下面的源码安装。

### 服务 Cookbook

逐个 checkpoint 的 Qwen 和 DeepSeek-V4-Flash 服务配置收录在
[v0.28.0-dev 服务 Cookbook](docs/cookbook/README.md) 中。能在单张 S5000 上装下的
checkpoint 优先使用 TP1；DeepSeek-V4-Flash 使用 TP8。这些配置是开发分支的起始配置，
上线前请针对实际镜像和业务流量重新验证。

### 新模型支持镜像

以下镜像包含近期新增的 MUSA 模型支持：

| 模型 | 模型卡片 | 镜像 |
|---|---|---|
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:qwen38-flash-next` |
| Hy4-preview | [tencent/Hy4-preview](https://huggingface.co/tencent/Hy4-preview) | `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:hy4-preview` |
| GLM-5.3-Flash | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:glm53-flash` |

### 软件包索引

`vllm-musa` 的依赖来自**两个**索引：

| 索引 | URL | 提供内容 |
|---|---|---|
| 摩尔线程 | `https://dl.mthreads.com/repo/api/pypi/pypi/simple` | `requirements/musa_private.txt` 中约束的 MUSA wheel — `torch`、`torch_musa`、`mate`、`flash_attn_3`、`flash_mla`、`deep-gemm`、`tilelang_musa`、`triton`、`apache-tvm-ffi` 等 |
| 公共 PyPI | `https://pypi.org/simple` 或其镜像 | `torchada` 以及 `requirements/build.txt` 与 `requirements/common.txt` 中的普通第三方 wheel |

大部分 MUSA wheel 并未发布到公共 PyPI，因此用 pip 默认索引安装
`requirements/musa.txt` 会失败并报
`No matching distribution found for torch==2.11.0.post1+musa5.2.0`。

MUSA wheel 必须在摩尔线程索引作为**唯一** `--index-url` 的前提下解析，因此下面的
安装步骤以“只装 MUSA wheel”的一步开始。

下面的第 3 步确实同时传入了两个索引。这样做是安全的，因为第 1 步已经按固定版本装好
了所有 MUSA wheel，该步不会重新解析任何 MUSA wheel，合并索引只用于补齐普通依赖。

### 从源码安装

1. 克隆仓库：

    ```bash
    git clone --branch v0.28.0-dev --single-branch \
        https://github.com/MooreThreads/vllm-musa.git
    cd vllm-musa
    ```

2. 选定两个索引：

    ```bash
    export MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
    export PYPI_INDEX_URL=https://pypi.org/simple
    ```

3. 分三步安装 Python 依赖：

    ```bash
    # 1. MUSA wheel，仅从摩尔线程索引安装。该步必须最先执行且带 --no-deps：
    #    torchada 和 transformers 声明的 `torch` 没有版本约束，若先解析它们会拉取
    #    公共 CUDA 版 torch 及数 GB 的 nvidia-cuda-* 依赖，覆盖 MUSA 版。
    pip install --no-deps --index-url "${MUSA_PIP_INDEX_URL}" \
        -r requirements/musa_private.txt

    # 2. 普通第三方 wheel，从公共 PyPI 安装。第 1 步已满足 `torch`。
    pip install --index-url "${PYPI_INDEX_URL}" \
        -r requirements/build.txt -r requirements/common.txt

    # 3. 补齐 MUSA wheel 自身的普通依赖（sympy、networkx 等）。第 1 步已固定所有
    #    MUSA wheel，这里不会重新解析它们。
    pip install --index-url "${MUSA_PIP_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        -r requirements/musa_private.txt
    ```

4. 安装摩尔线程 MUSA 的 vLLM 硬件插件。内置 vLLM 的依赖来自公共 PyPI，因此这里
   保持选用该索引：

    ```bash
    export PIP_INDEX_URL="${PYPI_INDEX_URL}"

    # 标准安装（安装 vLLM MUSA 插件和 vLLM）
    pip install . --no-build-isolation -v

    # 或可编辑安装（用于开发）
    pip install -e . --no-build-isolation -v
    ```

5. 验证安装：

    ```bash
    # 检查插件注册
    python -c "from vllm_musa import musa_platform_plugin; print('插件加载成功')"

    # 检查 MTML 设备管理
    python -c "from vllm_musa.platform import mtml_available; print(f'MTML 可用: {mtml_available}')"
    ```

### 环境变量

| 变量 | 描述 |
|------|------|
| `MUSA_VISIBLE_DEVICES` | 控制可见的 MUSA 设备（类似于 `CUDA_VISIBLE_DEVICES`） |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 多进程 worker 推荐设置 |
| `VLLM_MUSA_CUSTOM_OP_USE_NATIVE` | 使用 vLLM 自定义算子的原生实现（默认：`False`） |
| `VLLM_MUSA_WORKER_TERMINATION_TIMEOUT_S` | 控制 vLLM v1 worker 关闭超时时间（默认： `4s`） |
| `VLLM_MUSA_USE_CCACHE` | 当系统安装了 `ccache` 时，为原生扩展构建启用 ccache（默认：`1`） |
| `VLLM_MUSA_CCACHE` | 覆盖 `setup.py` 所使用的 ccache 可执行文件（默认：`PATH` 中的第一个 `ccache`） |
| `VLLM_MUSA_CCACHE_DIR` | 覆盖 `setup.py` 所使用的 ccache 目录（默认：`<repo>/.ccache`） |
| `VLLM_MUSA_CCACHE_MAXSIZE` | 可选的 ccache 最大容量，将作为 `CCACHE_MAXSIZE` 透传 |
| `VLLM_MUSA_REAL_MCC` | 覆盖被 ccache 包装的真实 MUSA 编译器（默认：自动探测到的 `mcc`） |

### 用于原生重新构建的 ccache

当 `PATH` 中存在 `ccache` 时，源码安装会自动让宿主 C++ 编译器和 MUSA `mcc` 走
ccache。生成的 `mcc` wrapper 会把 `.mu` 等 MUSA 专有输入规范化为可缓存的 `.cu`
副本，并对 ccache 隐藏 `-x musa`，同时仍将其传给 `mcc`。默认缓存目录为
`<repo>/.ccache`，因此在同一份检出中第二次执行
`pip install -e . --no-build-isolation -v` 可以复用已缓存的 `.cu`、`.mu` 和 C++
目标文件。

常用命令：

```bash
ccache --zero-stats
pip install -e . --no-build-isolation -v
ccache --show-stats
```

## 使用方法

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
├── requirements/               # 依赖版本约束（build、common、musa_private）
├── docker/                     # 镜像构建流程（musa.Dockerfile、build_image.sh）
├── third_party/                # PINS 及构建时克隆的上游 vLLM
├── build_utils/                # 构建辅助（ccache wrapper）
├── tools/                      # 同步、校验与补丁验证工具
├── example/                    # 使用示例
├── csrc/                       # C/C++ 源文件
├── docs/                       # 附加文档
├── vllm_musa/                  # 主包
│   ├── __init__.py             # 插件入口
│   ├── platform.py             # MUSA 平台实现
│   └── patches/                # 针对上游 vLLM 的补丁
│       ├── __init__.py         # 补丁应用逻辑
│       ├── series/             # 构建时的源码补丁序列
│       └── *.patch.py          # 导入时的对象补丁
└── tests/                      # 测试套件
    ├── conftest.py             # Pytest fixtures
    ├── test_musa.py            # 平台测试
    └── test_patches.py         # 补丁系统测试
```

## 补丁

本插件对上游 vLLM 的改动分为两类。绝大多数是源码改动，它们以
`git format-patch` 序列的形式存放在 `vllm_musa/patches/series/`，并在**构建时**
应用到固定版本的 vLLM 克隆上，因此安装完成的 vLLM 已经是打好补丁的。另有少量无法
表示为源码 diff 的实时对象 monkey-patch，在**导入时**运行
（`vllm_musa/patches/*.patch.py`），它们是仅有的运行时补丁。两种机制的详情请参阅
[patches/README.md](vllm_musa/patches/README.md)。

## 贡献

我们欢迎并重视任何形式的贡献与协作。提交前请设置 pre-commit hooks 以确保代码质量：

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

也可以手动运行检查：

```bash
make pre-commit    # 对所有文件运行 pre-commit hooks
make test          # 运行测试
make test-cov      # 运行覆盖率测试
```

## 联系我们

- 如有技术问题或功能需求，请通过 GitHub [Issues](https://github.com/MooreThreads/vllm-musa/issues) 提交。
- 提交 Bug 时，请运行 `vllm_collect_env`（或 `python -m vllm_musa.collect_env`）并将输出粘贴到 Issue 中，以帮助我们定位问题。

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
