# 安装

这里介绍 vLLM-MUSA 的详细安装方式；模型启动参数请先查看
[服务配置示例（英文）](cookbook/README.md)。

## 环境要求

- x86_64 Linux、CPython 3.10（固定版本的 MUSA wheel 针对这一组合发布）。
- 配备兼容 MUSA 驱动和工具包的摩尔线程（Moore Threads）GPU。
- vLLM-MUSA v0.28.0-dev 及对应的 PyTorch/MUSA wheel。

v0.28.0-dev 依赖栈使用 PyTorch 2.11.x。驱动、工具包和 wheel 应来自同一版本系列。

## 软件包索引

安装过程使用两个软件包索引：

| 索引 | 提供内容 |
|---|---|
| 摩尔线程 | `requirements/musa_private.txt` 中固定的 `torch`、`torch_musa`、`mate`、`flash_attn_3`、`flash_mla`、`deep-gemm`、`tilelang_musa`、`triton` 等 MUSA wheel |
| 公共 PyPI | `torchada`、普通构建/运行依赖以及内置 vLLM 所需的依赖 |

请先使用摩尔线程索引和 `--no-deps` 安装 MUSA wheel；否则未固定版本的
`torch` 可能会解析为公共 CUDA wheel，而不是 MUSA wheel。

## 容器

v0.28.0 推荐直接使用正式镜像：

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0
docker pull "${VLLM_MUSA_IMAGE}"
```

该 registry tag 是计划使用的正式镜像名称。如果暂未发布，请改为使用下文构建的
本地镜像。

镜像以 `vllm serve` 为入口点；在模型路径后追加
[服务配置示例（英文）](cookbook/README.md)中的引擎参数，并为 `docker run`
配置部署环境所需的 MUSA 运行时和模型卷参数。

需要源码构建时，在仓库根目录运行 `bash docker/build_image.sh`。若只需要
用于测试或交互的命令行镜像，请使用 `--target final`。构建参数和兼容性说明见
[docker/README.md](../docker/README.md)。

## 源码安装

MUSA wheel 位于摩尔线程软件包索引中。对于 v0.28.0 系列版本，请克隆对应的
源码分支（也可以替换为部署所用的确切 release tag 或 commit），再安装
MUSA wheel 和公共依赖：

```bash
git clone --branch v0.28.0-dev --single-branch \
  https://github.com/MooreThreads/vllm-musa.git
cd vllm-musa

export MUSA_PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple
export PYPI_INDEX_URL=https://pypi.org/simple

pip install --no-deps --index-url "${MUSA_PIP_INDEX_URL}" \
  -r requirements/musa_private.txt
pip install --index-url "${PYPI_INDEX_URL}" \
  -r requirements/build.txt -r requirements/common.txt
pip install --index-url "${MUSA_PIP_INDEX_URL}" \
  --extra-index-url "${PYPI_INDEX_URL}" \
  -r requirements/musa_private.txt

export PIP_INDEX_URL="${PYPI_INDEX_URL}"
pip install . --no-build-isolation -v
```

开发时使用可编辑安装：

```bash
pip install -e . --no-build-isolation -v
```

验证插件：

```bash
python -c "from vllm_musa import musa_platform_plugin; print('MUSA plugin loaded')"
python -c "from vllm_musa.platform import mtml_available; print(mtml_available)"
```

`requirements/musa_private.txt` 是 MUSA 软件栈的版本来源，不要用同名的公共
CUDA wheel 替换。

## 设备可见性

启动服务前设置 `MUSA_VISIBLE_DEVICES`；它是插件使用的权威设备列表：

```bash
export MUSA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=musa,musa_custom_ops
```

ccache、原生编译器和构建排障说明见
[配置参考（英文）](configuration.md)。
