# SPDX-License-Identifier: Apache-2.0
"""Cross-repository contract checks for torchada's in-place source porter."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_MUSA_STACKS = {
    "torch==2.9.1.post1+musa5.2.0": {
        "private": (
            "torch_musa==2.9.1.post1+musa5.2.0",
            "torchvision==0.24.1.post1+musa5.2.0",
            "torchaudio==2.9.1+musa5.2.0",
            "deep_ep==1.1.0+musa5.2.0torch2.9.1.post1s5000",
        ),
        "torchada": "torchada==0.1.77",
    },
    "torch==2.11.0.post1+musa5.2.0": {
        "private": (
            "torch_musa==2.11.0.post1+musa5.2.0",
            "torchvision==0.26.0.post1+musa5.2.0",
            "torchaudio==2.11.0+musa5.2.0",
            "deep_ep==1.1.0+musa5.2.0torch2.11.0.post1",
        ),
        "torchada": "torchada==0.1.83",
    },
}


def _select_musa_stack(private_requirements):
    torch_pins = sorted(
        line for line in private_requirements if line.startswith("torch==")
    )
    assert len(torch_pins) == 1, f"expected one torch pin, got {torch_pins}"
    torch_pin = torch_pins[0]
    assert torch_pin in SUPPORTED_MUSA_STACKS, f"unsupported MUSA stack: {torch_pin}"
    return SUPPORTED_MUSA_STACKS[torch_pin]


def _declared_musa_stack():
    private_requirements = set(
        (ROOT / "requirements" / "musa_private.txt").read_text().splitlines()
    )
    common_requirements = set(
        (ROOT / "requirements" / "common.txt").read_text().splitlines()
    )
    return (
        private_requirements,
        common_requirements,
        _select_musa_stack(private_requirements),
    )


def test_supported_musa_stack_contract_cases_are_explicit():
    assert set(SUPPORTED_MUSA_STACKS) == {
        "torch==2.9.1.post1+musa5.2.0",
        "torch==2.11.0.post1+musa5.2.0",
    }
    assert (
        SUPPORTED_MUSA_STACKS["torch==2.9.1.post1+musa5.2.0"]["torchada"]
        == "torchada==0.1.77"
    )
    assert (
        SUPPORTED_MUSA_STACKS["torch==2.11.0.post1+musa5.2.0"]["torchada"]
        == "torchada==0.1.83"
    )
    assert (
        "torchvision==0.24.1.post1+musa5.2.0"
        in SUPPORTED_MUSA_STACKS["torch==2.9.1.post1+musa5.2.0"]["private"]
    )
    assert (
        "torchvision==0.26.0.post1+musa5.2.0"
        in SUPPORTED_MUSA_STACKS["torch==2.11.0.post1+musa5.2.0"]["private"]
    )
    for torch_pin, expected in SUPPORTED_MUSA_STACKS.items():
        assert _select_musa_stack({torch_pin}) is expected


def test_torchada_floor_is_consistent():
    assert (
        'dynamic = ["dependencies", "version"]' in (ROOT / "pyproject.toml").read_text()
    )
    private_requirements, common_requirements, expected = _declared_musa_stack()
    assert set(expected["private"]).issubset(private_requirements)
    assert expected["torchada"] in common_requirements


def test_musa_image_runtime_dependency_contract():
    private_requirements = (
        (ROOT / "requirements" / "musa_private.txt").read_text().splitlines()
    )
    runtime_requirements = (
        (ROOT / "requirements" / "vllm_runtime_transitive.txt").read_text().splitlines()
    )
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()

    assert "triton==3.2.0" in private_requirements
    _, _, expected = _declared_musa_stack()
    assert set(expected["private"]).issubset(private_requirements)
    assert "fastapi[standard]" in runtime_requirements
    assert "pycountry" in runtime_requirements
    assert '"requirements/common.txt"' in dockerfile
    exact_import_gates = (
        ("torchada", "torchada"),
        ("torch", "torch"),
        ("torch_musa", "torch_musa"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("deep_ep", "deep_ep"),
    )
    for dist_name, module_name in exact_import_gates:
        gate = f'("{dist_name}", "{module_name}", requirement_prefix("{dist_name}"))'
        assert gate in dockerfile

    exact_version_gate = (
        'exact_version_dists = frozenset({"torchada", "torch", "torch_musa", '
        '"torchvision", "torchaudio", "deep_ep"})'
    )
    assert exact_version_gate in dockerfile
    torchada_gate = '("torchada", "torchada", requirement_prefix("torchada"))'
    torch_gate = '("torch", "torch", requirement_prefix("torch"))'
    assert torchada_gate in dockerfile
    assert dockerfile.index(torchada_gate) < dockerfile.index(torch_gate)
    import_statement = "importlib.import_module(module_name)"
    version_statement = "installed = version(dist_name)"
    assert import_statement in dockerfile
    assert dockerfile.index(import_statement) < dockerfile.index(version_statement)
    assert "if dist_name in exact_version_dists and installed != prefix:" in dockerfile
    assert (
        "if dist_name not in exact_version_dists and prefix "
        "and not installed.startswith(prefix):"
    ) in dockerfile
    assert '("triton", "triton", requirement_prefix("triton"))' in dockerfile
    assert '("uvloop", "uvloop", "")' in dockerfile
    assert '("pycountry", "pycountry", "")' in dockerfile


def test_musa_image_matches_upstream_workspace_and_includes_pytest():
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()
    test_requirements = (ROOT / "requirements" / "test.txt").read_text().splitlines()

    assert "WORKDIR /vllm-workspace" in dockerfile
    assert "COPY requirements/ /vllm-workspace/requirements/" in dockerfile
    assert "COPY . /vllm-workspace" in dockerfile
    assert "/workspace/vllm-musa" not in dockerfile
    assert "pytest" in test_requirements
    assert "-r requirements/test.txt" in dockerfile
    assert '("pytest", "pytest", "")' in dockerfile


def test_musa_image_stage_and_optional_component_contract():
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()
    build_script = (ROOT / "docker" / "build_image.sh").read_text()

    stage_markers = (
        "FROM apt_base AS devel",
        "FROM devel AS vllm_musa_deps",
        "FROM vllm_musa_deps AS vllm_musa_installed",
        "FROM vllm_musa_installed AS vllm_rs_build",
        "FROM vllm_musa_installed AS mooncake",
        "FROM mooncake AS final",
        "FROM final AS vllm-openai",
    )
    stage_positions = [dockerfile.index(marker) for marker in stage_markers]
    assert stage_positions == sorted(stage_positions)
    assert "FROM apt_base AS runtime" not in dockerfile

    base_stage = dockerfile.split("FROM base AS apt_base", 1)[0]
    for name in (
        "MUSA_HOME",
        "MTGPU_TARGET",
        "TORCH_MUSA_ARCH_LIST",
        "MATE_MUSA_ARCH_LIST",
    ):
        assert name not in base_stage

    deps_stage = dockerfile.split("FROM devel AS vllm_musa_deps", 1)[1].split(
        "FROM vllm_musa_deps AS vllm_musa_installed", 1
    )[0]
    assert "MTGPU_TARGET=mp_31" in deps_stage
    assert "TORCH_MUSA_ARCH_LIST=31" in deps_stage
    assert "MATE_MUSA_ARCH_LIST=3.1" in deps_stage

    mooncake_stage = dockerfile.split("FROM vllm_musa_installed AS mooncake", 1)[
        1
    ].split("FROM mooncake AS final", 1)[0]
    assert "MTHREADS_VISIBLE_DEVICES" not in mooncake_stage
    assert "ARG MOONCAKE_VERSION=0.3.12.post1" in mooncake_stage
    assert '"mooncake-transfer-engine-musa==${MOONCAKE_VERSION}"' in mooncake_stage
    assert '--index-url "${PYPI_INDEX_URL}"' in mooncake_stage
    assert "--only-binary=:all:" in mooncake_stage
    assert 'version("mooncake-transfer-engine-musa")' in mooncake_stage
    for source_build_token in (
        "MOONCAKE_REPO",
        "MOONCAKE_COMMIT",
        "git clone",
        "git submodule",
        "dependencies.sh",
        "cmake ..",
        "make install",
    ):
        assert source_build_token not in mooncake_stage

    assert 'MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.12.post1}"' in build_script
    assert '--build-arg MOONCAKE_VERSION="${MOONCAKE_VERSION}"' in build_script
    assert "MOONCAKE_REPO" not in build_script
    assert "MOONCAKE_COMMIT" not in build_script

    assert "ARG BUILD_VLLM_RS=1" in dockerfile
    assert "/tmp/vllm-rs-artifacts/build-mode" in dockerfile
    assert 'BUILD_VLLM_RS="${BUILD_VLLM_RS:-1}"' in build_script
    assert '--build-arg BUILD_VLLM_RS="${BUILD_VLLM_RS}"' in build_script

    final_stage, openai_stage = dockerfile.split("FROM final AS vllm-openai", 1)
    final_stage = final_stage.split("FROM mooncake AS final", 1)[1]
    assert 'CMD ["/bin/bash"]' in final_stage
    assert "ENTRYPOINT" not in final_stage
    assert 'ENTRYPOINT ["vllm", "serve"]' in openai_stage


def test_mooncake_uses_the_pinned_upstream_connector():
    distributed_init = (ROOT / "vllm_musa" / "distributed" / "__init__.py").read_text()
    package_init = (ROOT / "vllm_musa" / "__init__.py").read_text()
    mooncake_compat = (
        ROOT / "vllm_musa" / "distributed" / "mooncake_compat.py"
    ).read_text()
    manifest = (ROOT / "vllm_musa" / "patches" / "manifest.py").read_text()
    legacy_rebind = (
        ROOT
        / "vllm_musa"
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "mooncake_connector.py"
    )

    assert not legacy_rebind.exists()
    assert "import vllm_musa.distributed.kv_transfer" not in distributed_init
    assert "kv_connector/v1/mooncake_connector.py" not in manifest
    configure_index = package_init.index("    configure_legacy_device_filter()")
    patch_index = package_init.index("    _register_patches()", configure_index)
    modules_index = package_init.index("    _register_modules()", patch_index)
    assert configure_index < patch_index < modules_index
    assert '"MC_TE_FILTERS"' in mooncake_compat
    assert '"MOONCAKE_RDMA_DEVICES"' in mooncake_compat
    assert "import mooncake" not in mooncake_compat
    assert "from vllm" not in mooncake_compat


def test_mooncake_example_uses_current_proxy_and_scoped_cleanup():
    script = (
        ROOT / "example" / "disaggregated_serving" / "disaggregated_serving.sh"
    ).read_text()

    assert "third_party/vllm/examples/disaggregated/mooncake_connector" in script
    assert "mooncake_connector_proxy.py" in script
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT" in script
    assert "VLLM_ENFORCE_EAGER" in script
    assert '--max-num-seqs "${MAX_NUM_SEQS}"' in script
    assert "trap 'cleanup 143' TERM" in script
    assert '"transfer_id"' not in script  # The maintained proxy owns the protocol.
    assert "toy_proxy_server.py" not in script
    assert "pgrep" not in script
    assert "pkill" not in script
    assert "kill -- -$$" not in script


def test_mooncake_rdma_container_contract_is_explicit():
    example_readme = (ROOT / "docs" / "example" / "README.md").read_text()
    for token in (
        "--detach",
        "--entrypoint /bin/bash",
        "--network host",
        "sleep infinity",
        "/dev/infiniband:/dev/infiniband",
        "stat -c '%t %T'",
        '--device-cgroup-rule="c ${VERBS_MAJOR}:* rmw"',
        '--device-cgroup-rule="c ${RDMA_CM_MAJOR}:${RDMA_CM_MINOR} rmw"',
        "--env MC_FORCE_HCA=1",
        "--cap-add IPC_LOCK",
        "--ulimit memlock=-1:-1",
        "MC_TE_FILTERS",
    ):
        assert token in example_readme


def test_musa_image_provenance_labels_are_derived_from_source():
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()
    build_script = (ROOT / "docker" / "build_image.sh").read_text()

    expected_labels = (
        'org.opencontainers.image.source="https://github.com/MooreThreads/vllm-musa"',
        'org.opencontainers.image.revision="${VLLM_MUSA_COMMIT}"',
        'org.opencontainers.image.version="${VLLM_MUSA_REF}"',
        'com.mthreads.vllm.version="${VLLM_TAG}"',
    )
    for label in expected_labels:
        assert label in dockerfile
    assert "org.opencontainers.image.created" not in dockerfile

    assert "git rev-parse HEAD" in build_script
    assert (
        "git describe --tags --exact-match 2>/dev/null || " "git branch --show-current"
    ) in build_script
    assert 'awk -F= \'$1 == "VLLM_TAG"' in build_script
    for name in ("VLLM_MUSA_COMMIT", "VLLM_MUSA_REF", "VLLM_TAG"):
        assert f'--build-arg {name}="${{{name}}}"' in build_script


def test_vllm_rs_uses_rsproxy_for_cargo_without_overriding_upstream_toolchain():
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()
    cargo_config = (ROOT / "docker" / "cargo-config.toml").read_text()

    assert "https://sh.rustup.rs" in dockerfile
    assert "--default-toolchain none" in dockerfile
    assert "--default-toolchain 1.86.0" not in dockerfile
    assert "CARGO_NET_RETRY=10" in dockerfile
    assert "COPY docker/cargo-config.toml /root/.cargo/config.toml" in dockerfile
    assert 'replace-with = "rsproxy-sparse"' in cargo_config
    assert 'registry = "sparse+https://rsproxy.cn/index/"' in cargo_config


def test_musa_image_does_not_bake_a_device_id_list():
    """The image selects GPUs through the container runtime, not the driver.

    MUSA_VISIBLE_DEVICES is a device-id list for both the MUSA driver and vLLM.
    Baking "all" leaves the driver on every device (it ignores what it cannot
    parse) but makes vLLM fail engine start on int("all"), so the image must
    leave the variable alone and let the caller set it.
    """
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()
    # Drop comments, then fold backslash continuations so that a multi-line
    # `ENV A=1 \ <newline> B=2` block reads as the single instruction it is.
    body = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    env_instructions = [
        line
        for line in re.sub(r"\\\s*\n\s*", " ", body).splitlines()
        if line.lstrip().startswith(("ENV", "ARG"))
    ]
    assert not any("MUSA_VISIBLE_DEVICES" in line for line in env_instructions)
    assert any("MTHREADS_VISIBLE_DEVICES=all" in line for line in env_instructions)


def test_setup_finds_local_build_helpers_before_importing_them():
    setup = (ROOT / "setup.py").read_text()
    assert setup.index("sys.path.insert(0, str(root))") < setup.index(
        "from build_utils.ccache import"
    )


def test_setup_activates_torchada_without_installing_dependencies():
    setup = (ROOT / "setup.py").read_text()
    assert "ensure_torchada_installed" not in setup
    assert setup.index("import torchada") < setup.index("import torch\n")
    assert setup.index("import torchada") < setup.index(
        "from torch.utils.cpp_extension import"
    )


def test_archive_vllm_install_uses_upstream_version_override():
    setup = (ROOT / "setup.py").read_text()
    assert 'env.setdefault("VLLM_VERSION_OVERRIDE", "0.28.0")' in setup
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM" not in setup


def test_source_distribution_manifest_includes_setup_inputs():
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include requirements *.txt" in manifest
    assert "recursive-include build_utils *.py" in manifest
    assert "include third_party/PINS" in manifest


def test_runtime_jit_headers_are_packaged():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"jit_kernel/csrc/*.h"' in pyproject
    assert '"jit_kernel/tilelang/*.h"' in pyproject
    assert (ROOT / "vllm_musa/jit_kernel/tilelang/_atomic_helper.h").is_file()


def test_no_legacy_mirror_contract_remains():
    legacy_tokens = (
        "csrc_musa",
        "libtorch_stable_musa",
        "attention_musa",
        "quantization_musa",
        "per-file _musa",
    )
    paths = [ROOT / "third_party" / "PINS"]
    gitignore = ROOT / ".gitignore"
    if gitignore.exists():
        paths.append(gitignore)
    paths.extend(sorted((ROOT / "vllm_musa" / "patches" / "series").glob("*.patch")))

    offenders = []
    for path in paths:
        text = path.read_text(errors="replace")
        for token in legacy_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    assert not offenders, offenders


def test_native_sampler_includes_flashinfer_header_by_real_name():
    sampler = (ROOT / "csrc" / "musa" / "sampler.mu").read_text()
    assert "#include <flashinfer/sampling.cuh>" in sampler
    assert "#include <flashinfer/sampling.muh>" not in sampler
