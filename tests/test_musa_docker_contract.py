import subprocess
from pathlib import Path
from shutil import copyfile


def test_spawn_is_baked_into_the_shared_musa_runtime_layer():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()
    runtime_env = dockerfile.split("FROM devel AS", 1)[0]

    assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" in runtime_env
    assert dockerfile.count("VLLM_WORKER_MULTIPROC_METHOD") == 1


def test_omni_image_installs_with_the_foundation_and_headless_wheels():
    dockerfile = (Path(__file__).parents[1] / "docker" / "omni.Dockerfile").read_text()
    production = dockerfile.split("FROM omni-install AS omni\n", 1)[1]
    development = dockerfile.split("FROM omni-install AS test\n", 1)[1].split(
        "FROM omni-install AS omni\n", 1
    )[0]

    assert "--snapshot /opt/vllm-omni-foundation.txt" in dockerfile
    assert "pip download --require-hashes" in dockerfile
    assert "build_headless_wheels.py" in dockerfile
    assert "FROM omni-base AS omni-wheel" in dockerfile
    assert "/headless-wheels/*.whl /wheels/*.whl" in production
    assert "from=omni-wheel," in production
    assert "from=omni-wheel," not in development
    assert "--mode wheel" in production
    assert "-e /vllm-workspace/vllm-omni" in development
    assert "--mode editable" in development
    assert "python -m pip check" in production
    assert "python -m pip check" in development
    assert "pip uninstall" not in dockerfile
    assert 'CMD ["/bin/bash"]' in dockerfile.split("FROM omni-install AS test", 1)[0]


def test_omni_build_script_accepts_remote_refs(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_OMNI_CLONE", raising=False)
    root = Path(__file__).resolve().parents[1]
    for name in ("docker/build_omni_image.sh", "third_party/PINS"):
        (tmp_path / name).parent.mkdir()
        copyfile(root / name, tmp_path / name)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    command = [
        "bash",
        "docker/build_omni_image.sh",
        "--base",
        "example:base",
        "--dry-run",
    ]

    for args, expected in (
        ([], "origin/main"),
        (["--omni-ref", "some-branch"], "some-branch"),
    ):
        result = subprocess.run(
            command + args, cwd=tmp_path, check=True, capture_output=True, text=True
        )
        ref = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("vllm-omni ref")
        )
        assert ref.split()[-1] == expected


def test_production_image_installs_a_bundled_wheel_before_runtime_split():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "FROM vllm_rs_build AS vllm_wheel" in dockerfile
    assert "python -m build --wheel --no-isolation" in dockerfile
    assert "FROM vllm_musa_deps AS vllm_runtime" in dockerfile
    assert "COPY --from=vllm_wheel /tmp/vllm-wheel/" in dockerfile
    assert "python -m pip install" in dockerfile
    assert "--mount=" not in dockerfile


def test_production_and_source_targets_are_explicitly_separate():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "FROM mooncake AS final" in dockerfile
    assert "FROM final AS vllm-openai" in dockerfile
    assert "FROM vllm_musa_installed AS test" in dockerfile
    assert dockerfile.index("FROM vllm_musa_installed AS test") < dockerfile.index(
        "FROM mooncake AS final"
    )
    test_stage = dockerfile.split("FROM vllm_musa_installed AS test", 1)[1].split(
        "FROM vllm_runtime AS mooncake", 1
    )[0]
    assert "ENV MTHREADS_VISIBLE_DEVICES=all" in test_stage
    assert dockerfile.rstrip().endswith('ENTRYPOINT ["vllm", "serve"]')


def test_runtime_workspace_copies_the_musa_examples_and_benchmarks():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "COPY examples/ /vllm-workspace/examples/" in dockerfile
    assert "COPY benchmarks/ /vllm-workspace/benchmarks/" in dockerfile
    assert "/third_party/vllm/examples" not in dockerfile.replace(
        "/third_party/vllm/examples/disaggregated/mooncake_connector/"
        "mooncake_connector_proxy.py",
        "",
    )
    assert "/third_party/vllm/benchmarks" not in dockerfile
    assert "/vllm-workspace/torch_lib_versions.txt" in dockerfile
    assert "/vllm-workspace/collect_env.py" in dockerfile
    assert "/vllm-workspace/vllm_musa/collect_env.py" in dockerfile
    assert "/vllm-workspace/third_party/vllm/vllm/collect_env.py" not in dockerfile


def test_runtime_image_ships_the_proxy_used_by_both_mooncake_examples():
    root = Path(__file__).parents[1]
    dockerfile = (root / "docker" / "musa.Dockerfile").read_text()
    runtime = dockerfile.split("FROM vllm_musa_deps AS vllm_runtime", 1)[1]
    runtime = runtime.split("FROM vllm_musa_installed AS test", 1)[0]
    relative_path = (
        "examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    assert (
        "COPY --from=vllm_musa_installed "
        f"/vllm-workspace/third_party/vllm/{relative_path} "
        f"/vllm-workspace/{relative_path}"
    ) in " ".join(runtime.replace("\\\n", "").split())
    serving_examples = root / "examples/disaggregated/disaggregated_serving"
    for script in ("disaggregated_serving.sh", "mooncake_store_pd_serving.sh"):
        assert (
            f'PROXY_RELATIVE_PATH="{relative_path}"'
            in (serving_examples / script).read_text()
        )


def test_collect_env_reports_musa_distribution_and_provider_inventory():
    source = (Path(__file__).parents[1] / "vllm_musa" / "collect_env.py").read_text()

    assert '"vllm-musa"' in source
    assert '"mate-mubin"' in source
    assert '"flash-attn-3"' in source
    assert '"flash-mla"' in source
    assert '"deep-gemm"' in source
    assert '"flashinfer-python"' in source
    assert '"deep-ep"' in source
    assert '"tilelang-musa"' in source
    assert '"mooncake-transfer-engine-musa"' in source
    assert "get_provider_imports" in source
    assert "get_native_extensions" in source
    assert "importlib import metadata" in source
    assert "from vllm.envs import environment_variables" not in source
    assert "vLLM Version" not in source
    assert "get_vllm_version" not in source
    assert "vLLM import package path" in source
    assert "Backend Environment" not in source
    assert "{env_vars}" in source


def test_packaging_metadata_keeps_musa_extra_and_lora_resolvers():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    setup = (Path(__file__).parents[1] / "setup.py").read_text()

    assert 'dynamic = ["dependencies", "optional-dependencies"]' in pyproject
    assert 'musa = {file = ["requirements/musa_private.txt"]}' in pyproject
    assert "lora_filesystem_resolver =" in pyproject
    assert "lora_hf_hub_resolver =" in pyproject
    assert '"vllm_runtime_transitive.txt"' in setup


def test_source_tree_uses_the_upstream_examples_and_benchmarks_layout():
    root = Path(__file__).parents[1]

    assert (root / "examples" / "generate").is_dir()
    assert (root / "examples" / "disaggregated" / "disaggregated_serving").is_dir()
    assert not (root / "example").exists()
    assert (root / "benchmarks" / "kernels" / "fp8_block_splitk").is_dir()
    assert (root / "benchmarks" / "kernels" / "fused_moe").is_dir()
    assert not (root / "benchmarks" / "fp8_block_splitk").exists()
    assert not (root / "benchmarks" / "fused_moe").exists()


def test_vllm_rs_uses_the_pinned_vllm_build_script_path():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "bash tools/build_rust.sh" in dockerfile
    assert "bash build_rust.sh" not in dockerfile
    assert "tools/install_protoc.sh" not in dockerfile
    assert "_rust_tool_parser.abi3.so" not in dockerfile


def test_runtime_target_does_not_copy_the_editable_source_checkout():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()
    runtime = dockerfile.split("FROM vllm_musa_deps AS vllm_runtime", 1)[1]
    runtime = runtime.split("FROM vllm_runtime AS mooncake", 1)[0]

    assert "COPY . /vllm-workspace" not in runtime


def test_runtime_dependency_verifier_receives_requirement_files_without_source():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert (
        "COPY requirements/ /tmp/vllm-runtime-requirements/requirements/" in dockerfile
    )
    assert "/tmp/vllm-runtime-requirements/requirements/musa_private.txt" in dockerfile
    assert "VLLM_MUSA_REQUIREMENTS_ROOT=/tmp/vllm-runtime-requirements" in dockerfile


def test_wheel_stage_collects_native_extensions_from_installed_packages():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    wheel_stage = dockerfile.split("FROM vllm_rs_build AS vllm_wheel", 1)[1]
    wheel_stage = wheel_stage.split("# Install the prebuilt MUSA wheel", 1)[0]
    assert 'test -n "$(find /vllm-workspace/third_party/vllm/vllm' in wheel_stage


def test_numpy_runtime_constraint_allows_the_supported_numpy_1_series():
    requirements = (
        Path(__file__).parents[1] / "requirements" / "common.txt"
    ).read_text()
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "numpy<2" in requirements
    assert "numpy==1.26.4" not in requirements
    assert "python -m pip install 'numpy<2'" not in dockerfile
