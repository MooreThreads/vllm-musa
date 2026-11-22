from pathlib import Path


def test_spawn_is_baked_into_the_shared_musa_runtime_layer():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()
    runtime_env = dockerfile.split("FROM devel AS", 1)[0]

    assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" in runtime_env
    assert dockerfile.count("VLLM_WORKER_MULTIPROC_METHOD") == 1


def test_omni_image_restores_the_base_pinned_headless_opencv():
    dockerfile = (Path(__file__).parents[1] / "docker" / "omni.Dockerfile").read_text()

    assert "opencv-python-headless==${OPENCV_HEADLESS}" in dockerfile
    assert "opencv-python-headless==4.10.0.84" not in dockerfile
    assert "opencv-python==4.10.0.84" in dockerfile


def test_production_image_installs_a_bundled_wheel_before_runtime_split():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "FROM vllm_rs_build AS vllm_wheel" in dockerfile
    assert "python -m build --wheel --no-isolation" in dockerfile
    assert "FROM vllm_musa_deps AS vllm_runtime" in dockerfile
    assert "COPY --from=vllm_wheel /tmp/vllm-wheel/" in dockerfile
    assert "python -m pip install" in dockerfile


def test_production_and_source_targets_are_explicitly_separate():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "FROM mooncake AS final" in dockerfile
    assert "FROM final AS vllm-openai" in dockerfile
    assert "FROM vllm_musa_installed AS test" in dockerfile
    assert dockerfile.index("FROM vllm_musa_installed AS test") > dockerfile.index(
        "FROM final AS vllm-openai"
    )


def test_runtime_target_does_not_copy_the_editable_source_checkout():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()
    runtime = dockerfile.split("FROM vllm_musa_deps AS vllm_runtime", 1)[1]
    runtime = runtime.split("FROM vllm_musa_installed AS test", 1)[0]

    assert "COPY . /vllm-workspace" not in runtime
    assert "third_party/vllm/vllm" not in runtime


def test_runtime_dependency_verifier_receives_requirement_files_without_source():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "/tmp/vllm-runtime-requirements/common.txt" in dockerfile
    assert "/tmp/vllm-runtime-requirements/musa_private.txt" in dockerfile
    assert "VLLM_MUSA_REQUIREMENTS_ROOT=/tmp/vllm-runtime-requirements" in dockerfile


def test_wheel_stage_collects_native_extensions_from_installed_packages():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    wheel_stage = dockerfile.split("FROM vllm_rs_build AS vllm_wheel", 1)[1]
    wheel_stage = wheel_stage.split("# Install the prebuilt MUSA wheel", 1)[0]
    assert (
        "find /usr/local/lib/python3.12/site-packages" in wheel_stage
        and "/vllm-workspace/third_party/vllm" in wheel_stage
    )
    assert "-path \"*/${package}/*.so\"" in wheel_stage


def test_numpy_runtime_constraint_allows_the_supported_numpy_1_series():
    requirements = (Path(__file__).parents[1] / "requirements" / "common.txt").read_text()
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()

    assert "numpy<2" in requirements
    assert "numpy==1.26.4" not in requirements
    assert "python -m pip install 'numpy<2'" in dockerfile
