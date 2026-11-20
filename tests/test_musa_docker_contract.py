from pathlib import Path


def test_spawn_is_baked_into_the_shared_musa_runtime_layer():
    dockerfile = (Path(__file__).parents[1] / "docker" / "musa.Dockerfile").read_text()
    runtime_env = dockerfile.split("FROM devel AS", 1)[0]

    assert "VLLM_WORKER_MULTIPROC_METHOD=spawn" in runtime_env
    assert dockerfile.count("VLLM_WORKER_MULTIPROC_METHOD") == 1
    assert 'opencv-python-headless==${OPENCV_HEADLESS}' in dockerfile
    assert 'opencv-python-headless==4.10.0.84' not in dockerfile
