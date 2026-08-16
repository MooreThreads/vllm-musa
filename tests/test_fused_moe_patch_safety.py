from pathlib import Path

ROOT = Path(__file__).parents[1]
FAST_PATH_PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0087-MUSA-fused_moe_kernel-even_Ks-scalar-b_scale-fast-pa.patch"
)
SAFETY_PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0135-MUSA-bind-modular-MoE-plans-safely.patch"
)


def test_fused_moe_patch_keeps_masked_k_loads() -> None:
    text = SAFETY_PATCH.read_text()
    changed_source = "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )

    assert "and not current_platform.is_musa()" in text
    assert "tl.load(b_ptrs" not in changed_source
    assert "one-past MMU fault" in text


def test_fused_moe_patch_retains_scalar_scale_fast_path() -> None:
    text = FAST_PATH_PATCH.read_text()

    assert "if BLOCK_SIZE_N > group_n:" in text
    assert "a_scale[:, None] * b_scale" in text


def test_fused_moe_patch_uses_single_stage_for_musa_fp8_safety() -> None:
    text = SAFETY_PATCH.read_text()

    assert "+    if current_platform.is_musa() and use_fp8_w8a8:" in text
    assert '+        config["num_stages"] = 1' in text
    assert "Triton stages two and three can prefetch past the final" in text


def test_modular_runtime_plan_receipt_is_not_a_dispatch_receipt() -> None:
    binding = (
        ROOT
        / "vllm_musa"
        / "patches"
        / "series"
        / "0135-MUSA-bind-modular-MoE-plans-safely.patch"
    ).read_text()
    assert "MUSA fused-MoE plan binding receipt" not in binding
    assert "record_modular_fused_moe_runtime_plan_binding" in binding

    policy = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "fused_moe"
        / "dispatch_policy.py"
    ).read_text()
    assert "MUSA fused-MoE plan binding receipt" in policy
    assert "execution_backend=uncontrolled" in policy
