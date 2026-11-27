# SPDX-License-Identifier: Apache-2.0
"""Small TileLang compatibility helpers used by DeepSeek-V4 MUSA JIT paths."""

from __future__ import annotations

from typing import Any

_TILELANG_MUSA_OPT1_DEVICE_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-opt-level=1",
]

_TILELANG_MUSA_LS_DEVICE_COMPILE_FLAGS = [
    *_TILELANG_MUSA_OPT1_DEVICE_COMPILE_FLAGS,
    "-mllvm",
    "-mtgpu-load-store-opt=1",
    "-mllvm",
    "-mtgpu-fold-global-ldst=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "-mtgpu-store-cluster-mutation=1",
    "-mllvm",
    "-mtgpu-memory-sched-mutation=1",
]

_TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]

_TILELANG_MUSA_DSA_FULL_DEVICE_COMPILE_FLAGS = [
    *_TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS,
    "-mllvm",
    "-mtgpu-combine-instr-with-burst=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]


def _tilelang_musa_compile_profile_flags(
    default_profile: str | None = None,
) -> list[str] | None:
    # DeepSeek-V4 TileLang kernels use the validated DSA-full compilation
    # profile by default.  ``default_profile`` remains an internal API for
    # shape-specific helpers; no process environment is consulted.
    profile = (default_profile or "dsa_full").strip().lower()
    if profile in {"", "default", "none", "0"}:
        return None
    if profile == "opt1":
        return _TILELANG_MUSA_OPT1_DEVICE_COMPILE_FLAGS
    if profile == "ls":
        return _TILELANG_MUSA_LS_DEVICE_COMPILE_FLAGS
    if profile == "dsa":
        return _TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS
    if profile == "dsa_full":
        return _TILELANG_MUSA_DSA_FULL_DEVICE_COMPILE_FLAGS
    raise ValueError(
        f"Unsupported DeepSeek-V4 TileLang compile profile {profile!r}; "
        "expected one of default,opt1,ls,dsa,dsa_full"
    )


def _add_pass_config(
    pass_configs: dict[Any, Any],
    tilelang,
    key_name: str,
    value: Any,
) -> None:
    key = getattr(tilelang.PassConfigKey, key_name, None)
    if key is not None:
        pass_configs[key] = value


def _tilelang_musa_pass_configs(tilelang, *, compile_profile: str | None = None):
    """Return optional MUSA pass configs without requiring a specific TileLang build."""
    compile_flags = _tilelang_musa_compile_profile_flags(compile_profile)
    if compile_flags is None:
        return None

    pass_configs = {}
    _add_pass_config(pass_configs, tilelang, "TL_DISABLE_HOST_ASSERTS", True)
    if compile_flags is not None:
        _add_pass_config(
            pass_configs, tilelang, "TL_DEVICE_COMPILE_FLAGS", compile_flags
        )
    return pass_configs or None


def _tilelang_musa_burst_reduce_pass_configs(
    tilelang,
    *,
    compile_profile: str | None = None,
):
    pass_configs = {}
    _add_pass_config(pass_configs, tilelang, "TL_ENABLE_MUSA_BURST", True)
    _add_pass_config(pass_configs, tilelang, "TL_ENABLE_REDUCE_BURST", True)
    _add_pass_config(pass_configs, tilelang, "TL_DISABLE_HOST_ASSERTS", True)
    compile_flags = _tilelang_musa_compile_profile_flags(compile_profile)
    if compile_flags is not None:
        _add_pass_config(
            pass_configs, tilelang, "TL_DEVICE_COMPILE_FLAGS", compile_flags
        )
    return pass_configs or None


def _tilelang_musa_aggressive_pass_configs(
    tilelang,
    *,
    disable_index_promotion: bool = True,
    compile_profile: str | None = None,
):
    pass_configs = (
        _tilelang_musa_burst_reduce_pass_configs(
            tilelang,
            compile_profile=compile_profile,
        )
        or {}
    )
    _add_pass_config(pass_configs, tilelang, "TL_DISABLE_THREAD_STORAGE_SYNC", True)
    _add_pass_config(pass_configs, tilelang, "TL_DISABLE_SAFE_MEMORY_ACCESS", True)
    _add_pass_config(pass_configs, tilelang, "TL_ENABLE_LOWER_LDGSTG", True)
    _add_pass_config(pass_configs, tilelang, "TL_ENABLE_LOWER_LDGSTG_PREDICATED", True)
    if disable_index_promotion:
        _add_pass_config(
            pass_configs, tilelang, "TL_DISABLE_INDEX_TYPE_PROMOTION", True
        )
    _add_pass_config(pass_configs, tilelang, "TL_DISABLE_HOST_ASSERTS", True)
    return pass_configs or None


def _tilelang_musa_dsa_pass_configs(
    tilelang,
    *,
    full: bool = False,
    disable_index_promotion: bool = True,
):
    pass_configs = (
        _tilelang_musa_aggressive_pass_configs(
            tilelang,
            disable_index_promotion=disable_index_promotion,
        )
        or {}
    )
    flags = (
        _TILELANG_MUSA_DSA_FULL_DEVICE_COMPILE_FLAGS
        if full
        else _TILELANG_MUSA_DSA_DEVICE_COMPILE_FLAGS
    )
    _add_pass_config(pass_configs, tilelang, "TL_DEVICE_COMPILE_FLAGS", flags)
    return pass_configs or None


def _tilelang_jit(
    tilelang,
    name: str,
    pass_configs=None,
    *,
    target: str | None = "musa",
):
    if pass_configs is None:
        pass_configs = _tilelang_musa_pass_configs(tilelang)
    base_kwargs = {}
    if target is not None:
        base_kwargs["target"] = target
    if name:
        base_kwargs["name"] = name
    if pass_configs is not None:
        base_kwargs["pass_configs"] = pass_configs
    candidate_keys = (
        ("target", "name", "pass_configs"),
        ("target", "pass_configs"),
        ("target", "name"),
        ("target",),
        ("name", "pass_configs"),
        ("pass_configs",),
        ("name",),
        (),
    )
    candidates = [
        {key: base_kwargs[key] for key in keys if key in base_kwargs}
        for keys in candidate_keys
    ]

    seen = set()
    last_error: TypeError | None = None
    for kwargs in candidates:
        key = tuple(sorted(kwargs))
        if key in seen:
            continue
        seen.add(key)
        try:
            return tilelang.jit(**kwargs)
        except TypeError as exc:
            message = str(exc)
            if not any(
                text in message
                for text in ("name", "pass_configs", "target", "unexpected")
            ):
                raise
            last_error = exc
    if last_error is not None:
        raise last_error
    return tilelang.jit()


def _patch_tilelang_musa_wrapper() -> bool:
    """Patch TileLang's MUSA wrapper for host IR without ``T.call_packed``.

    The TileLang build in the current S5000 container can lower simple MUSA
    kernels, but its wrapper sorts device functions by searching the generated
    host IR for ``T.call_packed("<kernel>")``. Some lowered kernels do not
    contain that string, so compilation fails with ``ValueError: substring not
    found`` before the generated MUSA source is wrapped. Falling back to the
    device-module order preserves the upstream behavior for the single-kernel
    JIT helpers used here and keeps fixed TileLang builds untouched.
    """

    try:
        from tilelang import tvm
        from tilelang.jit.adapter.musa_wrapper import TLMUSASourceWrapper
        from tilelang.jit.adapter.utils import get_annotated_mod
    except Exception:
        return False

    if getattr(TLMUSASourceWrapper, "_vllm_musa_no_call_packed_patch", False):
        return True

    original_parse = TLMUSASourceWrapper.parse_source_information

    def patched_parse(self: Any):
        try:
            return original_parse(self)
        except ValueError as exc:
            if "substring not found" not in str(exc):
                raise

            if self.device_mod is None or self.host_mod is None:
                with tvm.transform.PassContext(opt_level=3, config=self.pass_configs):
                    device_mod, host_mod = get_annotated_mod(self.mod, self.target)
                self.device_mod = device_mod
                self.host_mod = host_mod

            assert len(self.device_mod.functions) >= 1, (
                "Device module should have at least one function."
            )
            assert len(self.host_mod.functions) == 1, (
                "Only support one function in host module."
            )

            block_info_map = {}
            grid_info_map = {}
            dynamic_smem_buf_map = {}
            use_cooperative_groups_map = {}
            function_names = []

            for g_var, func in self.device_mod.functions.items():
                block_info = [1, 1, 1]
                grid_info = [1, 1, 1]
                function_name = g_var.name_hint
                attrs = func.attrs
                dynamic_smem_buf = None
                use_cooperative_groups = False

                if "use_cooperative_groups" in attrs:
                    use_cooperative_groups = attrs["use_cooperative_groups"]
                if "dyn_shared_memory_buf" in attrs:
                    dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
                if "thread_extent" in attrs:
                    for tag, extent in attrs["thread_extent"].items():
                        if "threadIdx" in tag:
                            block_info["xyz".index(tag[-1])] = extent
                        elif "blockIdx" in tag:
                            grid_info["xyz".index(tag[-1])] = extent

                block_info_map[function_name] = block_info
                grid_info_map[function_name] = grid_info
                dynamic_smem_buf_map[function_name] = dynamic_smem_buf
                use_cooperative_groups_map[function_name] = use_cooperative_groups
                function_names.append(function_name)

            self.block_info = block_info_map
            self.grid_info = grid_info_map
            self.dynamic_smem_buf = dynamic_smem_buf_map
            self.use_cooperative_groups = use_cooperative_groups_map

            for _, func in self.host_mod.functions.items():
                if "tma_descriptor_args" in func.attrs:
                    self.tma_descriptor_args = func.attrs["tma_descriptor_args"]
                if "l2_persistent_map" in func.attrs:
                    for function_name in function_names:
                        self.l2_persistent_map[function_name] = func.attrs[
                            "l2_persistent_map"
                        ]

            self.function_names = function_names

    patched_parse._vllm_musa_original = original_parse
    TLMUSASourceWrapper.parse_source_information = patched_parse
    TLMUSASourceWrapper._vllm_musa_no_call_packed_patch = True
    return True
