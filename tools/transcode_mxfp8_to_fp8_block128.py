#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Transcode a serialized MXFP8 checkpoint to FP8 block-128.

MXFP8 stores E4M3 weights with one E8M0 scale byte per 32 values along the
input dimension.  The vLLM-MUSA block-FP8 path expects E4M3 weights with one
float32 multiplicative scale per 128x128 tile.  This tool converts between the
two layouts one safetensors shard at a time and never mutates the source model.

The output is written atomically, records resumable shard progress, rewrites
``config.json`` for ``quant_method=fp8``, and preserves non-quantized modules.
MXFP8 linears whose final two dimensions are not divisible by 128 are
dequantized to BF16 and added to ``ignored_layers``. The fallback propagates to
all logical shards of a fused linear, so Hy4 dequantizes both ``q_a_proj`` and
the 576-output ``kv_a_proj_with_mqa`` for 78 backbone layers plus one MTP
layer. Run ``self-test`` before the full conversion to gate the requantization
error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import torchada  # noqa: F401  # Must precede torch on a MUSA runtime.
except ImportError:
    pass

import torch

LOGGER = logging.getLogger("mxfp8_transcode")

E4M3_MAX = 448.0
MXFP8_GROUP_SIZE = 32
TARGET_BLOCK_SIZE = 128
PROGRESS_FILENAME = ".mxfp8-to-fp8-block128-progress.json"
PROGRESS_SCHEMA = 3
SPACE_SAFETY_FACTOR = 1.05
SPACE_HEADROOM_BYTES = 1 << 30


class TranscodeError(RuntimeError):
    """A checkpoint or conversion contract is invalid."""


def e8m0_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode raw E8M0 bytes to float32 multiplicative scales."""
    if scale_u8.dtype != torch.uint8:
        raise TranscodeError(f"MXFP8 scales must be uint8, got {scale_u8.dtype}")
    try:
        return scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    except (AttributeError, RuntimeError):
        return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def dequant_mxfp8(weight: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize an E4M3 weight with per-32 E8M0 scales to fp32.

    Leading dimensions (the packed-expert dimension in particular) are kept
    intact; the microscale grouping is always along the final input dimension.
    """
    _validate_mxfp8_pair("<tensor>", weight, scale_u8)
    scale = e8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(MXFP8_GROUP_SIZE, dim=scale.ndim - 1)
    return weight.to(torch.float32) * scale[..., : weight.shape[-1]]


def requant_fp8_block128(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D fp32 weight into E4M3 plus 128x128 float32 scales."""
    if weight.ndim != 2:
        raise TranscodeError(f"block-FP8 requires a 2D weight, got {weight.ndim}D")
    out_features, in_features = weight.shape
    out_blocks = (out_features + TARGET_BLOCK_SIZE - 1) // TARGET_BLOCK_SIZE
    in_blocks = (in_features + TARGET_BLOCK_SIZE - 1) // TARGET_BLOCK_SIZE

    padded = torch.zeros(
        (out_blocks * TARGET_BLOCK_SIZE, in_blocks * TARGET_BLOCK_SIZE),
        dtype=torch.float32,
        device=weight.device,
    )
    padded[:out_features, :in_features] = weight
    blocked = padded.reshape(
        out_blocks,
        TARGET_BLOCK_SIZE,
        in_blocks,
        TARGET_BLOCK_SIZE,
    )
    amax = blocked.abs().amax(dim=(1, 3))
    scale_inv = (amax / E4M3_MAX).clamp(min=1e-12)
    quantized = (blocked / scale_inv[:, None, :, None]).clamp(-E4M3_MAX, E4M3_MAX)
    weight_fp8 = quantized.to(torch.float8_e4m3fn).reshape(padded.shape)
    weight_fp8 = weight_fp8[:out_features, :in_features].contiguous()
    return weight_fp8, scale_inv.contiguous()


def dequant_fp8_block128(
    weight_fp8: torch.Tensor, scale_inv: torch.Tensor
) -> torch.Tensor:
    """Dequantize the target block-FP8 representation to fp32."""
    if weight_fp8.ndim != 2 or scale_inv.ndim != 2:
        raise TranscodeError("block-FP8 weight and scale must both be 2D")
    out_features, in_features = weight_fp8.shape
    scale = scale_inv.repeat_interleave(TARGET_BLOCK_SIZE, dim=0)
    scale = scale.repeat_interleave(TARGET_BLOCK_SIZE, dim=1)
    return weight_fp8.to(torch.float32) * scale[:out_features, :in_features]


def transcode_weight(
    weight: torch.Tensor, scale_u8: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target weight, target scale, and the fp32 source reference."""
    reference = dequant_mxfp8(weight, scale_u8)
    weight_fp8, scale_inv = requant_fp8_block128(reference)
    return weight_fp8, scale_inv, reference


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscodeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TranscodeError(f"expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(src: Path) -> tuple[dict[str, Any], dict[str, str], Path]:
    index_path = src / "model.safetensors.index.json"
    config_path = src / "config.json"
    index = _read_json(index_path)
    _read_json(config_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise TranscodeError(f"missing non-empty weight_map in {index_path}")
    if not all(
        isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
    ):
        raise TranscodeError(f"invalid weight_map entries in {index_path}")
    return index, dict(weight_map), index_path


def _quantized_pairs(
    weight_map: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """Return source weight/scale, target scale, and shard for MXFP8 pairs.

    ModelOpt checkpoints use ``foo.weight`` + ``foo.weight_scale`` for linear
    layers and ``experts.gate_up_proj`` + ``experts.gate_up_proj_scale`` for
    packed MoE tensors.  Older microscaling exports may already use a
    ``*_scale_inv`` suffix.  The block-FP8 target consistently uses
    ``<weight-key>_scale_inv``.
    """
    pairs: list[tuple[str, str, str, str]] = []
    for scale_key, scale_shard in weight_map.items():
        if scale_key.endswith("_scale_inv"):
            weight_key = scale_key.removesuffix("_scale_inv")
        elif scale_key.endswith("_scale"):
            weight_key = scale_key.removesuffix("_scale")
        else:
            continue
        shard = weight_map.get(weight_key)
        if shard is None:
            continue
        if scale_shard != shard:
            raise TranscodeError(
                f"{weight_key} and {scale_key} are in different shards: "
                f"{shard!r} vs {scale_shard!r}"
            )
        target_scale_key = f"{weight_key}_scale_inv"
        pairs.append((weight_key, scale_key, target_scale_key, shard))
    return sorted(pairs)


_FUSED_LINEAR_MODULE_GROUPS = (
    ("q_a_proj", "kv_a_proj_with_mqa"),
    ("gate_proj", "up_proj"),
)


def _plan_bf16_fallback_weights(
    src: Path,
    pairs: list[tuple[str, str, str, str]],
    exclusions: list[str],
) -> set[str]:
    """Plan BF16 fallbacks and propagate them across fused linear shards.

    vLLM rejects a fused linear when only some logical shards are quantized.
    Hy4 fuses ``q_a_proj`` (block-compatible) with
    ``kv_a_proj_with_mqa`` (576 output rows, not block-compatible), so both
    members must use the same BF16 fallback in the first correctness path.
    """
    from safetensors import safe_open

    pair_by_weight = {
        weight: (scale, target_scale, shard)
        for weight, scale, target_scale, shard in pairs
    }
    weights_by_shard: dict[str, list[str]] = {}
    for weight, _, _, shard in pairs:
        weights_by_shard.setdefault(shard, []).append(weight)

    fallback: set[str] = set()
    for shard, weights in weights_by_shard.items():
        with safe_open(src / shard, framework="pt", device="cpu") as handle:
            for weight in weights:
                shape = tuple(handle.get_slice(weight).get_shape())
                if len(shape) < 2:
                    raise TranscodeError(f"{weight}: expected a 2D-or-higher weight")
                if (
                    shape[-2] % TARGET_BLOCK_SIZE != 0
                    or shape[-1] % TARGET_BLOCK_SIZE != 0
                    or _is_excluded(weight, exclusions)
                ):
                    fallback.add(weight)

    changed = True
    while changed:
        changed = False
        for weight in tuple(pair_by_weight):
            module = _module_name(weight)
            for group in _FUSED_LINEAR_MODULE_GROUPS:
                matched = next(
                    (member for member in group if module.endswith(member)), None
                )
                if matched is None:
                    continue
                prefix = module[: -len(matched)]
                siblings = {
                    f"{prefix}{member}.weight"
                    for member in group
                    if f"{prefix}{member}.weight" in pair_by_weight
                }
                if siblings & fallback and not siblings <= fallback:
                    fallback.update(siblings)
                    changed = True
    return fallback


def _source_exclusions(config: dict[str, Any]) -> list[str]:
    quant_config = config.get("quantization_config", {})
    if not isinstance(quant_config, dict):
        raise TranscodeError("config.json quantization_config must be an object")
    nested = quant_config.get("quantization", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise TranscodeError("quantization_config.quantization must be an object")
    values = (
        nested.get("exclude_modules")
        or quant_config.get("exclude_modules")
        or quant_config.get("ignored_layers")
        or []
    )
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise TranscodeError("MXFP8 excluded modules must be a list of strings")
    return list(dict.fromkeys(values))


def _is_excluded(weight_key: str, exclusions: list[str]) -> bool:
    module = weight_key.removesuffix(".weight")
    return any(
        module == prefix or module.startswith(f"{prefix}.") for prefix in exclusions
    )


def _validate_mxfp8_pair(
    weight_key: str, weight: torch.Tensor, scale_u8: torch.Tensor
) -> None:
    if weight.dtype != torch.float8_e4m3fn:
        raise TranscodeError(
            f"{weight_key}: expected float8_e4m3fn weight, got {weight.dtype}"
        )
    if scale_u8.dtype != torch.uint8:
        raise TranscodeError(
            f"{weight_key}_scale_inv: expected uint8 E8M0, got {scale_u8.dtype}"
        )
    if weight.ndim < 2 or scale_u8.ndim != weight.ndim:
        raise TranscodeError(
            f"{weight_key}: expected matching 2D-or-higher weight/scale, got "
            f"{weight.ndim}D/{scale_u8.ndim}D"
        )
    expected = (
        *weight.shape[:-1],
        (weight.shape[-1] + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE,
    )
    if tuple(scale_u8.shape) != expected:
        raise TranscodeError(
            f"{weight_key}: expected per-32 scale shape {expected}, "
            f"got {tuple(scale_u8.shape)}"
        )


def _move_tensor(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if device == "cpu":
        return tensor
    return tensor.to(torch.device(device))


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _module_name(weight_key: str) -> str:
    return weight_key.removesuffix(".weight")


def _requires_bf16_fallback(weight: torch.Tensor) -> bool:
    """Return whether a serialized weight cannot use global 128x128 blocks."""
    return (
        weight.shape[-2] % TARGET_BLOCK_SIZE != 0
        or weight.shape[-1] % TARGET_BLOCK_SIZE != 0
    )


def _transcode_tensor(
    weight_key: str,
    weight: torch.Tensor,
    scale_u8: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcode a linear weight or a packed expert tensor with bounded memory."""
    _validate_mxfp8_pair(weight_key, weight, scale_u8)
    out_features, in_features = weight.shape[-2:]
    out_blocks = (out_features + TARGET_BLOCK_SIZE - 1) // TARGET_BLOCK_SIZE
    in_blocks = (in_features + TARGET_BLOCK_SIZE - 1) // TARGET_BLOCK_SIZE
    leading_shape = tuple(weight.shape[:-2])

    output_weight = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device="cpu")
    output_scale = torch.empty(
        (*leading_shape, out_blocks, in_blocks), dtype=torch.float32, device="cpu"
    )
    flat_weight = weight.reshape(-1, out_features, in_features)
    flat_source_scale = scale_u8.reshape(-1, out_features, scale_u8.shape[-1])
    flat_output_weight = output_weight.reshape(-1, out_features, in_features)
    flat_output_scale = output_scale.reshape(-1, out_blocks, in_blocks)

    # Packed MoE shards can contain hundreds of experts.  Convert one leading
    # slice at a time so the fp32 reference and padded tile never scale with E.
    for index in range(flat_weight.shape[0]):
        weight_slice = _move_tensor(flat_weight[index], device)
        scale_slice = _move_tensor(flat_source_scale[index], device)
        weight_fp8, scale_inv, _ = transcode_weight(weight_slice, scale_slice)
        flat_output_weight[index].copy_(weight_fp8.cpu())
        flat_output_scale[index].copy_(scale_inv.cpu())
    return output_weight, output_scale


def _dequantize_tensor_to_bf16(
    weight_key: str,
    weight: torch.Tensor,
    scale_u8: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Dequantize one linear or packed-expert MXFP8 tensor with bounded memory."""
    _validate_mxfp8_pair(weight_key, weight, scale_u8)
    out_features, in_features = weight.shape[-2:]
    leading_shape = tuple(weight.shape[:-2])
    output = torch.empty(weight.shape, dtype=torch.bfloat16, device="cpu")
    flat_weight = weight.reshape(-1, out_features, in_features)
    flat_scale = scale_u8.reshape(-1, out_features, scale_u8.shape[-1])
    flat_output = output.reshape(-1, out_features, in_features)
    for index in range(flat_weight.shape[0]):
        weight_slice = _move_tensor(flat_weight[index], device)
        scale_slice = _move_tensor(flat_scale[index], device)
        flat_output[index].copy_(
            dequant_mxfp8(weight_slice, scale_slice).to(torch.bfloat16).cpu()
        )
    return output.reshape(*leading_shape, out_features, in_features)


def inspect_checkpoint(src: Path, dst: Path | None = None) -> dict[str, Any]:
    """Validate checkpoint metadata and return a compact inventory."""
    _, weight_map, index_path = _checkpoint_metadata(src)
    config = _read_json(src / "config.json")
    quant_config = config.get("quantization_config", {})
    nested = (
        quant_config.get("quantization", {}) if isinstance(quant_config, dict) else {}
    )
    quant_algo = nested.get("quant_algo") if isinstance(nested, dict) else None
    pairs = _quantized_pairs(weight_map)
    shards = sorted(set(weight_map.values()))
    missing = [shard for shard in shards if not (src / shard).is_file()]
    if missing:
        raise TranscodeError(f"missing {len(missing)} checkpoint shards: {missing[:3]}")
    source_bytes = sum((src / shard).stat().st_size for shard in shards)
    inventory: dict[str, Any] = {
        "source": str(src.resolve()),
        "architecture": config.get("architectures"),
        "source_quant_algo": quant_algo or quant_config.get("quant_method"),
        "index_sha256": _sha256(index_path),
        "shards": len(shards),
        "quantized_weight_pairs": len(pairs),
        "excluded_modules": len(_source_exclusions(config)),
        "source_safetensors_bytes": source_bytes,
    }
    if dst is not None:
        probe = dst if dst.exists() else dst.parent
        probe.mkdir(parents=True, exist_ok=True)
        inventory["destination"] = str(dst.resolve())
        inventory["destination_free_bytes"] = shutil.disk_usage(probe).free
        inventory["estimated_required_free_bytes"] = (
            int(source_bytes * SPACE_SAFETY_FACTOR) + SPACE_HEADROOM_BYTES
        )
    return inventory


def self_test(src: Path, device: str, max_rel_l1: float) -> dict[str, Any]:
    """Transcode the first serialized MXFP8 tensor and measure round-trip error."""
    from safetensors import safe_open

    _, weight_map, _ = _checkpoint_metadata(src)
    pairs = _quantized_pairs(weight_map)
    if not pairs:
        raise TranscodeError("no MXFP8 weight/scale pair found")
    weight_key, scale_key, _, shard = pairs[0]
    with safe_open(src / shard, framework="pt", device="cpu") as handle:
        weight = _move_tensor(handle.get_tensor(weight_key), device)
        scale = _move_tensor(handle.get_tensor(scale_key), device)
    _validate_mxfp8_pair(weight_key, weight, scale)
    # A packed MoE tensor may be 3D; sample one expert for the scalar error
    # gate while still validating its complete pair shape above.
    sample_weight = weight.reshape(-1, *weight.shape[-2:])[0]
    sample_scale = scale.reshape(-1, *scale.shape[-2:])[0]
    weight_fp8, scale_inv = _transcode_tensor(
        weight_key, sample_weight, sample_scale, device
    )
    reference = dequant_mxfp8(sample_weight, sample_scale)
    restored = dequant_fp8_block128(weight_fp8, scale_inv)
    denominator = reference.abs().mean().clamp(min=1e-9)
    rel_l1 = ((reference - restored).abs().mean() / denominator).item()
    rel_max = (
        (reference - restored).abs().max() / reference.abs().max().clamp(min=1e-9)
    ).item()
    result = {
        "weight": weight_key,
        "weight_shape": list(sample_weight.shape),
        "source_scale_shape": list(scale.shape),
        "target_scale_shape": list(scale_inv.shape),
        "rel_l1": rel_l1,
        "rel_max": rel_max,
        "threshold": max_rel_l1,
        "passed": rel_l1 < max_rel_l1,
    }
    if not result["passed"]:
        raise TranscodeError(
            f"self-test failed: rel_L1={rel_l1:.6f} >= {max_rel_l1:.6f}"
        )
    return result


def _initial_progress(src: Path, index_sha256: str) -> dict[str, Any]:
    return {
        "schema": PROGRESS_SCHEMA,
        "algorithm": (
            "mxfp8-e4m3-e8m0-per32-to-fp8-e4m3-block128-"
            "with-fused-linear-bf16-fallback-v3"
        ),
        "source": str(src.resolve()),
        "source_index_sha256": index_sha256,
        "completed_shards": {},
        "status": "running",
    }


def _load_progress(
    path: Path, src: Path, index_sha256: str, resume: bool
) -> dict[str, Any]:
    if not path.exists():
        return _initial_progress(src, index_sha256)
    if not resume:
        raise TranscodeError(f"destination contains {path.name}; pass --resume")
    progress = _read_json(path)
    if progress.get("schema") != PROGRESS_SCHEMA:
        raise TranscodeError(f"unsupported progress schema in {path}")
    if progress.get("source_index_sha256") != index_sha256:
        raise TranscodeError("source index changed since the partial conversion")
    return progress


def _validate_resumed_shard(
    src_shard: Path,
    dst_shard: Path,
    expected_keys: list[str],
    record: dict[str, Any],
) -> bool:
    from safetensors import safe_open

    if not dst_shard.is_file():
        return False
    source_stat = src_shard.stat()
    if record.get("source_size") != source_stat.st_size:
        return False
    if record.get("source_mtime_ns") != source_stat.st_mtime_ns:
        return False
    if record.get("output_size") != dst_shard.stat().st_size:
        return False
    try:
        with safe_open(dst_shard, framework="pt", device="cpu") as handle:
            return sorted(handle.keys()) == sorted(expected_keys)
    except Exception:
        return False


def _ensure_destination(dst: Path, resume: bool) -> None:
    if dst.exists() and not dst.is_dir():
        raise TranscodeError(f"destination is not a directory: {dst}")
    if dst.exists() and any(dst.iterdir()) and not resume:
        raise TranscodeError(f"destination is not empty: {dst}; pass --resume")
    dst.mkdir(parents=True, exist_ok=True)


def _space_preflight(
    src: Path,
    dst: Path,
    shards: list[str],
    completed: dict[str, Any],
) -> None:
    remaining_source_bytes = sum(
        (src / shard).stat().st_size for shard in shards if shard not in completed
    )
    required = int(remaining_source_bytes * SPACE_SAFETY_FACTOR) + SPACE_HEADROOM_BYTES
    free = shutil.disk_usage(dst).free
    if free < required:
        raise TranscodeError(
            "insufficient destination space: "
            f"free={free} required~={required} remaining_source={remaining_source_bytes}"
        )


def _transcode_shard(
    src_shard: Path,
    dst_shard: Path,
    keys: list[str],
    weight_to_pair: dict[str, tuple[str, str]],
    bf16_fallback_weights: set[str],
    exclusions: list[str],
    device: str,
) -> tuple[int, int, int, list[str], list[str]]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    output: dict[str, torch.Tensor] = {}
    quantized = 0
    dequantized_modules: list[str] = []
    source_scale_keys_to_skip = {
        source_scale for source_scale, _ in weight_to_pair.values()
    }
    with safe_open(src_shard, framework="pt", device="cpu") as handle:
        present = set(handle.keys())
        for key in keys:
            if key in source_scale_keys_to_skip:
                continue
            pair = weight_to_pair.get(key)
            tensor = handle.get_tensor(key)
            if pair is not None:
                scale_key, target_scale_key = pair
            else:
                scale_key = target_scale_key = ""
            if pair is not None and scale_key in present:
                scale = handle.get_tensor(scale_key)
                if (
                    key in bf16_fallback_weights
                    or _is_excluded(key, exclusions)
                    or _requires_bf16_fallback(tensor)
                ):
                    output[key] = _dequantize_tensor_to_bf16(key, tensor, scale, device)
                    dequantized_modules.append(_module_name(key))
                    continue
                weight_fp8, scale_inv = _transcode_tensor(key, tensor, scale, device)
                output[key] = weight_fp8
                output[target_scale_key] = scale_inv
                quantized += 1
            else:
                output[key] = tensor.cpu()

    temporary = dst_shard.with_name(f".{dst_shard.name}.partial-{os.getpid()}")
    try:
        save_file(output, temporary, metadata={"format": "pt"})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, dst_shard)
    finally:
        temporary.unlink(missing_ok=True)
    tensor_bytes = sum(_tensor_bytes(tensor) for tensor in output.values())
    return (
        len(output),
        quantized,
        tensor_bytes,
        sorted(output),
        sorted(dequantized_modules),
    )


def _copy_auxiliary_files(src: Path, dst: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        PROGRESS_FILENAME,
    }
    for source in src.iterdir():
        if not source.is_file() or source.name in excluded:
            continue
        if source.suffix == ".safetensors":
            continue
        destination = dst / source.name
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def run_transcode(
    src: Path,
    dst: Path,
    device: str,
    resume: bool,
    skip_space_check: bool,
    max_rel_l1: float,
) -> dict[str, Any]:
    """Run the resumable shard-by-shard checkpoint conversion."""
    inventory = inspect_checkpoint(src, dst)
    self_test_result = self_test(src, device, max_rel_l1)
    _ensure_destination(dst, resume)

    index, weight_map, index_path = _checkpoint_metadata(src)
    config = _read_json(src / "config.json")
    exclusions = _source_exclusions(config)
    pairs = _quantized_pairs(weight_map)
    weight_to_pair = {
        weight: (source_scale, target_scale)
        for weight, source_scale, target_scale, _ in pairs
    }
    bf16_fallback_weights = _plan_bf16_fallback_weights(src, pairs, exclusions)
    planned_bf16_modules = sorted(
        _module_name(weight) for weight in bf16_fallback_weights
    )
    shards = sorted(set(weight_map.values()))
    progress_path = dst / PROGRESS_FILENAME
    progress = _load_progress(progress_path, src, _sha256(index_path), resume)
    recorded_plan = progress.get("planned_bf16_modules")
    if recorded_plan is not None and recorded_plan != planned_bf16_modules:
        raise TranscodeError("BF16 fallback plan changed since the partial conversion")
    progress["planned_bf16_modules"] = planned_bf16_modules
    completed = progress.setdefault("completed_shards", {})
    if not isinstance(completed, dict):
        raise TranscodeError(f"invalid completed_shards in {progress_path}")
    if not skip_space_check:
        _space_preflight(src, dst, shards, completed)

    total_quantized = 0
    total_dequantized = 0
    total_tensor_bytes = 0
    auto_dequantized_modules: set[str] = set()
    for position, shard in enumerate(shards, start=1):
        keys = sorted(
            key for key, mapped_shard in weight_map.items() if mapped_shard == shard
        )
        src_shard = src / shard
        dst_shard = dst / shard
        record = completed.get(shard)
        recorded_keys = record.get("output_keys") if isinstance(record, dict) else None
        if (
            isinstance(record, dict)
            and isinstance(recorded_keys, list)
            and all(isinstance(key, str) for key in recorded_keys)
            and _validate_resumed_shard(src_shard, dst_shard, recorded_keys, record)
        ):
            LOGGER.info("[%d/%d] resume %s", position, len(shards), shard)
            total_quantized += int(record.get("quantized_weights", 0))
            dequantized_modules = record.get("dequantized_modules", [])
            if not isinstance(dequantized_modules, list) or not all(
                isinstance(module, str) for module in dequantized_modules
            ):
                raise TranscodeError(f"invalid dequantized_modules for {shard}")
            auto_dequantized_modules.update(dequantized_modules)
            total_dequantized += len(dequantized_modules)
            total_tensor_bytes += int(record.get("tensor_bytes", 0))
            continue
        if dst_shard.exists():
            raise TranscodeError(
                f"existing shard is not a valid resumable output: {dst_shard}"
            )

        (
            tensor_count,
            quantized_count,
            tensor_bytes,
            output_keys,
            dequantized_modules,
        ) = _transcode_shard(
            src_shard,
            dst_shard,
            keys,
            weight_to_pair,
            bf16_fallback_weights,
            exclusions,
            device,
        )
        source_stat = src_shard.stat()
        completed[shard] = {
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "output_size": dst_shard.stat().st_size,
            "tensor_count": tensor_count,
            "quantized_weights": quantized_count,
            "dequantized_modules": dequantized_modules,
            "output_keys": output_keys,
            "tensor_bytes": tensor_bytes,
        }
        _atomic_write_json(progress_path, progress)
        total_quantized += quantized_count
        total_dequantized += len(dequantized_modules)
        auto_dequantized_modules.update(dequantized_modules)
        total_tensor_bytes += tensor_bytes
        LOGGER.info(
            "[%d/%d] wrote %s tensors=%d quantized=%d bf16_fallback=%d",
            position,
            len(shards),
            shard,
            tensor_count,
            quantized_count,
            len(dequantized_modules),
        )

    target_weight_map: dict[str, str] = {}
    for shard in shards:
        record = completed.get(shard)
        if not isinstance(record, dict):
            raise TranscodeError(f"missing completed progress for {shard}")
        output_keys = record.get("output_keys")
        if not isinstance(output_keys, list) or not all(
            isinstance(key, str) for key in output_keys
        ):
            raise TranscodeError(f"missing output_keys progress for {shard}")
        for key in output_keys:
            if key in target_weight_map:
                raise TranscodeError(f"duplicate target tensor key: {key}")
            target_weight_map[key] = shard
    target_index = {
        "metadata": {**index.get("metadata", {}), "total_size": total_tensor_bytes},
        "weight_map": target_weight_map,
    }
    target_config = dict(config)
    target_quant_config: dict[str, Any] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [TARGET_BLOCK_SIZE, TARGET_BLOCK_SIZE],
        "fmt": "e4m3",
    }
    target_exclusions = list(
        dict.fromkeys([*exclusions, *sorted(auto_dequantized_modules)])
    )
    if target_exclusions:
        target_quant_config["ignored_layers"] = target_exclusions
    target_config["quantization_config"] = target_quant_config

    _atomic_write_json(dst / "model.safetensors.index.json", target_index)
    _atomic_write_json(dst / "config.json", target_config)
    _copy_auxiliary_files(src, dst)
    progress.update(
        {
            "status": "complete",
            "target": str(dst.resolve()),
            "target_quantization": target_quant_config,
            "quantized_weights": total_quantized,
            "dequantized_weights": total_dequantized,
            "dequantized_modules": sorted(auto_dequantized_modules),
            "tensor_bytes": total_tensor_bytes,
            "self_test": self_test_result,
        }
    )
    _atomic_write_json(progress_path, progress)
    return {
        **inventory,
        "target": str(dst.resolve()),
        "target_shards": len(shards),
        "quantized_weights": total_quantized,
        "dequantized_weights": total_dequantized,
        "dequantized_modules": sorted(auto_dequantized_modules),
        "target_tensor_bytes": total_tensor_bytes,
        "self_test": self_test_result,
        "progress": str(progress_path),
        "passed": True,
    }


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate source metadata")
    inspect_parser.add_argument("--src", required=True, type=_path)
    inspect_parser.add_argument("--dst", type=_path)

    self_test_parser = subparsers.add_parser(
        "self-test", help="gate one-weight round-trip error"
    )
    self_test_parser.add_argument("--src", required=True, type=_path)
    self_test_parser.add_argument("--device", default="cpu")
    self_test_parser.add_argument("--max-rel-l1", type=float, default=0.08)

    run_parser = subparsers.add_parser("run", help="transcode all shards")
    run_parser.add_argument("--src", required=True, type=_path)
    run_parser.add_argument("--dst", required=True, type=_path)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--skip-space-check", action="store_true")
    run_parser.add_argument("--max-rel-l1", type=float, default=0.08)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        if args.command == "inspect":
            result = inspect_checkpoint(args.src, args.dst)
        elif args.command == "self-test":
            result = self_test(args.src, args.device, args.max_rel_l1)
        else:
            if args.src == args.dst:
                raise TranscodeError("source and destination must be different")
            result = run_transcode(
                args.src,
                args.dst,
                args.device,
                args.resume,
                args.skip_space_check,
                args.max_rel_l1,
            )
    except (OSError, TranscodeError, ValueError) as exc:
        LOGGER.error("FAIL mxfp8_to_fp8_block128 error=%s", exc)
        return 1

    LOGGER.info("%s", json.dumps(result, sort_keys=True))
    LOGGER.info("PASS mxfp8_to_fp8_block128 command=%s", args.command.replace("-", "_"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
