# SPDX-License-Identifier: Apache-2.0
"""Registration-only ``speculator_config`` surface for v0.34 Omni plugins.

vLLM 0.34 introduced :mod:`vllm.v1.worker.gpu.spec_decode.speculator_config` so
out-of-tree plugins can register custom speculative methods. ``vllm-omni``'s
``register_omni_models_to_vllm()`` imports ``register_speculative_method`` and
``SpeculatorConfig`` unconditionally during model registration to register the
CUDA-only Higgs Audio v3 INT8 drafter, even when speculative decoding is unused.

The pinned vLLM 0.32 runtime validates speculative methods through its static
``SpeculativeMethod`` literal and never consults this registry. Recording custom
registrations keeps Omni startup and CLI imports working while construction of a
``SpeculatorConfig`` fails explicitly if an unsupported custom speculative path
is invoked on 0.32.
"""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "SpeculatorConfig",
    "register_speculative_method",
    "registered_speculative_methods",
]


class SpeculatorConfig:
    """Base configuration marker for v0.34 custom speculators."""

    base_method: ClassVar[str | None] = None
    speculator_class: ClassVar[str | None] = None
    supported_options: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "SpeculatorConfig requires vLLM >= 0.34; custom speculative methods "
            "are unsupported on the pinned vLLM-MUSA 0.32 runtime"
        )


_CUSTOM_CONFIGS: dict[str, type[SpeculatorConfig] | str] = {}


def register_speculative_method(
    name: str, config_cls: type[SpeculatorConfig] | str
) -> None:
    """Record a custom speculative method config, matching vLLM 0.34's contract."""
    if not isinstance(config_cls, str) and not (
        isinstance(config_cls, type) and issubclass(config_cls, SpeculatorConfig)
    ):
        raise TypeError(
            f"The config of speculative method '{name}' must be a "
            "SpeculatorConfig subclass or qualified class name."
        )
    if name in _CUSTOM_CONFIGS and _CUSTOM_CONFIGS[name] != config_cls:
        raise ValueError(f"Speculative method '{name}' is already registered.")
    _CUSTOM_CONFIGS[name] = config_cls


def registered_speculative_methods() -> list[str]:
    """Return the sorted names of registered custom speculative methods."""
    return sorted(_CUSTOM_CONFIGS)
