# SPDX-License-Identifier: Apache-2.0

"""Minimal offline build capability for sealing engine plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .core import PLUGIN_IDENTITY, seal_plan_document
from .planner import BuildPolicy, build_plan_document


class JsonPlanBuilder:
    """Build or seal plans without importing a serving runtime."""

    metadata = PLUGIN_IDENTITY

    def build(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and seal one schema-v5 document."""

        return seal_plan_document(request)

    def build_from_timings(
        self,
        timings: Sequence[Mapping[str, Any]],
        *,
        plan_id: str,
        policy: BuildPolicy | None = None,
    ) -> dict[str, Any]:
        """Select tactics from timing caches and build a sealed schema-v5 plan."""

        return build_plan_document(
            timings,
            plan_id=plan_id,
            policy=policy,
        )
