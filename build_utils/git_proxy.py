# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route the build-time GitHub fetches through a mirror.

``setup.py`` materializes the pinned upstream vLLM and FlashInfer revisions from
github.com before it builds the MUSA extensions. On networks where github.com is
slow or unreachable, that fetch dominates a source install. Mirror prefixes such
as https://gh-proxy.org/ serve the same repository under
``<proxy>https://github.com/<owner>/<repo>.git``.

The mirror is selected with ``VLLM_MUSA_GIT_PROXY``:

* unset, empty, ``1``, ``true``, ``on``, ``yes`` or ``default`` -- use
  :data:`DEFAULT_PROXY` (gh-proxy.org);
* ``0``, ``false``, ``off``, ``no``, ``none`` or ``disabled`` -- fetch from
  github.com directly;
* anything else -- the mirror base URL to use instead, for example
  ``https://ghproxy.net/``.

Mirrored attempts also carry git's low-speed guard
(``VLLM_MUSA_GIT_PROXY_STALL_S``, default 60 seconds), so a mirror that accepts
the connection and then stops transferring fails the attempt instead of hanging
the build. A steady but slow transfer is not affected: the guard only fires when
the transfer stalls, which a total timeout cannot distinguish.

The rewrite is handed to git as ``url.<base>.insteadOf`` for the duration of one
command, so a materialized tree still records the canonical github.com remote and
no persistent git configuration is written.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

GITHUB_BASE = "https://github.com/"
DEFAULT_PROXY = "https://gh-proxy.org/"
PROXY_ENV = "VLLM_MUSA_GIT_PROXY"
STALL_ENV = "VLLM_MUSA_GIT_PROXY_STALL_S"
DEFAULT_STALL_S = 60

_DISABLED_VALUES = {"0", "false", "off", "no", "none", "disabled"}
_ENABLED_VALUES = {"1", "true", "on", "yes", "default"}


def resolve_proxy(value: str | None = None) -> str:
    """Return the mirror base to use, or ``""`` to fetch github.com directly.

    ``value`` overrides the environment; ``None`` reads ``VLLM_MUSA_GIT_PROXY``.
    An empty value counts as unset, so an exported-but-empty variable (a Docker
    ``ARG`` without a value, for example) keeps the default mirror.
    """
    if value is None:
        value = os.environ.get(PROXY_ENV)
    value = (value or "").strip()
    if value.lower() in _DISABLED_VALUES:
        return ""
    if not value or value.lower() in _ENABLED_VALUES:
        return DEFAULT_PROXY
    return value if value.endswith("/") else value + "/"


def resolve_stall_s(value: str | None = None) -> int:
    """Return the mirrored-attempt stall window in seconds (0 disables it).

    ``value`` overrides the environment; ``None`` reads :data:`STALL_ENV`.
    """
    if value is None:
        value = os.environ.get(STALL_ENV)
    value = (value or "").strip().lower()
    if not value or value in _ENABLED_VALUES:
        return DEFAULT_STALL_S
    if value in _DISABLED_VALUES:
        return 0
    return max(0, int(value))


def mirror_config_args(
    value: str | None = None, stall_s: str | None = None
) -> list[str]:
    """Return ``git -c`` arguments for a mirrored attempt.

    That is the github.com URL rewrite plus the low-speed guard. The result is
    empty when the mirror is disabled. The rewrite is scoped to
    :data:`GITHUB_BASE`, so remotes on any other host are left untouched.
    """
    proxy = resolve_proxy(value)
    if not proxy:
        return []
    args = ["-c", f"url.{proxy}{GITHUB_BASE}.insteadOf={GITHUB_BASE}"]
    stall = resolve_stall_s(stall_s)
    if stall:
        # 1 byte/sec across the stall window: any real progress clears the
        # guard, a dead transfer does not.
        args += ["-c", "http.lowSpeedLimit=1", "-c", f"http.lowSpeedTime={stall}"]
    return args


def candidate_commands(
    cmd: Sequence[str], value: str | None = None, stall_s: str | None = None
) -> list[list[str]]:
    """Return the variants of a network git command, in the order to try them.

    With the mirror enabled the first candidate goes through it and the second
    is the direct command, so a mirror outage degrades to a slow fetch instead
    of a failed build. With the mirror disabled there is a single candidate.
    """
    direct = [str(part) for part in cmd]
    args = mirror_config_args(value, stall_s)
    if not args:
        return [direct]
    return [[direct[0], *args, *direct[1:]], direct]
