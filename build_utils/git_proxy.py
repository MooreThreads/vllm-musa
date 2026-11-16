# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route the build-time GitHub clones through a mirror.

``setup.py`` clones the pinned upstream vLLM and FlashInfer revisions from
github.com before it builds the MUSA extensions. On networks where github.com
is slow or unreachable, that clone dominates a source install. Mirror prefixes
such as https://gh-proxy.com/ serve the same repository under
``<proxy>https://github.com/<owner>/<repo>.git``.

The mirror is selected with ``VLLM_MUSA_GIT_PROXY``:

* unset, ``1``, ``true``, ``on``, ``yes`` or ``default`` -- use
  :data:`DEFAULT_PROXY` (gh-proxy.com);
* ``0``, ``false``, ``off``, ``no``, ``none``, ``disabled`` or empty -- clone
  github.com directly;
* anything else -- the mirror base URL to use instead, for example
  ``https://ghproxy.net/``.

The rewrite is handed to git as ``url.<base>.insteadOf`` for the duration of one
command, so a cloned tree still records the canonical github.com remote and no
persistent git configuration is written.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

GITHUB_BASE = "https://github.com/"
DEFAULT_PROXY = "https://gh-proxy.com/"
PROXY_ENV = "VLLM_MUSA_GIT_PROXY"

_DISABLED_VALUES = {"", "0", "false", "off", "no", "none", "disabled"}
_ENABLED_VALUES = {"1", "true", "on", "yes", "default"}


def resolve_proxy(value: str | None = None) -> str:
    """Return the mirror base to use, or ``""`` to clone github.com directly.

    ``value`` overrides the environment; ``None`` reads ``VLLM_MUSA_GIT_PROXY``.
    """
    if value is None:
        value = os.environ.get(PROXY_ENV)
        if value is None:
            return DEFAULT_PROXY
    value = value.strip()
    if value.lower() in _DISABLED_VALUES:
        return ""
    if value.lower() in _ENABLED_VALUES:
        return DEFAULT_PROXY
    return value if value.endswith("/") else value + "/"


def proxy_config_args(value: str | None = None) -> list[str]:
    """Return ``git -c`` arguments that send github.com URLs to the mirror.

    The result is empty when the mirror is disabled. The rewrite is scoped to
    :data:`GITHUB_BASE`, so remotes on any other host are left untouched.
    """
    proxy = resolve_proxy(value)
    if not proxy:
        return []
    return ["-c", f"url.{proxy}{GITHUB_BASE}.insteadOf={GITHUB_BASE}"]


def candidate_commands(cmd: Sequence[str], value: str | None = None) -> list[list[str]]:
    """Return the variants of a network git command, in the order to try them.

    With the mirror enabled the first candidate goes through it and the second
    is the direct command, so a mirror outage degrades to a slow clone instead
    of a failed build. With the mirror disabled there is a single candidate.
    """
    direct = [str(part) for part in cmd]
    args = proxy_config_args(value)
    if not args:
        return [direct]
    return [[direct[0], *args, *direct[1:]], direct]
