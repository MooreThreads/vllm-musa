# SPDX-License-Identifier: Apache-2.0
"""Entry point for ``python -m vllm_musa.tactics``.

``-m`` imports this module rather than running the package's ``__init__``, so
the loop has to live behind a callable that this module can invoke.
"""

from . import main

raise SystemExit(main())
