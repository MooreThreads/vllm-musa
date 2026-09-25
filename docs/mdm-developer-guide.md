# vLLM-MUSA Developer Guide — the MUSA Divergence Manifest (MDM)

vLLM-MUSA is an out-of-tree plugin on top of upstream vLLM, plus ~90 source-level
divergences from upstream. Instead of patching an installed vLLM at runtime, the
**MUSA Divergence Manifest (MDM)** keeps those divergences as a build-time
`git format-patch` series applied to a *pinned* upstream vLLM clone. This guide
covers building, the developer edit loop, updating the pinned vLLM, and
(re)generating the patch series.

> Companion docs: `vllm_musa/patches/README.md` (the patch mechanism) and
> `vllm_musa/patches/series/README.md` (the series itself). This file is the
> *workflow* guide.

## Two flows (read this first)

Both share `third_party/vllm` but leave it in **different git states** — don't mix
them on the same checkout:

| Flow | What it does | Git state of `third_party/vllm` | Driver |
|---|---|---|---|
| **Build** (install/run) | clone upstream@pin → **`git apply`** the series (working tree) → build extensions → install | pin + uncommitted working-tree changes | `setup.py` → `vllm_musa/patches/build_apply.py` |
| **Maintenance** (author/update patches) | clone upstream@pin → **`git am -3`** the series (as commits) → edit + commit → **`git format-patch`** | pin + one commit per patch | `tools/musa_sync.py`, `Makefile.sync` |

`regen` works on **commits**, so you must be in the maintenance state (`git am`'d)
to (re)generate the series — see §4. A build leaves the clone `git apply`'d, which
is fine for running but not for `regen`.

## Repository layout

| Path | Role |
|---|---|
| `third_party/PINS` | **single source of truth** for upstream pins (`VLLM_TAG`, immutable `VLLM_COMMIT`, `FLASHINFER_COMMIT`); read by the build and sync tools so they can't desync |
| `third_party/vllm` | the cloned upstream vLLM (gitignored); build + maintenance workspace |
| `vllm_musa/patches/series/*.patch` | the build-time patch series (cat 1/2/3/4b), `git format-patch` output |
| `vllm_musa/patches/manifest.py` | declarative census of every divergence (id, category, path, phase) |
| `vllm_musa/patches/build_apply.py` | applies the series at build time (`git apply`, idempotent) |
| `vllm_musa/patches/*.patch.py` | cat-6 object/monkey-patches (each has a `def apply()`) |
| `vllm_musa/patches/module-drift/*.diff` | cat-4a drift tripwires (never applied; `verify` reports drift) |
| `tools/musa_sync.py` | maintenance driver: `apply` / `verify` / `check-series` / `rebase` / `regen` / `report` |
| `Makefile.sync` | thin make wrapper over `musa_sync` |
| `tools/patch_validate.py` | offline verify gate (any known subcommand passes through) |
| `tools/musa_verify/` | on-hardware verify harness (model smokes + unit tests) |

## 1. Build & install

Prereqs: a MUSA runtime/toolkit environment. Install the build, common, and
MUSA-private Python dependencies, including torch/torch_musa, from the
requirements entrypoints before installing `vllm-musa`.

```bash
# required before pip install . / pip install -e .
pip install -r requirements/build.txt -r requirements/musa.txt

# developer install (recommended) — vllm-musa editable:
pip install -e . --no-build-isolation -v

# end-user install — vllm-musa baked in:
pip install . --no-build-isolation -v
```

Both run `setup.py`, which clones upstream vLLM at the immutable
`third_party/PINS:VLLM_COMMIT` into
`third_party/vllm`, `git apply`s the series, builds the MUSA extensions
(`vllm._C`, `vllm._moe_C`, `vllm_musa._C`) from the patched csrc, and installs vLLM.

> **vLLM is always installed editable** (both commands above), so edits to
> `third_party/vllm` are live (see §2). It uses setuptools *compat* (path-based)
> mode — a `.pth` that adds `third_party/vllm` to `sys.path`, so `import vllm`
> resolves to the clone and wins over any system vLLM (e.g. a
> `--system-site-packages` venv). No clean venv required.

## 2. The developer edit loop (fast iteration)

vLLM is always editable, so edit the patched source in place — Python edits take
effect on the next run, no reinstall:

```bash
$EDITOR third_party/vllm/vllm/<file>.py     # live on the next run
```

(csrc / `.cu` edits still need a rebuild: `pip install -e . --no-build-isolation -v`.)

This loop is for **iterating**. To **capture** an edit as a tracked patch, fold it
into the series via the maintenance flow (§4) — that uses a separate, `git am`'d
checkout, so note your change, then reproduce it there as a commit.

## 3. Classify the divergence (six categories)

| # | Category | Representation | Use when |
|---|---|---|---|
| 1 | PY-EDIT | series `.patch` | python edit to an upstream vLLM file |
| 2 | CSRC-EDIT | series `.patch` | C++/CUDA edit to an upstream csrc file |
| 3 | CSRC-FILE | series `.patch` (whole-file diff) | full rewrite of an upstream csrc file (the whole-file diff lives in the series) |
| 4a | MOD-COPY (copy) | `module-drift/*.diff` tripwire (never applied) | a modified *copy* of an upstream module; `verify` warns if upstream drifts |
| 4b | MOD-COPY (rebind) | series `.patch` | a single-method rebind of an upstream module |
| 5 | NEW-MOD | plain tracked source | a genuinely-new MUSA module / native csrc |
| 6 | RUNTIME-OBJ | `vllm_musa/patches/<module>.patch.py` with `apply()` | a live-object monkey-patch with no source-diff form |

cat 1/2/3/4b live in `series/` (they're `git`-managed commits in the clone). cat-5/6
are plain files in `vllm_musa/`. Every cat-4a/5/6 divergence also needs a
`DivSpec(...)` row in `vllm_musa/patches/manifest.py` (keep the cat-6 expectation
sets in `tests/test_manifest.py` and `tests/test_patches.py` in sync).

## 4. Author or update a series patch (the maintenance flow)

`regen` = `git format-patch <pin>..HEAD`, so the clone must carry the series as
commits first:

```bash
# 1. fresh pinned clone with the series replayed as commits (git am -3):
make -f Makefile.sync clean apply-patches
#    (under the hood: python tools/musa_sync.py rebase $VLLM_COMMIT)

# 2. make your change in the clone and commit it:
$EDITOR third_party/vllm/vllm/<file>.py
git -C third_party/vllm add -A
git -C third_party/vllm commit -m "MUSA(<category>): <what/why>"   # new divergence = new commit
#    to MODIFY an existing divergence, edit its commit instead:
#    git -C third_party/vllm rebase -i $VLLM_COMMIT # mark the patch 'edit', amend, continue

# 3. regenerate the series from the commits:
make -f Makefile.sync format-patches
#    (under the hood: python tools/musa_sync.py regen)

# 4. (new cat-4a/5/6 divergence only) add the manifest entry / cat-6 .patch.py.

# 5. offline gate:
python tools/musa_sync.py verify
```

Then `pip install -e .` to run with the updated series.

## 5. Update to a new upstream vLLM version

```bash
# 1. bump the pin (single source of truth):
$EDITOR third_party/PINS                 # set VLLM_TAG + immutable VLLM_COMMIT

# 2. replay the series onto the new tag (true 3-way merge):
make -f Makefile.sync clean apply-patches
#    trivial upstream drift auto-3-way-merges; a real conflict halts at that patch.

# 3. resolve conflicts in third_party/vllm, then continue (repeat make apply-patches if it halts again):
$EDITOR third_party/vllm/<conflicted files>
git -C third_party/vllm add -A
git -C third_party/vllm am --continue

# 4. regenerate the series onto the new base:
make -f Makefile.sync format-patches

# 5. gate + smoke:
python tools/musa_sync.py verify         # every divergence clean / 0 need attention
tools/musa_verify/verify.sh              # on-hardware model smokes (see §6)

# 6. commit the bumped third_party/PINS + the regenerated series/.
```

cat-5/6 (new modules + object patches) need no rebase — `verify` existence-probes
their seams. cat-4a drift tripwires are regenerated separately (`musa_sync regen`,
`module` area) and flagged by `verify` if the upstream original drifted. Bump
`FLASHINFER_COMMIT` only deliberately — it is decoupled from the vLLM pins on purpose
(upstream's choice breaks the MUSA csrc build path).

## 6. Verify

- **Offline gate (no GPU):** `python tools/musa_sync.py verify` (alias:
  `python tools/patch_validate.py`). Clones upstream, `git apply --check`s every
  series diff, existence-probes cat-5/6 seams, checks cat-4a tripwires, and gates
  the series' generation form (`check-series`). Run before every bump / PR. A
  passing run reports every divergence as clean / `0 need attention` and ends with
  `=== musa_sync verify: PASS ===`.
- **Series form gate:** `python tools/musa_sync.py check-series [--repo <checkout>]`
  `[--replay] [--round-trip]`. Three modes, three costs:

  | invocation | what it proves | cost |
  |---|---|---|
  | `check-series` | *shape* only, no repository: every entry is a regular LF-only file with a non-empty slug, a canonical `git format-patch` mailbox (all-zero `From 000…` separator, `From:` author, RFC-2822 `Date:`, `Subject: [PATCH] ` whose slug matches the filename), ≥1 `index` line, and a diff `git apply --stat` can parse; numbering is unique and contiguous from `0001`; no two entries share a diff body | <1 s |
  | `… --repo <checkout>` | the above, plus every `index` anchor resolves to a **blob** in that checkout (or is a postimage an *earlier* entry produces) | <1 s |
  | `… --repo <checkout> --replay` | the above, plus `git am -3` of the whole series in a **disposable clone** of `--repo`, reporting the first entry git refuses with git's own error line | ~10 s per 171 entries |
  | `… --repo <checkout> --round-trip` | the above, plus `regen` itself in that clone: any entry whose bytes or filename `regen` would rewrite is a `round-trip-dirty` row | ~10 s |

  The default mode is the cheap half, and it is deliberately *not* the whole
  story: it cannot know whether a hunk still applies, and it cannot know whether
  `regen` would leave an entry byte-for-byte alone (an extra `Signed-off-by`, a
  deleted `---` diffstat or a changed author ident all pass it). Use `--replay`
  to ask git whether the series still applies to the pin and `--round-trip` to
  ask whether the series is literally what `regen` writes. Both are opt-in
  because both clone a repo and spawn one git process per entry.
  It fails closed on a missing or empty `series/`, on a symlink/directory/
  chmod-000 entry, and on an unusable `--repo`; a red series makes `verify` exit
  1 even when every divergence is clean.

  **Nothing invokes the gate automatically:** there is no in-repo CI workflow
  and no git hook that runs `check-series` or `patch_validate.py`, and the PR
  pipelines are external to this tree. Whatever gate runs, runs because a human
  or an external pipeline typed the command — so the cost table above is what
  decides which mode a reviewer should ask for.
- **On-hardware:** `tools/musa_verify/verify.sh` (configured via env vars
  `MUSA_HOST`, `MUSA_CONTAINER`, `MUSA_VENV`, … — never commit real values) runs the
  patch unit tests plus one functional server smoke per model, each pinned to its
  own MUSA device. `tools/musa_verify/unit_tests.sh` runs `tests/test_patches.py`.

### Exit codes and verdicts

Every `musa_sync` subcommand ends with one explicit verdict line, and the process
verdict is unambiguous:

| code | meaning |
|---|---|
| 0 | `=== musa_sync <cmd>: PASS ===` — nothing needs attention |
| 1 | `=== musa_sync <cmd>: FAIL (<counters>) ===` — see the rows above; `verify` prints the series-format section *before* the divergence summary so the last line always agrees with the exit code |
| 2 | usage/config error, not a verdict (e.g. `verify` with no resolvable `--target` and no `VLLM_COMMIT`/`VLLM_TAG` in `third_party/PINS`) |

## 7. Command reference

`python tools/musa_sync.py <cmd>`:

| cmd | what it does |
|---|---|
| `apply` | build-time: `git apply` the series to a cloned vLLM |
| `verify` | offline pre-bump gate: status of every divergence (+ the series-format gate) |
| `check-series [--repo PATH] [--replay] [--round-trip]` | gate the series' generation form: every entry must be a canonical `git am` mailbox with canonical numbering; `--repo` also resolves the `index` anchors against that checkout, `--replay` also `git am -3`s the series in a disposable clone of it, `--round-trip` also runs `regen` there and fails on any entry it would rewrite (see §6 for the cost of each) |
| `rebase <tag>` | `git am -3` the series onto `vllm@<tag>` (sets up commits for `regen`) |
| `regen` | regenerate `series/` from the clone's commits (`git format-patch`) |
| `report` | print the manifest census |

`make -f Makefile.sync <target>` (thin wrapper, prefers `VLLM_COMMIT` from `third_party/PINS`):

| target | what it does |
|---|---|
| `checkout` | clone + checkout the pinned upstream vLLM |
| `apply-patches` | `git am -3` the series (→ `musa_sync rebase`) |
| `format-patches` | regenerate the series (→ `musa_sync regen`) |
| `clean` | reset `third_party/vllm` for a clean re-apply |
