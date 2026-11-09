# vLLM MUSA Platform Patches

vLLM-MUSA carries two distinct kinds of change to upstream vLLM. Source edits
(the vast majority) are applied to a **cloned, pinned vLLM at build time** as a
`git format-patch` series. A tiny set of live-object monkey-patches that have no
source-diff form run at **import time**. There is no runtime source-patching and
no fallback path.

## 1. Source patches — build-time `git` series (the primary mechanism)

`setup.py` clones the pinned upstream vLLM into `third_party/vllm@<tag>` and
applies the diff series in `series/` to it before installing, so the installed
vLLM is already patched.

```
series/0001-MUSA-….patch …  the source-of-truth diff series (git format-patch)
series/upstream/0001-….patch upstream vLLM commits the pin predates, applied after series/
build_apply.py               stdlib-only build-time applier (idempotent git apply)
metadata/                    legacy omissions and functional validation requirements
Makefile.sync                cross-version regeneration (git am -3 + format-patch)
setup.py::_apply_musa_patch_series   wires build_apply into the build
```

- **Idempotency**: a patch that already
  `git apply --reverse --check`-es is skipped; otherwise it is applied forward;
  a forward failure is a **loud conflict** (the pinned vLLM moved — regenerate).
- Set `VLLM_MUSA_NO_BUILD_PATCH=1` to skip the build-time apply (e.g. when
  installing against an already-patched tree).

### Adding or changing a source patch

Edit the cloned vLLM and regenerate the series — do **not** hand-write a
`.patch` file:

```bash
make -f Makefile.sync checkout          # clone/reset third_party/vllm@<tag>
make -f Makefile.sync apply-patches      # git am -3 the current series
#   … edit third_party/vllm, git commit inside the clone …
make -f Makefile.sync format-patches     # regenerate series/ from the commits
git add vllm_musa/patches/series
```

To add an upstream vLLM commit the pin lacks, use
`python3 tools/musa_sync.py backport <sha>` instead: it lands in `series/upstream/`,
so a later pin bump drops it. See `docs/mdm-developer-guide.md` §4a.

### Bumping the pinned vLLM version

Plan first, so the upstream entries the new version already contains are left
out, then replay and regenerate. `docs/mdm-developer-guide.md` §5 has the full
procedure, including conflicts:

```bash
python3 tools/musa_sync.py upgrade-plan <new ref> --history <full vLLM clone>
python3 tools/musa_sync.py rebase <new ref> --drop-upstreamed --history <full vLLM clone>
#   … set VLLM_TAG and VLLM_COMMIT in third_party/PINS to <new ref> …
make -f Makefile.sync format-patches         # regenerate the series on the new base
git add third_party/PINS vllm_musa/patches/series   # commit pin + series
```

To inspect the source series' machine-checked ancestry and functional groups,
run `python3.10 tools/musa_sync.py report --metadata`. The JSON sidecars
contain required checks, not permanent pass flags; `.github/workflows/patch-series.yml`
runs offline replay/round-trip/source gates, not MUSA device or model tests.
See `series/README.md` for the 24 legacy dispositions and evidence limits.

## 2. Object patches — import-time `apply()` (live objects, not source)

A few MUSA changes patch **live Python objects** (monkey-patch a method, prime a
Triton kernel) rather than editing a source file, so they have no diff form and
must run in-process. They are the **only** runtime patches:

| File | What it does |
|---|---|
| `vllm__v1__spec_decode__eagle.patch.py` | Primes the Eagle3 draft kernel. |
| `vllm__distributed__parallel_state.patch.py` | Wires draft-TP=1 (`VLLM_MUSA_DRAFT_TP1=1`). |

Convention for an object patch file:

```python
# vllm__some__module.patch.py  — name maps to vllm.some.module (`__` -> `.`)
PATCHES: list = []            # empty: this file does NOT transform source

def apply() -> None:          # idempotent; called by apply_object_patches()
    ...                       # install the monkey-patch / prime the kernel
```

`vllm_musa.patches.apply_object_patches()` loads each `*.patch.py` in
deterministic name order and calls `apply()` where present. It is wired from
`vllm_musa.plugins.register_patches()`, which runs once per process before any
plugin part that touches vLLM, and is idempotent.

> A `*.patch.py` with a **non-empty** `PATCHES` list is now a mistake — source
> edits belong in `series/`. `patch_report()` flags such a file as
> `misplaced-source-patch`.

## 3. Patch report

`vllm_musa.patches.patch_report()` (also `vllm_musa.patch_report()`, surfaced by
`vllm_collect_env`) is a read-only audit of the object patches: their `apply()`
presence, status, and metadata. It never modifies anything. Build-time source
patches are not audited here — the installed vLLM is already patched.

## Notes

- `import torchada` must precede any `torch.cuda.*` / FlashAttention import; the
  package `__init__` does this first.
- Object-patch `apply()` functions must be idempotent (spawn workers re-import
  `vllm_musa`).
