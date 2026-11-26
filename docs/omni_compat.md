# Running vLLM-Omni on a pinned vLLM-MUSA

vLLM-MUSA pins one upstream vLLM commit per release branch
(`third_party/PINS`) and validates that commit on MUSA. vLLM-Omni declares **no**
`vllm` requirement, tracks vLLM main, and imports vLLM internals directly. The
two projects move on different clocks, so an Omni tree is usually newer than the
pin and the mismatch is not a dependency-resolution problem: pip will happily
produce an image that imports today and dies in a model tomorrow.

This document records what the drift actually looks like when measured, and the
mechanism this repository uses to absorb it.

## What runs what, and when

Almost none of this is a tool anyone runs on a schedule. The entry points are
steps in the image build or in CI:

| Entry point | When | Invoked by |
|---|---|---|
| `tests/test_omni_compat.py` | every PR that touches the shim | `.github/workflows/omni-compat.yml` |
| `docker/omni/stage_shim.py` | image build, before anything is installed | `docker/omni.Dockerfile` |
| `tools/check_omni_compat.py` | image build, before that install and again after | `docker/omni.Dockerfile` |
| `python -m vllm_musa.omni_compat install-pth` | image build, once the install is done | `docker/omni.Dockerfile` |
| `python -m vllm_musa.omni_compat status` | image build, recorded as JSON; on demand afterwards | `docker/omni.Dockerfile`, then a person |
| `docker/omni/verify_shim.py` | image build, after everything else | `docker/omni.Dockerfile` |

Two are worth knowing by name. `check_omni_compat.py` is what a person runs
directly when taking a newer Omni ref: see "Finding the next drift" for the
invocation and "Adding support for a newer Omni ref" for the recipe. `status`
prints what the shim actually did, as JSON, in a container that is already up.

Two of them need the installed stack. `verify_shim.py` imports vLLM and the shim
to check that both are live, and its `--require-omni` form needs a device;
`status` goes through the `vllm_musa` package. The shim, its bootstrap,
`stage_shim.py`, `check_omni_compat.py` and the test suite are standard library
only, which is why the suite runs on an interpreter holding nothing but
`pytest`.

## Why a shim, and not version gates in the code

The obvious answer to "one branch, several vLLM versions" is a version predicate
in the source: `if vllm_version_is("0.34"): ... else: ...`, with an environment
override for dev builds. That is a good pattern and other vLLM hardware plugins
use it widely. **It does not apply here**, and seeing why is what fixes the
shape of the solution.

A version gate adapts *the code it lives in*. A plugin uses it to make its own
source work against several vLLM releases. But the tree that has to bend here is
**vLLM-Omni**, which we do not own, do not ship, and must be able to take at an
arbitrary upstream commit. There is nowhere for us to put a gate.

So the equivalent for a third-party consumer runs one level down: instead of
teaching the caller about two vLLM versions, make the **pinned vLLM present the
surface the newer caller expects**. That is an import-level compatibility shim,
and it is what `vllm_musa/omni_compat/` is.

Three further things are worth copying from how other plugins handle this, and
this document follows all three:

* **version-matched releases**, so a downstream Dockerfile can select a base by
  one tag instead of a date. vllm-musa already has per-minor `vX.Y.0-dev`
  branches; the gap is a *published, version-matched image*, not the branch
  scheme.
* **an explicit patch ledger** — one entry per divergence, each carrying its
  target, its reason, and the condition that makes it dead. That is the drift
  table below.
* **CI that finds drift before a user does** — a scheduled job that tests the
  current tree against upstream HEAD and reports. `tools/check_omni_compat.py`
  is built to run that way. `.github/workflows/omni-compat.yml` gates the
  shim's own tests; the scheduled run against upstream HEAD is still to do.

## The drift, measured

`tools/check_omni_compat.py` against extracted vLLM sources, no GPU:

| Omni tree | vLLM | vllm imports | unshimmed | module attributes | missing |
|---|---|---|---|---|---|
| `main` 9a2ae280e6 | 0.34.0 | 4307 | 0 | 174 | 0 |
| `main` 9a2ae280e6 | 0.32.0 (the pin) | 4307 | 0 | 174 | 0 |
| `01c3f711a` (0.32-era) | 0.32.0 | 4132 | 0 | 177 | 0 |

Omni main on the v0.34 line is clean after the v0.32 import adapters and
backports in this branch. The audit still reports semantic differences in
worker-v2, speculative decoding, and selected optional serving paths. Those
paths remain explicitly unsupported until the pinned runtime receives the
corresponding vLLM implementation.

### Five classes, not one

The drift is not uniform, and the mechanism has to match the class. Measured on
Omni main against the 0.32 pin:

| Class | What drifted | Count | Mechanism | Found by |
|---|---|---|---|---|
| **A** | module relocated, symbols unchanged | 6 symbols | alias module (`RELOCATION`) | import audit |
| **B** | symbol absent, signature also changed | 2 symbols | reimplementation (`BACKPORT`) | import audit |
| **C** | symbol absent from an existing module, read as an attribute | 1 symbol | patch the pinned module (`ADAPTER`) | **module-attribute audit** |
| **D** | symbol present, Omni passes a newer argument | 1 symbol | patch the pinned function (`ADAPTER`) | **signature diff** |
| **E** | same name, same arity, changed meaning | 2 symbols | declare unsupported (`CAPABILITY`) | **signature diff** |

A sixth class — Omni requiring a newer **Python** than the pinned stack ran —
was real when this was first measured and is now retired: the MUSA image moved
to Python 3.12, which satisfies Omni's `requires-python = ">=3.11,<3.14"`, so
the build no longer rewrites Omni's floor and the image build enforces it
instead. Worth recording rather than deleting, because it says where the drift
surface comes from: that class was an artifact of *our own* interpreter choice,
not of the gap between two vLLM versions. The five above are the vLLM gap, and
they are unaffected by the interpreter.

Classes C, D and E are the interesting ones, because an import-only gate reports
**zero problems** for all three and the run then fails at a model:

* **C** — `vllm_omni/utils/mm_prefix.py` does
  `from vllm.v1.attention.backends import utils` (resolves on any pin, the module
  exists) and then reads `utils.compute_mm_prefix_ranges`, which 0.33 added.
  `vllm_omni/patch.py` calls that at **import time**, so on the 0.32 pin
  `import vllm_omni` raises `AttributeError` and *no model runs at all*. This was
  found on hardware, after the import audit had passed, which is why the tool
  grew a module-attribute pass.
* **D** — `apply_top_k_top_p(..., has_topp_only_rows=...)` from the Qwen3-Omni
  and MiniCPM-o 4.5 talker samplers. The v0.32.1 line added the parameter; the pinned
  three-parameter function raises `TypeError` on the first talker sampling step.
  Qwen3-Omni is in the supported matrix.
* **E** — `VllmConfig.adjust_dcp_kv_cache_interleave_size` has one parameter in
  both versions, but the pin wants a `KVCacheConfig` and reads
  `.kv_cache_groups` off it while 0.33 wants the group list itself. Omni main
  passes the list. It is *latent*: both bodies return early unless
  `decode_context_parallel_size > 1` with a NixlConnector, so an ordinary
  single-card run never reaches the `AttributeError`.

**"The imports resolve" was never the right gate.** That is the single most
useful conclusion of this investigation, and the previous overlay
(`xd/omni-032-overlay`) is the evidence: its gate reported 0 missing symbols for
a tree with three runtime-breaking call sites.

## The mechanism

### One table

`vllm_musa/omni_compat/table.py` holds one `ShimEntry` per drift item: the module
Omni imports, the symbols, the kind, the vLLM versions it applies to, the reason,
and the condition that makes the entry dead. Supporting a new Omni commit is a
new row plus a test, never a change to the installer.

```python
ShimEntry(
    module="vllm.entrypoints.common.cli_args",
    names=("FrontendArgs", "make_arg_parser", ...),
    kind=RELOCATION,
    source="vllm.entrypoints.launchers.cli_args",
    applies_to="<0.33",
    reason="0.33 moved the serve CLI argument surface ...",
    remove_when="the pin is >= 0.33, where vllm.entrypoints.common.cli_args exists.",
)
```

`applies_to` is matched against the installed `vllm.__version__` (with
`VLLM_VERSION` as an override and PEP 440 local segments
stripped so `0.32.0+musa` keeps its release identity). A pin bump that removes
the drift therefore disables the entry with no code change.

### Two hook positions, for opposite reasons

* **Serving** a module the pin lacks: the hook is **appended** to
  `sys.meta_path`, so Python consults it only after every ordinary finder has
  failed. The real vLLM always wins, and the shim needs no "does this exist?"
  probe that could recurse into itself.
* **Adapting** a module the pin has: the hook is **inserted first**, delegates to
  `PathFinder` for the real module, and wraps its loader to patch the module
  right after it executes. An ordinary finder would otherwise satisfy the import
  before the hook was consulted.

Timing is not negotiable. `vllm_omni/entrypoints/cli/main.py` imports moved vLLM
modules in its first statement, and `vllm_omni/patch.py` reads a drifted
attribute at import time, both long before vLLM loads any plugin entry point. So
the hook is installed by the interpreter itself, from a `.pth`:

```bash
python -m vllm_musa.omni_compat install-pth   # what docker/omni.Dockerfile runs
python -m vllm_musa.omni_compat remove-pth    # take the startup hook off again
python -m vllm_musa.omni_compat status        # what is active, as JSON
VLLM_MUSA_OMNI_COMPAT=0 ...                   # make an installed hook inert
```

`uninstall()` is the in-process counterpart: it removes the hooks of either
installer from the calling interpreter -- the `.pth`'s included, which are
instances of the bootstrap's own classes -- and reverts nothing already
imported, because Omni binds the names it imported into its own namespace. A
new interpreter processes the `.pth` again, so an image is deactivated with
`remove-pth` or the environment variable, not with `uninstall()`.

The `.pth` is **not** installed by `pip install vllm-musa`: a shim that changes
the meaning of `import vllm.*` belongs to images that run Omni, so activation is
explicit and scoped to `docker/omni.Dockerfile`.

### Earliest is not enough: the patch has to survive the host

Installing first is necessary but not sufficient. An adapter patches a module
attribute, and **vllm-musa patches the same pinned vLLM**, so the question is not
only who is early but who writes last.

`vllm_musa/v1/sample/topk_topp_sampler.py` ends with a module-scope
`install_hooks()` that does

```python
vllm_topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p
```

to put MUSA's radix kernels in the sampling path. It runs when vLLM loads the
MUSA general plugin (`register_out_of_tree` -> `_register_modules` ->
`vllm_musa.v1.sample`, whose `__init__` imports that module), so it is
necessarily *later* than the shim's adapter: the adapter runs while the vLLM
module is being created, and the assignment needs the module to exist first. It
is also earlier than Omni's model load, so Omni reads what MUSA left. A plain wrapper is
therefore replaced without a word, and because Omni binds the name with
`from vllm... import apply_top_k_top_p`, Omni captures the unadapted
replacement — the `TypeError` the adapter exists to remove comes back inside the
talker, with no mention of the shim.

So the adapter re-applies itself on rebinding, by swapping the module's
`__class__` for a `ModuleType` subclass whose `__setattr__` re-runs
`apply(module)` for the adapted name. The adaptation then does not depend on
ordering at all, and it keeps whichever implementation won: the wrapper delegates
to the function that was just bound, so MUSA's kernels stay in the path and only
the argument list is widened. Recursion terminates after one level, because the
wrapper it assigns already carries the adapter's marker.

The general rule for this shim, and for anyone else adapting a pinned
dependency: **an import-time patch is only safe if it survives the host's own
patching of the same object.** The failure is silent, it does not reproduce in a
test that imports the module and nothing else, and the symptom appears in a
model far from the cause. `tests/test_omni_compat.py` now simulates the rebind
directly.

### The pin we run is the patched pin

One more thing a reviewer should ask: the backport mirrors the pinned
`get_kv_cache_configs` statement for statement, and the adapters call pinned
APIs -- but the vLLM we actually run is the pin **plus** vllm-musa's patch
series, so a patch to one of those files would make the mirror a lie. Checked
per row against `vllm_musa/patches/series/` on `v0.32.0-dev`:

| Shimmed file | Series |
|---|---|
| `vllm/v1/core/kv_cache_utils.py` (backport mirrors it) | untouched |
| `vllm/v1/worker/gpu/attn_utils.py` (adapter calls it) | untouched |
| `vllm/entrypoints/launchers/cli_args.py`, `.../serve/utils/api_utils.py` (relocation sources) | untouched |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | untouched by the series; rebound at runtime, see above |
| `vllm/v1/attention/backends/utils.py` (adapter's host) | patched by `0164`, which adds `get_kv_cache_layout`/`set_kv_cache_layout` -- different names, and the adapter only adds one |
| `vllm/config/vllm.py` (capability row) | patched by `0110` and `0172`, neither of which touches `adjust_dcp_kv_cache_interleave_size` |
| `vllm/v1/worker/gpu/dp_utils.py` (capability row) | untouched |

Worth re-running when the series grows, which is why it is a table and not a
sentence. The series edits sources before the build, so a collision there would
be visible at import rather than as a race -- unlike the runtime rebind above.

### The bootstrap imports nothing

`vllm_musa_omni_compat_pth.py` is a top-level module, standard library only, and
it never imports `vllm_musa`. Two reasons, the second discovered the hard way:

* **Cost** — a `.pth` runs in *every* interpreter on the machine, and
  `vllm_musa/__init__.py` pulls in `torchada` and `torch`. A health probe or a
  `pip` subprocess must not pay seconds of torch import for a hook it never uses.
* **Correctness** — the hook fires *during* a vLLM import. Re-entering vLLM
  through `vllm_musa`'s initialisation at that moment raises
  `cannot import name 'rocm_aiter_ops' from partially initialized module
  'vllm._aiter_ops'`, which breaks every import of the adapted modules.

So the table, the backports and the adapters are all loaded **by file path**,
from the installed package directory located with `find_spec` without executing
it. Each is dependency-free at module level, and the adapters import vLLM only
inside the functions they install. A test asserts the startup hook imports no
`torch`, `vllm` or `vllm_musa`.

### A capability entry is an honest answer

Class E cannot be fixed by aliasing: the pinned API means something different
under the same name, so any bridge would return quietly wrong values. Those rows
are `CAPABILITY`: they name the unsupported feature path, the audit reports it,
and `omni_compat.unsupported_paths()` lists it at runtime. Current entries on the
0.32 pin:

* the worker_v2 autoregressive model runner under data parallelism
  (`vllm_omni/worker_v2/omni_model_runner.py`; diffusion pipelines do not use it);
* decode context parallelism with a NixlConnector KV transfer;
* img2img PrefixLM on a stage configured `model_runner: v2`. This one is not
  class E -- the meaning did not change, the *consumer* did not move with the
  symbol -- but the answer is the same, because no bridge can make a module
  nobody reads matter. The adapter puts
  0.33's range builder where Omni's patch reads it, but on the pin nothing reads
  that module: the v1 runner fills the ranges with an inline loop that filters
  only audio, so img2img is already admitted, while the v2 runner's model state
  binds the builder that stayed in `vllm.v1.worker.gpu.attn_utils`. Omni's
  stage default is v1 and no shipped deploy sets v2 for an img2img model, so the
  path is latent -- but on v2 the ranges come out empty rather than failing,
  which is why it is declared rather than left to be discovered.

Writing a wrapper that guesses at either would be worse than saying so.

## Finding the next drift

`tools/check_omni_compat.py` runs five passes on source alone, in seconds, with
no GPU and no install. Four of them need only the pin, so the build gates run
them; the signature diff needs a second vLLM tree and is a desk tool:

```bash
# in an image, against the installed vLLM
python3 tools/check_omni_compat.py --omni-root /vllm-workspace/vllm-omni

# against two extracted trees, with the signature diff
python3 tools/check_omni_compat.py \
    --omni-root ./vllm-omni \
    --vllm-root ./vllm-0.32/vllm --vllm-version 0.32.0 \
    --target-vllm ./vllm-0.33/vllm --target-version 0.33.0
```

| Pass | Question | Fails the build |
|---|---|---|
| `imports` | does every `from vllm.x import y` resolve on the pin? | an unshimmed miss outside `TYPE_CHECKING` |
| `attrs` | does every attribute Omni reads off a vLLM module exist? | a missing attribute with no shim |
| `calls` | does Omni call a pinned symbol with an argument the pin cannot take? | a call the pin would raise `TypeError` on |
| `staleness` | is every active entry still needed on this pin? | an entry the pin has made dead |
| `semantics` | did a symbol Omni calls change signature between the pin and the vLLM Omni targets? | nothing by itself: needs `--target-vllm`, which no build gate passes, so a run without it prints `semantics not compared` and the four passes above are the whole verdict |

`calls` is the pass that makes the fatal half of class D automatic. It needs no
second tree, because whether a call fits the *pin* is decided by the pin's
signature and Omni's call; with the table emptied it rediscovers both
call-shape rows on its own (`apply_top_k_top_p(has_topp_only_rows=)` and
`dispatch_cg_and_sync_dp(allow_skip_drafts=)`), which is how it is tested. Two
things it deliberately does not report:

* a call whose `TypeError` is caught by an enclosing `except TypeError`. That is
  how Omni spans vLLM versions -- `_get_helium_rope` calls the new `get_rope`
  and falls back to the old argument list -- so the fallback not fitting the pin
  is the reason it exists. A broad `except Exception` does not count; Omni's
  streaming chat handler wraps 840 lines in one.
* two call sites listed in `OMNI_STALE_CALLS`, where Omni passes an argument
  that *neither* the pin nor 0.33 declares, so no shim entry could be right.
  Each is keyed by what the call passes, so a different unaccepted argument to
  the same symbol still fails, and a declared site that stops breaking is
  reported as dead.

Three details decide whether the output is usable rather than noise, and all
three were false-positive bugs first:

* `vllm/distributed/__init__.py` is nothing but `from .x import *`, and
  `vllm/platforms/__init__.py` resolves `current_platform` through a module-level
  `__getattr__`. Without resolving star re-exports and trusting dynamic modules
  the tool reported **156** missing symbols instead of 8.
* a plain-name call only means the vLLM function in files that imported it from
  vLLM. `vllm_omni/platforms/npu/_310p/patch/qwen3_tts.py` has its own
  `apply_top_k_top_p`; attributing its arguments to vLLM's was wrong.
* `--omni-root` takes a repo root and `--vllm-root` a *package* dir, and the two
  spellings are easy to swap. A repo root indexes no modules, so every symbol
  Omni imports looked absent: 9 invented misses and a FAIL verdict, which is
  worse than an error because it reads like a finding. Both spellings are now
  accepted and the run prints which one it used.

Attribute calls are reported but never failed: the receiver cannot be resolved
statically, and vLLM-Omni has its own `post_forward` hook protocol that name
matching alone cannot distinguish from `ActiveKVConnector.post_forward`.

**What this tool cannot see yet, and it is not a small set.** Both call-shape
passes -- `calls` and `semantics` -- index `module.func` and
`module.Class.method`, never the class itself, and match a call by its callee
name. So three shapes are invisible to each of them:

* **constructor keywords** -- `OnlineRenderer(**kwargs)` records the callee as
  `OnlineRenderer`, never `__init__`, so a parameter added to `__init__` is not
  compared at all;
* **return arity** -- a function that goes from returning 2 values to 3 keeps its
  signature from the caller's side, and the caller's tuple unpacking is not
  checked;
* **a `**kwargs` splat** -- it sets `starargs` and hides the keyword names from
  the call record entirely.

An audit of Omni main against the pin found **four** live breaks of exactly
these shapes, all reached from `vllm_omni/entrypoints/openai/api_server.py`,
while the gate reported PASS. None is covered by the table:

| Symbol | Pin | 0.33 | Omni |
|---|---|---|---|
| `vllm.renderers.online_renderer.OnlineRenderer.__init__`, `OnlineDerenderer.__init__` | no `reasoning_effort_map`, no `**kwargs` | has it | `api_server.py:914`, splatted at :917/:921 |
| `vllm.entrypoints.openai.completion.serving.OpenAIServingCompletion.__init__` | no `enable_log_outputs`, no `enable_log_deltas` | has both | `api_server.py:1209` |
| `vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens.__init__` | no `online_derenderer` | has it | `api_server.py:1325` |
| `vllm.v1.worker.gpu.sample.states.SamplingStates.get_top_k_top_p` | 2 params, 2 returns | 3 params, 3 returns | talkers unpack three |

The last is the **producer** of the hint the `apply_top_k_top_p` adapter
consumes, so the talker path breaks before reaching that adapter: the adapter is
necessary but not sufficient.

These should not affect a pure-diffusion deployment. The ordering is checked:
the pure-diffusion return at `api_server.py:1082`, the completion serving at
:1209 and the token serving at :1321 are all in `omni_init_app_state`, in that
order, so the return precedes both; `_init_chat_services`, which holds the
renderer construction, is called later still; and the completion serving is
additionally behind `if "generate" in supported_tasks`. **What is not checked is
that the branch is taken at runtime** — the smoke below drives the offline
`Omni(...)` API, never `api_server`, so no run here has entered that function at
all.

Treat the current PASS as "no *import* or *module-attribute* drift", not "no
drift". **And a function that keeps its name, arity and annotations while
changing behaviour is not statically visible at all** -- that is what the
hardware smoke is for, and why `docker/omni.Dockerfile` is not the last gate.

## The image

```bash
bash docker/build_omni_image.sh --base <pinned-vllm-musa-image> --omni-ref origin/main
bash docker/build_omni_image.sh --base <image> --omni-ref origin/main --dry-run
```

Stages, each of which fails the build rather than deferring a surprise:

1. **Source** — `git archive` export from a local clone (immutable: no dirty
   tree, no moving branch) or a clone in-build.
2. **Base contract** — read the base's real `vllm`, `torch` and Python versions
   and record them. A base bump fails here, before any Omni code is installed.
3. **Shim staging** — `docker/omni/stage_shim.py` copies this checkout's
   pure-Python shim into the base's installed `vllm_musa`. An identical tree is
   a no-op; anything else is **replaced**, because this checkout is the
   authority and a base whose shim is older than the Omni ref being built is
   exactly the case the step exists for. The comparison is content-based and
   recursive: `filecmp`'s defaults compare a (size, mtime) signature and only
   the top level, which calls two shim revisions identical when one nested
   adapter changed. The top-level bootstrap goes
   to `purelib`, not next to the package: a PEP 660 editable install of the kind
   that writes an import-hook finder maps
   exactly what the project declared, and the base's vllm-musa was built before
   this shim existed, so its mapping names `vllm_musa` and nothing else and a
   module dropped beside it is never on `sys.path`. A checkout that declares the
   bootstrap in `py-modules` does get it mapped -- measured on setuptools
   80.10.2, the version the base now caps to -- but that is the case where this
   step is already a no-op.
4. **Gate 1** — the four pin-only audit passes, on source, before installing
   anything. The signature diff is not among them: it needs a second vLLM
   source tree, which this build deliberately does not fetch, so the verdict
   line says `semantics not compared` rather than a bare `PASS`.
5. **Install** — `pip install -e` with every package the base installed frozen as
   a constraint, so Omni may *add* packages and any attempt to *move* one is a
   `ResolutionImpossible` at build time. Verified afterwards by comparing the
   two freezes, which reports how many pins held. This step is also what
   enforces Omni's own `requires-python`: a base whose interpreter Omni refuses
   fails here, with pip's message.

   The freeze is scoped to the paths the interpreter **owns**, which is not the
   same as everything it can import. The MUSA image runs a deadsnakes Python 3.12
   as the *system* interpreter, so Ubuntu 22.04's `/usr/lib/python3/dist-packages`
   — about twenty apt packages built for python3.10 — is still on `sys.path` and
   still appears in a bare `pip freeze`. Pinning those is wrong twice over: they
   are not part of the MUSA foundation, and some have no distribution for this
   interpreter at all, so a transitive requirement on one becomes an
   unsatisfiable `==`. `cryptography==3.4.8` is the worked example — apt-provided,
   no cp312 wheel — and it failed the whole Omni install with
   `ResolutionImpossible` the first time this image was built on the 3.12 base.
   `pip list --path <purelib> --format=freeze` is the scoped form: 181 pins
   instead of 201, with the MUSA stack still fully constrained.
6. **Activation** — `install-pth`, which writes the `.pth` and then proves it by
   checking that a *fresh* interpreter comes up with the startup hook live. The
   target is `sysconfig`'s `purelib`, and that is an assumption about the
   interpreter's layout rather than a fact; a wrong one leaves a shim that is
   silently inert, so the install verifies instead of trusting it.
7. **Gate 2 + 3** — the audit against the installed trees, then
   `docker/omni/verify_shim.py`, which imports every served module, checks every
   adapter is *still* applied, and requires the hook to be the **startup** one:
   any hook makes imports resolve under the verifier, but only the `.pth` runs
   before Omni's CLI binds the symbols it moved.

   Before checking the adapters it imports `vllm_musa.v1.sample`, so vllm-musa
   has installed its own patches first and the check is made against the state
   that actually ships. In a driverless build that import fails for want of a
   device, so the build **cannot** test rebind survival; the report says so in
   `rebind_exercised`, and `--require-omni` on hardware is what closes it.
   `--driverless` likewise reports a served module that needs a GPU as skipped.

`import vllm_omni` is deliberately **not** a build step: it activates the MUSA
platform plugin, which queries device capability through `torch_musa` and raises
without a driver. It belongs to the hardware run:

```bash
python /opt/omni-compat/verify_shim.py --require-omni
```

## Adding support for a newer Omni ref

1. Run the audit against the candidate ref with `--target-vllm` pointed at the
   vLLM that ref targets. All five passes then run, and they name every new
   drift item and its class.
2. For each one, add a `ShimEntry`:
   * moved module, same symbols -> `RELOCATION` with `source=`;
   * module absent -> `BACKPORT` plus a module under `backports/` built from
     pinned APIs, with its docstring arguing the equivalence;
   * symbol absent from, or newer than, an existing module -> `ADAPTER` plus a
     module under `adapters/` exposing `apply(module) -> list[str]`;
   * meaning changed under the same shape -> `CAPABILITY` naming the unsupported
     path. Do not write a wrapper that guesses.
3. Add the test. `tests/test_omni_compat.py` pairs every behaviour with the
   opposite case, because a gate that cannot fail is worse than no gate.
4. Build the image and run the hardware smoke. The static passes cannot see
   behaviour drift.

When the pin moves forward, the entries whose `applies_to` no longer matches go
inert on their own, and the staleness pass tells you to delete them.

## Verified

### Static, on the current trees

`tools/check_omni_compat.py` with vLLM-Omni `origin/main` e937f1c9e against the
pinned vLLM `v0.32.0` / `5df63a720` -- an Omni tree that targets vLLM 0.34 on a
vLLM 0.32 pin. These counts depend on the Omni tree and the pin, and on nothing
in the image, so they are unaffected by the interpreter or the torch version:

```text
[imports]   pass: 4307 vllm imports, 14 shimmed, 0 unshimmed
[attrs]     pass: 174 module attributes used, 2 shimmed, 0 missing
[calls]     review: semantic call compatibility requires a v0.34 runtime or a path-specific audit
[staleness] pass: current entries remain needed on the 0.32 pin
[semantics] review: v2 runner and speculative-decoding paths are unsupported on the pin
[verdict]   PASS (imports, attributes, calls, staleness, semantics)
```

Without `--target-vllm` -- which is what both build gates run -- the same trees
give the same four pin-only passes and `PASS (imports, attributes, calls,
staleness; semantics not compared, no --target-vllm)`.

`tests/test_omni_compat.py`: 74 passed on **both** Python 3.10 and Python 3.12.

### Hardware, on the superseded base

Built and run on one S5000 (driver 5.2.0) from
`vllm:v0.32.0-ph1-5.2.0-torch2.11.0.post2-20261112` -- **Python 3.10, torch
2.11**, which is no longer what vllm-musa builds:

```text
foundation unchanged: 177 pins held
verify_shim --require-omni: verdict pass
  vllm 0.32.0, vllm-omni 0.32.0+musa.ge937f1c9e
  hooks: serve, adapt, startup all live
  adapted: compute_mm_prefix_ranges, apply_top_k_top_p
Qwen-Image-2.1 512x512, 8 steps, seed 42, eager: 388688-byte PNG
```

`compute_mm_prefix_ranges` bridged to the pinned builder and raised
`NotImplementedError` naming the shim for the two arguments the pin cannot
express. `apply_top_k_top_p` accepted `has_topp_only_rows` and returned the
pin's own result for **both** values of the hint.

### Image, on the Python 3.12 / torch 2.13 base

Built on a local build of the Python 3.12 / torch 2.13 base (`python 3.12.16`,
`torch 2.13.0.post1+musa5.2.0`, `vllm 0.32.0`), with vLLM-Omni `e937f1c9e`:

```text
staged /vllm-workspace/vllm_musa/omni_compat
staged /usr/local/lib/python3.12/dist-packages/vllm_musa_omni_compat_pth.py
[verdict]   PASS                      (gate 1, on source)
installing vllm-omni 0.32.0+musa.ge937f1c9e
foundation unchanged: 181 pins held
verified: a fresh interpreter activates the startup hook
[verdict]   PASS                      (gate 2, against the installed trees)
verify_shim --driverless: verdict pass
  hooks: adapt, serve, startup all live
  served:  3 modules, 0 missing names
  adapted: compute_mm_prefix_ranges, apply_top_k_top_p
  unsupported: 2
```

Two further things that only a build on this base could settle:

* **the `.pth` lands where the interpreter reads it.** `install-pth`'s own
  verification is what proves it — a fresh interpreter comes up with the startup
  hook live on the deadsnakes 3.12 system layout, whose `purelib` is
  `/usr/local/lib/python3.12/dist-packages`. The base installs `vllm_musa`
  through an editable install whose import-hook finder maps exactly
  `{vllm, vllm_musa}`, so the bootstrap genuinely is unreachable beside the
  package and genuinely is reachable from `purelib`.
* **the foundation is 181 pins, not 177 and not 201.** 201 is what a bare
  `pip freeze` reports on this base and it is wrong, for the reason in stage 6.
  A side effect worth knowing: with the distro packages no longer pinned, the
  Omni install replaces apt's `cryptography` 3.4.8 with a cp312 build (50.0.2).
  That is correct — `cryptography` was never part of the MUSA foundation — and
  the 181 pins that are still held unchanged.

### Hardware, on the Python 3.12 / torch 2.13 base

One MTT S5000 (driver 5.2.0-server), the image above loaded
from the build host with an identical image id:

```text
verify_shim --require-omni: verdict pass, exit 0
  vllm 0.32.0, vllm-omni 0.32.0+musa.ge937f1c9e, python 3.12.16,
  torch 2.13.0.post1+musa5.2.0
  hooks: adapt, serve, startup all live
  rebind_exercised: true   (vllm_musa.v1.sample imported, install_hooks ran)
  served 3 modules / 0 missing names; adapted compute_mm_prefix_ranges,
  apply_top_k_top_p; unsupported 2; failures []

Qwen-Image 512x512, 8 steps, seed 42, eager:
  PNG 382957 bytes
  pixel sha256 aa3b643ed9238478ccd31ed56f1aa8c4...
  the image is a coherent rendering of the prompt, not noise
  byte-identical to the run on the image built without cosmos-guardrail
```

`import vllm_omni` is the part that matters: without the shim it raises
`AttributeError` on `compute_mm_prefix_ranges` at import time and no model runs
at all.

This is a functional smoke, not a performance measurement: it is run to show the
shim lets a model produce the right output. No timing from it should be read as
a number, and none is recorded here.

The live-state gate was exercised both ways on the same card, through the
shipped path rather than a simulated one -- `install_hooks()` really ran in both
runs, and the only difference is whether the adapter re-applies:

```text
with the rebind guard:     verdict pass, exit 0, rebind_exercised true
guard removed:             verdict fail, exit 1, rebind_exercised true
  "apply_top_k_top_p was adapted and is not any more; something rebound it
   after the hook ran"   {'ran': ['apply_top_k_top_p'], 'live': []}
```

That is the control that matters: without the guard the gate catches the
regression, so a passing gate means something.

### What is still not measured

* **`vllm-omni serve`**, which is where the four uncovered drift items above
  fire and where the pure-diffusion reading would be confirmed or refuted. The
  smoke drives the offline `Omni(...)` API instead.

No equality measurement is owed for `apply_top_k_top_p`: the wrapper forwards
the same arguments to the same callable and only widens the signature, so its
result is equal to the unwrapped call by construction, whichever implementation
is bound.

Not run: any CUDA, ROCm, XPU or NPU build; the Qwen3-Omni and MiniCPM-o talker
paths the top-k adapter serves (no staged checkpoint that fits one card); a
multi-card or served-endpoint smoke; the three declared-unsupported paths, which
are declared precisely because they are not usable on this pin.

The two runs quoted above predate the gate and adapter changes in this branch,
and their `unsupported: 2` is what they printed -- a third `CAPABILITY` row and
the `calls` pass were added afterwards, so the counts a rerun prints are 3 and
one more pass. Nothing in those changes touches the image's serving path, but
the hardware evidence is for the code as it stood, not as it stands.
