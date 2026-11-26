# vLLM-MUSA Developer Guide — the MUSA Divergence Manifest (MDM)

vLLM-MUSA is an out-of-tree plugin on top of upstream vLLM, plus ~200
source-level divergences from upstream (`musa_sync report` prints the manifest
census; `vllm_musa/patches/series/README.md` states how many are `series/`
entries). Instead of patching an installed vLLM at runtime, the
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
| `vllm_musa/patches/series/upstream/*.patch` | upstream vLLM commits the pin predates (`git cherry-pick -x`), applied after `series/*.patch`; see §4a |
| `vllm_musa/patches/manifest.py` | declarative census of every divergence (id, category, path, phase) |
| `vllm_musa/patches/build_apply.py` | applies the series at build time (`git apply`, idempotent) |
| `vllm_musa/patches/*.patch.py` | cat-6 object/monkey-patches (each has a `def apply()`) |
| `vllm_musa/patches/module-drift/*.diff` | cat-4a drift tripwires (never applied; `verify` reports drift) |
| `tools/musa_sync.py` | maintenance driver: `apply` / `verify` / `check-series` / `rebase` / `upgrade-plan` / `backport` / `regen` / `report` |
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

# developer install — vllm-musa and the patched upstream vLLM editable:
pip install -e . --no-build-isolation -v

# source-tree wheel install — vllm-musa bundles patched upstream vLLM:
pip install . --no-build-isolation -v
```

The editable command is the development workflow. It clones upstream vLLM at the immutable
`third_party/PINS:VLLM_COMMIT` into
`third_party/vllm`, `git apply`s the series, builds the MUSA extensions
(`vllm._C`, `vllm._moe_C`, `vllm_musa._C`) from the patched csrc, and installs vLLM.

The non-editable command builds and installs the bundled wheel. It does not
provide a live `third_party/vllm` source path. The production Docker targets
use this wheel path. The Docker `test` target retains the editable source
workflow for development.

> In the editable workflow, setuptools compat mode adds `third_party/vllm` to
> `sys.path`, so `import vllm` resolves to the patched clone. In the wheel
> workflow, `vllm` and `vllm_musa` are both provided by the installed
> `vllm-musa` distribution.

## 2. The developer edit loop (fast iteration)

After the editable install, edit the patched source in place. Python edits take
effect on the next run, without reinstalling:

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

A new commit becomes the next `series/` entry. A commit that changes a file an
`upstream/` entry changes is kept in `upstream/` after it instead, as a MUSA fix
(§4a). To change what an upstream entry does, add a MUSA commit on top rather than
editing the cherry-picked commit.

## 4a. Backport an upstream vLLM commit

Use this when vLLM-MUSA needs an upstream vLLM change that the pinned `VLLM_COMMIT`
does not have yet. `backport` records it as an **upstream entry** in
`series/upstream/`, so the pin bump that brings the change in drops the entry
automatically (§5). Do not commit an upstream change as a MUSA patch, and do not
write the `.patch` file by hand.

How the two directories work:

- `series/NNNN-*.patch` are the MUSA entries and `series/upstream/NNNN-*.patch` the
  upstream entries. The build and `rebase` apply every MUSA entry first, then
  `upstream/` in order. Each directory is numbered from `0001` on its own, so adding or
  removing an upstream entry renumbers only `upstream/`.
- An upstream entry is the `git cherry-pick -x` of one upstream commit, kept
  verbatim: the upstream subject (a squash-merged PR ends it in `(#N)`) and git's
  `(cherry picked from commit <sha>)` line, which `upgrade-plan` reads.
- An `upstream/` entry without that line is a **MUSA fix** for the upstream entries
  before it, e.g. adapting a backported test to the pin.
- `check-series` fails a cherry-pick in `series/` (`misplaced-cherry-pick`) and an
  `upstream/` whose first entry is not one (`not-a-cherry-pick`).

Steps:

```bash
# 1. find the upstream commit: the commit that merged the PR into vLLM main (vLLM
#    squash-merges, so one PR is one commit). Use its full 40-character sha. If it
#    needs other upstream commits the pin lacks, backport those first, oldest first.

# 2. put the series into the clone as commits (the maintenance state, §4):
python tools/musa_sync.py rebase                 # or: make -f Makefile.sync clean apply-patches

# 3. cherry-pick it on top of the series and regenerate the series:
python tools/musa_sync.py backport <sha>                        # fetches <sha> from upstream
python tools/musa_sync.py backport <sha> --from /path/to/vllm   # or from a local vLLM clone
#    -> "backported <sha12> as upstream/NNNN", then "=== musa_sync regen: PASS ==="
#    The only new file is series/upstream/NNNN-<subject>.patch; no other entry changes.
```

`backport` refuses to start while `third_party/vllm` does not hold exactly the current
series on the pin (run `rebase`, or `regen` first to keep commits you made there), has
uncommitted changes, or has an unfinished `git am`, `cherry-pick` or `rebase`. It also
refuses a commit that is already an upstream entry, and fails without adding anything
when the cherry-pick is empty because the series already has the change.

If the cherry-pick conflicts, `backport` stops with `CONFLICT: cherry-pick <sha12>`.
Resolve it with the smallest change that makes the upstream commit apply to the pin,
then finish it:

```bash
$EDITOR third_party/vllm/<conflicted files>
git -C third_party/vllm add <resolved files>
git -C third_party/vllm cherry-pick --continue
python tools/musa_sync.py regen
# or give up: git -C third_party/vllm cherry-pick --abort
```

The entry then differs from its upstream commit, so the pin bump that brings that
commit in reviews it instead of dropping it.

If the backport needs a MUSA adaptation to build or pass on the pin (a renamed field, a
MUSA-only code path), keep the cherry-pick verbatim and commit the adaptation on top as
an ordinary MUSA commit, without a cherry-pick line:

```bash
$EDITOR third_party/vllm/<file>
git -C third_party/vllm commit -am "MUSA: <what the adaptation does>"
python tools/musa_sync.py regen
```

When the adaptation changes a file that an upstream entry changes, `regen` keeps it in
`upstream/` right after the upstream entries and prints
`note: upstream/NNNN-… changes a file that an upstream commit changes, …`; otherwise it
becomes the next MUSA entry in `series/`. `upstream/0003` is an example: it renames a
field in the backported DeepSeek-V4 test to the pin's name.

Then record and check it:

```bash
# 4. update vllm_musa/patches/series/README.md: the "**N upstream entries**" count (and
#    "Currently **N patches**" if a MUSA entry was added; the tests check both), plus a
#    sentence on what the backport does and the upstream commit or PR it is.

# 5. gate:
python tools/musa_sync.py check-series
python tools/musa_sync.py check-series --repo third_party/vllm --round-trip
git -C third_party/vllm worktree add --detach /tmp/vllm-pin <VLLM_COMMIT>
python tools/musa_sync.py check-series --repo /tmp/vllm-pin --replay
git -C third_party/vllm worktree remove /tmp/vllm-pin
python tools/musa_sync.py upgrade-plan           # against the current pin: keeps every entry
python -m pytest tests/test_manifest_completeness.py tests/test_manifest.py \
    tests/test_musa_sync.py tests/test_musa_sync_upstream.py

# 6. build and test: the install log lists "MUSA build patch: applied upstream/NNNN-…"
#    last. Run the backported change's own upstream tests on the patched tree and the
#    regression the change affects.
pip install -e . --no-build-isolation -v

# 7. commit the new entry and the README:
git add vllm_musa/patches/series/upstream/NNNN-*.patch vllm_musa/patches/series/README.md
git commit -m "fix(series): backport <upstream subject> (#N)"
```

To remove an upstream entry, together with any MUSA fix after it that only served it,
delete it and regenerate; only `upstream/` is renumbered:

```bash
git rm vllm_musa/patches/series/upstream/NNNN-*.patch
python tools/musa_sync.py rebase
python tools/musa_sync.py regen
git add vllm_musa/patches/series                 # then update the README count
```

## 5. Update to a new upstream vLLM version

A bump replays the series onto the new ref. `upgrade-plan` first decides, entry by
entry, what the new ref already contains, so the upstream entries it has are left out
instead of stopping `git am -3`. Keep `third_party/PINS` on the **old** pin until
step 3: the plan searches the history between the old pin and the new ref.

`--history` needs a clone of upstream vLLM with the full history of the new ref
(`git clone https://github.com/vllm-project/vllm.git`); `third_party/vllm` is usually
shallow and cannot place most commits. `--fetch` fetches missing commits into it.

```bash
# 1. plan the bump (read-only):
python tools/musa_sync.py upgrade-plan <new ref> --history /path/to/vllm
```

It prints a row for every entry it does not keep (`--all` lists every entry) and exits
1 while any entry needs review. For example, from the v0.28.0 pin to `v0.30.0`:

```text
=== musa_sync upgrade-plan: 2cf0a6915ce5 -> ced6857afa0e (history: /path/to/vllm) ===
  drop    upstream  upstream/0001-Core-Fix-ValueError-on-KV-load-failure-with-a-hybrid.patch
                    c351fd3c6495 is in the target history; #50388 merged as c351fd3c6495
  drop    upstream  upstream/0002-Bugfix-Mooncake-Report-failed-remote-KV-loads-to-the.patch
                    2c7ee87223f0 is in the target history; #50984 merged as 2c7ee87223f0
  review  fix       upstream/0003-MUSA-run-the-DeepSeek-V4-KV-load-failure-test-on-the.patch
                    a MUSA change on top of upstream/0001-…, which is drop: drop it with it, or keep it as a MUSA patch if still needed
--- 180 keep / 2 drop / 1 review ---
=== musa_sync upgrade-plan: FAIL (review=1) ===
```

An upstream entry counts as present in the new ref when the ref contains its commit,
a commit whose subject ends in its `(#N)` (how release branches carry it), a
`(cherry picked from commit …)` of it, or a commit with the same `git patch-id`. The
series is also applied in order to the pristine new ref. Settle every `review`:

| group | reason (abridged) | what to do |
|---|---|---|
| `upstream` | `…; the entry differs from its upstream commit` | it was adapted when backported. Delete it; if part of the adaptation is still needed, re-add that part as a MUSA commit after step 2 |
| `upstream` | `…, reverted by <sha>` or `…, but the change still applies: reverted?` | upstream undid the change after it landed: keep the entry if MUSA still needs it, otherwise delete it |
| `upstream` | `… (pass --history <full vLLM clone>)`, `its upstream commit is not in the history` | the history cannot place it: pass a full clone, or add `--fetch` |
| `fix` | `a MUSA change on top of upstream/NNNN-…, which is drop` | delete it if it only adapted that entry; keep it if MUSA still needs it |
| `musa` | `the change is already in the target: upstreamed?` | upstream now has the change: delete it |

To delete an entry, `git rm` its file and its sentence in the series README. Re-run
`upgrade-plan` until it passes, or until only reviews you decided to keep remain.

```bash
# 2. replay onto the new ref, leaving out the drop entries:
python tools/musa_sync.py rebase <new ref> --drop-upstreamed --history /path/to/vllm
```

`rebase` lists the entries it leaves out, then `git am -3`s the rest, MUSA entries
first and `upstream/` last. Trivial upstream drift merges by itself; a real conflict
stops at that entry with `CONFLICT: <entry>`. Running `rebase` again starts over from
the new ref, so resolve the conflict and queue every later entry in one `git am`; a
later conflict then stops that same `git am`, and `git am --continue` carries on:

```bash
$EDITOR third_party/vllm/<conflicted files>
git -C third_party/vllm add <resolved files>
git -C third_party/vllm am --continue

E=0042-MUSA-example.patch    # the entry after "CONFLICT:" (upstream ones read upstream/NNNN-…)
S=$PWD/vllm_musa/patches/series
rest=$(printf '%s\n' "$S"/*.patch "$S"/upstream/*.patch | sed "0,\#/$E\$#d" |
       grep -v -e '/upstream/0001-' -e '/upstream/0002-')   # leave out the entries rebase dropped
git -C third_party/vllm am -3 --keep-non-patch $rest
```

```bash
# 3. point the pin at the new ref, then regenerate the series from the commits:
$EDITOR third_party/PINS                 # VLLM_TAG and the immutable VLLM_COMMIT of <new ref>
python tools/musa_sync.py regen          # or: make -f Makefile.sync format-patches
```

`regen` exports the commits after the `PINS` commit, and refuses to run while `PINS`
names a different commit than the one the series sits on. Each directory is
renumbered on its own, so dropped upstream entries never renumber the MUSA entries.

```bash
# 4. update what names the old version: the counts, descriptions and release label in
#    vllm_musa/patches/series/README.md, tests/test_musa_sync.py::test_read_pin and the
#    branch notes in README.md / README_CN.md (`git grep -n <old tag>` finds them). Gate:
python tools/musa_sync.py check-series --repo third_party/vllm --round-trip
python tools/musa_sync.py upgrade-plan --history /path/to/vllm   # the new pin: keeps every entry
python tools/musa_sync.py verify                                 # every divergence clean
python -m pytest tests/test_manifest_completeness.py tests/test_manifest.py \
    tests/test_musa_sync.py tests/test_musa_sync_upstream.py
pip install -e . --no-build-isolation -v
tools/musa_verify/verify.sh                                      # on-hardware model smokes (§6)

# 5. commit third_party/PINS, the regenerated series and the files step 4 changed:
git add third_party/PINS vllm_musa/patches/series tests/test_musa_sync.py README.md README_CN.md
git commit -m "build: bump vLLM to <tag>"
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
  passing run reports `0 need attention` and ends with `=== musa_sync verify:
  PASS ===`. The row *status* is not uniform: the `series/` entries and the cat-4a
  tripwires read `clean`, while the cat-5/6 seam probes read `present` (an
  existence probe, not a patch-state check).
- **Series form gate:** `python tools/musa_sync.py check-series [--repo <checkout>]`
  `[--replay | --round-trip]`. Four modes, four costs, and each one needs a
  checkout in a specific state:

  | invocation | needs | what it proves | cost |
  |---|---|---|---|
  | `check-series` | nothing | *shape* only: every entry is a regular file, not the whole-patch CRLF form (measured per line: a lone CR on a structural line, and a CR that is *content* — an added line of a CRLF file — are both accepted by `git apply` and `git am`, so they are not reported), a non-empty slug, a canonical `git format-patch` mailbox (all-zero `From 000…` separator, `From:` author, RFC-2822 `Date:`, `Subject: [PATCH] ` whose slug matches what `git format-patch` would name it — RFC-2047 decoding, `.`-run collapsing and the 52-character cap included), an `index` line **when the entry has a hunk**, and a diff `git apply --stat` can parse; numbering is unique and contiguous from `0001` in each directory; no two entries share a diff body; a cherry-pick sits only in `upstream/`, which starts with one | <1 s |
  | `… --repo <checkout>` | a checkout of the pin | the above, plus every `index` anchor resolves to a **blob** there, or is a preimage an *earlier* entry declares as its postimage. The checkout is validated first, so a typo'd path, `/dev/null`, a file or an empty repository is a `repo-unusable` row even for a series that declares no preimage at all | <1 s |
  | `… --repo <checkout> --replay` | a checkout **at the pin** | the above, plus what only a replay can settle: `git am -3` of the whole series in a **disposable clone** (first entry git refuses, with git's own error line → `replay-failed`; an entry that applies *nothing* → `replay-noop`, the signature of a checkout that already holds the series); every `(path, postimage)` an entry declares must be the blob **at that path in the commit that entry created** (`exemption-unearned` — an invented id, a blob a *later* entry produces, or the pin's own copy of the file, is not what this entry produced); and the series also applies the way the **build** applies it, sequentially with `git apply --recount -p1` (`build-path-conflict` — `git am -3` rescues a stale hunk with a 3-way merge, `build_apply.py` does not) | ~19 s per 171 entries |
  | `… --repo <checkout> --round-trip` | a checkout that **holds** the series (`rebase` first) | the fixed point, non-tautologically: it runs exactly what `cmd_regen` runs (`git format-patch --no-signature --no-numbered --zero-commit` from `VLLM_COMMIT`/`VLLM_TAG` plus the canonical author rewrite) **in the checkout you passed**, and rows any entry whose bytes (`round-trip-dirty`) or name (`regen` names this entry …) it would rewrite, or any count mismatch (`round-trip-count`, one row that says the checkout does not hold *this* series). Whether the checkout holds the series is decided by pairing the entries **position for position** — entry *i* of the series against entry *i* of `regen`'s output, at least half of them agreeing on the subject slug — and a checkout that fails that is `round-trip-unverifiable`: one row, rather than a pass and rather than a per-entry cascade of "`regen` names this entry …" rows. The positional form matters in both directions: a set intersection would pair a foreign history that merely reuses half the slugs (measured: a `round-trip-dirty` row for every entry of a state), and would refuse a series whose subjects repeat (measured: 4 entries sharing one slug read as a single-slug set) | <1 s |

  `--replay` and `--round-trip` cannot be combined: they need contradictory
  checkout states (one before the series is applied, one after), and asking for
  both is a usage error (exit 2).

  The default mode is the cheap half, and it is deliberately *not* the whole
  story: it cannot know whether a hunk still applies, and it cannot know whether
  `regen` would leave an entry byte-for-byte alone (an extra `Signed-off-by`, a
  deleted `---` diffstat or a changed author ident all pass it). Where it must
  defer it says so in this document: an anchor "exempted" by an earlier entry's
  declaration is only proven by `--replay`, and the regen fixed point is only
  proven by `--round-trip` against a checkout that holds the series.
  `--repo` hands it an existing checkout instead of a fresh clone, and that checkout must
  be **at the pin and unpatched**: the cat-4a/4b probes compare its own files, so a checkout
  that already holds the series reports 35 divergences (`1 clean / 210 total`). Reset it with
  `git -C <repo> reset --hard <pin>` before probing.
  It fails closed on a missing or empty `series/`, on a symlink/directory/
  chmod-000/fifo/0-byte entry, and on an unusable `--repo` (validated before it
  is consulted, so a typo'd path cannot pass a series that declares no anchor); a
  red series makes `verify` exit 1 even when every divergence is clean. A usage error prints
  `=== musa_sync check-series: FAIL (usage) ===`.
  [Patch series gates](../.github/workflows/patch-series.yml) runs `check-series`,
  `check-series --replay`, `verify`, and `check-series --round-trip` in its
  `offline-series` job when a change matches the workflow's paths. Replay and
  verification use a pristine checkout at `VLLM_COMMIT`. The round-trip check
  uses a separate checkout with the full series applied.
- **On-hardware:** `tools/musa_verify/verify.sh` (configured via env vars
  `MUSA_HOST`, `MUSA_CONTAINER`, `MUSA_VENV`, … — never commit real values) runs the
  patch unit tests plus one functional server smoke per model, each pinned to its
  own MUSA device. `tools/musa_verify/unit_tests.sh` runs `tests/test_patches.py`.

### Exit codes and verdicts

Every `musa_sync` **gate** subcommand (`check-series`, `verify`, `regen`,
`rebase`, `upgrade-plan`, `backport`) that has something to report ends with one explicit verdict line, and
the process verdict is unambiguous (`regen --area module` is a mode of `regen`,
not a subcommand). Three paths print only an `ERROR:` line — a usage/config error
(exit 2), a missing `git` on `PATH` (exit 1) and `regen`'s internal checks on the
patches it generated (exit 1) — so a caller parsing output must read the exit
code. The two reporting commands do not print a verdict at all: `report` prints
the manifest census and `apply` prints `--- N applied, … ---`:

| code | meaning |
|---|---|
| 0 | `=== musa_sync <cmd>: PASS ===` — nothing needs attention |
| 1 | `=== musa_sync <cmd>: FAIL (<counters>) ===` — see the rows above; `verify` prints the series-format section *before* the divergence summary so the last line always agrees with the exit code |
| 2 | usage/config error, not a verdict: no resolvable target (`verify`, `regen`, `rebase` with no `--target`/`VLLM_COMMIT`/`VLLM_TAG`), a target the `--repo` checkout does not contain, `--replay`/`--round-trip` without `--repo`, or both at once |

## 7. Command reference

`python tools/musa_sync.py <cmd>`:

| cmd | what it does |
|---|---|
| `apply` | build-time: `git apply` the series to a cloned vLLM |
| `verify` | offline pre-bump gate: status of every divergence (+ the series-format gate) |
| `check-series [--repo PATH] [--replay \| --round-trip]` | gate the series' generation form: every entry must be a canonical `git am` mailbox with canonical numbering; `--repo` also resolves the `index` anchors against that checkout (at the pin), `--replay` also replays the series and probes the build path there, `--round-trip` instead compares the series with what `regen` writes from a checkout that holds it (see §6 for the state each mode needs) |
| `rebase [<ref>] [--drop-upstreamed --history PATH]` | `git am -3` the series onto `vllm@<ref>` (default: the pin), MUSA entries first and `upstream/` last (sets up commits for `regen`); `--drop-upstreamed` leaves out the entries `upgrade-plan` drops |
| `upgrade-plan [<ref>] [--history PATH] [--fetch] [--all] [--no-content]` | `keep` / `drop` / `review` for every entry on a bump to `<ref>` (default: the current pin); exits 1 while an entry needs review (§5) |
| `backport <sha> [--from PATH] [--no-regen]` | cherry-pick an upstream vLLM commit (`-x`) onto the series and regenerate it as the next `upstream/` entry (§4a) |
| `regen` | regenerate `series/` and `series/upstream/` from the clone's commits (`git format-patch`) |
| `report` | print the manifest census, upstream entries included |

`make -f Makefile.sync <target>` (thin wrapper, prefers `VLLM_COMMIT` from `third_party/PINS`):

| target | what it does |
|---|---|
| `checkout` | clone + checkout the pinned upstream vLLM |
| `apply-patches` | `git am -3` the series (→ `musa_sync rebase`) |
| `format-patches` | regenerate the series (→ `musa_sync regen`) |
| `clean` | reset `third_party/vllm` for a clean re-apply |
