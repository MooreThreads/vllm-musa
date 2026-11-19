# Cookbook site

Turns `docs/cookbook/**/*.md` into a browsable recipe catalog — the
`recipes.vllm.ai` shape (card grid, filters, one page per recipe, "copy this
`vllm serve` line") — without adding a Node toolchain or a front-end build.

```
docs/cookbook/                     build_site.py                _site/
  README.md            ───────▶     parse markdown       ──▶     index.html
  qwen/qwen3-8b-fp8.md              extract metadata             qwen/index.html
  deepseek/…                        render pages                 qwen/qwen3-8b-fp8.html
                                    emit JSON catalog            api/recipes.json
                                                                 src/… (sources)
```

Two Python packages are required: `markdown` and `pygments`.

## Run it locally

From the repository root:

```console
$ python -m pip install 'markdown>=3.3,<4' 'pygments>=2.15'
$ python tools/cookbook-site/build_site.py --source docs/cookbook --out _site
$ python -m http.server -d _site 8000
```

`build_site.py` with no `--source` also finds `docs/cookbook` on its own. To
build a cookbook that only exists on another ref, without checking it out:

```console
$ python tools/cookbook-site/build_site.py \
    --git-repo . --git-ref origin/codex/v0240-cookbook \
    --ref codex/v0240-cookbook --out _site
```

`make site`, `make serve PORT=8000` and `make clean` wrap the same commands.

## What it publishes

```
index.html              catalog: search, family/precision/architecture/modality/spec filters
<vendor>/index.html     one page per model family
<vendor>/<slug>.html    recipe page: overview, at-a-glance sidebar, TOC, copy buttons
getting-started.html    docs/cookbook/README.md with links rewritten to the site
api/recipes.json        the whole catalog (plus api/recipes/<vendor>/<slug>.json)
api/recipes.schema.json JSON Schema for the catalog
src/…                   the source markdown (--embed-sources), so "view source" works
```

Everything is relative-linked, so the same tree works at a domain root, under a
project path (`/vllm-musa/`), or from `file://`. There is no server component:
GitHub Pages serves it as-is.

## Where the metadata comes from

Nothing is hand-maintained; the page structure is the schema.

| Field | Source |
|---|---|
| title, vendor, slug | `# H1`, parent directory, file name |
| summary | first paragraph of `## Overview` |
| category / spec / precision / topology | rows of the `## At a glance` table, with a fallback to the file name, title and overview text |
| GPUs | `Hardware` row, multiplying `N nodes` by `Mx` when both are given |
| copy-this command | first bash block under `## Launching the server`, `## Environment`, `## Single node`, or `## Prefill node` |

A recipe that omits a field simply gets no badge and no filter entry — the build
never guesses a value it cannot read, and it fails loudly on a missing
`# H1` or a non-recipe Markdown file.

## CI

`.github/workflows/cookbook-site.yml` builds on every push to `v0.32.0-dev`
that touches `docs/cookbook/**` or this directory, and on pull requests that do
(build only — the link check still runs). It publishes through
`actions/deploy-pages`, which requires Pages to be configured with
**Source: GitHub Actions**.

`workflow_dispatch` takes an optional `cookbook_ref` input, so a recipe branch
can be previewed on the real site before it is merged:

```console
$ gh workflow run cookbook-site.yml -f cookbook_ref=my/recipe-branch
```

## Scope

This is a presentation layer over the Markdown, not a replacement for
`vllm-project/recipes`. It has no per-hardware verified/supported matrix, no
precision/topology tab combinations, and no interactive command builder; those
need structured data (the upstream `SCHEMA.md` YAML), not prose. The generator
stays compatible with that move: adding a front-matter block or a sibling
`<slug>.yaml` would let the fenced blocks be regenerated from structured flags,
and `api/recipes.json` already exposes the shape consumers would read.
