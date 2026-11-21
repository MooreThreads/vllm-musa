#!/usr/bin/env python3
"""Build a static recipes website from the vLLM-MUSA cookbook markdown.

The cookbook source of truth is ``docs/cookbook/**/*.md`` in the vllm-musa
repository.  Every recipe is a plain markdown file with a ``# Title`` and a
``## At a glance`` table; this generator turns that tree into a
recipes.vllm.ai-style static site:

    index.html                          catalog: search + filters + cards
    <vendor>/index.html                 per-vendor listing
    <vendor>/<slug>.html                recipe page (markdown + at-a-glance + TOC)
    getting-started.html                the cookbook README
    404.html
    api/recipes.json                    machine-readable catalog
    api/recipes/<vendor>/<slug>.json    one record per recipe
    api/recipes.schema.json             JSON Schema for the catalog

No Node, no bundler, no CDN: the output is plain HTML/CSS/JS that can be
opened with ``file://``, served by any static server, or published to GitHub
Pages / GitLab Pages / nginx.

Usage
-----
    # from a checked-out worktree
    ./build_site.py --source ../vllm-musa-v0280-cookbook/docs/cookbook --out site

    # straight from a git ref (the cookbook lives on feature branches)
    ./build_site.py --git-repo ../vllm-musa --git-ref xd/nemotron35-mtp6-cookbook \
                    --out site

Requirements: python3.10+, ``markdown`` (3.x) and ``pygments`` (2.x).
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import markdown
    from markdown.extensions.codehilite import CodeHiliteExtension
except ImportError:  # pragma: no cover - dependency guard
    sys.exit(
        "error: the 'markdown' package is required (pip install markdown pygments)"
    )
try:
    import pygments  # noqa: F401  (codehilite needs it at render time)
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("error: the 'pygments' package is required (pip install pygments)")

HERE = Path(__file__).resolve().parent
ASSETS_SRC = HERE / "assets"

# ---------------------------------------------------------------------------
# Settings that a fork is likely to touch
# ---------------------------------------------------------------------------

DEFAULT_SITE_NAME = "vLLM-MUSA Recipes"
DEFAULT_TAGLINE = (
    "Pick a model, get a vllm serve line that runs on Moore Threads S5000."
)
DEFAULT_REPO_URL = "https://github.com/MooreThreads/vllm-musa"
DEFAULT_REF = "main"
DEFAULT_HARDWARE = "Moore Threads S5000 (MUSA)"

VENDOR_LABELS = {
    "qwen": "Qwen",
    "deepseek": "DeepSeek",
    "glm": "GLM",
    "nemotron": "NVIDIA Nemotron",
    "llama": "Llama",
    "minimax": "MiniMax",
    "mimo": "MiMo",
    "internlm": "InternLM",
    "kimi": "Kimi",
    "ocr": "OCR",
    "diffusiongemma": "DiffusionGemma",
}

VENDOR_BLURB = {
    "qwen": "Dense, MoE and vision-language Qwen checkpoints on one to eight S5000 GPUs.",
    "deepseek": "MLA/MoE DeepSeek deployments, including multi-node and PD-disaggregated profiles.",
    "glm": "GLM serving topologies with PP2, full-decode graphs and thinking-mode templates.",
    "nemotron": "Hybrid Mamba-2/attention Nemotron profiles tuned for per-token decode latency.",
    "ocr": "Document, chart and scene-text extraction models served through vLLM.",
    "diffusiongemma": "Block-diffusion text generation: parallel denoising instead of one token at a time.",
}

# Sections whose first code block is the command the "Copy launch command"
# button should pick up.
LAUNCH_RE = re.compile(r"\bvllm\s+serve\b|\bvllm_musa\b|\bdocker\s+run\b")

PRIMARY_SECTION_HINTS = (
    "launching the server",
    "environment",
    "recommended graph profile",
    "single node",
    "prefill node",
)

ALERT_RE = re.compile(
    r"<blockquote>\s*<p>\[!(TIP|NOTE|IMPORTANT|WARNING|CAUTION)\]\s*(.*?)</p>\s*</blockquote>",
    re.S,
)
FENCE_RE = re.compile(r"^```([\w+-]*)\s*$")
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)
NESTED_GLANCE_RE = re.compile(r"^#{3,6}\s+at a glance\s*$", re.I | re.M)
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
MULTIMODAL_RE = re.compile(
    r"\bvl\b|\bvision\b|multimodal|video|audio|\bimage\b|\bocr\b", re.I
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Section:
    title: str
    body: str


@dataclass
class Recipe:
    vendor: str
    slug: str
    title: str
    source_path: str  # relative to the cookbook root, e.g. qwen/qwen3-8b-fp8.md
    overview: str = ""
    overview_md: str = ""
    at_a_glance: list[tuple[str, str]] = field(default_factory=list)
    body_md: str = ""
    body_html: str = ""
    toc: list[dict] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    primary_block: int = -1
    lang_sequence: list[str] = field(default_factory=list)

    # derived facets -------------------------------------------------------
    @property
    def glance(self) -> dict[str, str]:
        return {k: v for k, v in self.at_a_glance}

    @property
    def identifier(self) -> str:
        return f"{self.vendor}/{self.slug}"

    @property
    def page_path(self) -> str:
        return f"{self.vendor}/{self.slug}.html"

    def facet(self, *names: str) -> str:
        glance = self.glance
        for name in names:
            for key, value in glance.items():
                if key.lower() == name.lower():
                    return value
        for name in names:
            for key, value in glance.items():
                if key.lower().startswith(name.lower()):
                    return value
        return ""

    def facets(self) -> dict[str, str]:
        return recipe_facets(self)


def plain(text: str) -> str:
    """Strip the inline markdown that survives into facet values."""
    text = MD_LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", plain(text).lower()).strip("-")
    return slug or "section"


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


def split_sections(text: str) -> list[Section]:
    """Split markdown on H2 headings, ignoring ``#`` lines inside fences."""
    sections: list[Section] = []
    title, buf = "", []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        if not in_fence and H2_RE.match(line):
            sections.append(Section(title, "\n".join(buf)))
            title, buf = H2_RE.match(line).group(1).strip(), []
            continue
        buf.append(line)
    sections.append(Section(title, "\n".join(buf)))
    return sections


def parse_table(lines: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or all(set(c) <= set("-: ") for c in cells):
            continue
        if cells[0].lower() in {"setting", "field", "option"}:
            continue
        rows.append((plain(cells[0]), cells[1].strip()))
    return rows


def first_paragraph(md: str) -> str:
    buf: list[str] = []
    in_fence = False
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        if not in_fence and not stripped:
            if buf:
                break
            continue
        if not in_fence and stripped.startswith(("#", ">", "|", "-", "*", "!")):
            if buf:
                break
            continue
        buf.append(stripped)
    return plain(" ".join(buf))


def fence_languages(md: str) -> list[str]:
    langs, in_fence = [], False
    for line in md.splitlines():
        m = FENCE_RE.match(line.strip())
        if not m and not line.strip().startswith("```"):
            continue
        if not in_fence:
            langs.append(m.group(1) or "text")
            in_fence = True
        else:
            in_fence = False
    return langs


def fence_blocks(md: str) -> list[tuple[str, str]]:
    """(language, content) for every fenced block, in document order."""
    blocks: list[tuple[str, str]] = []
    lang, buf, in_fence = "", [], False
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                blocks.append((lang, "\n".join(buf)))
                lang, buf, in_fence = "", [], False
            else:
                match = FENCE_RE.match(stripped)
                lang, in_fence = (match.group(1) if match else "") or "text", True
            continue
        if in_fence:
            buf.append(line)
    if in_fence:
        blocks.append((lang, "\n".join(buf)))
    return blocks


def primary_block_index(sections: list[Section]) -> int:
    """Index of the code block that carries the launch command.

    Content beats heading text: a block that actually starts a server
    (``vllm serve``, ``docker run``) wins over one that only exports variables,
    even when the exports sit under a hint section. Without that ordering,
    ``## Environment`` (exports only) shadowed GLM-5.3's serve line under
    ``## Recommended graph profile`` and the copy button handed out exports.
    """
    ranked: list[tuple[int, int, int, int]] = []  # (launches, lang, hint, order)
    order = 0
    for section in sections:
        title = section.title.lower()
        hint = next(
            (i for i, needle in enumerate(PRIMARY_SECTION_HINTS) if needle in title),
            len(PRIMARY_SECTION_HINTS) + 1 if not title else len(PRIMARY_SECTION_HINTS),
        )
        for lang, content in fence_blocks(section.body):
            launches = 0 if LAUNCH_RE.search(content) else 1
            shell = 0 if lang in ("bash", "shell", "sh", "console") else 1
            ranked.append((launches, shell, hint, order))
            order += 1
    if not ranked:
        return -1
    return min(ranked)[3]


def parse_recipe(root: Path, path: Path) -> Recipe:
    raw = path.read_text(encoding="utf-8")
    rel = path.relative_to(root).as_posix()
    vendor = rel.split("/")[0] if "/" in rel else "general"
    slug = path.stem

    title = slug
    body = raw
    m = H1_RE.search(raw)
    if m:
        title = plain(m.group(1))
        body = raw[m.end() :]

    sections = split_sections(body)
    glance: list[tuple[str, str]] = []
    overview = ""
    overview_md = ""
    for section in sections:
        key = section.title.lower()
        if key == "at a glance" and not glance:
            glance = parse_table(section.body.splitlines())
        elif key == "overview" and not overview_md:
            overview_md = section.body.strip()
            overview = first_paragraph(section.body)
    if not glance:
        # Multi-profile recipes put "At a glance" under each profile as an H3.
        nested = NESTED_GLANCE_RE.search(body)
        if nested:
            glance = parse_table(body[nested.end() :].splitlines())
    if not overview:
        overview = first_paragraph(body)

    recipe = Recipe(
        vendor=vendor,
        slug=slug,
        title=title,
        source_path=rel,
        overview=overview,
        overview_md=overview_md,
        at_a_glance=glance,
        sections=[s.title for s in sections if s.title],
        primary_block=primary_block_index(sections),
    )

    # Rebuild the body without the H1 and without the "At a glance" table: the
    # recipe page renders that table in the sidebar, not twice.
    parts: list[str] = []
    for section in sections:
        if section.title.lower() in ("at a glance", "overview"):
            continue
        if section.title:
            parts.append(f"## {section.title}")
        if section.body.strip():
            parts.append(section.body.strip())
    recipe.body_md = "\n\n".join(parts) + "\n"
    recipe.lang_sequence = fence_languages(recipe.body_md)
    recipe.body_html, recipe.toc = render_markdown(recipe.body_md)
    return recipe


# ---------------------------------------------------------------------------
# Rendering markdown
# ---------------------------------------------------------------------------


def make_markdown() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "attr_list",
            "sane_lists",
            "md_in_html",
            CodeHiliteExtension(css_class="highlight", guess_lang=False),
            "toc",
        ],
        extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
        output_format="html5",
    )


def render_markdown(text: str) -> tuple[str, list[dict]]:
    md = make_markdown()
    body = md.convert(text)
    return body, md.toc_tokens


def decorate_html(body: str, langs: list[str], primary_block: int) -> str:
    """Post-process converted markdown: alerts, code chrome, primary anchor."""

    def _alert(match: re.Match) -> str:
        kind = match.group(1).lower()
        inner = match.group(2).strip().replace("\n", " ")
        label = {
            "tip": "Tip",
            "note": "Note",
            "important": "Important",
            "warning": "Warning",
            "caution": "Caution",
        }[kind]
        return (
            f'<div class="callout callout-{kind}">'
            f'<span class="callout-title">{label}</span>'
            f'<div class="callout-body">{inner}</div></div>'
        )

    body = ALERT_RE.sub(_alert, body)

    # Re-wrap every highlighted code block with a bar carrying the language
    # and a copy button.  Order of <div class="highlight"> matches fence order.
    blocks = [0]

    def _code(match: re.Match) -> str:
        index = blocks[0]
        blocks[0] += 1
        lang = langs[index] if index < len(langs) else "text"
        anchor = ' id="primary-command"' if index == primary_block else ""
        return (
            f'<div class="codeblock"{anchor}>'
            f'<div class="codeblock-bar"><span class="codeblock-lang">{html.escape(lang)}</span>'
            f'<button class="copy-btn" type="button" data-copy>Copy</button></div>'
            f"{match.group(0)}</div>"
        )

    body = re.sub(
        r'<div class="highlight">.*?</div>\s*(?=<|$)', _code, body, flags=re.S
    )
    return body


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------


def build_timestamp() -> str:
    """Reproducible build stamp: honour SOURCE_DATE_EPOCH when it is set."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and epoch.isdigit():
        return datetime.fromtimestamp(int(epoch), timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def rel_link(from_page: str, to_page: str) -> str:
    start = posixpath.dirname(from_page) or "."
    return posixpath.relpath(to_page, start)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{esc(plain(text))}</span>'


def recipe_badges(recipe: Recipe) -> str:
    out = []
    precision = recipe.facet("Precision")
    if precision:
        out.append(badge(precision, "precision"))
    hardware = recipe.facet("Hardware")
    if hardware:
        # "2 nodes, 8x S5000 per node" is too long for a badge and hides the
        # total, so collapse it to the GPU count it actually means.
        nodes = re.search(r"(\d+)\s*nodes?", hardware, re.I)
        per_node = re.search(r"(\d+)\s*[x\u00d7]\s*([A-Za-z0-9.]+)", hardware)
        if nodes and per_node:
            hardware = (
                f"{int(nodes.group(1)) * int(per_node.group(1))}x {per_node.group(2)}"
            )
        out.append(badge(hardware, "hardware"))
    tp = recipe.facet("Tensor parallelism")
    if tp:
        out.append(badge(tp, "tp"))
    spec = plain(recipe.facet("Speculative decoding", "Candidate profile"))
    if spec and spec.lower() not in ("unknown",):
        out.append(badge(spec, "spec" if not spec.lower().startswith("off") else "off"))
    arch = recipe.facet("Architecture")
    if arch and "moe" in arch.lower():
        out.append(badge("MoE", "arch"))
    if re.search(
        r"\bvl\b|vision|image|audio|omni", recipe.title + " " + recipe.overview, re.I
    ):
        out.append(badge("Multimodal", "arch"))
    return "".join(out)


def card(recipe: Recipe, page: str) -> str:
    url = rel_link(page, recipe.page_path)
    hardware = recipe.facet("Hardware")
    tp = recipe.facet("Tensor parallelism")
    ctx = recipe.facet("Maximum context")
    meta = " · ".join(x for x in (tp, ctx) if x)
    facets = " ".join(
        f'data-{key}="{esc(value)}"' for key, value in recipe.facets().items()
    )
    return f"""<article class="card" {facets} data-vendor="{esc(recipe.vendor)}">
  <a class="card-hit" href="{esc(url)}"></a>
  <header class="card-head">
    <h3>{esc(recipe.title)}</h3>
    <span class="card-vendor">{esc(vendor_label(recipe.vendor))}</span>
  </header>
  <div class="badges">{recipe_badges(recipe)}</div>
  <p class="card-desc">{esc(recipe.overview)}</p>
  <footer class="card-foot">
    <span class="card-meta">{esc(" · ".join(x for x in (hardware, meta) if x))}</span>
    <span class="card-open">Open recipe <span aria-hidden="true">→</span></span>
  </footer>
</article>"""


def vendor_label(key: str) -> str:
    return VENDOR_LABELS.get(key, key.replace("-", " ").title())


def nav_html(page: str, meta: "SiteMeta", vendors: list[str]) -> str:
    items = [
        ("index.html", "Recipes"),
        ("getting-started.html", "Getting started"),
        ("api/recipes.json", "JSON API"),
    ]
    links = []
    for target, label in items:
        active = ' aria-current="page"' if target == page else ""
        links.append(
            f'<a class="nav-link" href="{esc(rel_link(page, target))}"{active}>{label}</a>'
        )
    vendor_items = "".join(
        f'<a role="menuitem" href="{esc(rel_link(page, f"{v}/index.html"))}">'
        f"{esc(vendor_label(v))}</a>"
        for v in vendors
    )
    return f"""<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="{esc(rel_link(page, "index.html"))}">
      <span class="brand-mark" aria-hidden="true">M</span>
      <span class="brand-text">{esc(meta.name)}</span>
    </a>
    <nav class="nav" aria-label="Main">
      {''.join(links)}
      <div class="menu">
        <button class="nav-link menu-button" type="button" aria-haspopup="true" aria-expanded="false">Vendors<span aria-hidden="true">▾</span></button>
        <div class="menu-items" role="menu">{vendor_items}</div>
      </div>
    </nav>
    <div class="topbar-actions">
      <a class="ghost-btn" href="{esc(meta.repo_url)}" target="_blank" rel="noopener">GitHub ↗</a>
      <button class="icon-btn" type="button" id="theme-toggle" aria-label="Toggle color theme">◐</button>
    </div>
  </div>
</header>"""


@dataclass
class SiteMeta:
    name: str
    tagline: str
    repo_url: str
    ref: str
    hardware: str
    version: str
    source_label: str
    generated_at: str
    cookbook_root: str
    commit: str = ""
    embed_sources: bool = False


def shell(
    *,
    page: str,
    title: str,
    body: str,
    meta: SiteMeta,
    vendors: list[str],
    description: str = "",
    body_class: str = "",
) -> str:
    root = rel_link(page, ".")
    root = "" if root == "." else root + "/"
    full_title = title if title == meta.name else f"{title} · {meta.name}"
    desc = description or meta.tagline
    return f"""<!DOCTYPE html>
<html lang="en" data-page="{esc(page)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(full_title)}</title>
<meta name="description" content="{esc(desc)}">
<meta property="og:title" content="{esc(full_title)}">
<meta property="og:description" content="{esc(desc)}">
<link rel="stylesheet" href="{root}assets/site.css">
<link rel="stylesheet" href="{root}assets/pygments-light.css">
<link rel="stylesheet" href="{root}assets/pygments-dark.css">
<script>try{{var q=new URLSearchParams(location.search).get('theme');var t=q==='dark'||q==='light'?q:localStorage.getItem('theme');if(!t){{t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}}document.documentElement.setAttribute('data-theme',t);}}catch(e){{document.documentElement.setAttribute('data-theme','light');}}</script>
</head>
<body class="{esc(body_class)}">
{nav_html(page, meta, vendors)}
{body}
<footer class="footer">
  <div class="footer-inner">
    <p>{esc(meta.name)} — generated {esc(meta.generated_at)} from <code>{esc(meta.source_label)}</code>{f" at <code>{esc(meta.commit[:10])}</code>" if meta.commit else ""}.</p>
    <p>Recipes are starting configurations, not production guarantees. Revalidate image, checkpoint and workload before you ship.</p>
    <p><a href="{esc(meta.repo_url)}" target="_blank" rel="noopener">Source repository ↗</a> · <a href="{esc(rel_link(page, "api/recipes.json"))}">Machine-readable catalog</a></p>
  </div>
</footer>
<script src="{root}assets/site.js"></script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Recipe facets (filters, JSON API)
# ---------------------------------------------------------------------------


def recipe_facets(recipe: Recipe) -> dict[str, str]:
    """Filter values the catalog JavaScript matches against."""
    precision = plain(recipe.facet("Precision"))
    if not precision:
        names = {"FP8": "fp8", "BF16": "bf16", "FP4": "fp4", "INT8": "int8"}
        haystacks = (recipe.slug, recipe.title, recipe.overview)
        for label, needle in names.items():
            if any(needle in hay.lower() for hay in haystacks):
                precision = label
                break
    if not precision:
        precision = "Unspecified"

    arch_text = " ".join(
        (
            recipe.facet("Architecture"),
            recipe.title,
            recipe.overview,
            recipe.body_md,
        )
    ).lower()
    if (
        "moe" in arch_text
        or "expert" in arch_text
        or re.search(r"\ba\d+b\b", recipe.title.lower())
    ):
        arch = "MoE"
    else:
        arch = "Dense"

    spec = plain(recipe.facet("Speculative decoding", "Candidate profile"))
    lowered_spec = spec.lower()
    if not spec:
        spec_key = ""
    elif "off" in lowered_spec:
        spec_key = "Off"
    elif "mtp" in lowered_spec or "eagle" in lowered_spec or "dspark" in lowered_spec:
        spec_key = "MTP / spec"
    else:
        spec_key = spec

    hardware = plain(recipe.facet("Hardware"))
    per_node = re.search(r"(\d+)\s*[x×]", hardware)
    nodes = re.search(r"(\d+)\s*nodes?\b", hardware, re.I)
    gpus = ""
    if per_node:
        gpus = str(int(per_node.group(1)) * (int(nodes.group(1)) if nodes else 1))
    task = (
        "Multimodal"
        if MULTIMODAL_RE.search(
            recipe.title + " " + recipe.overview + " " + recipe.facet("Architecture")
        )
        else "Text"
    )

    return {
        "precision": precision,
        "arch": arch,
        "task": task,
        "spec": spec_key,
        "gpus": gpus,
        "tp": plain(recipe.facet("Tensor parallelism")),
    }


def recipe_record(recipe: Recipe) -> dict:
    return {
        "id": recipe.identifier,
        "title": recipe.title,
        "vendor": recipe.vendor,
        "vendor_label": vendor_label(recipe.vendor),
        "slug": recipe.slug,
        "url": recipe.page_path,
        "source": recipe.source_path,
        "description": recipe.overview,
        "facets": recipe.facets(),
        "at_a_glance": [{"key": k, "value": plain(v)} for k, v in recipe.at_a_glance],
        "sections": recipe.sections,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def render_index(recipes: list[Recipe], meta: SiteMeta, vendors: list[str]) -> str:
    page = "index.html"
    by_vendor: dict[str, int] = {}
    for recipe in recipes:
        by_vendor[recipe.vendor] = by_vendor.get(recipe.vendor, 0) + 1

    def values(key: str) -> list[str]:
        seen = {recipe.facets()[key] for recipe in recipes}
        return sorted(v for v in seen if v)

    def options(key: str) -> str:
        return "".join(
            f'<option value="{esc(v)}">{esc(v)}</option>' for v in values(key)
        )

    vendor_links = "".join(
        f'<a class="vendor-chip" href="{esc(vendor)}/index.html">'
        f'<span class="vendor-chip-name">{esc(vendor_label(vendor))}</span>'
        f'<span class="vendor-chip-count">{by_vendor[vendor]}</span></a>'
        for vendor in vendors
    )

    cards = "\n".join(card(recipe, page) for recipe in recipes)
    body = f"""<main class="wrap">
  <section class="hero">
    <p class="eyebrow">{esc(meta.hardware)}{(' · ' + esc(meta.version)) if meta.version else ''}</p>
    <h1>{esc(meta.name)}</h1>
    <p class="hero-sub">{esc(meta.tagline)}</p>
    <div class="hero-actions">
      <a class="primary-btn" href="#catalog">Browse {len(recipes)} recipes</a>
      <a class="ghost-btn" href="getting-started.html">Before you start</a>
    </div>
    <p class="hero-meta">{len(recipes)} recipes · {len(vendors)} model families · copy-ready <code>vllm serve</code> commands</p>
  </section>

  <section class="vendors">
    <h2 class="section-title">Browse by family</h2>
    <div class="vendor-row">{vendor_links}</div>
  </section>

  <section class="catalog" id="catalog">
    <div class="catalog-head">
      <h2 class="section-title">All recipes</h2>
      <p class="catalog-count" id="catalog-count">{len(recipes)} recipes</p>
    </div>
    <div class="controls">
      <label class="search">
        <span class="search-icon" aria-hidden="true">⌕</span>
        <input id="search" type="search" placeholder="Search model, precision, hardware…" autocomplete="off" spellcheck="false" aria-label="Search recipes">
        <kbd>/</kbd>
      </label>
      <label class="select"><span>Family</span>
        <select id="filter-vendor" data-key="vendor"><option value="">All</option>
          {"".join(f'<option value="{esc(v)}">{esc(vendor_label(v))}</option>' for v in vendors)}
        </select>
      </label>
      <label class="select"><span>Precision</span>
        <select id="filter-precision" data-key="precision"><option value="">All</option>{options("precision")}</select>
      </label>
      <label class="select"><span>Architecture</span>
        <select id="filter-arch" data-key="arch"><option value="">All</option>{options("arch")}</select>
      </label>
      <label class="select"><span>Modality</span>
        <select id="filter-task" data-key="task"><option value="">All</option>{options("task")}</select>
      </label>
      <label class="select"><span>Spec. decoding</span>
        <select id="filter-spec" data-key="spec"><option value="">All</option>{options("spec")}</select>
      </label>
      <button class="ghost-btn reset-btn" type="button" id="reset">Reset</button>
    </div>
    <div class="grid" id="grid">
{cards}
    </div>
    <p class="empty" id="empty" hidden>No recipe matches those filters. Try clearing the search.</p>
  </section>
</main>"""
    return shell(
        page=page,
        title=meta.name,
        body=body,
        meta=meta,
        vendors=vendors,
        description=meta.tagline,
    )


def render_vendor(
    vendor: str, recipes: list[Recipe], meta: SiteMeta, vendors: list[str]
) -> str:
    page = f"{vendor}/index.html"
    cards = "\n".join(card(recipe, page) for recipe in recipes)
    blurb = VENDOR_BLURB.get(vendor, "")
    body = f"""<main class="wrap">
  <nav class="crumbs" aria-label="Breadcrumb">
    <a href="../index.html">Recipes</a><span aria-hidden="true">/</span><span>{esc(vendor_label(vendor))}</span>
  </nav>
  <section class="page-head">
    <h1>{esc(vendor_label(vendor))}</h1>
    <p class="hero-sub">{esc(blurb or f"{len(recipes)} serving recipes for {vendor_label(vendor)} on S5000.")}</p>
    <p class="hero-meta">{len(recipes)} recipe{'s' if len(recipes) != 1 else ''}</p>
  </section>
  <div class="grid">
{cards}
  </div>
</main>"""
    return shell(
        page=page,
        title=vendor_label(vendor),
        body=body,
        meta=meta,
        vendors=vendors,
        description=blurb or f"{vendor_label(vendor)} recipes for vLLM-MUSA on S5000.",
    )


def toc_html(tokens: list[dict]) -> str:
    def walk(items: list[dict]) -> str:
        parts = []
        for item in items:
            children = walk(item.get("children") or [])
            parts.append(
                f'<li><a href="#{esc(item["id"])}">{esc(plain(item["name"]))}</a>{children}</li>'
            )
        return f"<ul>{''.join(parts)}</ul>" if parts else ""

    return walk(tokens) if tokens else ""


def glance_html(recipe: Recipe) -> str:
    if not recipe.at_a_glance:
        return ""
    rows = "".join(
        f'<div class="spec"><dt>{esc(k)}</dt><dd>{inline_md(v)}</dd></div>'
        for k, v in recipe.at_a_glance
    )
    return f'<div class="specs"><dl>{rows}</dl></div>'


def inline_md(text: str) -> str:
    """Render inline markdown for table values (code spans, emphasis)."""
    out = esc(text)
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


def render_recipe(
    recipe: Recipe,
    recipes: list[Recipe],
    meta: SiteMeta,
    vendors: list[str],
) -> str:
    page = recipe.page_path
    siblings = [r for r in recipes if r.vendor == recipe.vendor]
    index = siblings.index(recipe)
    prev_recipe = siblings[index - 1] if index > 0 else None
    next_recipe = siblings[index + 1] if index + 1 < len(siblings) else None

    def pager(recipe_: Recipe | None, direction: str) -> str:
        if recipe_ is None:
            return ""
        arrow = "←" if direction == "prev" else "→"
        cls = "pager-prev" if direction == "prev" else "pager-next"
        return (
            f'<a class="pager {cls}" href="{esc(rel_link(page, recipe_.page_path))}">'
            f'<span class="pager-dir">{arrow} {direction.title()}</span>'
            f'<span class="pager-title">{esc(recipe_.title)}</span></a>'
        )

    if meta.embed_sources:
        source_url = rel_link(page, f"src/{recipe.source_path}")
    else:
        source_url = (
            f"{meta.repo_url}/blob/{meta.ref}/docs/cookbook/{recipe.source_path}"
        )
    json_url = rel_link(page, f"api/recipes/{recipe.vendor}/{recipe.slug}.json")
    has_command = recipe.primary_block >= 0
    body_html = decorate_html(
        recipe.body_html, recipe.lang_sequence, recipe.primary_block
    )
    body_html = rewrite_links(body_html, page, recipe.source_path, recipes)
    intro_html, _ = render_markdown(recipe.overview_md or recipe.overview)
    intro_html = decorate_html(intro_html, [], -1)

    body = f"""<main class="wrap recipe-layout">
  <nav class="crumbs" aria-label="Breadcrumb">
    <a href="{esc(rel_link(page, "index.html"))}">Recipes</a><span aria-hidden="true">/</span>
    <a href="{esc(rel_link(page, f"{recipe.vendor}/index.html"))}">{esc(vendor_label(recipe.vendor))}</a><span aria-hidden="true">/</span>
    <span>{esc(recipe.title)}</span>
  </nav>
  <div class="recipe-grid">
    <article class="recipe">
      <header class="recipe-head">
        <h1>{esc(recipe.title)}</h1>
        <div class="hero-sub intro">{intro_html}</div>
        <div class="badges">{recipe_badges(recipe)}</div>
      </header>
      <div class="prose">
{body_html}
      </div>
      <nav class="pager-row" aria-label="Recipe navigation">
        {pager(prev_recipe, "prev")}
        {pager(next_recipe, "next")}
      </nav>
    </article>
    <aside class="aside">
      <div class="aside-card">
        <h2 class="aside-title">At a glance</h2>
        {glance_html(recipe)}
      </div>
      {f'<button class="primary-btn copy-command" type="button" id="copy-command">Copy launch command</button>' if has_command else ''}
      <div class="aside-card">
        <h2 class="aside-title">On this page</h2>
        <nav class="toc">{toc_html(recipe.toc) or "<p class='muted'>No sections</p>"}</nav>
      </div>
      <div class="aside-links">
        <a href="{esc(source_url)}" target="_blank" rel="noopener">View markdown source ↗</a>
        <a href="{esc(json_url)}">Recipe JSON</a>
        <a href="{esc(rel_link(page, "index.html"))}#catalog">All recipes</a>
      </div>
    </aside>
  </div>
</main>"""
    return shell(
        page=page,
        title=recipe.title,
        body=body,
        meta=meta,
        vendors=vendors,
        description=recipe.overview[:180],
        body_class="recipe-page",
    )


def render_readme(
    readme_md: str,
    recipes: list[Recipe],
    meta: SiteMeta,
    vendors: list[str],
) -> str:
    page = "getting-started.html"
    readme_body = re.sub(r"^#\s+.*?\n", "", readme_md, count=1)
    body_html, toc = render_markdown(readme_body)
    body_html = decorate_html(body_html, fence_languages(readme_body), -1)
    body_html = rewrite_links(body_html, page, "README.md", recipes)
    by_vendor = {}
    for recipe in recipes:
        by_vendor.setdefault(recipe.vendor, []).append(recipe)
    listings = "".join(
        f'<div class="start-vendor"><h3>{esc(vendor_label(v))}</h3><ul>'
        + "".join(
            f'<li><a href="{esc(recipe.page_path)}">{esc(recipe.title)}</a>'
            f'<span class="muted">{esc(recipe.facet("Hardware"))}</span></li>'
            for recipe in by_vendor[v]
        )
        + "</ul></div>"
        for v in vendors
        if v in by_vendor
    )
    body = f"""<main class="wrap recipe-layout">
  <nav class="crumbs" aria-label="Breadcrumb">
    <a href="index.html">Recipes</a><span aria-hidden="true">/</span><span>Getting started</span>
  </nav>
  <div class="recipe-grid">
    <article class="recipe">
      <header class="recipe-head">
        <h1>Getting started</h1>
        <p class="hero-sub">How to read a recipe, pick the image, and validate a server before you trust a profile.</p>
      </header>
      <div class="prose">
{body_html}
      </div>
      <section class="start-index">
        <h2>All recipes by family</h2>
        <div class="start-vendors">{listings}</div>
      </section>
    </article>
    <aside class="aside">
      <div class="aside-card">
        <h2 class="aside-title">On this page</h2>
        <nav class="toc">{toc_html(toc) or "<p class='muted'>No sections</p>"}</nav>
      </div>
      <div class="aside-links">
        <a href="index.html#catalog">Browse the catalog</a>
        <a href="api/recipes.json">JSON catalog</a>
      </div>
    </aside>
  </div>
</main>"""
    return shell(
        page=page,
        title="Getting started",
        body=body,
        meta=meta,
        vendors=vendors,
        description="How to read a vLLM-MUSA serving recipe and validate a deployment on S5000.",
    )


def rewrite_links(
    body_html: str, page: str, source_path: str, recipes: list[Recipe]
) -> str:
    """Point ``*.md`` links at generated pages; fall back to the repo blob URL."""
    by_source = {recipe.source_path: recipe.page_path for recipe in recipes}
    by_source["README.md"] = "getting-started.html"
    for vendor in {recipe.vendor for recipe in recipes}:
        by_source[f"{vendor}/README.md"] = f"{vendor}/index.html"

    def _repl(match: re.Match) -> str:
        href = match.group(1)
        if re.match(r"^[a-z]+:", href) or href.startswith("#"):
            return match.group(0)
        path, _, anchor = href.partition("#")
        if not path.endswith(".md"):
            return match.group(0)
        target = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_path), path)
        )
        anchor = f"#{anchor}" if anchor else ""
        if target in by_source:
            return f'href="{esc(rel_link(page, by_source[target]))}{anchor}"'
        return match.group(0)

    return re.sub(r'href="([^"]+)"', _repl, body_html)


# ---------------------------------------------------------------------------
# Assets and JSON payloads
# ---------------------------------------------------------------------------


def pygments_css(theme: str, style: str) -> str:
    from pygments.formatters import HtmlFormatter

    css = HtmlFormatter(style=style).get_style_defs(".highlight")
    lines = []
    for line in css.splitlines():
        if line.startswith("."):
            lines.append(f'html[data-theme="{theme}"] {line}')
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def write_assets(out: Path) -> None:
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("site.css", "site.js"):
        src = ASSETS_SRC / name
        if not src.exists():
            sys.exit(f"error: missing asset {src}")
        shutil.copyfile(src, assets / name)
    (assets / "pygments-light.css").write_text(
        pygments_css("light", "friendly"), encoding="utf-8"
    )
    (assets / "pygments-dark.css").write_text(
        pygments_css("dark", "monokai"), encoding="utf-8"
    )


def catalog_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "vLLM-MUSA recipe catalog",
        "type": "object",
        "required": ["generated_at", "count", "recipes"],
        "properties": {
            "generated_at": {"type": "string"},
            "count": {"type": "integer", "minimum": 0},
            "source": {"type": "object"},
            "recipes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "title", "vendor", "url", "facets"],
                    "properties": {
                        "id": {"type": "string", "pattern": "^[a-z0-9-]+/[a-z0-9.-]+$"},
                        "title": {"type": "string"},
                        "vendor": {"type": "string"},
                        "slug": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "description": {"type": "string"},
                        "facets": {
                            "type": "object",
                            "properties": {
                                "precision": {"type": "string"},
                                "arch": {"type": "string"},
                                "spec": {"type": "string"},
                                "gpus": {"type": "string"},
                                "tp": {"type": "string"},
                            },
                        },
                        "at_a_glance": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["key", "value"],
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                        },
                        "sections": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def git_toplevel(path: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def git_commit(repo: Path, ref: str) -> str:
    """Resolve `ref` to a commit id so the output records exactly what built it."""
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def load_from_git(repo: Path, ref: str) -> tuple[Path, str]:
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        sys.exit(f"error: {repo} is not a git repository")
    tmp = Path(tempfile.mkdtemp(prefix="cookbook-site-"))
    result = subprocess.run(
        ["git", "-C", str(repo), "archive", ref, "docs/cookbook"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(
            f"error: git archive {ref}:docs/cookbook failed\n"
            + result.stderr.decode(errors="replace")
        )
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
        tar.extractall(tmp)
    return tmp / "docs" / "cookbook", f"{repo.name}:{ref}"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


CODE_SAMPLE_RE = re.compile(r"<pre>.*?</pre>", re.S)


INDEX_LINK_RE = re.compile(r"\]\(([A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.md)\)")


def structure_problems(root: Path, recipes: list[Recipe]) -> list[tuple[str, str]]:
    """Cookbook problems that still render, but as a page nobody can trust.

    The link check in verify() only sees the generated site; these run before
    anything is written, so a pull request that adds a recipe without an
    overview, a table or a launch block fails while the diff is still open.
    """
    problems: list[tuple[str, str]] = []
    for recipe in recipes:
        if "overview" not in {title.lower() for title in recipe.sections}:
            problems.append((recipe.source_path, "missing a '## Overview' section"))
        if not recipe.at_a_glance:
            problems.append((recipe.source_path, "missing an '## At a glance' table"))
        if recipe.primary_block < 0:
            problems.append(
                (
                    recipe.source_path,
                    "no fenced block launches a server ('vllm serve' or 'docker run')",
                )
            )
        if recipe.slug != recipe.slug.lower() or " " in recipe.slug:
            problems.append((recipe.source_path, "file name must be lowercase ASCII"))

    index = root / "README.md"
    if index.is_file():
        linked = {
            link
            for link in INDEX_LINK_RE.findall(index.read_text(encoding="utf-8"))
            if not link.endswith("/README.md")  # vendor sub-indexes are not recipes
        }
        listed = {recipe.source_path for recipe in recipes}
        for missing in sorted(listed - linked):
            problems.append(("README.md", f"the index does not link to '{missing}'"))
        for dead in sorted(linked - listed):
            problems.append(
                ("README.md", f"the index links to '{dead}', which is not a recipe")
            )
    return problems


def verify(out: Path) -> list[str]:
    problems: list[str] = []
    for page in out.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        # Sample code keeps literal quotes ("src=\"images/x.jpg\""), so a naive
        # attribute scan reports links that no browser would ever request.
        text = CODE_SAMPLE_RE.sub("", text)
        for attr, value in re.findall(r'(href|src)="([^"]+)"', text):
            if re.match(r"^(https?:|mailto:|data:|#)", value):
                continue
            target = (page.parent / value.split("#")[0]).resolve()
            if value.split("#")[0] and not target.exists():
                problems.append(f"{page.relative_to(out)} -> {value}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a static recipes website from the vLLM-MUSA cookbook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source",
        type=Path,
        help="directory holding the cookbook (contains README.md and <vendor>/*.md)",
    )
    source.add_argument(
        "--git-repo",
        type=Path,
        help="repository to read the cookbook from (use with --git-ref)",
    )
    parser.add_argument(
        "--git-ref",
        default="HEAD",
        help="git ref holding docs/cookbook (default: HEAD)",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "site", help="output directory"
    )
    parser.add_argument(
        "--embed-sources",
        action="store_true",
        help="copy the recipe markdown into <out>/src and link source views there "
        "(makes the published tree self-contained)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep the temporary checkout created by --git-repo (debugging)",
    )
    parser.add_argument("--name", default=DEFAULT_SITE_NAME, help="site name")
    parser.add_argument("--tagline", default=DEFAULT_TAGLINE, help="hero tagline")
    parser.add_argument(
        "--repo-url", default=DEFAULT_REPO_URL, help="source repository URL"
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="label for the built ref (default: the git ref, the source commit, or 'main')",
    )
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE, help="hardware label")
    args = parser.parse_args(argv)

    tmp_root = None
    commit = ""
    if args.git_repo:
        repo = args.git_repo.resolve()
        root, _ = load_from_git(repo, args.git_ref)
        commit = git_commit(repo, args.git_ref)
        tmp_root = root.parents[1]
    elif args.source:
        root = args.source.resolve()
    else:  # auto-detect a sibling worktree next to this script
        candidates = [
            Path("../../docs/cookbook"),  # tools/cookbook-site inside the repo
            Path("docs/cookbook"),  # run from the repository root
            Path("../vllm-musa/docs/cookbook"),  # sibling checkout
            Path("../vllm-musa-v0280-cookbook/docs/cookbook"),
        ]
        found = next((c for c in candidates if (HERE / c).resolve().is_dir()), None)
        if not found:
            parser.error("no --source/--git-repo given and no sibling cookbook found")
        root = (HERE / found).resolve()

    # The label ends up in the footer, so keep it repository-relative (a CI
    # runner's absolute path would leak into the published page).
    if args.git_repo:
        ref_hint = args.ref or args.git_ref
        source_label = f"{args.git_repo.resolve().name}:{ref_hint}"
    else:
        repo = git_toplevel(root)
        ref_hint = args.ref or (git_commit(repo, "HEAD")[:12] if repo else DEFAULT_REF)
        if repo and root.is_relative_to(repo):
            source_label = root.relative_to(repo).as_posix()
        else:
            source_label = str(root)
        if repo:
            commit = git_commit(repo, "HEAD")

    if not (root / "README.md").exists():
        sys.exit(f"error: {root} does not look like a cookbook (no README.md)")

    readme_md = (root / "README.md").read_text(encoding="utf-8")
    sources = [
        path for path in sorted(root.glob("*/*.md")) if path.name.lower() != "readme.md"
    ]
    untitled = [
        path for path in sources if not H1_RE.search(path.read_text(encoding="utf-8"))
    ]
    if untitled:
        for path in untitled:
            print(
                f"::error file={path.relative_to(root)}::recipe is missing a '# Title' heading"
            )
        sys.exit(f"error: {len(untitled)} recipe(s) without an H1 title")
    recipes = [parse_recipe(root, path) for path in sources]
    if not recipes:
        sys.exit(f"error: no recipes found under {root}")

    structure = structure_problems(root, recipes)
    if structure:
        prefix = f"{args.source.as_posix()}/" if args.source else ""
        for source, message in structure:
            print(f"::error file={prefix}{source}::{message}")
        sys.exit(f"error: {len(structure)} cookbook structure problem(s)")

    order = {vendor: i for i, vendor in enumerate(VENDOR_LABELS)}
    recipes.sort(key=lambda r: (order.get(r.vendor, 99), r.vendor, r.title.lower()))
    vendors = sorted({r.vendor for r in recipes}, key=lambda v: (order.get(v, 99), v))

    version_match = re.search(r"vLLM-MUSA\s+(v[\w.\-+]+)", readme_md)
    meta = SiteMeta(
        name=args.name,
        tagline=args.tagline,
        repo_url=args.repo_url,
        ref=ref_hint,
        hardware=args.hardware,
        version=version_match.group(1) if version_match else "",
        source_label=source_label,
        generated_at=build_timestamp(),
        cookbook_root=source_label,
        commit=commit,
        embed_sources=args.embed_sources,
    )

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_assets(out)

    # pages -----------------------------------------------------------------
    (out / "index.html").write_text(
        render_index(recipes, meta, vendors), encoding="utf-8"
    )
    (out / "getting-started.html").write_text(
        render_readme(readme_md, recipes, meta, vendors), encoding="utf-8"
    )
    for vendor in vendors:
        vendor_dir = out / vendor
        vendor_dir.mkdir(parents=True, exist_ok=True)
        vendor_recipes = [r for r in recipes if r.vendor == vendor]
        (vendor_dir / "index.html").write_text(
            render_vendor(vendor, vendor_recipes, meta, vendors), encoding="utf-8"
        )
    for recipe in recipes:
        page = out / recipe.page_path
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_recipe(recipe, recipes, meta, vendors), encoding="utf-8")

    # 404 -------------------------------------------------------------------
    not_found = shell(
        page="404.html",
        title="Not found",
        body="""<main class="wrap narrow">
  <h1>404</h1>
  <p class="hero-sub">That recipe is not in this catalog.</p>
  <p><a class="primary-btn" href="index.html">Back to all recipes</a></p>
</main>""",
        meta=meta,
        vendors=vendors,
    )
    (out / "404.html").write_text(not_found, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    # JSON API --------------------------------------------------------------
    api = out / "api"
    api.mkdir(parents=True, exist_ok=True)
    catalog = {
        "generated_at": meta.generated_at,
        "count": len(recipes),
        "source": {
            "repository": meta.repo_url,
            "ref": meta.ref,
            "commit": meta.commit,
            "cookbook_root": meta.cookbook_root,
        },
        "site": {"name": meta.name, "hardware": meta.hardware, "version": meta.version},
        "recipes": [recipe_record(r) for r in recipes],
    }
    (api / "recipes.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (api / "recipes.schema.json").write_text(
        json.dumps(catalog_schema(), indent=2) + "\n", encoding="utf-8"
    )
    for recipe in recipes:
        path = api / "recipes" / recipe.vendor / f"{recipe.slug}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(recipe_record(recipe), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if args.embed_sources:
        for md in root.rglob("*.md"):
            target = out / "src" / md.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(md, target)

    problems = verify(out)
    if tmp_root and not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"built {len(recipes)} recipes from {source_label}")
    print(f"output: {out}")
    print(
        f"pages : {len(list(out.rglob('*.html')))} html, "
        f"{len(list(out.rglob('*.json')))} json"
    )
    if problems:
        # The workflow uploads only when this exits 0, so a cookbook edit that
        # drops a recipe fails the run instead of publishing a dead link.
        print(f"error: {len(problems)} internal link(s) point outside the site:")
        for problem in problems[:20]:
            print(f"  {problem}")
        print(f"::error::{len(problems)} broken internal link(s) in the generated site")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
