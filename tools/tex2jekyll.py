#!/usr/bin/env python3
"""Render latex/bulk-rna/main.tex as a Jekyll post.

The .tex is the source of truth -- it is what people send pull requests
against. The generated markdown is committed too, because GitHub Pages only
runs whitelisted plugins and so cannot convert LaTeX at build time.

Never hand-edit the generated post; edit the .tex and re-run this.

Usage: python3 tools/tex2jekyll.py
"""

import datetime
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEX = REPO / "latex" / "bulk-rna" / "main.tex"

# Jekyll skips posts dated in the future, and GitHub Pages only rebuilds on
# push -- so a future date means the post silently never appears. Keep this
# on or before the day you push.
PUB_DATE = "2026-08-30"
PUB_TIME = "09:00:00 +0200"
SLUG = "bulk-rna-patient-representations"

POST = REPO / "_posts" / f"{PUB_DATE}-{SLUG}.md"

# Kramdown wants $$...$$ for math. Inside raw HTML blocks kramdown does not
# parse content at all, so math there has to use the single-$ delimiters that
# MathJax's tex2jax picks up directly from the DOM (see _includes/katex.html).
DISPLAY, INLINE, HTML_INLINE = "display", "inline", "html-inline"

SYMBOLS = {
    "modelbetter": '<span class="eval-yes">&#10003;</span>',
    "modelnotbetter": '<span class="eval-no">&#215;</span>',
    "modelconditional": '<span class="eval-partial">&#9651;</span>',
    "modeluntested": '<span class="eval-none">&mdash;</span>',
}

FRONT_MATTER = """---
layout: post
title:  "{title}"
date:   {date} {time}
permalink: biology/{slug}
tags: [Deep Learning, Bulk RNA-seq, Transcriptomics, Gene Expression, Computational Biology]
categories: [Biology]
image: /assets/bulk-rna/card.png
no_preview_image: true
excerpt: "{excerpt}"

---
{{% include katex.html %}}

<!-- GENERATED FILE -- DO NOT EDIT.
     Source: latex/bulk-rna/main.tex
     Regenerate with: python3 tools/tex2jekyll.py -->
"""

# _includes/social-metatags.html truncates the description at 152 characters,
# so anything longer gets cut mid-word in the Open Graph card that LinkedIn
# and Twitter render. Keep this under that limit -- see the assert in main().
EXCERPT = (
    "How patients are represented from bulk RNA-seq — from expression "
    "vectors and PCA to pathway graphs and foundation models — and whether "
    "it helps."
)
EXCERPT_LIMIT = 152


def slugify(text):
    text = re.sub(r"<[^>]+>", "", text)
    # Entities would otherwise smuggle their spelling into the id, turning
    # "gene&ndash;pathway" into "genendashpathway".
    text = re.sub(r"&(?:ndash|mdash);", "-", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "-", text).strip("-")


class MathStore:
    """Swap math out for placeholders so text rules cannot corrupt it."""

    def __init__(self):
        self.items = []

    def stash(self, body, kind):
        body = body.strip()
        if kind != DISPLAY:
            # Kramdown only recognises inline math on a single line, and the
            # .tex hard-wraps mid-expression.
            body = re.sub(r"\s+", " ", body)
        # A bare "|" makes kramdown read the whole line as a table row, which
        # swallows the surrounding prose into <td> cells. \vert renders the
        # same and is invisible to the table parser.
        body = re.sub(r"(?<!\\)\|", r"\\vert ", body)
        self.items.append((body, kind))
        return f"\x00MATH{len(self.items) - 1}\x00"

    def restore(self, text):
        def sub(match):
            body, kind = self.items[int(match.group(1))]
            if kind == DISPLAY:
                return f"\n\n$$\n{body}\n$$\n\n"
            if kind == HTML_INLINE:
                return f"${body}$"
            return f"$${body}$$"

        return re.sub(r"\x00MATH(\d+)\x00", sub, text)


def protect_math(text, math, inline_kind=INLINE):
    text = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        lambda m: math.stash(m.group(1), DISPLAY),
        text,
        flags=re.DOTALL,
    )
    return re.sub(
        r"\$([^$]+)\$", lambda m: math.stash(m.group(1), inline_kind), text
    )


def parse_bibliography(tex):
    """Return {cite_key: (number, rendered_html)} in citation-list order."""
    block = re.search(
        r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}",
        tex,
        re.DOTALL,
    ).group(1)

    entries = {}
    for number, chunk in enumerate(re.split(r"\\bibitem\{", block)[1:], start=1):
        key, body = chunk.split("}", 1)
        body = body.replace("\\newblock", " ")
        body = re.sub(r"\\emph\{([^}]*)\}", r"<em>\1</em>", body)
        entries[key] = (number, inline_text(body).strip())
    return entries


def inline_text(text):
    """LaTeX text-mode conventions -> HTML/markdown equivalents."""
    replacements = [
        (r"\\'\{?([aeiouAEIOU])\}?", r"\1"),  # Rodr\'iguez -> Rodriguez
        (r"``", '"'),
        (r"''", '"'),
        (r"---", "&mdash;"),
        (r"--", "&ndash;"),
        (r"\\&", "&amp;"),
        (r"\\%", "%"),
        (r"\\_", "_"),
        (r"~", " "),
        (r"\\ ", " "),
        (r"\\,", " "),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return re.sub(r"[ \t]+", " ", text)


def convert_table(tex, math, bib):
    """The longtable is the most-shared artifact; render it as a real table."""
    block = re.search(
        r"\\begin\{longtable\}.*?\\endfoot(.*?)\\end\{longtable\}", tex, re.DOTALL
    ).group(1)

    rows = []
    for raw in block.split(r"\\"):
        if not raw.strip():
            continue
        cells = []
        for cell in raw.split("&"):
            cell = protect_math(cell, math, inline_kind=HTML_INLINE)
            cell = re.sub(
                r"\\(model\w+)\{?\}?", lambda m: SYMBOLS[m.group(1)], cell
            )
            cell = re.sub(
                r"\\cite\{([^}]*)\}",
                lambda m: cite_html(m.group(1), bib),
                cell,
            )
            cells.append(inline_text(cell).strip())
        if len(cells) == 5:
            rows.append(cells)

    head = ["Study", "Dataset", "Task", "Modeling method", "Direct gain?"]
    html = ['<div class="table-scroll">', '<table id="table-1" class="eval-table">']
    html.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead>")
    html.append("<tbody>")
    for row in rows:
        cls = ' class="eval-cell"'
        tds = "".join(
            f"<td{cls if i == 4 else ''}>{c}</td>" for i, c in enumerate(row)
        )
        html.append(f"<tr>{tds}</tr>")
    html.append("</tbody></table></div>")
    html.append(
        '<p class="table-caption"><strong>Table 1.</strong> Representative '
        "evaluations of bulk-transcriptomic modeling. Results are not directly "
        "comparable across rows.</p>"
    )
    return "\n".join(html), len(rows)


def cite_html(keys, bib):
    out = []
    for key in [k.strip() for k in keys.split(",")]:
        number = bib[key][0]
        out.append(f'<a href="#ref-{key}" class="cite">[{number}]</a>')
    return "".join(out)


def convert_body(tex, math, bib, table_html):
    body = re.search(r"\\endgroup(.*?)\\begin\{thebibliography\}", tex, re.DOTALL)
    text = body.group(1)

    # Drop the table's LaTeX formatting scaffolding and splice in the HTML.
    text = re.sub(
        r"\\begingroup\s*\\footnotesize.*?\\end\{longtable\}\s*\\endgroup",
        "\x00TABLE\x00",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"^\s*\\(setlength|renewcommand|begingroup|endgroup).*$", "", text,
                  flags=re.MULTILINE)

    text = protect_math(text, math)

    for command, level in (
        ("subsubsection", "H3"),
        ("subsection", "H2"),
        ("section", "H1"),
    ):
        text = re.sub(
            r"\\%s\{([^}]*)\}" % command,
            lambda m, lv=level: f"\x00{lv}\x00{m.group(1)}",
            text,
        )

    text = re.sub(
        r"\\begin\{itemize\}(.*?)\\end\{itemize\}",
        lambda m: convert_list(m.group(1)),
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r"Table~\\ref\{tab:bulk-evaluations\}", "[Table 1](#table-1)", text)
    text = re.sub(r"\\cite\{([^}]*)\}", lambda m: cite_html(m.group(1), bib), text)
    text = re.sub(r"\\(model\w+)\{?\}?", lambda m: SYMBOLS[m.group(1)], text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", text)
    text = inline_text(text)

    return unwrap_paragraphs(text).replace("\x00TABLE\x00", table_html)


def convert_list(block):
    items = [re.sub(r"\s+", " ", i).strip() for i in block.split(r"\item") if i.strip()]
    return "\n\n" + "\n".join(f"- {i}" for i in items) + "\n\n"


def unwrap_paragraphs(text):
    """LaTeX hard-wraps at ~78 chars; markdown reads better unwrapped."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith(("-", "\x00TABLE")):
            out.append(para)
        else:
            out.append(re.sub(r"\s*\n\s*", " ", para))
    return "\n\n".join(out)


def apply_headings(text):
    """Emit explicit {#id} anchors so the TOC links cannot drift."""
    toc = []

    def sub(match):
        level, title = match.group(1), match.group(2).strip()
        slug = slugify(title)
        if level == "H1":
            toc.append(f"- [{title}](#{slug})")
        elif level == "H2":
            toc.append(f"  - [{title}](#{slug})")
        hashes = {"H1": "#", "H2": "##", "H3": "###"}[level]
        return f"{hashes} {title} {{#{slug}}}"

    text = re.sub(r"\x00(H[123])\x00([^\n]*)", sub, text)
    return text, "\n".join(toc)


def render_references(bib):
    lines = ['\n# References {#references}\n', '<ol class="references">']
    for key, (_, body) in sorted(bib.items(), key=lambda kv: kv[1][0]):
        lines.append(f'<li id="ref-{key}">{body}</li>')
    lines.append("</ol>")
    return "\n".join(lines)


def main():
    if len(EXCERPT) > EXCERPT_LIMIT:
        raise SystemExit(
            f"EXCERPT is {len(EXCERPT)} chars; social-metatags.html truncates "
            f"at {EXCERPT_LIMIT}, which would cut the link-preview description "
            "mid-word. Shorten it."
        )

    tex = TEX.read_text(encoding="utf-8")
    math = MathStore()

    title = " ".join(
        re.search(r"\\title\{(.*?)\}", tex, re.DOTALL).group(1).split()
    )
    bib = parse_bibliography(tex)
    table_html, n_rows = convert_table(tex, math, bib)

    text = convert_body(tex, math, bib, table_html)
    text, toc = apply_headings(text)
    text = math.restore(text)

    post = (
        FRONT_MATTER.format(
            title=title,
            excerpt=EXCERPT,
            date=PUB_DATE,
            time=PUB_TIME,
            slug=SLUG,
        )
        + "\n"
        + toc
        + "\n- [References](#references)\n\n"
        + text
        + "\n"
        + render_references(bib)
        + "\n"
    )
    post = re.sub(r"\n{3,}", "\n\n", post)
    POST.write_text(post, encoding="utf-8")

    print(f"wrote {POST.relative_to(REPO)}")
    print(f"  title      {title}")
    print(f"  refs       {len(bib)}")
    print(f"  table rows {n_rows}")
    print(f"  equations  {sum(1 for _, k in math.items if k == DISPLAY)}")
    print(f"  words      {len(post.split())}")
    print(f"  excerpt    {len(EXCERPT)}/{EXCERPT_LIMIT} chars")


if __name__ == "__main__":
    main()
