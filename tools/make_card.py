#!/usr/bin/env python3
"""Draw the link-preview card for the bulk RNA-seq post.

The site-wide og:image is a landscape photo, which in a LinkedIn or Twitter
feed makes a technical review look like a lifestyle post. This renders a
1200x627 card (the 1.91:1 ratio both platforms crop to) using the blog's own
palette, so the preview matches what the reader lands on.

Usage: python3 tools/make_card.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets" / "bulk-rna" / "card.png"

W, H = 1200, 627
MARGIN = 78

# Straight from assets/css/style.scss.
BG = "#ffffff"
PANEL = "#fafbfc"
ACCENT = "#1e6bb8"
TEXT = "#24292e"
MUTED = "#6a737d"
BORDER = "#e1e4e8"

FONTS = Path("/usr/share/fonts/truetype/dejavu")
SERIF_BOLD = FONTS / "DejaVuSerif-Bold.ttf"
SERIF = FONTS / "DejaVuSerif.ttf"
SANS = FONTS / "DejaVuSans.ttf"
SANS_BOLD = FONTS / "DejaVuSans-Bold.ttf"

TITLE = "Patient Representations\nfrom Bulk RNA-seq"
SUBTITLE = "From gene expression to foundation models"
# The post's actual arc, which is also its argument.
CHAIN = ["expression vectors", "PCA", "pathways", "graphs", "foundation models"]
DOMAIN = "jacobgil.github.io"


def font(path, size):
    return ImageFont.truetype(str(path), size)


def width(draw, text, f):
    return draw.textbbox((0, 0), text, font=f)[2]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Accent rule down the left edge, echoing the post's link colour.
    d.rectangle([0, 0, 10, H], fill=ACCENT)

    f_domain = font(SANS_BOLD, 21)
    f_title = font(SERIF_BOLD, 62)
    f_sub = font(SERIF, 31)
    f_chain = font(SANS, 20)

    # Footer strip: the progression the post walks through, as chips.
    strip_top = H - 132

    # Centre the text block in the space above the strip rather than hanging it
    # from the top margin, which leaves a dead band across the middle.
    title_lines = TITLE.split("\n")
    block_h = 62 + 78 * len(title_lines) + 14 + 40
    y = (strip_top - block_h) // 2

    d.text((MARGIN, y), DOMAIN.upper(), font=f_domain, fill=ACCENT)

    y += 62
    for line in title_lines:
        d.text((MARGIN, y), line, font=f_title, fill=TEXT)
        y += 78

    y += 14
    d.text((MARGIN, y), SUBTITLE, font=f_sub, fill=MUTED)
    d.rectangle([0, strip_top, W, H], fill=PANEL)
    d.line([(0, strip_top), (W, strip_top)], fill=BORDER, width=1)

    x = MARGIN
    cy = strip_top + 66
    for i, item in enumerate(CHAIN):
        if i:
            d.text((x, cy - 12), "→", font=font(SANS, 22), fill=ACCENT)
            x += width(d, "→", font(SANS, 22)) + 18

        w = width(d, item, f_chain)
        pad = 13
        d.rounded_rectangle(
            [x - pad, cy - 21, x + w + pad, cy + 21], radius=6, fill="#e8f1fa"
        )
        d.text((x, cy - 12), item, font=f_chain, fill=ACCENT)
        x += w + 18

    img.save(OUT, "PNG", optimize=True)
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO)}  {W}x{H}  {kb:.0f} KB")


if __name__ == "__main__":
    main()
