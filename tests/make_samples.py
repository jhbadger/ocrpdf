# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf>=1.24", "pillow>=10"]
# ///
"""Generate sample PDFs for exercising ocrpdf.py.

    uv run --script tests/make_samples.py OUTDIR

Writes OUTDIR/digital.pdf (born-digital text + an embedded image + a vector
chart) and OUTDIR/scanned.pdf (the same pages rasterised to images only).
"""

import io
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

LOREM = (
    "The quick brown fox jumps over the lazy dog while the committee deliberates "
    "on matters of considerable importance. Every afternoon the river carried "
    "small boats past the old mill, and the miller counted them carefully. "
)


def picture() -> bytes:
    im = Image.new("RGB", (600, 400), "white")
    d = ImageDraw.Draw(im)
    for i in range(0, 400, 8):
        d.line([(0, i), (600, 400 - i)], fill=(30 + i // 3, 90, 200 - i // 3), width=4)
    d.ellipse([200, 100, 400, 300], fill=(220, 120, 40), outline="black", width=5)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def header(page, n):
    page.insert_text((72, 40), "A SAMPLE BOOK", fontsize=9)
    page.insert_text((300, 770), str(n), fontsize=10)


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()

    # Page 1: chapter heading, paragraphs, a picture, a list.
    p = doc.new_page()
    header(p, 1)
    p.insert_htmlbox(
        pymupdf.Rect(72, 60, 540, 330),
        "<h1 style='font-size:28px'>Chapter One</h1>"
        f"<p style='font-size:11px'>{LOREM * 3}</p>"
        f"<p style='font-size:11px;text-indent:20px'>{LOREM * 2}</p>",
    )
    p.insert_image(pymupdf.Rect(156, 340, 456, 540), stream=picture())
    p.insert_htmlbox(
        pymupdf.Rect(72, 555, 540, 750),
        "<p style='font-size:11px'>Figure 1: A picture of a sunset over lines.</p>"
        "<ul style='font-size:11px'><li>First point about foxes.</li>"
        "<li>Second point about dogs.</li></ul>"
        f"<p style='font-size:11px'>{LOREM * 2} The paragraph then carries on over the page break and into the next "
        "page where it continues without",
    )

    # Page 2: continued paragraph, a heading, then two columns.
    p = doc.new_page()
    header(p, 2)
    p.insert_htmlbox(
        pymupdf.Rect(72, 60, 540, 160),
        f"<p style='font-size:11px'>interruption until it finally ends here. {LOREM}</p>"
        "<h2 style='font-size:18px'>Two Columns</h2>",
    )
    left = "".join(f"<p style='font-size:11px'>Left column paragraph {i}. {LOREM}</p>" for i in range(1, 4))
    right = "".join(f"<p style='font-size:11px'>Right column paragraph {i}. {LOREM}</p>" for i in range(1, 4))
    p.insert_htmlbox(pymupdf.Rect(72, 170, 296, 560), left)
    p.insert_htmlbox(pymupdf.Rect(316, 170, 540, 560), right)
    # A vector bar chart.
    for i, hgt in enumerate([60, 120, 90, 150, 40]):
        p.draw_rect(pymupdf.Rect(200 + i * 45, 740 - hgt, 230 + i * 45, 740), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    p.draw_line((190, 740), (430, 740), color=(0, 0, 0), width=1.5)
    p.draw_line((190, 740), (190, 580), color=(0, 0, 0), width=1.5)

    doc.save(outdir / "digital.pdf")

    # Scanned copy: each page becomes a single greyscale image.
    scan = pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
        sp = scan.new_page(width=page.rect.width, height=page.rect.height)
        sp.insert_image(sp.rect, stream=pix.tobytes("png"))
    scan.save(outdir / "scanned.pdf")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
