# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf>=1.24", "pillow>=10"]
# ///
"""Generate sample PDFs for exercising ocrpdf.py.

    uv run --script tests/make_samples.py OUTDIR

Writes OUTDIR/digital.pdf (born-digital) and OUTDIR/scanned.pdf (the same
pages rasterised to images only, with the same bookmarks). Between them they
cover: a chapter heading, paragraphs with a first-line indent, a picture with
a caption, a bullet list, a paragraph split across a page break, two columns,
a vector chart, superscripts/subscripts, a footnote, a plain-text table with
a caption, monospaced code, cross-references, running heads, page numbers,
bookmarks and an index.
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
        "<h1 style='font-size:28px'>Chapter 1</h1>"
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

    # Page 3: super/subscripts and a footnote, a text table, code, references.
    p = doc.new_page()
    header(p, 3)
    p.insert_htmlbox(
        pymupdf.Rect(72, 60, 540, 230),
        "<h2 style='font-size:18px'>Results</h2>"
        "<p style='font-size:11px'>Water is H<sub>2</sub>O and the yield was 10<sup>3</sup> per litre. "
        "As shown in figure 1 and table 1, the results hold.<sup>1</sup> The rest of this paragraph is "
        "long enough to wrap onto a second line of text in the box.</p>",
    )
    for y, cells in zip(range(260, 340, 16), [("Item", "Count", "Share"), ("Apples", "12", "40%"),
                                              ("Pears", "9", "30%"), ("Plums", "6", "20%"), ("Figs", "3", "10%")]):
        for x, cell in zip((90, 300, 420), cells):
            p.insert_text((x, y), cell, fontsize=10)
    p.insert_text((72, 360), "Table 1: Fruit counts from the orchard.", fontsize=10)
    p.insert_text((72, 400), "def answer():", fontsize=10, fontname="cour")
    p.insert_text((72 + 24, 414), "return 42", fontsize=10, fontname="cour")
    p.insert_htmlbox(
        pymupdf.Rect(72, 440, 540, 520),
        "<p style='font-size:11px'>See chapter 1 for the background to all of this.</p>",
    )
    p.insert_htmlbox(
        pymupdf.Rect(72, 700, 540, 740),
        "<p style='font-size:8px'><sup>1</sup> A footnote about the results.</p>",
    )

    # Page 4: an index, which is dropped by default.
    p = doc.new_page()
    header(p, 4)
    p.insert_htmlbox(
        pymupdf.Rect(72, 60, 540, 400),
        "<h1 style='font-size:28px'>Index</h1>"
        + "".join(f"<p style='font-size:11px'>{e}</p>" for e in ("apples, 3", "foxes, 1, 2", "pears, 3", "rivers, 1")),
    )

    toc = [[1, "Chapter 1", 1], [2, "Two Columns", 2], [2, "Results", 3], [1, "Index", 4]]
    doc.set_toc(toc)
    doc.subset_fonts()
    doc.save(outdir / "digital.pdf", garbage=4, deflate=True)

    # Scanned copy: each page becomes a single greyscale image.
    scan = pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
        sp = scan.new_page(width=page.rect.width, height=page.rect.height)
        sp.insert_image(sp.rect, stream=pix.tobytes("jpeg", jpg_quality=75))
    scan.set_toc(toc)
    scan.save(outdir / "scanned.pdf", garbage=4, deflate=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
