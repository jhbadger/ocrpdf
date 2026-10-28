"""Tests for ocrpdf.py.

Run them with:

    uv run --script tests/run_tests.py            # check
    uv run --script tests/run_tests.py --update   # rewrite expected outputs

The regression tests convert the small synthetic PDFs in tests/fixtures/ and
compare the Markdown with tests/fixtures/expected/. OCR results come from the
cache in tests/fixtures/ocr-cache/, so the tests never call Vision and run
anywhere; ``--update`` refreshes that cache (needs Vision, i.e. a normal
Terminal, not a sandbox).
"""

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
EXPECTED = FIXTURES / "expected"
UPDATE = os.environ.get("OCRPDF_UPDATE") == "1"

sys.path.insert(0, str(ROOT))
import ocrpdf  # noqa: E402
from ocrpdf import Block, Box, Line  # noqa: E402


@pytest.fixture(autouse=True)
def no_spellchecker(monkeypatch):
    """Use a fixed word list instead of the system spell checker."""
    words = {"example", "data", "driven", "side", "effects", "exam", "ple"}
    monkeypatch.setattr(ocrpdf, "is_word", lambda w: w.lower() in words)


def line(text, x0, y0, x1, y1, **kw):
    return Line(text, 1.0, Box(x0, y0, x1, y1), **kw)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def test_join_text_dehyphenates():
    assert ocrpdf.join_text("an exam-", "ple of it") == "an example of it"


def test_join_text_keeps_real_compounds():
    assert ocrpdf.join_text("the data-", "driven way") == "the data-driven way"


def test_join_text_inside_emphasis():
    assert ocrpdf.join_text("*Myco-*", "*bacterium*") == "*Mycobacterium*"


def test_join_text_plain():
    assert ocrpdf.join_text("one", "two") == "one two"


def test_escaping():
    assert ocrpdf.md_escape("a*b_c [d] <e> x^2 ~y") == r"a\*b\_c \[d\] \<e\> x\^2 \~y"
    assert ocrpdf.escape_block_start("1. not a list") == r"1\. not a list"
    assert ocrpdf.escape_block_start("# not a heading") == r"\# not a heading"


def test_clean_text_ligatures_and_accents():
    assert ocrpdf.clean_text("eﬀect Ke¸smir") == "effect Keşmir"


def test_styled_md_merges_adjacent_subscripts():
    plain = (False, False, False, "")
    sub_plain = (False, False, False, "sub")
    sub_italic = (False, True, False, "sub")
    groups = [["p", (False, True, False, "")], ["9", sub_plain], ["V", sub_italic], [" = 0.4", plain]]
    assert ocrpdf.styled_md(groups) == "*p*~9*V*~ = 0.4"


def test_styled_md_no_emphasis_on_symbols():
    assert ocrpdf.styled_md([["−2", (False, True, False, "")]]) == "−2"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def test_xy_cut_reads_columns_in_order():
    lines = [line(f"L{i}", 50, 100 + 15 * i, 280, 112 + 15 * i) for i in range(5)]
    lines += [line(f"R{i}", 320, 100 + 15 * i, 550, 112 + 15 * i) for i in range(5)]
    items = [ocrpdf.Item(ln.box, line=ln) for ln in lines]
    order = [it.line.text for it in ocrpdf.xy_cut(items, 10, 6)]
    assert order == [f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)]


def test_xy_cut_does_not_split_a_single_row():
    lines = [line("2.1", 40, 100, 55, 112), line("Section Title", 67, 100, 300, 112)]
    items = [ocrpdf.Item(ln.box, line=ln) for ln in lines]
    out = ocrpdf.xy_cut(items, 10, 6)
    assert [it.line.text for it in out] == ["2.1 Section Title"]


def test_markdown_table():
    rows = [("Item", "Count"), ("Apples", "12"), ("Pears", "9")]
    lines = [line(a, 90, 100 + 16 * i, 130, 110 + 16 * i) for i, (a, _) in enumerate(rows)]
    lines += [line(b, 300, 100 + 16 * i, 320, 110 + 16 * i) for i, (_, b) in enumerate(rows)]
    assert ocrpdf.markdown_table(lines, 10) == (
        "| Item | Count |\n|---|---:|\n| Apples | 12 |\n| Pears | 9 |"
    )


def test_contents_page_needs_rising_page_numbers():
    toc = [line(f"Chapter {i} title", 50, 100 + 15 * i, 300, 112 + 15 * i) for i in range(10)]
    toc += [line(str(10 * i + 1), 400, 100 + 15 * i, 410, 112 + 15 * i) for i in range(10)]
    assert ocrpdf.is_contents_page(toc, 3, 300)
    table = [line(f"Disease {i}", 50, 100 + 15 * i, 300, 112 + 15 * i) for i in range(10)]
    table += [line(str((7 * i) % 10), 400, 100 + 15 * i, 410, 112 + 15 * i) for i in range(10)]
    assert not ocrpdf.is_contents_page(table, 3, 300)


def test_index_pages_from_toc():
    toc = [[1, "Chapter 1", 1], [1, "References", 10], [1, "Index", 20]]
    assert ocrpdf.index_pages_from_toc(toc, 25) == set(range(20, 26))


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_captions_ids_and_links():
    blocks = [
        Block("figure", "images/a.jpg", page=1),
        Block("para", "Figure 3.9: A curve.", page=1),
        Block("para", "As figure 3.9 shows, and see chapter 2.", page=1),
        Block("heading", "Chapter 2: Later", level=1, page=2),
    ]
    ocrpdf.attach_captions(blocks)
    assert blocks[0].caption == "Figure 3.9: A curve."
    ocrpdf.assign_ids(blocks)
    ocrpdf.link_references(blocks)
    assert blocks[0].ident == "fig-3-9"
    assert blocks[1].text == "As [figure 3.9](#fig-3-9) shows, and see [chapter 2](#chap-2)."


def test_footnotes():
    m = ocrpdf.Metrics(body_h=12, body_size=10)
    p = ocrpdf.PageData(1, 600, 800)
    p.lines = [
        line("Body text.1", 50, 100, 400, 112, md="Body text.^1^", size=10),
        line("1 The note.", 50, 700, 300, 708, md="^1^ The note.", size=8),
    ]
    notes = ocrpdf.extract_footnotes(p, m)
    assert notes == {"1": "The note."}
    assert [ln.text for ln in p.lines] == ["Body text.1"]
    blocks = [Block("para", "Body text.^1^", page=1)]
    defs = ocrpdf.link_footnotes(blocks, notes, 1)
    assert blocks[0].text == "Body text.[^p1-1]"
    assert defs[0].text == "[^p1-1]: The note."


def test_small_print_without_marker_is_not_a_footnote():
    m = ocrpdf.Metrics(body_h=12, body_size=10)
    p = ocrpdf.PageData(1, 600, 800)
    p.lines = [
        line("Body text.", 50, 100, 400, 112, size=10),
        line("1 Not referenced.", 50, 700, 300, 708, md="^1^ Not referenced.", size=8),
    ]
    assert ocrpdf.extract_footnotes(p, m) == {}
    assert len(p.lines) == 2


# --------------------------------------------------------------------------
# Regression: whole documents
# --------------------------------------------------------------------------

CASES = {
    "digital": ("digital.pdf", []),
    "digital-ocr": ("digital.pdf", ["--text-layer", "never"]),
    "scanned": ("scanned.pdf", []),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_regression(name, tmp_path):
    pdf, extra = CASES[name]
    out = tmp_path / name
    cache = ["--cache-dir", str(FIXTURES / "ocr-cache")] + ([] if UPDATE else ["--cache-only"])
    assert ocrpdf.main([str(FIXTURES / pdf), "-o", str(out), "--jobs", "1", *cache, *extra]) == 0
    md = (out / Path(pdf).with_suffix(".md").name).read_text()
    expected = EXPECTED / f"{name}.md"
    if UPDATE:
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text(md)
    assert md == expected.read_text()
    # Every image the Markdown refers to was written.
    for ref in re.findall(r"\]\((images/[^)]+)\)", md):
        assert (out / ref).exists()
