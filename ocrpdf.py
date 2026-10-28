#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pymupdf>=1.24",
#     "pillow>=10",
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "pyobjc-framework-Vision>=10",
#     "pyobjc-framework-Quartz>=10",
#     "pyobjc-framework-Cocoa>=10",
# ]
# ///
"""Convert a PDF to Markdown with JPEG figures, entirely on this Mac.

Text comes from the PDF's own text layer when it has a usable one, and
otherwise from Apple's on-device Vision OCR. The output is a Markdown file
plus an ``images/`` directory of JPEG figures, referenced from the Markdown
at the point where they appear in the reading order. It is intended as input
for pandoc (``pandoc book.md -o book.epub``) or similar tools.

Everything runs locally:
  * text recognition uses macOS Vision (VNRecognizeTextRequest), which runs
    on-device;
  * PDF parsing, rendering and image handling use PyMuPDF / Pillow / numpy /
    scipy;
  * Python-level networking is disabled at start-up as a belt-and-braces guard.

Usage:
    ./ocrpdf.py book.pdf                 # writes book/book.md + book/images/
    ./ocrpdf.py book.pdf -o out --pages 1-20 --lang en-US
    ./ocrpdf.py --list-languages
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _forbid_network() -> None:
    """Make any Python-level network attempt fail loudly."""

    def blocked(*_a, **_k):
        raise RuntimeError("ocrpdf: network access is disabled")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]


_forbid_network()

import numpy as np  # noqa: E402
import objc  # noqa: E402
import pymupdf  # noqa: E402
import Vision  # noqa: E402
from Foundation import NSData  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402


# --------------------------------------------------------------------------
# Data model (all coordinates are PDF points, origin top-left of the page)
# --------------------------------------------------------------------------


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def union(self, o: "Box") -> "Box":
        return Box(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1))

    def intersection_area(self, o: "Box") -> float:
        w = min(self.x1, o.x1) - max(self.x0, o.x0)
        h = min(self.y1, o.y1) - max(self.y0, o.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def expand(self, d: float) -> "Box":
        return Box(self.x0 - d, self.y0 - d, self.x1 + d, self.y1 + d)

    def near(self, o: "Box", gap: float) -> bool:
        return self.expand(gap).intersection_area(o) > 0


@dataclass
class Line:
    text: str  # plain text
    conf: float
    box: Box
    md: str = ""  # Markdown, with inline emphasis/code when known
    size: float | None = None  # font size (text layer only)
    bold: bool = False  # every character bold (text layer only)
    mono: bool = False  # every character monospaced (text layer only)

    def __post_init__(self):
        if not self.md:
            self.md = md_escape(self.text)


@dataclass
class Figure:
    box: Box
    path: str  # relative path of the JPEG (or a data: URI)


@dataclass
class Table:
    box: Box
    md: str  # pipe table


@dataclass
class PageData:
    number: int  # 1-based
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    source: str = "ocr"  # "ocr", "text", "skipped" (contents) or "index"


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

_CLEAN = str.maketrans(
    {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st", "\u00a0": " "}
)
# Spacing accents that TeX-made PDFs put before the letter ("\u00b8s" for "\u015f").
_ACCENTS = {
    "\u00b8": "\u0327", "\u00b4": "\u0301", "\u00a8": "\u0308", "\u02c6": "\u0302", "\u02dc": "\u0303", "\u02c7": "\u030c",
    "\u02d8": "\u0306", "\u02da": "\u030a", "\u00af": "\u0304", "\u02d9": "\u0307", "\u02dd": "\u030b",
}
_ACCENT_RE = re.compile(f"([{''.join(_ACCENTS)}])([A-Za-z])")


def clean_text(s: str) -> str:
    s = s.translate(_CLEAN)
    s = _ACCENT_RE.sub(lambda m: unicodedata.normalize("NFC", m.group(2) + _ACCENTS[m.group(1)]), s)
    return re.sub("\u00ad(?=.)", "", s)  # soft hyphens, except at a line end


_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>^~])")


def md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def escape_block_start(md: str) -> str:
    """Escape what Markdown would read as block syntax at a paragraph start."""
    md = re.sub(r"^(#{1,6}\s|[+\-=]\s|>)", r"\\\1", md)
    return re.sub(r"^(\d+)([.)])(\s)", r"\1\\\2\3", md)


def strip_emphasis(md: str) -> str:
    return re.sub(r"(?<!\\)[*`^~]", "", md)


def plain_of(md: str) -> str:
    return re.sub(r"\\(.)", r"\1", strip_emphasis(md))


TERMINAL = tuple('.!?:;"”’)]»…')


def ends_sentence(md: str) -> bool:
    return plain_of(md).rstrip().endswith(TERMINAL)


def continues(prev: str, nxt: str) -> bool:
    """Does text ``nxt`` look like the continuation of ``prev``?"""
    return not ends_sentence(prev) and plain_of(nxt).lstrip()[:1].islower()


_speller = None
_known_words: dict[str, bool | None] = {}


def is_word(word: str) -> bool | None:
    """Ask the macOS spell checker (local) whether ``word`` is a word."""
    global _speller
    if word in _known_words:
        return _known_words[word]
    if _speller is None:
        try:
            from AppKit import NSSpellChecker

            _speller = NSSpellChecker.sharedSpellChecker()
        except Exception:
            _speller = False
    ok = None
    if _speller:
        try:
            ok = _speller.checkSpellingOfString_startingAt_(word, 0).length == 0
        except Exception:
            ok = None
    _known_words[word] = ok
    return ok


_HYPHEN_END = re.compile("(\\w+)[-\u00ad](\\**)$")
_LOWER_START = re.compile(r"(\**)([a-zà-ÿ]\w*)")


def join_text(a: str, b: str) -> str:
    """Join two consecutive lines of Markdown, undoing end-of-line hyphenation.

    "exam-" + "ple" becomes "example"; "data-" + "driven" stays hyphenated
    when the spell checker knows both halves but not the joined word.
    """
    if not a:
        return b
    m, n = _HYPHEN_END.search(a), _LOWER_START.match(b)
    if m and n and m.group(2) == n.group(1):
        left, right = m.group(1), n.group(2)
        keep = is_word(left + right) is False and is_word(left) and is_word(right)
        head = a[: m.end(1)] + ("-" if keep else "")
        return head + b[len(n.group(1)) :]
    return a + " " + b


# --------------------------------------------------------------------------
# Text layer
# --------------------------------------------------------------------------


def _rotated(rect: pymupdf.Rect, page: pymupdf.Page) -> Box:
    if page.rotation:
        rect = rect * page.rotation_matrix
    return Box(rect.x0, rect.y0, rect.x1, rect.y1)


def span_style(span: dict, size: float, baseline: float) -> tuple[bool, bool, bool, str]:
    """(bold, italic, mono, script) from the font name, flags and position.

    ``script`` is "sup" or "sub" for small text raised above or dropped below
    the line's baseline, else "".
    """
    f, fl = span["font"].lower(), span["flags"]
    bold = bool(fl & 16) or any(k in f for k in ("bold", "demi", "black", "heavy"))
    italic = bool(fl & 2) or "italic" in f or "oblique" in f
    mono = bool(fl & 8) or any(k in f for k in ("mono", "courier", "typewriter", "consol"))
    script = ""
    if span["size"] <= 0.85 * size:
        y = span["origin"][1]
        if y < baseline - 0.2 * size:
            script = "sup"
        elif y > baseline + 0.1 * size:
            script = "sub"
    return bold, italic, mono, script


def styled_md(groups: list[list]) -> str:
    """Markdown for a line's style groups: emphasis, code, ^sup^ and ~sub~."""
    out: list[str] = []
    run: list[str] | None = None  # [script, text] of an open ^...^ / ~...~ run

    def close_run():
        nonlocal run
        if run:
            mark = "^" if run[0] == "sup" else "~"
            out.append(mark + run[1].replace(" ", "\\ ") + mark)  # pandoc needs escaped spaces
        run = None

    for text, (bold, italic, mono, script) in groups:
        core = text.strip()
        if not core:
            close_run()
            out.append(text)
            continue
        lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]
        if mono:
            s = f"`{core}`" if "`" not in core else f"`` {core} ``"
        else:
            s = md_escape(core)
            if not any(c.isalpha() for c in core):
                pass  # math-font minus signs, brackets, digits: no emphasis
            elif bold and italic:
                s = f"***{s}***"
            elif bold:
                s = f"**{s}**"
            elif italic:
                s = f"*{s}*"
        if script and run and run[0] == script and not lead:
            run[1] += s  # adjacent pieces of one sub/superscript in different fonts
        else:
            close_run()
            out.append(lead)
            if script:
                run = [script, s]
            else:
                out.append(s)
        if trail:
            close_run()
            out.append(trail)
    close_run()
    return "".join(out)


def text_layer_lines(page: pymupdf.Page) -> list[Line]:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES
    lines: list[Line] = []
    for block in page.get_text("dict", flags=flags)["blocks"]:
        for ln in block.get("lines", []):
            if abs(ln["dir"][1]) > 0.05:  # rotated text: axis labels and the like
                continue
            groups: list[list] = []
            sizes: Counter[float] = Counter()
            for s in ln["spans"]:
                sizes[round(s["size"], 1)] += len(s["text"].strip())
            if not sizes or not sum(sizes.values()):
                continue
            main_size = sizes.most_common(1)[0][0]
            baseline = statistics.median(s["origin"][1] for s in ln["spans"] if s["size"] >= 0.9 * main_size)
            prev_x1 = None
            for s in ln["spans"]:
                t = clean_text(s["text"])
                if not t:
                    continue
                style = span_style(s, main_size, baseline)
                if not t.strip() and groups:
                    style = groups[-1][1]
                if (
                    prev_x1 is not None
                    and s["bbox"][0] - prev_x1 > 0.15 * s["size"]
                    and not groups[-1][0].endswith(" ")
                    and not t.startswith(" ")
                ):
                    t = " " + t
                if groups and groups[-1][1] == style:
                    groups[-1][0] += t
                else:
                    groups.append([t, style])
                prev_x1 = s["bbox"][2]
            text = "".join(g[0] for g in groups).strip()
            if not text:
                continue
            inked = [g for g in groups if g[0].strip()]
            lines.append(
                Line(
                    text,
                    1.0,
                    _rotated(pymupdf.Rect(ln["bbox"]), page),
                    md=styled_md(groups).strip(),
                    size=main_size,
                    bold=all(g[1][0] for g in inked),
                    mono=all(g[1][2] for g in inked),
                )
            )
    return lines


def usable_text(lines: list[Line], min_letters: float = 0.5) -> bool:
    """Is a page's text layer real text (not empty, not mis-encoded glyphs)?"""
    text = "".join(ln.text for ln in lines).replace(" ", "")
    if len(text) < 30:
        return False
    letters = sum(c.isalpha() for c in text)
    junk = sum(c == "�" or not c.isprintable() for c in text)
    return letters >= min_letters * len(text) and junk < 0.01 * len(text)


def invisible_text(page: pymupdf.Page) -> bool:
    """Is most of the page's text drawn invisibly (a hidden OCR layer)?"""
    try:
        trace = page.get_texttrace()
    except Exception:
        return False
    total = sum(len(t["chars"]) for t in trace)
    hidden = sum(len(t["chars"]) for t in trace if t.get("type") == 3 or t.get("opacity", 1) == 0)
    return total > 0 and hidden >= 0.5 * total


def document_text_ok(doc: pymupdf.Document) -> bool:
    """Does the document as a whole have a trustworthy text layer?"""
    sample = [doc[i] for i in range(0, doc.page_count, max(1, doc.page_count // 20))]
    with_text = [ls for ls in (text_layer_lines(p) for p in sample) if len("".join(ln.text for ln in ls)) >= 30]
    return bool(with_text) and sum(usable_text(ls) for ls in with_text) >= 0.6 * len(with_text)


# --------------------------------------------------------------------------
# OCR with Apple Vision
# --------------------------------------------------------------------------


def supported_languages(accurate: bool = True) -> list[str]:
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelAccurate if accurate else Vision.VNRequestTextRecognitionLevelFast
    )
    langs, _err = req.supportedRecognitionLanguagesAndReturnError_(None)
    return list(langs or [])


def ocr_image(png: bytes, width: float, height: float, langs: list[str] | None, fast: bool) -> list[Line]:
    """Run Vision text recognition on a PNG; return lines in page coordinates."""
    data = NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelFast if fast else Vision.VNRequestTextRecognitionLevelAccurate
    )
    req.setUsesLanguageCorrection_(True)
    if langs:
        req.setRecognitionLanguages_(langs)
    elif hasattr(req, "setAutomaticallyDetectsLanguage_"):
        req.setAutomaticallyDetectsLanguage_(True)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(
            f"Vision OCR failed ({err}). If this runs inside a sandbox, Vision's system services may be blocked."
        )

    lines: list[Line] = []
    for obs in req.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        cand = cands[0]
        text = clean_text(str(cand.string())).strip()
        if not text:
            continue
        bb = obs.boundingBox()  # normalised, origin bottom-left
        x0 = bb.origin.x * width
        x1 = (bb.origin.x + bb.size.width) * width
        y0 = (1.0 - bb.origin.y - bb.size.height) * height
        y1 = (1.0 - bb.origin.y) * height
        lines.append(Line(text, float(cand.confidence()), Box(x0, y0, x1, y1)))
    return lines


def drop_phantom_lines(lines: list[Line]) -> list[Line]:
    """Remove Vision's occasional hallucinated lines.

    These come back with middling confidence and an implausible box: much
    taller than the surrounding text (e.g. "ill." read off a bar chart) or
    lying on top of other lines.
    """
    if len(lines) < 3:
        return lines
    med_h = statistics.median(ln.box.h for ln in lines)
    keep = []
    for ln in lines:
        if ln.conf < 0.6:
            overlap = sum(ln.box.intersection_area(o.box) for o in lines if o is not ln)
            if ln.box.h > 1.8 * med_h or overlap > 0.3 * ln.box.area:
                continue
        keep.append(ln)
    return keep


# --------------------------------------------------------------------------
# Rows, decorations, contents pages, tables
# --------------------------------------------------------------------------


def group_rows(items: list, box: Callable = lambda i: i.box) -> list[list]:
    """Group items whose boxes overlap vertically into rows, top to bottom."""
    rows: list[list] = []
    row_box: Box | None = None
    for it in sorted(items, key=lambda i: box(i).cy):
        b = box(it)
        if rows and row_box is not None:
            overlap = min(row_box.y1, b.y1) - max(row_box.y0, b.y0)
            if overlap > 0.5 * min(row_box.h, b.h):
                rows[-1].append(it)
                row_box = row_box.union(b)
                continue
        rows.append([it])
        row_box = b
    for r in rows:
        r.sort(key=lambda i: box(i).x0)
    return rows


def _image_key(b: Box) -> tuple:
    return (round(b.x0 / 2), round(b.y0 / 2), round(b.x1 / 2), round(b.y1 / 2))


def embedded_image_boxes(page: pymupdf.Page) -> list[Box]:
    boxes = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.is_empty or r.is_infinite:
            continue
        boxes.append(_rotated(r, page))
    return boxes


def find_decorations(doc: pymupdf.Document) -> set[tuple]:
    """Image positions repeated on many pages: banners, watermarks, ornaments."""
    counts: Counter[tuple] = Counter()
    for page in doc:
        small = 0.3 * page.rect.width * page.rect.height  # not page-sized scans or backgrounds
        counts.update({_image_key(b) for b in embedded_image_boxes(page) if b.area < small})
    return {k for k, n in counts.items() if n >= max(3, doc.page_count // 50)}


CONTENTS_TITLE = re.compile(r"^(table of )?contents$", re.IGNORECASE)
ENDS_WITH_PAGE = re.compile(r"(\d+|\b[ivxlcdm]+)$", re.IGNORECASE)


def is_contents_page(lines: list[Line], index: int, page_count: int) -> bool:
    """A table of contents: most rows end in page numbers, and those increase."""
    if 0.15 * page_count <= index:
        return False
    rows = [" ".join(ln.text for ln in r).strip() for r in group_rows(lines)]
    rows = [r for r in rows if sum(c.isalpha() for c in r) >= 3]
    if len(rows) < 8:
        return False
    ends = [ENDS_WITH_PAGE.search(r) for r in rows]
    ratio = sum(bool(e) for e in ends) / len(rows)
    nums = [int(e.group(1)) for e in ends if e and e.group(1).isdigit()]
    rising = sum(b >= a for a, b in zip(nums, nums[1:])) >= 0.8 * (len(nums) - 1) if len(nums) > 1 else True
    titled = any(CONTENTS_TITLE.match(r) for r in rows)
    return rising and (ratio >= 0.5 or (titled and ratio >= 0.3))


def find_text_tables(lines: list[Line], med_h: float, text_w: float) -> list[Box]:
    """Tables set as plain text: runs of rows made of separate short cells."""
    rows = group_rows(lines)
    multi = []
    for r in rows:
        cells = [r[0].box]
        for ln in r[1:]:
            if ln.box.x0 - cells[-1].x1 >= 1.5 * med_h:
                cells.append(ln.box)
            else:
                cells[-1] = cells[-1].union(ln.box)
        multi.append(len(cells) >= 2 and any(c.w < 0.3 * text_w for c in cells))

    tables = []
    run: list[int] = []

    def close_run():
        if sum(multi[i] for i in run) >= 4:
            b = None
            for i in run:
                for ln in rows[i]:
                    b = ln.box if b is None else b.union(ln.box)
            tables.append(b.expand(0.3 * med_h))

    for i, r in enumerate(rows):
        top = min(ln.box.y0 for ln in r)
        prev_bottom = max(ln.box.y1 for ln in rows[run[-1]]) if run else None
        contiguous = prev_bottom is not None and top - prev_bottom < 2 * med_h
        if multi[i] and (contiguous or not run):
            run.append(i)
        elif multi[i]:
            close_run()
            run = [i]
        elif run and contiguous and i + 1 < len(rows) and multi[i + 1]:
            run.append(i)  # a single label row inside a table
        else:
            close_run()
            run = []
    close_run()
    return tables


# --------------------------------------------------------------------------
# Figure detection
# --------------------------------------------------------------------------


def _visible(drawing: dict, page_rect: pymupdf.Rect) -> bool:
    def white(c) -> bool:
        return c is None or all(v >= 0.95 for v in c)

    r = drawing["rect"]
    on_page = r.x1 > page_rect.x0 and r.x0 < page_rect.x1 and r.y1 > page_rect.y0 and r.y0 < page_rect.y1
    return on_page and not (white(drawing.get("fill")) and white(drawing.get("color")))


def drawing_cluster_boxes(page: pymupdf.Page) -> list[Box]:
    """Bounding boxes of clusters of visible vector drawings (charts, diagrams).

    White background boxes and crop marks outside the page are ignored.
    """
    try:
        drawings = [d for d in page.get_drawings() if _visible(d, page.rect)]
        rects = page.cluster_drawings(drawings=drawings) if drawings else []
    except Exception:
        return []
    return [_rotated(r, page) for r in rects]


def ink_blob_boxes(img: Image.Image, page_w: float, page_h: float, text: list[Line]) -> list[Box]:
    """Find non-text graphics on a rasterised (scanned) page.

    Text recognised by Vision is masked out; what remains is dilated and split
    into connected components, and the large ones are treated as figures.
    """
    scale = max(1, round(img.width / (page_w / 72 * 75)))  # analyse at ~75 dpi
    small = img.convert("L").resize((img.width // scale, img.height // scale), Image.BOX)
    gray = np.asarray(small)
    hpx, wpx = gray.shape
    sx, sy = wpx / page_w, hpx / page_h

    ink = gray < 200
    for ln in text:
        b = ln.box
        ink[
            max(0, int(b.y0 * sy) - 1) : int(b.y1 * sy) + 2,
            max(0, int(b.x0 * sx) - 1) : int(b.x1 * sx) + 2,
        ] = False
    # Scans often have dark borders; ignore a thin frame around the page.
    mx, my = max(1, int(wpx * 0.015)), max(1, int(hpx * 0.015))
    ink[:my, :] = ink[-my:, :] = False
    ink[:, :mx] = ink[:, -mx:] = False

    grown = ndimage.binary_dilation(ink, iterations=4)
    labels, _n = ndimage.label(grown)
    boxes = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        ys, xs = sl
        w, h = xs.stop - xs.start, ys.stop - ys.start
        if w < wpx * 0.06 or h < hpx * 0.04:
            continue
        if w * h < wpx * hpx * 0.015 or w * h > wpx * hpx * 0.95:
            continue
        if w / h > 15 or h / w > 15:  # rules and borders
            continue
        density = ink[sl].mean()
        if density < 0.02:
            continue
        boxes.append(Box(xs.start / sx, ys.start / sy, xs.stop / sx, ys.stop / sy))
    return boxes


def merge_boxes(boxes: list[Box], gap: float) -> list[Box]:
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        for b in boxes:
            for i, o in enumerate(out):
                if b.near(o, gap):
                    out[i] = o.union(b)
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def text_coverage(box: Box, lines: list[Line]) -> float:
    if box.area <= 0:
        return 1.0
    return sum(box.intersection_area(ln.box) for ln in lines) / box.area


def find_figures(
    page: pymupdf.Page,
    get_img: Callable[[], Image.Image],
    lines: list[Line],
    mode: str,
    images: list[Box],
    scanned: bool,
) -> list[Box]:
    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    candidates: list[Box] = []

    if mode in ("auto", "ink") and (scanned or mode == "ink"):
        candidates += ink_blob_boxes(get_img(), pw, ph, lines)
    if mode in ("auto", "embedded"):
        candidates += [b for b in images if b.area >= 0.004 * page_area and (b.area < 0.6 * page_area or not scanned)]
        candidates += [
            b
            for b in drawing_cluster_boxes(page)
            if b.area >= 0.01 * page_area and b.w >= 0.08 * pw and b.h >= 0.04 * ph and text_coverage(b, lines) <= 0.4
        ]

    figs = []
    for b in merge_boxes(candidates, gap=0.01 * min(pw, ph)):
        b = Box(max(0, b.x0), max(0, b.y0), min(pw, b.x1), min(ph, b.y1))
        if b.area < 0.004 * page_area or b.area > 0.97 * page_area:
            continue
        # Shaded sidebars / text boxes are mostly text, not pictures.
        if text_coverage(b, lines) > 0.4:
            continue
        figs.append(b)
    return figs


CAPTION_RE = re.compile(r"^(fig(ure)?|table|plate|chart|exhibit|scheme)\.?\s*[\dIVX]", re.IGNORECASE)


def is_label(ln: Line, fig: Box) -> bool:
    """Short text hugging a figure: axis ticks, legends, panel letters."""
    t = ln.text.strip()
    if CAPTION_RE.match(t) or re.search(r"[A-Za-z]{2,}\.$", t) or ln.box.w > 0.6 * fig.w:
        return False
    letters = sum(c.isalpha() for c in t)
    return (len(t.split()) <= 3 and len(t) <= 20) or letters <= 0.4 * len(t)


def absorb_labels(figs: list[Box], lines: list[Line], smalls: list[Box], gap: float) -> list[Box]:
    figs = list(figs)
    changed = True
    while changed:
        changed = False
        for i, f in enumerate(figs):
            for ln in lines:
                if not f.contains_point(ln.box.cx, ln.box.cy) and f.near(ln.box, gap) and is_label(ln, f):
                    f = f.union(ln.box)
                    changed = True
            for b in smalls:
                if f.near(b, gap) and f.intersection_area(b) < b.area:
                    f = f.union(b)
                    changed = True
            figs[i] = f
    return merge_boxes(figs, gap=0)


# --------------------------------------------------------------------------
# Markdown tables
# --------------------------------------------------------------------------


def markdown_table(lines: list[Line], med_h: float) -> str | None:
    """A pipe table for a text table whose cells line up in clear columns.

    Returns None when the layout is irregular (a body line straddling two
    columns, too many columns); the caller then saves the table as an image.
    """
    rows = group_rows(lines)
    if len(rows) < 2:
        return None

    # Columns are the x-ranges separated by whitespace in every row.
    spans = sorted((ln.box.x0, ln.box.x1) for ln in lines)
    cols = [list(spans[0])]
    for s, e in spans[1:]:
        if s - cols[-1][1] >= 0.8 * med_h:
            cols.append([s, e])
        else:
            cols[-1][1] = max(cols[-1][1], e)
    if not 2 <= len(cols) <= 10:
        return None

    def column(b: Box) -> int:
        return min(range(len(cols)), key=lambda k: 0 if cols[k][0] <= b.cx <= cols[k][1] else
                   min(abs(b.cx - cols[k][0]), abs(b.cx - cols[k][1])))

    grid: list[list[str]] = []
    for n, r in enumerate(rows):
        row = [""] * len(cols)
        for ln in r:
            i = column(ln.box)
            straddles = sum(ln.box.x0 < e - med_h and ln.box.x1 > s + med_h for s, e in cols) > 1
            if straddles and n > 0:
                return None  # a merged cell in the body: not a simple grid
            text = ln.md.replace("|", "\\|")
            if not row[i] and ln.box.x0 - cols[i][0] > med_h and i == 0:
                text = "\u00a0" * 4 + text  # indented row label
            row[i] = (row[i] + " " + text).strip() if row[i] else text
        if grid and not row[0] and sum(bool(x) for x in row) == 1:
            k = next(i for i, x in enumerate(row) if x)
            grid[-1][k] = (grid[-1][k] + " " + row[k]).strip()  # wrapped cell text
            continue
        grid.append(row)

    def numeric(s: str) -> bool:
        return bool(re.fullmatch(r"[-+−–]?[\d.,]+%?", plain_of(s).strip()))

    align = []
    for i in range(len(cols)):
        body = [r[i] for r in grid[1:] if r[i]]
        align.append("---:" if body and sum(map(numeric, body)) >= 0.7 * len(body) else "---")
    out = ["| " + " | ".join(grid[0]) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(out)


# --------------------------------------------------------------------------
# OCR cache
# --------------------------------------------------------------------------


def page_fingerprint(page: pymupdf.Page, args: argparse.Namespace) -> str:
    """Identify what a page looks like: its content, images, fonts and OCR settings."""
    doc = page.parent
    h = hashlib.sha256(page.read_contents())
    xrefs = [img[0] for img in page.get_images(full=True)] + [f[0] for f in page.get_fonts(full=True)]
    for xref in xrefs:
        try:
            h.update(doc.xref_stream_raw(xref) or b"")
        except Exception:
            pass
    h.update(json.dumps([page.rotation, round(page.rect.width, 1), round(page.rect.height, 1),
                         args.dpi, args.lang, args.fast]).encode())
    return h.hexdigest()[:32]


def ocr_cached(png: bytes, width: float, height: float, args: argparse.Namespace, key: str | None) -> list[Line]:
    """Vision OCR, reusing a cached result for an identical page."""
    path = args.cache_dir / f"{key}.json" if args.cache_dir and key else None
    if path and path.exists():
        return [Line(t, c, Box(*b)) for t, c, b in json.loads(path.read_text())]
    if args.cache_only:
        raise RuntimeError(f"OCR cache miss ({key}) and --cache-only was given")
    with objc.autorelease_pool():
        lines = ocr_image(png, width, height, args.lang, args.fast)
    if path and lines:  # an empty result may be Vision failing silently: don't keep it
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [[ln.text, ln.conf, [ln.box.x0, ln.box.y0, ln.box.x1, ln.box.y1]] for ln in lines]
        path.write_text(json.dumps(data, ensure_ascii=False))
    return lines


# --------------------------------------------------------------------------
# Pass 1: text, figures and tables per page
# --------------------------------------------------------------------------


@dataclass
class PagePrep:
    """A page between rendering (main thread) and finishing (after OCR)."""

    page: pymupdf.Page
    pd: PageData
    lines: list[Line]
    images: list[Box]
    decor: list[Box]
    big_image: bool
    pix: pymupdf.Pixmap | None = None
    png: bytes | None = None
    key: str | None = None

    def get_pix(self, dpi: int) -> pymupdf.Pixmap:
        if self.pix is None:
            self.pix = self.page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        return self.pix


def prepare_page(page: pymupdf.Page, args: argparse.Namespace, decorations: set[tuple]) -> PagePrep:
    """Choose text layer or OCR for a page; render it if OCR is needed."""
    pw, ph = page.rect.width, page.rect.height
    pd = PageData(page.number + 1, pw, ph)
    all_images = embedded_image_boxes(page)
    decor = [b for b in all_images if _image_key(b) in decorations]
    images = [b for b in all_images if _image_key(b) not in decorations]
    big_image = any(b.area >= 0.85 * pw * ph for b in images)

    # Use the text layer unless it is unusable, or it is the invisible layer
    # another OCR program put over a scan (then Vision redoes it).
    lines: list[Line] = []
    if args.text_layer != "never":
        lines = text_layer_lines(page)
        if args.text_layer == "always":
            pd.source = "text" if lines else "ocr"
        elif usable_text(lines, 0.25 if args.doc_text_ok else 0.5) and not (big_image and invisible_text(page)):
            pd.source = "text"
    prep = PagePrep(page, pd, lines if pd.source == "text" else [], images, decor, big_image)
    if pd.source == "ocr":
        prep.png = prep.get_pix(args.dpi).tobytes("png")
        prep.key = page_fingerprint(page, args)
    return prep


def finish_page(
    prep: PagePrep,
    ocr_lines: list[Line] | None,
    args: argparse.Namespace,
    image_dir: Path,
    image_prefix: str,
) -> PageData:
    """Find figures and tables, crop them, and keep the remaining text."""
    page, pd = prep.page, prep.pd
    pw, ph = pd.width, pd.height
    page_area = pw * ph

    def get_img() -> Image.Image:
        pix = prep.get_pix(args.dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    scanned = prep.big_image and pd.source == "ocr"
    lines = prep.lines
    if ocr_lines is not None:
        lines = drop_phantom_lines([ln for ln in ocr_lines if ln.conf >= args.min_confidence])
    lines = [ln for ln in lines if not any(d.expand(2).contains_point(ln.box.cx, ln.box.cy) for d in prep.decor)]

    if not args.keep_contents and is_contents_page(lines, page.number, page.parent.page_count):
        pd.source = "skipped"
        return pd
    # Without an "Index" bookmark, an "Index" title near the end starts the index.
    if not args.keep_index and not args.index_pages and is_index_start(lines, page.number, page.parent.page_count):
        args.index_open = True
    if args.index_open:
        pd.source = "index"
        return pd

    med_h = statistics.median([ln.box.h for ln in lines]) if lines else 10.0
    long_lines = [ln.box.w for ln in lines if len(ln.text) >= 40]
    text_w = statistics.median(long_lines) if long_lines else 0.7 * pw

    boxes = [] if args.figures == "none" else find_figures(page, get_img, lines, args.figures, prep.images, scanned)
    if args.tables != "text":
        for tb in find_text_tables([ln for ln in lines if not ln.mono], med_h, text_w):
            inside = [ln for ln in lines if tb.contains_point(ln.box.cx, ln.box.cy)]
            md = None
            if args.tables == "markdown" or (args.tables == "auto" and pd.source == "text"):
                md = markdown_table(inside, med_h)
            if md:
                pd.tables.append(Table(tb, md))
            else:
                boxes.append(tb)
    boxes = merge_boxes(boxes, gap=0.01 * min(pw, ph))
    smalls = [b for b in prep.images if b.area < 0.004 * page_area]
    boxes = absorb_labels(boxes, lines, smalls, gap=1.2 * med_h)
    boxes = [b.expand(0.004 * min(pw, ph)) for b in boxes]
    boxes.sort(key=lambda b: (b.y0, b.x0))

    if boxes:
        pix = prep.get_pix(args.dpi)
        img = get_img()
        kx, ky = pix.width / pw, pix.height / ph
    for i, fb in enumerate(boxes, 1):
        crop = img.crop(
            (
                max(0, int(fb.x0 * kx)),
                max(0, int(fb.y0 * ky)),
                min(pix.width, int(fb.x1 * kx + 0.5)),
                min(pix.height, int(fb.y1 * ky + 0.5)),
            )
        )
        if args.inline_images:
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=args.jpeg_quality, optimize=True)
            ref = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        else:
            name = f"page{pd.number:04d}_fig{i}.jpg"
            crop.save(image_dir / name, "JPEG", quality=args.jpeg_quality, optimize=True)
            ref = f"{image_prefix}/{name}"
        pd.figures.append(Figure(fb, ref))

    # Text inside a figure or table (axis labels, cells) is part of it.
    regions = [f.box for f in pd.figures] + [t.box for t in pd.tables]
    if not args.keep_figure_text:
        lines = [ln for ln in lines if not any(r.contains_point(ln.box.cx, ln.box.cy) for r in regions)]
    pd.lines = lines
    prep.pix = prep.png = None  # free memory
    return pd


INDEX_TITLE = re.compile(r"^(general |subject )?index$", re.IGNORECASE)


def is_index_start(lines: list[Line], index: int, page_count: int) -> bool:
    """A page near the end of the book that opens with the title "Index"."""
    if index < 0.75 * page_count:
        return False
    top = sorted(lines, key=lambda ln: ln.box.y0)[:3]
    return any(INDEX_TITLE.match(ln.text.strip()) for ln in top)


def index_pages_from_toc(toc: list[list], page_count: int) -> set[int]:
    """Pages covered by an "Index" bookmark, up to the next bookmark at its level."""
    pages: set[int] = set()
    for i, (lvl, title, pg) in enumerate(toc):
        if INDEX_TITLE.match(title.strip()):
            end = next((p for lv, _t, p in toc[i + 1 :] if lv <= lvl and p > pg), page_count + 1)
            pages.update(range(pg, end))
    return pages


# --------------------------------------------------------------------------
# Pass 2: running headers/footers, reading order, paragraphs
# --------------------------------------------------------------------------

PAGE_NUM_RE = re.compile(r"^(page\s+)?([\divxlcdm]+)(\s+(of|/)\s+\d+)?$", re.IGNORECASE)


def _norm_header(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip(" .-–—|#")


def remove_running_heads(pages: list[PageData], zone: float = 0.1) -> None:
    """Drop page numbers, lines beside them, and lines repeated in the margins."""

    def in_zone(ln: Line, p: PageData) -> bool:
        return ln.box.cy < p.height * zone or ln.box.cy > p.height * (1 - zone)

    def is_page_num(ln: Line) -> bool:
        return bool(PAGE_NUM_RE.match(ln.text.strip(" .-–—|")))

    def beside(ln: Line, nums: list[Line]) -> bool:
        return any(min(ln.box.y1, n.box.y1) - max(ln.box.y0, n.box.y0) > 0.5 * min(ln.box.h, n.box.h) for n in nums)

    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({_norm_header(ln.text) for ln in p.lines if in_zone(ln, p)})
    for p in pages:
        nums = [ln for ln in p.lines if in_zone(ln, p) and is_page_num(ln)]
        p.lines = [
            ln
            for ln in p.lines
            if not (in_zone(ln, p) and (is_page_num(ln) or beside(ln, nums) or counts[_norm_header(ln.text)] >= 2))
        ]


@dataclass
class Item:
    box: Box
    line: Line | None = None
    figure: Figure | None = None
    code: str | None = None
    table: Table | None = None
    col: Box | None = None  # extent of the column the item was placed in


def _gaps(items: list[Item], axis: int, min_gap: float) -> list[tuple[float, float]]:
    spans = sorted(((it.box.x0, it.box.x1) if axis == 0 else (it.box.y0, it.box.y1)) for it in items)
    gaps = []
    end = spans[0][1]
    for s, e in spans[1:]:
        if s - end >= min_gap:
            gaps.append((end, s))
        end = max(end, e)
    return gaps


def _bbox(items: list) -> Box:
    b = items[0].box
    for it in items[1:]:
        b = b.union(it.box)
    return b


def merge_row(lines: list[Line]) -> Line:
    if len(lines) == 1:
        return lines[0]
    sizes = [ln.size for ln in lines if ln.size]
    return Line(
        " ".join(ln.text for ln in lines),
        min(ln.conf for ln in lines),
        _bbox(lines),
        md=" ".join(ln.md for ln in lines),
        size=max(sizes) if sizes else None,
        bold=all(ln.bold for ln in lines),
        mono=all(ln.mono for ln in lines),
    )


def xy_cut(items: list[Item], min_gx: float, min_gy: float, col: Box | None = None) -> list[Item]:
    """Recursive XY-cut: columns first, then the widest horizontal gap."""
    if not items:
        return []
    col = col or _bbox(items)
    if len(items) == 1:
        items[0].col = col
        return items

    gaps = _gaps(items, 0, min_gx)
    if gaps:
        edges = [-1e9] + [(a + b) / 2 for a, b in gaps] + [1e9]
        parts = [[it for it in items if lo <= it.box.cx < hi] for lo, hi in zip(edges, edges[1:])]
        parts = [part for part in parts if part]
        # A lone line is not a column: "2.1 | Title", "Label. | Text ...".
        if not any(len(part) == 1 and part[0].line for part in parts):
            out = []
            for part in parts:
                out += xy_cut(part, min_gx, min_gy, _bbox(part))
            return out

    gaps = _gaps(items, 1, min_gy)
    if gaps:
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        cut = (a + b) / 2
        top = [it for it in items if it.box.cy < cut]
        bottom = [it for it in items if it.box.cy >= cut]
        return xy_cut(top, min_gx, min_gy, col) + xy_cut(bottom, min_gx, min_gy, col)

    # No cut possible: read each row left to right.
    out = []
    for row in group_rows(items):
        out += [i for i in row if not i.line]
        texts = [i.line for i in row if i.line]
        if texts:
            merged = merge_row(texts)
            out.append(Item(merged.box, line=merged))
    for it in out:
        it.col = col
    return out


def code_items(lines: list[Line]) -> list[Item]:
    """Group monospaced lines into preformatted blocks, keeping their layout."""
    mono = sorted((ln for ln in lines if ln.mono), key=lambda ln: ln.box.y0)
    groups: list[list[Line]] = []
    for ln in mono:
        if groups and ln.box.y0 <= _bbox(groups[-1]).y1 + 1.5 * ln.box.h:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    items = []
    for g in groups:
        charw = 0.6 * statistics.median(ln.size or 0.8 * ln.box.h for ln in g)
        x0 = min(ln.box.x0 for ln in g)
        rows = []
        for row in group_rows(g):
            s = ""
            for ln in row:
                col = round((ln.box.x0 - x0) / charw)
                s = s + " " * (col - len(s)) if col > len(s) else (s + " " if s else s)
                s += ln.text
            rows.append(s.rstrip())
        items.append(Item(_bbox(g), code="\n".join(rows)))
    return items


BULLET_RE = re.compile(r"^\s*([•●◦▪▫■□‣∙·\-–*])\s+")
# The same, on Markdown: the bullet glyph may be wrapped in emphasis or escaped.
BULLET_MD_RE = re.compile(r"^\s*(?:\*{1,3}[•●◦▪▫■□‣∙·–-]\*{1,3}|[•●◦▪▫■□‣∙·–-]|\\\*)\s+")
NUMBERED_RE = re.compile(r"^\s*(\d{1,3}|[a-zA-Z])[.)]\s+")
CHAPTER_RE = re.compile(r"^(chapter|part|appendix|book|lecture)\s+(\d+|[ivxlcdm]+|[a-z]+)$", re.IGNORECASE)


@dataclass
class Block:
    kind: str  # "para", "heading", "list", "figure", "table", "code", "footnote"
    text: str = ""  # Markdown (the image path for figures)
    level: int = 0  # heading level; 0 = minor heading, level decided later
    page: int = 0
    x0: float = 0.0  # left edge of the first line
    nlines: int = 0
    hanging: bool = False
    caption: str = ""  # figures and tables
    ident: str = ""  # anchor id for cross-references


@dataclass
class Metrics:
    body_h: float  # median line-box height of body text
    body_size: float | None  # median font size of body text (text layer)


def size_ratio(ln: Line, m: Metrics) -> float:
    if ln.size and m.body_size:
        return ln.size / m.body_size
    return ln.box.h / m.body_h


def heading_level(r: float) -> int:
    if r < 1.35:
        return 0
    if r >= 2.0:
        return 1
    if r >= 1.6:
        return 2
    return 3


def _gap(ordered: list[Item], idx: int, step: int) -> float:
    """Vertical whitespace between item ``idx`` and its neighbour in ``step``."""
    j = idx + step
    if not 0 <= j < len(ordered) or not ordered[j].line:
        return 1e9
    a, b = ordered[idx], ordered[j]
    if a.col != b.col:
        return 1e9
    gap = b.box.y0 - a.box.y1 if step > 0 else a.box.y0 - b.box.y1
    return gap if gap > -0.5 * a.box.h else 1e9  # negative: moved to another column


def heading_candidate(ordered: list[Item], idx: int, m: Metrics) -> int | None:
    """Heading level for a line (0 = minor heading), or None for body text."""
    it = ordered[idx]
    ln = it.line
    text = ln.text.strip()
    if len(text.split()) > 15 or text.endswith((",", ";")) or ln.mono:
        return None
    r = size_ratio(ln, m)
    level = heading_level(r)
    if level:
        return level
    col = it.col or it.box
    standalone = _gap(ordered, idx, -1) >= 0.6 * m.body_h and (
        _gap(ordered, idx, 1) >= 0.25 * m.body_h
        or (idx + 1 < len(ordered) and ordered[idx + 1].line and ordered[idx + 1].line.bold == ln.bold)
    )
    if not standalone or text.endswith(".") or it.box.w > 0.8 * col.w:
        return None
    if ln.bold and len(text.split()) <= 12:
        return 0
    if r >= 1.15:
        return 0
    return None


def build_blocks(p: PageData, m: Metrics) -> list[Block]:
    code = code_items(p.lines)
    items = [Item(ln.box, line=ln) for ln in p.lines if not ln.mono]
    items += [Item(f.box, figure=f) for f in p.figures] + [Item(t.box, table=t) for t in p.tables] + code
    if not items:
        return []
    ordered = xy_cut(items, min_gx=max(0.012 * p.width, 0.6 * m.body_h), min_gy=0.5 * m.body_h)

    blocks: list[Block] = []
    cur: Block | None = None
    prev: Item | None = None

    def flush():
        nonlocal cur
        if cur and cur.text:
            blocks.append(cur)
        cur = None

    for idx, it in enumerate(ordered):
        if not it.line:
            flush()
            if it.figure:
                blocks.append(Block("figure", it.figure.path, page=p.number))
            elif it.table:
                blocks.append(Block("table", it.table.md, page=p.number))
            else:
                blocks.append(Block("code", it.code, page=p.number))
            prev = None
            continue

        ln = it.line
        b = ln.box
        level = heading_candidate(ordered, idx, m)
        if level is not None:
            heading_md = md_escape(ln.text)
            if (
                cur
                and cur.kind == "heading"
                and cur.level == level
                and prev
                and 0 <= b.y0 - prev.box.y1 < 0.8 * b.h
            ):
                cur.text = join_text(cur.text, heading_md)
            elif _gap(ordered, idx, -1) >= 0.4 * m.body_h:
                flush()
                cur = Block("heading", heading_md, level, p.number)
            else:
                level = None
            if level is not None:
                prev = it
                continue

        text = ln.md
        bullet = BULLET_RE.match(ln.text) or NUMBERED_RE.match(ln.text)
        if bullet and not BULLET_RE.match(ln.text) and cur and cur.kind == "para" and not ends_sentence(cur.text):
            bullet = None  # "... the score is" / "5. However ..." continues the sentence
        new_para = cur is None or cur.kind == "heading" or bullet is not None
        if not new_para and prev is not None:
            pb = prev.box
            col = it.col or b
            indent = m.body_h * 0.8
            if b.y0 < pb.y0 - 0.5 * m.body_h or prev.col != it.col:
                new_para = not continues(cur.text, text)  # column or page region change
            elif b.y0 - pb.y1 > 0.5 * m.body_h:
                new_para = True
            elif cur.kind == "list":
                new_para = b.x0 < cur.x0 - 0.5 * m.body_h
            elif b.x0 - pb.x0 > indent and b.x0 - col.x0 > indent:
                if cur.nlines == 1 and pb.x1 >= col.x1 - 0.1 * col.w:
                    cur.hanging = True  # hanging indent: first line out, the rest in
                else:
                    new_para = True  # first-line indent of a new paragraph
            elif cur.hanging and b.x0 < pb.x0 - indent:
                new_para = True  # back out of a hanging-indent paragraph
            elif pb.x1 < col.x1 - 0.15 * col.w and ends_sentence(cur.text):
                new_para = True  # previous line ended short
        if new_para:
            flush()
            if bullet and BULLET_RE.match(ln.text):
                cur = Block("list", BULLET_MD_RE.sub("", text, count=1), page=p.number)
            elif bullet:
                cur = Block("list", text, level=1, page=p.number)  # numbered: keep the label
            else:
                cur = Block("para", text, page=p.number)
            cur.x0 = b.x0
            cur.nlines = 1
        else:
            cur.text = join_text(cur.text, text)
            cur.nlines += 1
        prev = it
    flush()
    return blocks


# --------------------------------------------------------------------------
# Headings from the PDF outline (bookmarks), chapter labels, levels
# --------------------------------------------------------------------------


def norm_title(s: str) -> str:
    s = plain_of(s).lower().strip()
    s = re.sub(r"^(chapter|part|appendix|section)\s+[\divxlc]+[.:]?\s*", "", s)
    s = re.sub(r"^[\d.]+\s*", "", s)
    return re.sub(r"[\W_]+", "", s)


def apply_headings(per_page: list[list[Block]], toc: list[list]) -> None:
    """Assign heading levels.

    With a PDF outline, blocks whose text matches a bookmark on (or next to)
    their page become headings at the bookmark's depth, and other headings
    are placed below the current bookmark level. Without one, numbered
    headings ("2.1", "2.2.1") get their depth from the number.
    """
    entries = [
        {"level": min(lvl, 6), "norm": norm_title(title), "page": pg, "used": False}
        for lvl, title, pg in toc
        if norm_title(title)
    ]
    by_page: dict[int, list[dict]] = {}
    for e in entries:
        for pg in (e["page"] - 1, e["page"], e["page"] + 1):
            by_page.setdefault(pg, []).append(e)

    outline_level = 0
    for blocks in per_page:
        for b in blocks:
            if b.kind not in ("para", "heading") or len(plain_of(b.text).split()) > 20:
                continue
            nb = norm_title(b.text)
            match = None
            for e in by_page.get(b.page, []):
                if e["used"] or not nb or len(nb) > 2 * len(e["norm"]):
                    continue
                if nb == e["norm"] or difflib.SequenceMatcher(None, nb, e["norm"]).ratio() >= 0.88:
                    match = e
                    break
            if match:
                match["used"] = True
                b.kind, b.level = "heading", match["level"]
                b.text = strip_emphasis(b.text)
                outline_level = match["level"]
            elif b.kind == "heading":
                if entries:
                    if b.level == 0 and not outline_level:
                        b.kind = "para"  # bold lines before the first bookmark: title pages
                    elif b.level == 0 or outline_level:
                        b.level = max(b.level, min(outline_level + 1, 6))
                else:
                    num = re.match(r"^(\d+(?:\.\d+)*)\.?\s", plain_of(b.text))
                    if num and b.level in (0, 3):
                        b.level = min(6, max(2, num.group(1).count(".") + 1))
                    elif b.level == 0:
                        b.level = 3

        # "Chapter 2" followed by its title becomes one heading.
        out: list[Block] = []
        for b in blocks:
            if (
                b.kind == "heading"
                and out
                and out[-1].kind in ("para", "heading")
                and CHAPTER_RE.match(plain_of(out[-1].text).strip())
            ):
                label = plain_of(out.pop().text).strip()
                b.text = f"{md_escape(label)}: {b.text}"
            out.append(b)
        blocks[:] = out


# --------------------------------------------------------------------------
# Footnotes
# --------------------------------------------------------------------------

FOOTNOTE_START = re.compile(r"^\^(\d{1,3}|\\\*|[†‡§¶])\^\s*|^(\d{1,3}|\\\*|[†‡§¶])\s+")


def extract_footnotes(p: PageData, m: Metrics) -> dict[str, str]:
    """Take footnotes off the bottom of a text-layer page.

    Footnotes are small-print lines at the foot of the page, each starting
    with a marker that also appears as a superscript in the body text. They
    are only taken when every marker is found, so a caption or small print
    is never mistaken for one.
    """
    if not m.body_size:
        return {}
    ls = sorted(p.lines, key=lambda ln: ln.box.y0)
    i = len(ls)
    while i > 0 and ls[i - 1].size and ls[i - 1].size <= 0.9 * m.body_size and ls[i - 1].box.y0 > 0.5 * p.height:
        i -= 1
    while i < len(ls) and not FOOTNOTE_START.match(ls[i].md):
        i += 1
    zone, body = ls[i:], ls[:i]
    if not zone:
        return {}
    notes: list[list[str]] = []
    for ln in zone:
        mm = FOOTNOTE_START.match(ln.md)
        if mm:
            notes.append([mm.group(1) or mm.group(2), ln.md[mm.end() :]])
        else:
            notes[-1][1] = join_text(notes[-1][1], ln.md)
    body_md = " ".join(ln.md for ln in body)
    if not all(f"^{mk}^" in body_md for mk, _ in notes):
        return {}
    zone_ids = {id(ln) for ln in zone}
    p.lines = [ln for ln in p.lines if id(ln) not in zone_ids]
    return dict(notes)


def link_footnotes(blocks: list[Block], notes: dict[str, str], page: int) -> list[Block]:
    """Turn ^marker^ in the page's text into footnote references; return the notes."""
    out = []
    for k, (marker, text) in enumerate(notes.items(), 1):
        label = f"p{page}-{k}"
        for b in blocks:
            if b.kind in ("para", "list", "heading") and f"^{marker}^" in b.text:
                b.text = b.text.replace(f"^{marker}^", f"[^{label}]", 1)
                out.append(Block("footnote", f"[^{label}]: {text}", page=page))
                break
    return out


# --------------------------------------------------------------------------
# Captions, anchors and cross-references
# --------------------------------------------------------------------------

REF_KINDS = {"fig": "fig", "figure": "fig", "table": "tab", "plate": "plate", "chart": "fig", "scheme": "fig",
             "exhibit": "fig", "chapter": "chap", "section": "sec"}
CAPTION_ID_RE = re.compile(r"^(fig(?:ure)?|table|plate|chart|scheme|exhibit)\.?\s*(\d+(?:\.\d+)*[a-z]?)", re.IGNORECASE)
REF_RE = re.compile(r"\b(fig(?:ure)?s?\.?|tables?|plates?|chapters?|sections?)(\s+)(\d+(?:\.\d+)*)", re.IGNORECASE)


def ref_id(word: str, number: str) -> str:
    kind = REF_KINDS.get(word.lower().rstrip(".").rstrip("s"), "")
    if not kind and word.lower().startswith("fig"):
        kind = "fig"
    return f"{kind}-{number.replace('.', '-')}" if kind else ""


def attach_captions(blocks: list[Block]) -> None:
    """Move "Figure 3.9: ..." paragraphs into the figure or table they caption.

    Captions usually follow the figure; a "Table ..." caption may also come
    just before its table.
    """
    out: list[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind in ("figure", "table"):
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            prv = out[-1] if out else None
            if nxt and nxt.kind == "para" and CAPTION_RE.match(plain_of(nxt.text).strip()):
                b.caption = nxt.text
                i += 1
            elif (
                prv
                and prv.kind == "para"
                and not prv.caption
                and re.match(r"^table\b", plain_of(prv.text).strip(), re.IGNORECASE)
                and CAPTION_RE.match(plain_of(prv.text).strip())
            ):
                b.caption = out.pop().text
        out.append(b)
        i += 1
    blocks[:] = out


def assign_ids(blocks: list[Block]) -> None:
    """Give captioned figures/tables and numbered headings anchor ids."""
    seen: set[str] = set()
    for b in blocks:
        ident = ""
        if b.kind in ("figure", "table") and b.caption:
            m = CAPTION_ID_RE.match(plain_of(b.caption).strip())
            ident = ref_id(m.group(1), m.group(2)) if m else ""
        elif b.kind == "heading":
            t = plain_of(b.text).strip()
            m = re.match(r"^chapter\s+(\d+)\b", t, re.IGNORECASE) or (
                re.match(r"^(\d+)\.?\s+\D", t) if b.level == 1 else None
            )
            if m:
                ident = f"chap-{m.group(1)}"
            elif m := re.match(r"^(\d+(?:\.\d+)+)\.?\s", t):
                ident = f"sec-{m.group(1).replace('.', '-')}"
        if ident and ident not in seen:
            b.ident = ident
            seen.add(ident)


def link_references(blocks: list[Block]) -> None:
    """Link "figure 3.9", "table 2.1", "chapter 4", "section 2.1" to their targets."""
    targets = {b.ident for b in blocks if b.ident}

    def link(text: str, own: str) -> str:
        parts = re.split(r"(`[^`]*`)", text)  # leave code spans alone
        for i in range(0, len(parts), 2):
            def sub(m: re.Match) -> str:
                ident = ref_id(m.group(1), m.group(3))
                if ident in targets and ident != own:
                    return f"[{m.group(0)}](#{ident})"
                return m.group(0)

            parts[i] = REF_RE.sub(sub, parts[i])
        return "".join(parts)

    for b in blocks:
        if b.kind in ("para", "list"):
            b.text = link(b.text, "")
        elif b.caption:
            b.caption = link(b.caption, b.ident)


# --------------------------------------------------------------------------
# Markdown output
# --------------------------------------------------------------------------


def render_markdown(blocks: list[Block], page_markers: bool, front_matter: dict[str, str]) -> str:
    out: list[str] = []
    if front_matter:
        out.append("---")
        out += [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in front_matter.items()]
        out.append("---\n")
    last_page = None
    prev_kind = None
    for b in blocks:
        if page_markers and b.page != last_page:
            out.append(f"<!-- page {b.page} -->\n")
            prev_kind = None
        last_page = b.page
        if b.kind == "heading":
            attr = f" {{#{b.ident}}}" if b.ident else ""
            out.append(f"{'#' * max(1, b.level)} {b.text}{attr}\n")
        elif b.kind == "figure":
            attr = f"{{#{b.ident}}}" if b.ident else ""
            out.append(f"![{b.caption}]({b.text}){attr}\n")
        elif b.kind == "table":
            if b.ident:
                out.append(f"[]{{#{b.ident}}}\n")
            out.append(b.text + "\n")
            if b.caption:
                out.append(f"Table: {b.caption}\n")
        elif b.kind == "footnote":
            out.append(b.text + "\n")
        elif b.kind == "code":
            fence = "~~~" if "```" in b.text else "```"
            out.append(f"{fence}\n{b.text}\n{fence}\n")
        elif b.kind == "list":
            if b.level:  # numbered item: "3. text" or "a) text"
                m = NUMBERED_RE.match(b.text)
                item = f"{b.text[: m.end()].strip()} {b.text[m.end():]}" if m else b.text
            else:
                item = f"- {b.text}"
            if prev_kind == "list" and out and out[-1].endswith("\n"):
                out[-1] = out[-1].rstrip("\n")  # keep list items tight
            out.append(item + "\n")
        else:
            out.append(escape_block_start(b.text) + "\n")
        prev_kind = b.kind
    return "\n".join(out).rstrip() + "\n"


def stitch_pages(per_page: list[list[Block]]) -> list[Block]:
    """Concatenate pages, re-joining paragraphs split by a page break.

    Figures at the top of the next page are moved after the re-joined
    paragraph so the sentence is not interrupted.
    """
    out: list[Block] = []
    for blocks in per_page:
        blocks = list(blocks)
        last = out[-1] if out else None
        if last and last.kind in ("para", "list"):
            lead_figs = []
            while blocks and blocks[0].kind in ("figure", "table"):
                lead_figs.append(blocks.pop(0))
            if blocks and blocks[0].kind == "para" and continues(last.text, blocks[0].text):
                last.text = join_text(last.text, blocks.pop(0).text)
            blocks = lead_figs + blocks
        out += blocks
    return out


def valid_meta(s: str | None) -> str:
    s = (s or "").strip()
    if not s or not all(c.isprintable() for c in s) or sum(c.isalpha() for c in s) < 0.5 * len(s):
        return ""
    return s


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_pages(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages += range(int(a or 1) - 1, min(n, int(b or n)))
        elif part:
            pages.append(int(part) - 1)
    return [p for p in dict.fromkeys(pages) if 0 <= p < n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a PDF to Markdown with JPEG figures, locally (text layer or macOS Vision OCR)."
    )
    ap.add_argument("pdf", nargs="?", type=Path, help="input PDF")
    ap.add_argument("-o", "--outdir", type=Path, help="output directory (default: <pdf dir>/<pdf stem>)")
    ap.add_argument("--pages", help="pages to process, e.g. '1-10,15' (1-based)")
    ap.add_argument(
        "--text-layer",
        choices=["auto", "always", "never"],
        default="auto",
        help="use the PDF's own text instead of OCR: auto = when a page has usable text and is not a scan "
        "(default); always = whenever there is any; never = always OCR",
    )
    ap.add_argument("--dpi", type=int, default=300, help="render resolution for OCR and figures (default 300)")
    ap.add_argument(
        "--lang",
        action="append",
        help="OCR language, repeatable, e.g. --lang en-US --lang fr-FR (default: auto-detect)",
    )
    ap.add_argument("--fast", action="store_true", help="use Vision's fast (less accurate) recogniser")
    ap.add_argument("--min-confidence", type=float, default=0.3, help="drop OCR lines below this confidence")
    ap.add_argument(
        "--figures",
        choices=["auto", "embedded", "ink", "none"],
        default="auto",
        help="figure detection: auto = embedded images/vector drawings, plus ink analysis on scanned pages; "
        "embedded = only embedded images/drawings; ink = image analysis on every page; none = text only",
    )
    ap.add_argument(
        "--tables",
        choices=["auto", "markdown", "image", "text"],
        default="auto",
        help="tables set as text: auto = Markdown tables from the text layer when the columns are clear, "
        "otherwise images (default); markdown = also try OCR'd tables; image = always images; text = leave as text",
    )
    ap.add_argument("--keep-index", action="store_true", help="keep the book's index (dropped by default)")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel OCR jobs (default 4)")
    ap.add_argument("--cache-dir", type=Path, help="OCR cache directory (default: <outdir>/.ocr-cache)")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the OCR cache")
    ap.add_argument("--cache-only", action="store_true", help=argparse.SUPPRESS)  # tests: never call Vision
    ap.add_argument("--epub", action="store_true", help="also build an EPUB with pandoc")
    ap.add_argument("--cover", type=int, default=1, help="page rendered as the EPUB cover (0 = none; default 1)")
    ap.add_argument("--no-outline", action="store_true", help="ignore the PDF's bookmarks when finding headings")
    ap.add_argument("--keep-contents", action="store_true", help="keep table-of-contents pages")
    ap.add_argument("--keep-figure-text", action="store_true", help="also emit text found inside figures")
    ap.add_argument("--keep-headers", action="store_true", help="keep running headers, footers and page numbers")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--inline-images", action="store_true", help="embed JPEGs as base64 data URIs in the Markdown")
    ap.add_argument("--page-markers", action="store_true", help="insert <!-- page N --> comments")
    ap.add_argument("--no-front-matter", action="store_true", help="omit the YAML title/author block")
    ap.add_argument("--list-languages", action="store_true", help="list Vision's recognition languages and exit")
    args = ap.parse_args(argv)

    if args.list_languages:
        print("\n".join(supported_languages(not args.fast)))
        return 0
    if not args.pdf:
        ap.error("a PDF is required")

    doc = pymupdf.open(args.pdf)
    outdir = args.outdir or args.pdf.with_suffix("")
    image_dir = outdir / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.inline_images:
        image_dir.mkdir(exist_ok=True)
        for old in image_dir.glob("page*_fig*.jpg"):  # left over from a previous run
            old.unlink()

    decorations = find_decorations(doc)
    args.doc_text_ok = args.text_layer != "never" and document_text_ok(doc)
    args.cache_dir = None if args.no_cache else (args.cache_dir or outdir / ".ocr-cache")
    toc = [] if args.no_outline else doc.get_toc()
    args.index_pages = set() if args.keep_index else index_pages_from_toc(doc.get_toc(), doc.page_count)
    args.index_open = False
    page_numbers = parse_pages(args.pages, doc.page_count)

    # Pages are rendered and finished in order on this thread (PyMuPDF is not
    # thread-safe); Vision OCR runs for several pages at once in a pool.
    pages: list[PageData] = []
    pending: deque = deque()

    def finish_one():
        prep, fut = pending.popleft()
        if isinstance(prep, PageData):
            pages.append(prep)
            return
        ocr = fut.result() if fut else None
        with objc.autorelease_pool():
            pages.append(finish_page(prep, ocr, args, image_dir, "images"))
        print(f"\rPage {prep.pd.number} ({len(pages)}/{len(page_numbers)})", end="", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for pno in page_numbers:
            page = doc[pno]
            if pno + 1 in args.index_pages:
                pending.append((PageData(pno + 1, page.rect.width, page.rect.height, source="index"), None))
            else:
                with objc.autorelease_pool():
                    prep = prepare_page(page, args, decorations)
                fut = None
                if prep.png is not None:
                    fut = pool.submit(ocr_cached, prep.png, prep.pd.width, prep.pd.height, args, prep.key)
                pending.append((prep, fut))
            while pending and (len(pending) > 2 * max(1, args.jobs) or pending[0][1] is None):
                finish_one()
        while pending:
            finish_one()
    print(file=sys.stderr)

    used = Counter(p.source for p in pages)
    skipped = ", ".join(str(p.number) for p in pages if p.source == "skipped")
    print(
        f"{used['text']} pages from the text layer, {used['ocr']} by OCR, "
        f"{used['skipped']} contents pages skipped{': ' + skipped if skipped else ''}, "
        f"{used['index']} index pages dropped",
        file=sys.stderr,
    )
    pages = [p for p in pages if p.source in ("text", "ocr")]

    if not args.keep_headers:
        remove_running_heads(pages)

    all_lines = [ln for p in pages for ln in p.lines]
    heights = [ln.box.h for ln in all_lines if len(ln.text) >= 20] or [ln.box.h for ln in all_lines] or [12.0]
    sizes = [ln.size for ln in all_lines if ln.size and len(ln.text) >= 20]
    metrics = Metrics(statistics.median(heights), statistics.median(sizes) if sizes else None)

    per_page, notes = [], []
    for p in pages:
        page_notes = extract_footnotes(p, metrics)
        blocks = build_blocks(p, metrics)
        notes += link_footnotes(blocks, page_notes, p.number)
        per_page.append(blocks)
    apply_headings(per_page, toc)
    for blocks in per_page:
        attach_captions(blocks)
    blocks = stitch_pages(per_page)
    assign_ids(blocks)
    link_references(blocks)
    blocks += notes

    front = {}
    if not args.no_front_matter:
        meta = doc.metadata or {}
        first_h1 = next((plain_of(b.text) for b in blocks if b.kind == "heading" and b.level == 1), "")
        front["title"] = valid_meta(meta.get("title")) or first_h1 or args.pdf.stem
        if valid_meta(meta.get("author")):
            front["author"] = valid_meta(meta.get("author"))

    md_path = outdir / f"{args.pdf.stem}.md"
    md_path.write_text(render_markdown(blocks, args.page_markers, front), encoding="utf-8")
    nfig = sum(len(p.figures) for p in pages)
    ntab = sum(len(p.tables) for p in pages)
    print(
        f"Wrote {md_path} ({len(pages)} pages, {nfig} images, {ntab} Markdown tables, {len(notes)} footnotes)",
        file=sys.stderr,
    )

    if args.epub:
        return build_epub(doc, md_path, args)
    return 0


def build_epub(doc: pymupdf.Document, md_path: Path, args: argparse.Namespace) -> int:
    """Run pandoc (locally) on the Markdown, with a rendered page as the cover."""
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("pandoc not found; install it (e.g. brew install pandoc) to use --epub", file=sys.stderr)
        return 1
    outdir = md_path.parent
    cmd = [pandoc, md_path.name, "-o", md_path.with_suffix(".epub").name]
    if 1 <= args.cover <= doc.page_count:
        doc[args.cover - 1].get_pixmap(dpi=150).pil_save(outdir / "cover.jpg", quality=90)
        cmd += ["--epub-cover-image", "cover.jpg"]
    result = subprocess.run(cmd, cwd=outdir)
    if result.returncode == 0:
        print(f"Wrote {md_path.with_suffix('.epub')}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
