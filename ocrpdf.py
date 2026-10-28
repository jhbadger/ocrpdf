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
# ]
# ///
"""OCR a PDF with Apple's on-device Vision framework and write Markdown.

The output is a Markdown file plus an ``images/`` directory of JPEG figures,
referenced from the Markdown at the point where they appear in the reading
order. The result is intended as input for pandoc (``pandoc book.md -o
book.epub``) or similar tools.

Everything runs locally:
  * text recognition uses macOS Vision (VNRecognizeTextRequest), which runs
    on-device;
  * PDF rendering and image handling use PyMuPDF / Pillow / numpy / scipy;
  * Python-level networking is disabled at start-up as a belt-and-braces guard.

Usage:
    ./ocrpdf.py book.pdf                 # writes book/book.md + book/images/
    ./ocrpdf.py book.pdf -o out --pages 1-20 --lang en-US
    ./ocrpdf.py --list-languages
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import socket
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


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
    text: str
    conf: float
    box: Box


@dataclass
class Figure:
    box: Box
    path: str  # relative path of the JPEG (or a data: URI)


@dataclass
class PageData:
    number: int  # 1-based
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)


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
        text = str(cand.string()).strip()
        if not text:
            continue
        bb = obs.boundingBox()  # normalised, origin bottom-left
        x0 = bb.origin.x * width
        x1 = (bb.origin.x + bb.size.width) * width
        y0 = (1.0 - bb.origin.y - bb.size.height) * height
        y1 = (1.0 - bb.origin.y) * height
        lines.append(Line(text, float(cand.confidence()), Box(x0, y0, x1, y1)))
    return lines


# --------------------------------------------------------------------------
# Figure detection
# --------------------------------------------------------------------------


def _rotated(rect: pymupdf.Rect, page: pymupdf.Page) -> Box:
    if page.rotation:
        rect = rect * page.rotation_matrix
    return Box(rect.x0, rect.y0, rect.x1, rect.y1)


def embedded_image_boxes(page: pymupdf.Page) -> list[Box]:
    boxes = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.is_empty or r.is_infinite:
            continue
        boxes.append(_rotated(r, page))
    return boxes


def drawing_cluster_boxes(page: pymupdf.Page) -> list[Box]:
    """Bounding boxes of clusters of vector drawings (charts, diagrams)."""
    try:
        rects = page.cluster_drawings()
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
    page: pymupdf.Page, img: Image.Image, lines: list[Line], mode: str
) -> list[Box]:
    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    candidates: list[Box] = []

    images = embedded_image_boxes(page)
    scanned = any(b.area >= 0.6 * page_area for b in images)

    if mode in ("auto", "ink") and (scanned or mode == "ink"):
        candidates += ink_blob_boxes(img, pw, ph, lines)
    if mode in ("auto", "embedded"):
        candidates += [b for b in images if 0.004 * page_area <= b.area < 0.6 * page_area]
        candidates += [
            b
            for b in drawing_cluster_boxes(page)
            if b.area >= 0.01 * page_area and b.w >= 0.08 * pw and b.h >= 0.04 * ph
        ]

    page_box = Box(0, 0, pw, ph)
    figs = []
    for b in merge_boxes(candidates, gap=0.01 * min(pw, ph)):
        b = Box(max(0, b.x0), max(0, b.y0), min(pw, b.x1), min(ph, b.y1))
        if b.area < 0.004 * page_area or b.area > 0.97 * page_box.area:
            continue
        # Shaded sidebars / text boxes are mostly text, not pictures.
        if text_coverage(b, lines) > 0.4:
            continue
        figs.append(b.expand(0.004 * min(pw, ph)))
    return figs


# --------------------------------------------------------------------------
# Pass 1: render, OCR and extract figures per page
# --------------------------------------------------------------------------


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


def process_page(
    page: pymupdf.Page,
    args: argparse.Namespace,
    image_dir: Path,
    image_prefix: str,
) -> PageData:
    pw, ph = page.rect.width, page.rect.height
    pix = page.get_pixmap(dpi=args.dpi, colorspace=pymupdf.csRGB, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    lines = ocr_image(pix.tobytes("png"), pw, ph, args.lang, args.fast)
    lines = drop_phantom_lines([ln for ln in lines if ln.conf >= args.min_confidence])

    pd = PageData(page.number + 1, pw, ph)
    fig_boxes = [] if args.figures == "none" else find_figures(page, img, lines, args.figures)
    fig_boxes.sort(key=lambda b: (b.y0, b.x0))

    kx, ky = pix.width / pw, pix.height / ph
    for i, fb in enumerate(fig_boxes, 1):
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

    # Text inside a figure (axis labels, text in photos) belongs to the figure.
    if not args.keep_figure_text:
        lines = [
            ln for ln in lines if not any(f.box.contains_point(ln.box.cx, ln.box.cy) for f in pd.figures)
        ]
    pd.lines = lines
    return pd


# --------------------------------------------------------------------------
# Pass 2: running headers/footers, reading order, paragraphs
# --------------------------------------------------------------------------

PAGE_NUM_RE = re.compile(
    r"^(page\s+)?([\divxlcdm]+)(\s+(of|/)\s+\d+)?$", re.IGNORECASE
)


def _norm_header(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip(" .-–—|")


def remove_running_heads(pages: list[PageData], zone: float = 0.1) -> None:
    def in_zone(ln: Line, p: PageData) -> bool:
        return ln.box.cy < p.height * zone or ln.box.cy > p.height * (1 - zone)

    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({_norm_header(ln.text) for ln in p.lines if in_zone(ln, p)})
    threshold = max(2, int(0.2 * len(pages)))
    for p in pages:
        p.lines = [
            ln
            for ln in p.lines
            if not (
                in_zone(ln, p)
                and (
                    PAGE_NUM_RE.match(ln.text.strip(" .-–—|"))
                    or counts[_norm_header(ln.text)] >= threshold
                )
            )
        ]


@dataclass
class Item:
    box: Box
    line: Line | None = None
    figure: Figure | None = None
    col: Box | None = None  # extent of the column the item was placed in


def _gaps(items: list[Item], axis: int, min_gap: float) -> list[tuple[float, float]]:
    spans = sorted(
        ((it.box.x0, it.box.x1) if axis == 0 else (it.box.y0, it.box.y1)) for it in items
    )
    gaps = []
    end = spans[0][1]
    for s, e in spans[1:]:
        if s - end >= min_gap:
            gaps.append((end, s))
        end = max(end, e)
    return gaps


def _bbox(items: list[Item]) -> Box:
    b = items[0].box
    for it in items[1:]:
        b = b.union(it.box)
    return b


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
        out = []
        edges = [-1e9] + [(a + b) / 2 for a, b in gaps] + [1e9]
        for lo, hi in zip(edges, edges[1:]):
            part = [it for it in items if lo <= it.box.cx < hi]
            if part:
                out += xy_cut(part, min_gx, min_gy, _bbox(part))
        return out

    gaps = _gaps(items, 1, min_gy)
    if gaps:
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        cut = (a + b) / 2
        top = [it for it in items if it.box.cy < cut]
        bottom = [it for it in items if it.box.cy >= cut]
        return xy_cut(top, min_gx, min_gy, col) + xy_cut(bottom, min_gx, min_gy, col)

    # No cut possible: group into rows and read each row left to right.
    rows: list[list[Item]] = []
    for it in sorted(items, key=lambda i: i.box.cy):
        if rows:
            r = _bbox(rows[-1])
            overlap = min(r.y1, it.box.y1) - max(r.y0, it.box.y0)
            if overlap > 0.5 * min(r.h, it.box.h):
                rows[-1].append(it)
                continue
        rows.append([it])
    out = []
    for row in rows:
        row.sort(key=lambda i: i.box.x0)
        texts = [i for i in row if i.line]
        out += [i for i in row if i.figure]
        if texts:
            merged = Line(
                " ".join(i.line.text for i in texts),
                min(i.line.conf for i in texts),
                _bbox(texts),
            )
            out.append(Item(merged.box, line=merged))
    for it in out:
        it.col = col
    return out


TERMINAL = tuple('.!?:;"”’)]»…')
BULLET_RE = re.compile(r"^\s*([•●◦▪▫■□‣∙·\-–*])\s+")
NUMBERED_RE = re.compile(r"^\s*(\d{1,3}|[a-zA-Z])[.)]\s+")


def join_text(a: str, b: str) -> str:
    if not a:
        return b
    if re.search(r"[A-Za-zÀ-ÿ][-­]$", a) and b[:1].islower():
        return a[:-1] + b
    return a + " " + b


def continues(prev: str, nxt: str) -> bool:
    return not prev.rstrip().endswith(TERMINAL) and nxt[:1].islower()


@dataclass
class Block:
    kind: str  # "para", "heading", "list", "figure"
    text: str = ""
    level: int = 0
    page: int = 0


def heading_level(h: float, body_h: float) -> int:
    r = h / body_h
    if r < 1.35:
        return 0
    if r >= 2.0:
        return 1
    if r >= 1.6:
        return 2
    return 3


def is_minor_heading(ordered: list[Item], idx: int, body_h: float) -> bool:
    """A slightly enlarged, short line standing apart from its neighbours."""
    it = ordered[idx]
    b, col = it.box, it.col or it.box
    if b.h < 1.15 * body_h or b.w > 0.7 * col.w or it.line.text.rstrip().endswith((".", ",", ";")):
        return False
    prev = ordered[idx - 1] if idx > 0 else None
    nxt = ordered[idx + 1] if idx + 1 < len(ordered) else None
    gap_above = b.y0 - prev.box.y1 if prev and prev.line and prev.box.y1 <= b.y0 else 1e9
    gap_below = nxt.box.y0 - b.y1 if nxt and nxt.line and nxt.box.y0 >= b.y1 else 1e9
    return gap_above >= 0.6 * body_h and gap_below >= 0.4 * body_h


def build_blocks(p: PageData, body_h: float) -> list[Block]:
    items = [Item(ln.box, line=ln) for ln in p.lines] + [Item(f.box, figure=f) for f in p.figures]
    if not items:
        return []
    ordered = xy_cut(items, min_gx=max(0.012 * p.width, 0.6 * body_h), min_gy=0.5 * body_h)

    blocks: list[Block] = []
    cur: Block | None = None
    prev: Item | None = None

    def flush():
        nonlocal cur
        if cur and cur.text:
            blocks.append(cur)
        cur = None

    for idx, it in enumerate(ordered):
        if it.figure:
            flush()
            blocks.append(Block("figure", it.figure.path, page=p.number))
            prev = None
            continue

        ln = it.line
        text = ln.text
        level = 0
        if len(text.split()) <= 15:
            level = heading_level(ln.box.h, body_h) or (3 if is_minor_heading(ordered, idx, body_h) else 0)
        if level:
            if cur and cur.kind == "heading" and cur.level == level and prev and (
                ln.box.y0 - prev.box.y1 < 0.8 * ln.box.h
            ):
                cur.text = join_text(cur.text, text)
            else:
                flush()
                cur = Block("heading", text, level, p.number)
            prev = it
            continue

        bullet = BULLET_RE.match(text) or NUMBERED_RE.match(text)
        new_para = cur is None or cur.kind == "heading" or bullet is not None
        if not new_para and prev is not None:
            pb, b = prev.box, ln.box
            col = it.col or b
            moved_up = b.y0 < pb.y0 - 0.5 * body_h
            if moved_up or (prev.col is not it.col and prev.col != it.col):
                new_para = not continues(cur.text, text)
            elif b.y0 - pb.y1 > 0.75 * body_h:
                new_para = True
            elif b.x0 - pb.x0 > 0.8 * body_h and b.x0 - col.x0 > 0.8 * body_h:
                new_para = True  # first-line indent
            elif pb.x1 < col.x1 - 0.15 * col.w and cur.text.rstrip().endswith(TERMINAL):
                new_para = True  # previous line ended short
        if new_para:
            flush()
            if bullet:
                if BULLET_RE.match(text):
                    text = BULLET_RE.sub("", text, count=1)
                    cur = Block("list", text, page=p.number)
                else:
                    cur = Block("list", text, level=1, page=p.number)  # numbered: keep label
            else:
                cur = Block("para", text, page=p.number)
        else:
            cur.text = join_text(cur.text, text)
        prev = it
    flush()
    return blocks


# --------------------------------------------------------------------------
# Markdown output
# --------------------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>])")


def md_escape(text: str) -> str:
    text = _MD_SPECIAL.sub(r"\\\1", text)
    text = re.sub(r"^(#+|[+\-=]|\d+[.)])(\s)", r"\\\1\2", text)
    return text


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
        last_page = b.page
        if b.kind == "heading":
            out.append(f"{'#' * b.level} {md_escape(b.text)}\n")
        elif b.kind == "figure":
            out.append(f"![]({b.text})\n")
        elif b.kind == "list":
            if b.level:  # numbered item: "3. text" or "a) text"
                m = NUMBERED_RE.match(b.text)
                label, rest = b.text[: m.end()].strip(), b.text[m.end() :]
                item = f"{label} {md_escape(rest)}"
            else:
                item = f"- {md_escape(b.text)}"
            if prev_kind == "list" and out and out[-1].endswith("\n"):
                out[-1] = out[-1].rstrip("\n")  # keep list items tight
            out.append(item + "\n")
        else:
            out.append(md_escape(b.text) + "\n")
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
            while blocks and blocks[0].kind == "figure":
                lead_figs.append(blocks.pop(0))
            if blocks and blocks[0].kind == "para" and continues(last.text, blocks[0].text):
                last.text = join_text(last.text, blocks.pop(0).text)
            blocks = lead_figs + blocks
        out += blocks
    return out


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
        description="OCR a PDF locally with macOS Vision and produce Markdown with JPEG figures."
    )
    ap.add_argument("pdf", nargs="?", type=Path, help="input PDF")
    ap.add_argument("-o", "--outdir", type=Path, help="output directory (default: <pdf dir>/<pdf stem>)")
    ap.add_argument("--pages", help="pages to process, e.g. '1-10,15' (1-based)")
    ap.add_argument("--dpi", type=int, default=300, help="render resolution for OCR and figures (default 300)")
    ap.add_argument(
        "--lang",
        action="append",
        help="recognition language, repeatable, e.g. --lang en-US --lang fr-FR (default: auto-detect)",
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
    ap.add_argument("--keep-figure-text", action="store_true", help="also emit OCR text found inside figures")
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

    page_numbers = parse_pages(args.pages, doc.page_count)
    pages: list[PageData] = []
    for k, pno in enumerate(page_numbers, 1):
        print(f"\rOCR page {pno + 1} ({k}/{len(page_numbers)})", end="", file=sys.stderr, flush=True)
        with objc.autorelease_pool():
            pages.append(process_page(doc[pno], args, image_dir, "images"))
    print(file=sys.stderr)

    if not args.keep_headers:
        remove_running_heads(pages)

    heights = [ln.box.h for p in pages for ln in p.lines if len(ln.text) >= 20]
    heights = heights or [ln.box.h for p in pages for ln in p.lines] or [12.0]
    body_h = statistics.median(heights)

    blocks = stitch_pages([build_blocks(p, body_h) for p in pages])

    front = {}
    if not args.no_front_matter:
        meta = doc.metadata or {}
        front["title"] = (meta.get("title") or "").strip() or args.pdf.stem
        if (meta.get("author") or "").strip():
            front["author"] = meta["author"].strip()

    md_path = outdir / f"{args.pdf.stem}.md"
    md_path.write_text(render_markdown(blocks, args.page_markers, front), encoding="utf-8")
    nfig = sum(len(p.figures) for p in pages)
    print(f"Wrote {md_path} ({len(pages)} pages, {nfig} figures)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
