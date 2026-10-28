# ocrpdf

OCR a PDF entirely on your Mac and produce Markdown with JPEG figures placed
where they occur in the text, ready for `pandoc` → EPUB.

- **Text recognition**: Apple's Vision framework (`VNRecognizeTextRequest`,
  the same on-device engine as Live Text), called through PyObjC. Nothing
  is uploaded.
- **Rendering / images**: PyMuPDF, Pillow, numpy, scipy.
- **No network**: Python-level sockets are disabled when the script starts.
  (`uv` downloads the dependencies the first time you run it; after that it
  runs from the local cache.)

## Usage

```bash
./ocrpdf.py book.pdf
```

This writes `book/book.md` and `book/images/page0001_fig1.jpg`, … next to
the PDF. Then:

```bash
cd book && pandoc book.md -o book.epub
```

Useful options (`./ocrpdf.py -h` lists them all):

| option | effect |
|---|---|
| `-o DIR` | output directory |
| `--pages 1-20,25` | only these pages |
| `--lang en-US --lang de-DE` | recognition languages (default: auto-detect); `--list-languages` shows them |
| `--dpi 300` | render resolution for OCR and figure crops |
| `--fast` | Vision's faster, less accurate recogniser |
| `--figures auto\|embedded\|ink\|none` | how figures are found (see below) |
| `--inline-images` | embed JPEGs as base64 `data:` URIs instead of separate files |
| `--page-markers` | add `<!-- page N -->` comments |
| `--keep-headers` | keep running headers/footers and page numbers |
| `--keep-figure-text` | also emit OCR text found inside figures (labels etc.) |
| `--no-front-matter` | omit the YAML `title`/`author` block (taken from PDF metadata) |

## How it works

1. Each page is rendered at `--dpi` and run through Vision, which returns text
   lines with bounding boxes.
2. **Figures**:
   - *Born-digital pages*: embedded images and clusters of vector drawings
     (charts) become figures.
   - *Scanned pages* (one image covers the page): recognised text is masked
     out and the remaining large blobs of ink become figures.
   - Regions that are mostly text (shaded sidebars) are ignored. Each figure
     is cropped from the render and saved as JPEG; OCR text inside it is
     dropped.
3. **Reading order**: recursive XY-cut over lines and figures (columns first,
   then horizontal bands), so multi-column layouts read correctly and figures
   land between the right paragraphs.
4. **Structure**:
   - Paragraphs: split on vertical gaps, first-line indents and short final
     lines; hyphenation at line ends is re-joined.
   - Headings: from line height relative to body text. Bullet and numbered
     lines become lists.
   - Running heads and page numbers: lines in the top/bottom 10% that repeat
     across pages are removed.
   - Paragraphs split by a page break are re-joined.

## Limitations

- Headings that differ only by weight (bold, same size) are not detected.
- Tables come out as plain lines of text, or as a figure if they are drawn
  with rules.
- Figure detection on scans is heuristic. If it misses or over-detects on a
  particular book, try `--figures embedded`/`ink`, or tweak the thresholds
  in `ink_blob_boxes()`.
- Vision does not run inside restrictive sandboxes (it needs system
  services); run the script from a normal Terminal.

## Development

- `tests/make_samples.py OUTDIR` generates a born-digital and a scanned test
  PDF (two columns, an image, a vector chart, running heads, a paragraph split
  across pages).
- `uv run --script tests/dump_lines.py file.pdf PAGE` prints Vision's raw
  lines for one page.
