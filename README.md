# ocrpdf

Convert a PDF to Markdown entirely on your Mac, with JPEG figures placed
where they occur in the text, ready for `pandoc` → EPUB.

- **Text**: taken from the PDF's own text layer when it has a usable one
  (exact, fast, and it keeps bold, italic, monospace, superscripts and
  subscripts). Otherwise each page is OCR'd with Apple's Vision framework
  (`VNRecognizeTextRequest`, the same on-device engine as Live Text),
  called through PyObjC, several pages at a time. Nothing is uploaded.
- **Rendering / images**: PyMuPDF, Pillow, numpy, scipy.
- **No network**: Python-level sockets are disabled when the script starts.
  (`uv` downloads the dependencies the first time you run it; after that it
  runs from the local cache.)

## Usage

```bash
./ocrpdf.py book.pdf --epub
```

This writes `book/book.md`, `book/images/page0001_fig1.jpg`, … and, with
`--epub`, `book/book.epub` (built by a local `pandoc`, with page 1 as the
cover). Without `--epub` you can run pandoc yourself:

```bash
cd book && pandoc book.md -o book.epub
```

Useful options (`./ocrpdf.py -h` lists them all):

| option | effect |
|---|---|
| `-o DIR` | output directory |
| `--pages 1-20,25` | only these pages |
| `--epub`, `--cover N` | also build an EPUB; page N as its cover (`0` = none) |
| `--text-layer auto\|always\|never` | use the PDF's text (default `auto`: when usable and not a hidden OCR layer over a scan); `never` forces OCR |
| `--lang en-US --lang de-DE` | OCR languages (default: auto-detect); `--list-languages` shows them |
| `--jobs 4` | pages OCR'd in parallel |
| `--cache-dir DIR`, `--no-cache` | where OCR results are cached (default `<outdir>/.ocr-cache`) |
| `--dpi 300` | render resolution for OCR (lower values cost accuracy) |
| `--image-dpi 200`, `--jpeg-quality 75` | figure image resolution and quality (see below) |
| `--fast` | Vision's faster, less accurate recogniser |
| `--figures auto\|embedded\|ink\|none` | how figures are found (see below) |
| `--tables auto\|markdown\|image\|text` | text tables: Markdown tables when the text layer has clear columns, else images (default `auto`) |
| `--no-outline` | don't use the PDF's bookmarks for headings |
| `--keep-contents` | keep table-of-contents pages (skipped by default; the EPUB makes its own) |
| `--keep-index` | keep the index (dropped by default; its page numbers mean nothing in an EPUB) |
| `--inline-images` | embed JPEGs as base64 `data:` URIs instead of separate files |
| `--page-markers` | add `<!-- page N -->` comments |
| `--keep-headers` | keep running headers/footers and page numbers |
| `--keep-figure-text` | also emit text found inside figures (labels etc.) |
| `--no-front-matter` | omit the YAML `title`/`author` block |

## How it works

1. **Text** for each page comes from the text layer (with font size, bold,
   italic, monospace and sub/superscript position) or from Vision OCR of a
   render at `--dpi`.
   - OCR runs in a thread pool, and results are cached per page, so
     re-running after a change only redoes the layout.
   - Ligatures and TeX-style accents are normalised.
   - Banners and watermarks repeated on many pages are ignored.
2. **Figures and tables**:
   - *Born-digital pages*: embedded images and clusters of visible vector
     drawings (charts) become figures. Short labels hugging a figure, like
     axis ticks, are pulled into it.
   - *Scanned pages*: recognised text is masked out and the remaining large
     blobs of ink become figures.
   - *Tables set as text* (rows of separate short cells) become Markdown
     tables when their columns are clear in the text layer, otherwise
     images.
   - Each figure is cropped as a JPEG from a render at `--image-dpi`;
     text inside it is dropped.
3. **Reading order**: recursive XY-cut over lines and figures (columns first,
   then horizontal bands), so multi-column layouts read correctly and figures
   land between the right paragraphs.
4. **Structure**:
   - Headings: when the PDF has bookmarks, text matching a bookmark becomes
     a heading at the bookmark's depth, and "Chapter 2" + title merge into
     one `#` heading. Other bold or enlarged standalone lines become
     subheadings. Without bookmarks, numbering ("2.1", "2.2.1") sets the
     depth.
   - Paragraphs: split on vertical gaps, first-line indents and short final
     lines. Hanging indents are kept together.
   - Bullets and numbered items become lists.
   - Monospaced blocks become code blocks with their alignment preserved.
   - Superscripts and subscripts become pandoc's `^sup^` and `~sub~`.
   - Footnotes: small print at the foot of a page, whose markers appear as
     superscripts in the text, becomes pandoc footnotes. Text layer only.
   - Line-end hyphens are removed, but the macOS spell checker keeps
     compounds like "data-driven".
   - Running heads and page numbers are removed: lines that repeat in the
     margins, or share a row with a page number.
   - Contents pages and the index are dropped, and paragraphs split by a
     page break are re-joined.
5. **Captions and links**:
   - A "Figure 3.9: …" / "Table 2.1: …" / "Plate 8 …" paragraph next to a
     figure or table becomes its caption, with an anchor (`#fig-3-9`,
     `#tab-2-1`, `#plate-8`).
   - Chapters and numbered sections get anchors too (`#chap-4`, `#sec-2-1`).
   - Mentions like "see figure 3.9" or "chapter 4" become links to them,
     which pandoc keeps working across the EPUB's chapter files.

## Image size

Figures are rendered at `--image-dpi` (default 200) and saved at
`--jpeg-quality` (default 75), independently of the 300 dpi used for OCR.
Common e-readers are about 1070–1270 pixels wide, and 200 dpi makes a
full-width figure from a 6-inch-wide book about 1150 px wide. On a 330-page
book with 140 figures, the images take:

| `--image-dpi` | quality 85 | quality 75 | quality 65 |
|---|---:|---:|---:|
| 300 | 16.9 MB | 13.7 MB | 11.9 MB |
| 200 | 8.9 MB | 7.2 MB | 6.3 MB |
| 150 | 5.9 MB | 4.8 MB | 4.2 MB |

`--image-dpi 150 --jpeg-quality 65` is the smallest that keeps diagram labels
readable; use 300 if the EPUB is mainly for tablets or zooming.

## Limitations

- In OCR mode there is no font information, so there are no superscripts,
  footnotes or code blocks, and headings that differ only by weight are
  only found via bookmarks or numbering.
- Tables with merged or irregular cells are saved as images.
- Figure detection on scans is heuristic. If it misses or over-detects on a
  particular book, try `--figures embedded`/`ink`, or tweak the thresholds
  in `ink_blob_boxes()`.
- Vision and the spell checker do not run inside restrictive sandboxes
  (they need system services); run the script from a normal Terminal.

## Development

- `uv run --script tests/run_tests.py` runs the unit and regression tests.
  - The regression tests convert the synthetic PDFs in `tests/fixtures/` and
    compare the Markdown with `tests/fixtures/expected/`.
  - OCR results come from `tests/fixtures/ocr-cache/`, so the tests never
    call Vision.
  - After an intended change in output, run with `--update` and review the
    diff of `tests/fixtures/expected/`.
- `tests/make_samples.py OUTDIR` regenerates the synthetic PDFs. If you
  regenerate the fixtures, also run the tests with `--update` to refresh
  the OCR cache.
- `tests/snapshot.py save|diff book.pdf` keeps a before/after snapshot of a
  real book under `out/snapshots/` (git-ignored), to see how a change
  affects it.
- `uv run --script tests/dump_lines.py file.pdf PAGE` prints Vision's raw
  lines for one page.
