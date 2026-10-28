# ocrpdf

Convert a PDF to Markdown entirely on your Mac, with JPEG figures placed
where they occur in the text, ready for `pandoc` → EPUB.

- **Text**: taken from the PDF's own text layer when it has a usable one
  (exact, fast, and it keeps bold, italic and monospaced text). Otherwise
  each page is OCR'd with Apple's Vision framework
  (`VNRecognizeTextRequest`, the same on-device engine as Live Text),
  called through PyObjC. Nothing is uploaded.
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
| `--text-layer auto\|always\|never` | use the PDF's text (default `auto`: when usable and not a hidden OCR layer over a scan); `never` forces OCR |
| `--lang en-US --lang de-DE` | OCR languages (default: auto-detect); `--list-languages` shows them |
| `--dpi 300` | render resolution for OCR and figure crops |
| `--fast` | Vision's faster, less accurate recogniser |
| `--figures auto\|embedded\|ink\|none` | how figures are found (see below) |
| `--tables image\|text` | save tables set as text as images (default) or leave them as text |
| `--no-outline` | don't use the PDF's bookmarks for headings |
| `--keep-contents` | keep table-of-contents pages (skipped by default; the EPUB makes its own) |
| `--inline-images` | embed JPEGs as base64 `data:` URIs instead of separate files |
| `--page-markers` | add `<!-- page N -->` comments |
| `--keep-headers` | keep running headers/footers and page numbers |
| `--keep-figure-text` | also emit text found inside figures (labels etc.) |
| `--no-front-matter` | omit the YAML `title`/`author` block |

## How it works

1. **Text** for each page comes from the text layer (with font size, bold,
   italic and monospace) or from Vision OCR of a render at `--dpi`.
   - Ligatures and TeX-style accents are normalised.
   - Banners and watermarks repeated on many pages are ignored.
2. **Figures**:
   - *Born-digital pages*: embedded images and clusters of visible vector
     drawings (charts) become figures. Short labels hugging a figure, like
     axis ticks, are pulled into it.
   - *Scanned pages*: recognised text is masked out and the remaining large
     blobs of ink become figures.
   - *Tables set as text* (rows of separate short cells) are saved as
     images too.
   - Each figure is cropped from the render as a JPEG; text inside it is
     dropped.
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
   - Line-end hyphens are removed, but the macOS spell checker keeps
     compounds like "data-driven".
   - Running heads and page numbers are removed: lines that repeat in the
     margins, or share a row with a page number.
   - Contents pages are skipped, and paragraphs split by a page break are
     re-joined.

## Limitations

- In OCR mode, headings that differ only by weight (bold, same size) are
  only found via bookmarks or numbering.
- Tables become images, not Markdown tables.
- Figure detection on scans is heuristic. If it misses or over-detects on a
  particular book, try `--figures embedded`/`ink`, or tweak the thresholds
  in `ink_blob_boxes()`.
- Vision and the spell checker do not run inside restrictive sandboxes
  (they need system services); run the script from a normal Terminal.

## Development

- `tests/make_samples.py OUTDIR` generates a born-digital and a scanned test
  PDF (two columns, an image, a vector chart, running heads, a paragraph split
  across pages).
- `uv run --script tests/dump_lines.py file.pdf PAGE` prints Vision's raw
  lines for one page.
