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
"""Before/after snapshots of real books, which can't be committed as fixtures.

    uv run --script tests/snapshot.py save book.pdf [ocrpdf options...]
    uv run --script tests/snapshot.py diff book.pdf [ocrpdf options...]

``save`` converts the book into out/snapshots/<stem>/baseline; ``diff``
converts it again into .../current and shows how the Markdown changed. OCR
results are cached in out/snapshots/<stem>/ocr-cache, so repeated runs only
re-do the layout.
"""

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ocrpdf  # noqa: E402


def convert(pdf: Path, outdir: Path, extra: list[str]) -> str:
    cache = outdir.parent / "ocr-cache"
    if ocrpdf.main([str(pdf), "-o", str(outdir), "--cache-dir", str(cache), *extra]) != 0:
        sys.exit("conversion failed")
    return (outdir / f"{pdf.stem}.md").read_text()


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("save", "diff"):
        sys.exit(__doc__)
    action, pdf, extra = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    base = ROOT / "out" / "snapshots" / pdf.stem
    if action == "save":
        convert(pdf, base / "baseline", extra)
        print(f"Saved baseline in {base / 'baseline'}")
        return
    baseline = base / "baseline" / f"{pdf.stem}.md"
    if not baseline.exists():
        sys.exit(f"no baseline; run: tests/snapshot.py save {pdf}")
    before = baseline.read_text().splitlines(keepends=True)
    after = convert(pdf, base / "current", extra).splitlines(keepends=True)
    diff = list(difflib.unified_diff(before, after, "baseline", "current", n=1))
    sys.stdout.writelines(diff)
    print(f"\n{sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} lines added, "
          f"{sum(1 for d in diff if d.startswith('-') and not d.startswith('---'))} removed")


if __name__ == "__main__":
    main()
