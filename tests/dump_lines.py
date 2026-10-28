# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf>=1.24", "pyobjc-framework-Vision>=10", "pyobjc-framework-Quartz>=10", "pillow", "numpy", "scipy"]
# ///
"""Debug helper: print Vision's raw OCR lines for one page as JSON.

    uv run --script tests/dump_lines.py file.pdf PAGE [DPI]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ocrpdf, pymupdf

doc = pymupdf.open(sys.argv[1])
page = doc[int(sys.argv[2]) - 1]
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
for ln in ocrpdf.ocr_image(pix.tobytes("png"), page.rect.width, page.rect.height, None, False):
    b = ln.box
    print(json.dumps([ln.text, round(ln.conf, 2), round(b.x0), round(b.y0), round(b.x1), round(b.y1)]))
