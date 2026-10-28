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
#     "pytest>=8",
# ]
# ///
"""Run the test suite.

    uv run --script tests/run_tests.py            # check
    uv run --script tests/run_tests.py --update   # rewrite expected outputs and the OCR cache
"""

import os
import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--update" in args:
        args.remove("--update")
        os.environ["OCRPDF_UPDATE"] = "1"
    here = Path(__file__).resolve().parent
    sys.exit(pytest.main([str(here), "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning", *args]))
