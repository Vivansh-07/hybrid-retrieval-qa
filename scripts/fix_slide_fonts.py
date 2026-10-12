#!/usr/bin/env python
"""Make the downloaded slide deck render properly on any Windows PC.

    python scripts/fix_slide_fonts.py "<downloaded deck>.pptx" Slides.pptx

The deck uses three web fonts: Source Serif 4, IBM Plex Sans and IBM Plex Mono. The
PowerPoint download names them but does not embed them, so on a PC without them PowerPoint
substitutes other fonts. Checked on 11 Oct: the formula on slide 5 ran together, and words
such as "What" and "limits" rendered with gaps inside them. This writes a copy that names
the deck's own fallback fonts instead (Georgia, Arial and Courier New, which every Windows
PC has). It also stops text boxes that hold a single word or number from wrapping, because
the wider fallback digits otherwise split "0.723" over two lines (slide 8). The input file
is not changed.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

# the fallbacks the deck itself names, e.g. font-family: 'Source Serif 4', Georgia, serif
FALLBACKS = {
    "Source Serif 4": "Georgia",
    "IBM Plex Sans": "Arial",
    "IBM Plex Mono": "Courier New",
}
SLIDE = re.compile(r"ppt/slides/slide\d+\.xml$")


def _no_wrap_single_words(xml: str) -> tuple[str, int]:
    """Set wrap="none" on text boxes whose whole text is one word: it cannot wrap usefully."""
    shapes = re.split(r"(?=<p:sp>)", xml)
    changed = 0
    for i, shape in enumerate(shapes):
        text = "".join(re.findall(r"<a:t>([^<]*)</a:t>", shape)).strip()
        if text and not re.search(r"\s", text) and len(text) <= 12 and '<a:bodyPr wrap="square"' in shape:
            shapes[i] = shape.replace('<a:bodyPr wrap="square"', '<a:bodyPr wrap="none"', 1)
            changed += 1
    return "".join(shapes), changed


def fix(src: Path, dst: Path) -> dict:
    """Write the fixed copy of `src` to `dst`; return how many fonts and boxes were changed."""
    counts = {"fonts": 0, "no_wrap_boxes": 0}
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".xml"):
                xml = data.decode("utf-8")
                for font, fallback in FALLBACKS.items():
                    counts["fonts"] += xml.count(f'typeface="{font}"')
                    xml = xml.replace(f'typeface="{font}"', f'typeface="{fallback}"')
                if SLIDE.match(item.filename):
                    xml, n = _no_wrap_single_words(xml)
                    counts["no_wrap_boxes"] += n
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the deck as downloaded (.pptx)")
    ap.add_argument("dst", help="where to write the fixed copy, e.g. Slides.pptx")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        ap.error("write the fixed copy to a different file; the download is kept as it is")
    counts = fix(src, dst)
    print(f"written {dst}: {counts['fonts']} font references replaced, "
          f"{counts['no_wrap_boxes']} single-word text boxes set not to wrap")


if __name__ == "__main__":
    main()
