"""scripts/fix_slide_fonts.py: web fonts become the deck's fallbacks, single words stop wrapping."""

import importlib.util
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SLIDE = (
    '<p:sld><p:sp><p:txBody><a:bodyPr wrap="square"/><a:p><a:r>'
    '<a:rPr><a:latin typeface="Source Serif 4"/></a:rPr><a:t>0.723</a:t></a:r></a:p></p:txBody></p:sp>'
    '<p:sp><p:txBody><a:bodyPr wrap="square"/><a:p><a:r>'
    '<a:rPr><a:latin typeface="IBM Plex Sans"/></a:rPr><a:t>Stronger encoder, dense alone</a:t></a:r></a:p>'
    '</p:txBody></p:sp></p:sld>'
)


def _fix_slide_fonts():
    spec = importlib.util.spec_from_file_location("fix_slide_fonts", REPO / "scripts" / "fix_slide_fonts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fonts_replaced_and_single_words_do_not_wrap(tmp_path):
    src, dst = tmp_path / "download.pptx", tmp_path / "Slides.pptx"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("ppt/slides/slide8.xml", SLIDE)
        z.writestr("ppt/theme/theme1.xml", '<a:latin typeface="IBM Plex Mono"/>')
        z.writestr("ppt/media/image1.png", b"\x89PNG not xml")

    counts = _fix_slide_fonts().fix(src, dst)

    with zipfile.ZipFile(dst) as z:
        slide = z.read("ppt/slides/slide8.xml").decode("utf-8")
        assert 'typeface="Georgia"' in slide and 'typeface="Arial"' in slide
        assert 'typeface="Courier New"' in z.read("ppt/theme/theme1.xml").decode("utf-8")
        assert z.read("ppt/media/image1.png") == b"\x89PNG not xml"          # binaries untouched
    shapes = slide.split("<p:sp>")
    assert 'wrap="none"' in shapes[1]          # "0.723": one word, must not split
    assert 'wrap="square"' in shapes[2]        # a sentence still wraps
    assert counts == {"fonts": 3, "no_wrap_boxes": 1}
    assert "Source Serif 4" in zipfile.ZipFile(src).read("ppt/slides/slide8.xml").decode("utf-8")  # input kept
