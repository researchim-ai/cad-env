"""Pinned technical-font outlines; physical cap height is in drawing millimeters."""

import hashlib
from functools import lru_cache
from html import escape

from .font_cache import font_path, require_font

FONT = font_path()


@lru_cache(maxsize=1)
def font_data():
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont

    font = TTFont(require_font())
    glyphs = font.getGlyphSet()
    pen = BoundsPen(glyphs)
    glyphs["H"].draw(pen)
    return font, glyphs, pen.bounds[3] - pen.bounds[1]


@lru_cache(maxsize=1024)
def glyph(character):
    from fontTools.pens.svgPathPen import SVGPathPen

    font, glyphs, _cap = font_data()
    name = font.getBestCmap().get(ord(character))
    if name is None:
        raise ValueError("Technical font lacks character " + repr(character))
    pen = SVGPathPen(glyphs)
    glyphs[name].draw(pen)
    return pen.getCommands(), font["hmtx"][name][0]


def outline_text(x, y, value, requested_height, anchor="start", **attrs):
    value = str(value)
    height = min((2.5, 3.5, 5, 7, 10), key=lambda h: abs(h - requested_height))
    _, _, cap = font_data()
    scale = height / cap
    width = sum(glyph(c)[1] for c in value) * scale
    shift = 0 if anchor == "start" else -width / 2 if anchor == "middle" else -width
    x += shift
    parts = []
    offset = 0
    for character in value:
        path, advance = glyph(character)
        if path:
            parts.append(f'<path transform="translate({offset} 0)" d="{path}"/>')
        offset += advance
    extra = " ".join(
        f'{k.replace("_", "-")}="{escape(str(v), quote=True)}"'
        for k, v in attrs.items()
        if k not in ("stroke",)
    )
    digest = hashlib.sha256(FONT.read_bytes()).hexdigest()
    return (
        f'<g data-text="{escape(value, quote=True)}" data-cap-height-mm="{height}" data-font-sha256="{digest}" data-text-width-mm="{width}" '
        f'transform="translate({x:g} {y:g}) scale({scale} {-scale})" fill="black" stroke="none" {extra}>'
        + "".join(parts)
        + "</g>"
    )
