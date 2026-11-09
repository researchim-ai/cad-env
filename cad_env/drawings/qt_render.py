"""System Qt renderer: SVG -> vector PDF + page PNG, without a display server."""

import json
import sys
from pathlib import Path

from PySide2 import QtCore, QtGui, QtSvg


def main(request_file):
    request = json.loads(Path(request_file).read_text())
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(
        ["cad-env-drawings"]
    )
    writer = QtGui.QPdfWriter(request["pdf"])
    layout = QtGui.QPageLayout(
        QtGui.QPageSize(QtGui.QPageSize.A3),
        QtGui.QPageLayout.Landscape,
        QtCore.QMarginsF(0, 0, 0, 0),
        QtGui.QPageLayout.Millimeter,
    )
    if not writer.setPageLayout(layout):
        raise RuntimeError("Cannot set A3 landscape page layout")
    writer.setResolution(144)
    writer.setTitle(request["title"])
    painter = QtGui.QPainter(writer)
    if not painter.isActive():
        raise RuntimeError("Cannot create PDF painter")
    for index, path in enumerate(request["svgs"]):
        renderer = QtSvg.QSvgRenderer(str(path))
        if not renderer.isValid():
            raise ValueError("Invalid SVG: " + str(path))
        if index:
            writer.newPage()
        renderer.render(painter, QtCore.QRectF(0, 0, writer.width(), writer.height()))
        width = request.get("png_width", 2400)
        height = round(width * 297 / 420)
        image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        image.fill(QtCore.Qt.white)
        ip = QtGui.QPainter(image)
        renderer.render(ip, QtCore.QRectF(0, 0, width, height))
        ip.end()
        if not image.save(str(Path(path).with_suffix(".png"))):
            raise RuntimeError("Cannot write PNG")
    painter.end()
    del app


if __name__ == "__main__":
    main(sys.argv[1])
