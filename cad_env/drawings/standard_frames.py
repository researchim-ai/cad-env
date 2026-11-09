"""Separate title-block renderers for ISO data fields and Russian ESKD forms."""

from .profiles import ACTIVE, METADATA

TRANSLATIONS = {
    "ORTHOGONAL VIEWS AND SECTION": "ВИДЫ И РАЗРЕЗ",
    "Model-derived hidden-line removal; identical scale for all views.": "Проекции модели. Единый масштаб изображений.",
    "FEATURE AND DRILLING SCHEDULE": "ТАБЛИЦА ЭЛЕМЕНТОВ И ОТВЕРСТИЙ",
    "Coordinates in the global model XY system. Diameters and depths in mm.": "Координаты X, Y в системе модели. Размеры в мм.",
    "PAD: height upward from support. HOLE: depth downward from support top.": "Выступ: высота от опоры вверх. Отверстие: глубина от опоры вниз.",
    "CB: counterbore diameter / depth from the same support top.": "Цековка: диаметр / глубина от той же поверхности опоры.",
    "Do not merge holes with different depths into a single through cut.": "Отверстия различной глубины не объединять сквозным вырезом.",
    "All listed rows are required. Names identify feature supports, not operation order.": "Все строки обязательны. Имена обозначают элементы и опоры.",
    "No fit, surface finish, thread or material requirements are inferred from nominal geometry.": "Посадки, шероховатость, резьба и материал не заданы.",
    "TYPE": "ТИП",
    "FEATURE": "ЭЛЕМЕНТ",
    "SUPPORT": "ОПОРА",
    "SIZE": "РАЗМЕР",
    "H / DEPTH": "ВЫС. / ГЛУБ.",
    "CB (D / DEPTH)": "ЦЕКОВКА (⌀ / ГЛУБ.)",
    "BASE": "ОСН.",
    "PAD": "ВЫСТУП",
    "HOLE": "ОТВ.",
}
TITLES = {
    "heterogeneous_plate": ("Drilled plate", "Плита с отверстиями"),
    "multi_support_fixture": ("Fixture body", "Корпус приспособления"),
    "terraced_flange_stack": ("Stepped flange", "Ступенчатый фланец"),
}


def render_frame(identity, title, sheet, total, scale):
    from .lettering import font_data, glyph
    from .sheets import line, text

    profile = ACTIVE.get()
    m = METADATA.get() or {}
    left, top, right, bottom = profile.margins
    drawing_title = (
        m.get("title")
        or TITLES.get(title, (title.replace("_", " "), title.replace("_", " ")))[
            profile.key == "gost"
        ]
    )
    designation = m.get("designation") or identity

    def field(value, width, height):
        """Fail rather than clipping document metadata or inventing abbreviations."""
        value = value or ""
        cap = font_data()[2]
        actual = sum(glyph(c)[1] for c in value) * height / cap
        if actual > width:
            raise ValueError(f"Title-block value exceeds {width} mm cell: {value!r}")
        return value

    body = f'<rect id="standard-border" x="{left}" y="{top}" width="{420 - left - right}" height="{297 - top - bottom}" fill="none" stroke-width="{profile.wide}"/>'

    def box(x, y, w, h, kind):
        return f'<rect id="{kind}" x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke-width="{profile.wide}"/>'

    if profile.key == "iso":
        x, y = 230, 237
        body += box(x, y, 180, 50, "iso-title-block")
        for offset in (10, 22, 34, 44):
            body += line(x, y + offset, x + 180, y + offset)
        for offset, begin, end in [
            (110, 0, 10),
            (120, 22, 34),
            (140, 22, 34),
            (60, 34, 44),
            (120, 34, 44),
            (90, 44, 50),
            (130, 44, 50),
            (150, 44, 50),
        ]:
            body += line(x + offset, y + begin, x + offset, y + end)
        for label, value, xx, yy in [
            ("OWNER", m.get("owner", ""), x + 2, y + 3),
            ("DOCUMENT TYPE", m.get("document_type", ""), x + 112, y + 3),
            ("TITLE", drawing_title, x + 2, y + 13),
            ("IDENTIFICATION", designation, x + 2, y + 25),
            ("REV", m.get("revision", ""), x + 122, y + 25),
            ("SHEET", f"{sheet}/{total}", x + 142, y + 25),
            ("CREATOR", m.get("creator", ""), x + 2, y + 37),
            ("APPROVAL", m.get("approver", ""), x + 62, y + 37),
            ("ISSUE DATE", m.get("issue_date", ""), x + 122, y + 37),
        ]:
            width = {
                "OWNER": 106,
                "DOCUMENT TYPE": 66,
                "TITLE": 176,
                "IDENTIFICATION": 116,
                "REV": 16,
                "SHEET": 36,
            }.get(label, 56)
            body += text(xx, yy, label, 2.5) + text(
                xx, yy + 5, field(value, width, 3.5), 3.5
            )
        body += text(x + 2, y + 48, m.get("status", "DRAFT"), 2.5)
        body += (
            text(x + 92, y + 48, "SCALE " + scale, 2.5)
            + text(x + 132, y + 48, "mm", 2.5)
            + text(x + 152, y + 48, "1st angle", 2.5)
        )
        body += text(407, 293, "A3", 2.5, "end")
    else:
        x = 230
        if sheet == 1:
            y = 237
            body += box(x, y, 185, 55, "gost-form-1")
            body += line(x + 65, y, x + 65, y + 55)
            # Change rows and signing rows, in 5 mm increments.
            for i in range(1, 11):
                body += line(x, y + 5 * i, x + 65, y + 5 * i)
            for xx in (7, 17, 40, 55):
                body += line(x + xx, y, x + xx, y + 30)
            for xx in (17, 40, 55):
                body += line(x + xx, y + 30, x + xx, y + 55)
            for offset in (15, 40):
                body += line(x + 65, y + offset, x + 135, y + offset)
            body += line(x + 135, y + 15, x + 185, y + 15) + line(
                x + 65, y + 15, x + 185, y + 15
            )
            body += line(x + 135, y + 30, x + 185, y + 30) + line(
                x + 135, y + 35, x + 185, y + 35
            )
            body += line(x + 135, y + 15, x + 135, y + 55)
            for xx in (150, 167):
                body += line(x + xx, y + 15, x + xx, y + 30)
            body += line(x + 135, y + 20, x + 185, y + 20)
            body += line(x + 155, y + 30, x + 155, y + 35)
            body += line(x + 175, y, x + 175, y + 15)
            body += (
                text(x + 67, y + 10, field(designation, 106, 5), 5)
                + text(x + 176, y + 3, "Вер.", 2.5)
                + text(x + 176, y + 10, m.get("version", ""), 3.5)
            )
            body += text(x + 100, y + 27, field(drawing_title, 66, 3.5), 3.5, "middle")
            body += text(
                x + 100, y + 49, field(m.get("material"), 66, 3.5), 3.5, "middle"
            )
            for label, xx, value in [
                ("Лит.", x + 137, m.get("liter", "")),
                ("Масса", x + 152, m.get("mass", "")),
                ("Масштаб", x + 169, scale),
            ]:
                body += text(xx, y + 19, label, 2.5) + text(xx, y + 27, value, 2.5)
            body += text(x + 137, y + 34, f"Лист {sheet}", 2.5) + text(
                x + 157, y + 34, f"Листов {total}", 2.5
            )
            body += text(x + 160, y + 46, field(m.get("owner"), 46, 2.5), 2.5, "middle")
            for i, (role, key) in enumerate(
                [
                    ("Разраб.", "developer"),
                    ("Пров.", "checker"),
                    ("Т. контр.", "technologist"),
                    ("Н. контр.", "norm_checker"),
                    ("Утв.", "approval"),
                ]
            ):
                body += text(x + 1, y + 34 + i * 5, role, 2.5) + text(
                    x + 18, y + 34 + i * 5, field(m.get(key), 20, 2.5), 2.5
                )
            for label, offset in [
                ("Изм.", 1),
                ("Лист", 8),
                ("№ док.", 18),
                ("Подп.", 41),
                ("Дата", 56),
            ]:
                body += text(x + offset, y + 29, label, 2.5)
        else:
            y = 277
            body += box(x, y, 185, 15, "gost-form-2a")
            for yy in (5, 10):
                body += line(x, y + yy, x + 65, y + yy)
            for xx in (7, 17, 40, 55, 65, 165, 175):
                body += line(x + xx, y, x + xx, y + 15)
            body += (
                text(x + 67, y + 10, field(designation, 96, 5), 5)
                + text(x + 166, y + 4, "Вер.", 2.5)
                + text(x + 166, y + 11, m.get("version", ""), 3.5)
            )
            body += text(x + 176, y + 4, "Лист", 2.5) + text(
                x + 179, y + 11, sheet, 3.5, "middle"
            )
            for label, offset in [
                ("Изм.", 1),
                ("Лист", 8),
                ("№ док.", 18),
                ("Подп.", 41),
                ("Дата", 56),
            ]:
                body += text(x + offset, y + 4, label, 2.5)
        # Required duplicate designation on the opposite corner, rotated 180 degrees.
        body += '<g id="gost-field-26" transform="translate(90 19) rotate(180)">'
        body += (
            box(0, 0, 70, 14, "gost-duplicate-designation")
            + text(35, 9, field(designation, 66, 3.5), 3.5, "middle")
            + "</g>"
        )
        # Inventory strip; content deliberately blank unless real records supplied.
        body += '<g id="gost-inventory-strip" transform="translate(8 292) rotate(-90)">'
        body += box(0, 0, 145, 12, "gost-inventory-fields")
        for xx in (25, 60, 85, 110):
            body += line(xx, 0, xx, 12)
        for xx, label in [
            (1, "Инв. № подл."),
            (26, "Подп. и дата"),
            (61, "Взам. инв."),
            (86, "Инв. № дубл."),
            (111, "Подп. и дата"),
        ]:
            body += text(xx, 4, label, 2.5)
        body += text(1, 10, m.get("inventory_number", ""), 2.5) + "</g>"
        body += text(412, 296, "Формат А3", 2.5, "end")
    # Do not put invented signatures or release approval into a synthetic example.
    disclaimer = (
        "DRAFT - CONFORMANCE REVIEW PENDING"
        if profile.key == "iso"
        else "ПРОЕКТ - НОРМОКОНТРОЛЬ НЕ ЗАВЕРШЕН"
    )
    body += text(24, 265, disclaimer, 3.5)
    body += text(
        24,
        273,
        "Nominal geometry only"
        if profile.key == "iso"
        else "Только номинальная геометрия",
        2.5,
    )
    return body
