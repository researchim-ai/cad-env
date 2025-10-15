"""
Шаблоны FreeCAD Python кода
"""

from typing import Dict, Any, Optional


class FreeCADTemplates:
    """
    Шаблоны для генерации FreeCAD Python кода
    """
    
    def create_box(self, length: float = 10, width: float = 10, height: float = 10) -> str:
        """Создание коробки"""
        return f"""# Создание коробки
box = doc.addObject("Part::Box", "Box_{length}x{width}x{height}")
box.Length = {length}
box.Width = {width}
box.Height = {height}
doc.recompute()"""
    
    def create_cylinder(self, radius: float = 5, height: float = 10) -> str:
        """Создание цилиндра"""
        return f"""# Создание цилиндра
cylinder = doc.addObject("Part::Cylinder", "Cylinder_r{radius}_h{height}")
cylinder.Radius = {radius}
cylinder.Height = {height}
doc.recompute()"""
    
    def create_sphere(self, radius: float = 5) -> str:
        """Создание сферы"""
        return f"""# Создание сферы
sphere = doc.addObject("Part::Sphere", "Sphere_r{radius}")
sphere.Radius = {radius}
doc.recompute()"""
    
    def create_cone(self, radius1: float = 5, radius2: float = 0, height: float = 10) -> str:
        """Создание конуса"""
        return f"""# Создание конуса
cone = doc.addObject("Part::Cone", "Cone_r1{radius1}_r2{radius2}_h{height}")
cone.Radius1 = {radius1}
cone.Radius2 = {radius2}
cone.Height = {height}
doc.recompute()"""
    
    def create_torus(self, radius1: float = 10, radius2: float = 3) -> str:
        """Создание тора"""
        return f"""# Создание тора
torus = doc.addObject("Part::Torus", "Torus_r1{radius1}_r2{radius2}")
torus.Radius1 = {radius1}
torus.Radius2 = {radius2}
doc.recompute()"""
    
    def rotate(self, angle: float = 90, axis: str = "Z") -> str:
        """Поворот объекта"""
        return f"""# Поворот объекта
import math
rotation_angle = math.radians({angle})
if "{axis}".upper() == "X":
    rotation_axis = FreeCAD.Vector(1, 0, 0)
elif "{axis}".upper() == "Y":
    rotation_axis = FreeCAD.Vector(0, 1, 0)
else:  # Z
    rotation_axis = FreeCAD.Vector(0, 0, 1)

# Применяем поворот к последнему объекту
if doc.Objects:
    last_obj = doc.Objects[-1]
    last_obj.Placement = last_obj.Placement * FreeCAD.Placement(FreeCAD.Vector(0,0,0), FreeCAD.Rotation(rotation_axis, rotation_angle))
doc.recompute()"""
    
    def translate(self, x: float = 0, y: float = 0, z: float = 0) -> str:
        """Перемещение объекта"""
        return f"""# Перемещение объекта
translation_vector = FreeCAD.Vector({x}, {y}, {z})

# Применяем перемещение к последнему объекту
if doc.Objects:
    last_obj = doc.Objects[-1]
    last_obj.Placement = last_obj.Placement * FreeCAD.Placement(translation_vector, FreeCAD.Rotation())
doc.recompute()"""
    
    def scale(self, factor: float = 2) -> str:
        """Масштабирование объекта"""
        return f"""# Масштабирование объекта
scale_factor = {factor}

# Применяем масштабирование к последнему объекту
if doc.Objects:
    last_obj = doc.Objects[-1]
    last_obj.Placement = last_obj.Placement * FreeCAD.Placement(FreeCAD.Vector(0,0,0), FreeCAD.Rotation(), FreeCAD.Vector(scale_factor, scale_factor, scale_factor))
doc.recompute()"""
    
    def extrude(self, distance: float = 5) -> str:
        """Выдавливание объекта"""
        return f"""# Выдавливание объекта
extrude_distance = {distance}

# Создаем выдавливание последнего объекта
if doc.Objects:
    last_obj = doc.Objects[-1]
    # Создаем плоскость для выдавливания
    face = last_obj.Shape.Faces[0] if hasattr(last_obj.Shape, 'Faces') and last_obj.Shape.Faces else None
    if face:
        extrude_obj = doc.addObject("Part::Extrusion", "Extrusion")
        extrude_obj.Base = last_obj
        extrude_obj.Dir = FreeCAD.Vector(0, 0, extrude_distance)
        doc.recompute()"""
    
    def union(self) -> str:
        """Объединение объектов"""
        return """# Объединение объектов
if len(doc.Objects) >= 2:
    union_obj = doc.addObject("Part::Fuse", "Union")
    union_obj.Base = doc.Objects[-2]
    union_obj.Tool = doc.Objects[-1]
    doc.recompute()"""
    
    def cut(self) -> str:
        """Вычитание объектов"""
        return """# Вычитание объектов
if len(doc.Objects) >= 2:
    cut_obj = doc.addObject("Part::Cut", "Cut")
    cut_obj.Base = doc.Objects[-2]
    cut_obj.Tool = doc.Objects[-1]
    doc.recompute()"""
    
    def intersection(self) -> str:
        """Пересечение объектов"""
        return """# Пересечение объектов
if len(doc.Objects) >= 2:
    intersection_obj = doc.addObject("Part::Common", "Intersection")
    intersection_obj.Base = doc.Objects[-2]
    intersection_obj.Tool = doc.Objects[-1]
    doc.recompute()"""
    
    def create_sketch(self, points: list) -> str:
        """Создание эскиза"""
        points_str = ", ".join([f"FreeCAD.Vector({p[0]}, {p[1]}, {p[2]})" for p in points])
        return f"""# Создание эскиза
sketch = doc.addObject("Sketcher::SketchObject", "Sketch")
sketch.addGeometry(Part.LineSegment({points_str}))
doc.recompute()"""
    
    def create_loft(self, profiles: list) -> str:
        """Создание лофта"""
        return f"""# Создание лофта
loft = doc.addObject("Part::Loft", "Loft")
loft.Sections = {profiles}
doc.recompute()"""
    
    def create_sweep(self, profile: str, path: str) -> str:
        """Создание развертки"""
        return f"""# Создание развертки
sweep = doc.addObject("Part::Sweep", "Sweep")
sweep.Sections = [{profile}]
sweep.Spine = {path}
doc.recompute()"""
    
    def create_fillet(self, radius: float = 1) -> str:
        """Создание скругления"""
        return f"""# Создание скругления
fillet_radius = {radius}
if doc.Objects:
    last_obj = doc.Objects[-1]
    # Применяем скругление к ребрам
    edges = last_obj.Shape.Edges
    if edges:
        fillet = doc.addObject("Part::Fillet", "Fillet")
        fillet.Base = last_obj
        # Настройка скругления для всех ребер
        for i, edge in enumerate(edges):
            fillet.addEdge(i, fillet_radius)
        doc.recompute()"""
    
    def create_chamfer(self, distance: float = 1) -> str:
        """Создание фаски"""
        return f"""# Создание фаски
chamfer_distance = {distance}
if doc.Objects:
    last_obj = doc.Objects[-1]
    # Применяем фаску к ребрам
    edges = last_obj.Shape.Edges
    if edges:
        chamfer = doc.addObject("Part::Chamfer", "Chamfer")
        chamfer.Base = last_obj
        # Настройка фаски для всех ребер
        for i, edge in enumerate(edges):
            chamfer.addEdge(i, chamfer_distance)
        doc.recompute()"""
