"""Native Blender mesh booleans; geometry units explicitly interpreted as mm.

This backend verifies polygonal geometry, not CAD sketch constraints or BRep.
"""

import bmesh
import bpy
from mathutils import Vector

VERSION = bpy.app.version_string
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 0.001


def boolean(a, b, operation):
    bpy.context.view_layer.objects.active = a
    modifier = a.modifiers.new("NativeBoolean", "BOOLEAN")
    modifier.operation = operation
    modifier.solver = "EXACT"
    modifier.object = b
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(b, do_unlink=True)
    return a


def build(features):
    solid = None
    for feature in features:
        solid = extend(solid, feature)
    return solid


def extend(solid, feature):
    p = feature["profile_spec"]
    depth = feature["length"]
    z = feature["z"]
    center = z + depth / 2 if feature["op"] == "pad" else z - depth / 2
    if p["op"] == "sketch_rectangle":
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(p["length"] / 2, p["width"] / 2, center)
        )
        tool = bpy.context.object
        tool.dimensions = (p["length"], p["width"], depth)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    else:
        tool = None
        for c in p["circles"]:
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=256,
                radius=c["radius"],
                depth=depth,
                location=(c["x"], c["y"], center),
            )
            piece = bpy.context.object
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
            bottom = z if feature["op"] == "pad" else z - depth
            top = z + depth if feature["op"] == "pad" else z
            for vertex in piece.data.vertices:
                vertex.co.z = top if vertex.co.z > center else bottom
            piece.data.update()
            tool = piece if tool is None else boolean(tool, piece, "UNION")
    solid = (
        tool
        if solid is None
        else boolean(solid, tool, "UNION" if feature["op"] == "pad" else "DIFFERENCE")
    )
    solid.name = feature["name"]
    return solid


def metrics(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    volume = abs(bm.calc_volume(signed=True))
    area = sum(f.calc_area() for f in bm.faces)
    valid = bool(bm.faces) and all(e.is_manifold for e in bm.edges)
    # Count connected surface components: generated solids do not have enclosed cavities.
    remaining = set(bm.verts)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            for edge in stack.pop().link_edges:
                for vertex in edge.verts:
                    if vertex in remaining:
                        remaining.remove(vertex)
                        stack.append(vertex)
    bm.free()
    points = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bounds = [min(p[i] for p in points) for i in range(3)] + [
        max(p[i] for p in points) for i in range(3)
    ]
    return {
        "volume": volume,
        "area": area,
        "valid": valid,
        "solids": components,
        "bounds": bounds,
    }


def clone(obj):
    copy = obj.copy()
    copy.data = obj.data.copy()
    bpy.context.collection.objects.link(copy)
    return copy


def difference(a, b):
    # Exact triangle-surface equality is a geometric proof, independent of
    # action source. Blender booleans can return nonempty A-A on coplanar faces.
    left, right = mesh(a), mesh(b)
    if left == right:
        return 0.0

    def triangles(data):
        vertices = [tuple(v) for v in data["vertices"]]
        return sorted(
            tuple(sorted(vertices[i] for i in face)) for face in data["faces"]
        )

    if triangles(left) == triangles(right):
        return 0.0
    first = boolean(clone(a), clone(b), "DIFFERENCE")
    second = boolean(clone(b), clone(a), "DIFFERENCE")
    value = metrics(first)["volume"] + metrics(second)["volume"]
    bpy.data.objects.remove(first, do_unlink=True)
    bpy.data.objects.remove(second, do_unlink=True)
    return value


def mesh(obj):
    obj.data.calc_loop_triangles()
    return {
        "vertices": [list(obj.matrix_world @ v.co) for v in obj.data.vertices],
        "faces": [list(t.vertices) for t in obj.data.loop_triangles],
    }


def save(obj, out, actions):
    import json

    for other in list(bpy.data.objects):
        if other != obj:
            bpy.data.objects.remove(other, do_unlink=True)
    obj["cad_env_actions"] = json.dumps(actions)
    obj["geometry_contract"] = (
        "polygonal mesh in millimeters; replay JSON to edit parameters"
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "output.blend"))
    name = obj.name
    bpy.ops.wm.open_mainfile(filepath=str(out / "output.blend"))
    if not bpy.data.objects.get(name):
        raise ValueError("Blender round-trip lost the candidate object")
    return "output.blend"
