import math

import bmesh
import bpy

from cad_env.multicad import blender_engine as base

from .schema import arc_points

VERSION = bpy.app.version_string
metrics = base.metrics
mesh = base.mesh
difference = base.difference


def objmesh(name, vertices, faces):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    bm.to_mesh(data)
    bm.free()
    return obj


def extrude(points, height, name):
    if points[0] == points[-1]:
        points = points[:-1]
    n = len(points)
    vertices = [(x, y, z) for z in [0, height] for x, y in points]
    faces = [list(reversed(range(n))), list(range(n, 2 * n))]
    faces.extend([[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)])
    return objmesh(name, vertices, faces)


def lathe(points, angle, name):
    full = angle == 360
    n = 256 if full else max(16, math.ceil(angle / 360 * 256))
    vertices = []
    rings = []
    for r, z in points:
        if r == 0:
            index = len(vertices)
            vertices.append((0, 0, z))
            rings.append([index] * (n if full else n + 1))
        else:
            ring = []
            for j in range(n if full else n + 1):
                a = math.radians(angle) * j / n
                ring.append(len(vertices))
                vertices.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(ring)
    faces = []
    for i in range(len(points)):
        for j in range(n):
            k = (j + 1) % n if full else j + 1
            raw = [
                rings[i][j],
                rings[(i + 1) % len(points)][j],
                rings[(i + 1) % len(points)][k],
                rings[i][k],
            ]
            face = list(dict.fromkeys(raw))
            if len(face) >= 3:
                faces.append(face)
    if not full:
        faces.extend([[r[0] for r in rings], [r[-1] for r in reversed(rings)]])
    return objmesh(name, vertices, faces)


def build(program):
    objects = {}
    for a in program:
        op = a["op"]
        if op == "finish":
            vertices = []
            faces = []
            for name in a["targets"]:
                data = mesh(objects[name])
                offset = len(vertices)
                vertices.extend(data["vertices"])
                faces.extend([[i + offset for i in f] for f in data["faces"]])
            return objmesh("Final", vertices, faces)
        if op in {"box", "cylinder"}:
            origin = a.get("origin", [0, 0, 0])
            profile = (
                {"op": "sketch_rectangle", "length": a["length"], "width": a["width"]}
                if op == "box"
                else {
                    "op": "sketch_circles",
                    "circles": [{"x": 0, "y": 0, "radius": a["radius"]}],
                }
            )
            obj = base.build(
                [
                    {
                        "op": "pad",
                        "name": a["name"],
                        "profile_spec": profile,
                        "z": origin[2],
                        "length": a["height"],
                    }
                ]
            )
            obj.location.x += origin[0]
            obj.location.y += origin[1]
            bpy.context.view_layer.update()
        elif op == "polygon_extrude":
            obj = extrude(a["points"], a["height"], a["name"])
        elif op == "arc_extrude":
            points = [a["start"]]
            start = a["start"]
            for segment in a["segments"]:
                points.extend(
                    arc_points(start, segment["mid"], segment["to"])
                    if segment["type"] == "arc"
                    else [segment["to"]]
                )
                start = segment["to"]
            obj = extrude(points, a["height"], a["name"])
        elif op == "revolve_profile":
            obj = lathe(a["points"], a["angle"], a["name"])
        elif op in {"fillet", "chamfer", "shell"}:
            obj = base.clone(objects[a["source"]])
            bpy.context.view_layer.objects.active = obj
            if op == "shell":
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                top = max(v.co.z for v in bm.verts)
                faces = [
                    f
                    for f in bm.faces
                    if all(abs(v.co.z - top) < 1e-6 for v in f.verts)
                ]
                if not faces:
                    raise ValueError("Top opening not found")
                bmesh.ops.delete(bm, geom=faces, context="FACES")
                bm.to_mesh(obj.data)
                bm.free()
                modifier = obj.modifiers.new("Shell", "SOLIDIFY")
                modifier.thickness = a["thickness"]
                modifier.offset = -1
                modifier.use_even_offset = True
            else:
                modifier = obj.modifiers.new("DressUp", "BEVEL")
                modifier.width = a.get("radius", a.get("size"))
                modifier.segments = 12 if op == "fillet" else 1
                modifier.limit_method = "ANGLE"
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        elif op in {"fuse", "cut"}:
            obj = base.boolean(
                base.clone(objects[a["left"]]),
                base.clone(objects[a["right"]]),
                "UNION" if op == "fuse" else "DIFFERENCE",
            )
        objects[a["name"]] = obj
        if not metrics(obj)["valid"] or metrics(obj)["volume"] <= 0:
            raise ValueError("Invalid/empty solid feature")


def save(obj, out, program):
    return base.save(obj, out, program)
