"""Deterministic orthographic Z-buffer renders, never GUI screenshots.

Native B-Rep tessellation supplies triangles; a real depth buffer is necessary
for holes and overlapping faces (average-depth painter sorting is insufficient).
"""

import json
from pathlib import Path

VIEWS = {"iso": (25, -55), "top": (90, -90), "front": (0, -90), "right": (0, 0)}


def render_mesh(mesh_path, output_dir, prefix="view", size=384, views=None):
    import numpy as np
    from PIL import Image

    mesh = json.loads(Path(mesh_path).read_text())
    vertices = np.asarray(mesh["vertices"], dtype=float).reshape(-1, 3)
    triangles = np.asarray(mesh["faces"], dtype=int).reshape(-1, 3)
    faces = vertices[triangles]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.asarray([0.4, -0.5, 0.8])
    light /= np.linalg.norm(light)
    shades = 0.5 + 0.5 * np.abs(normals @ light)
    colors = np.rint(shades[:, None] * np.asarray([155, 185, 210])).astype(np.uint8)
    lines = [
        np.asarray(line, dtype=float) for line in mesh.get("lines", []) if len(line) > 1
    ]
    points = [vertices] if len(vertices) else []
    points += lines
    framing = (
        np.concatenate(points) if points else np.asarray([[-10, -10, 0], [10, 10, 0]])
    )
    center = (framing.max(axis=0) + framing.min(axis=0)) / 2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, cameras = [], {}
    # 2x supersampling improves the small circular openings used in VLM targets.
    resolution = size * 2
    selected = list(VIEWS) if views is None else list(views)
    if not selected or set(selected) - set(VIEWS):
        raise ValueError("Choose at least one supported orthographic view")
    for name in selected:
        elevation, azimuth = VIEWS[name]
        el, az = np.deg2rad([elevation, azimuth])
        toward_camera = np.asarray(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        right = np.asarray([-np.sin(az), np.cos(az), 0])
        up = np.cross(toward_camera, right)
        basis = np.stack([right, up, toward_camera], axis=1)
        projected = (vertices - center) @ basis
        frame_projection = (framing - center) @ basis
        span = max(float(np.ptp(frame_projection[:, :2], axis=0).max()) * 1.18, 1.0)
        projected[:, 0] = (projected[:, 0] / span + 0.5) * resolution
        projected[:, 1] = (0.5 - projected[:, 1] / span) * resolution
        rgb = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
        depth = np.full((resolution, resolution), -np.inf)
        for index, tri in enumerate(projected[triangles]):
            x0, y0, z0 = tri[0]
            x1, y1, z1 = tri[1]
            x2, y2, z2 = tri[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1e-9:
                continue
            xmin, ymin = np.maximum(np.floor(tri[:, :2].min(axis=0)).astype(int), 0)
            xmax, ymax = np.minimum(
                np.ceil(tri[:, :2].max(axis=0)).astype(int), resolution - 1
            )
            if xmax < xmin or ymax < ymin:
                continue
            xx, yy = np.meshgrid(
                np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5
            )
            a = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
            b = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
            c = 1 - a - b
            z = a * z0 + b * z1 + c * z2
            region = depth[ymin : ymax + 1, xmin : xmax + 1]
            visible = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9) & (z > region)
            region[visible] = z[visible]
            rgb[ymin : ymax + 1, xmin : xmax + 1][visible] = colors[index]
        # Sketch edges are part of the current public observation, not target labels.
        for line in lines:
            screen = (line - center) @ basis
            screen[:, 0] = (screen[:, 0] / span + 0.5) * resolution
            screen[:, 1] = (0.5 - screen[:, 1] / span) * resolution
            for p, q in zip(screen[:-1], screen[1:]):
                steps = max(2, int(np.max(np.abs(q[:2] - p[:2]))) + 1)
                samples = np.linspace(p, q, steps)
                xy = np.rint(samples[:, :2]).astype(int)
                valid = (
                    (xy[:, 0] >= 0)
                    & (xy[:, 0] < resolution)
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] < resolution)
                )
                xy, samples = xy[valid], samples[valid]
                for dx, dy in ((0, 0), (1, 0), (0, 1)):
                    x = np.clip(xy[:, 0] + dx, 0, resolution - 1)
                    y = np.clip(xy[:, 1] + dy, 0, resolution - 1)
                    visible = samples[:, 2] >= depth[y, x] - 0.05
                    rgb[y[visible], x[visible]] = [35, 75, 110]
        path = output_dir / f"{prefix}-{name}.png"
        Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS).save(path)
        paths.append(str(path.resolve()))
        cameras[name] = {
            "projection": "orthographic",
            "elevation_deg": elevation,
            "azimuth_deg": azimuth,
            "center_mm": center.tolist(),
            "span_mm": span,
            "resolution_px": size,
            "basis_columns": basis.tolist(),
        }
    (output_dir / f"{prefix}-cameras.json").write_text(json.dumps(cameras, indent=2))
    return paths
