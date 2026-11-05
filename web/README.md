# cad-env website

Buildless, self-hosted project website. No package installation, external fonts,
CDN libraries, analytics or credentials. Includes an accessible interactive WebGL
viewer of three actual native FreeCAD TRAIN reference meshes, teacher action
sequences, recorded benchmark evidence and an architecture/setup guide.

From the repository root:

```bash
python -m http.server 8000 --directory web
```

Open http://localhost:8000. Use HTTP rather than `file://` for mesh/JSON loading.
Drag the model to orbit, use arrow keys while focused, toggle wireframe, zoom,
and select a different reference part in the explorer. Mobile navigation and
copy-code buttons are functional. Rendering occurs only when the camera changes;
there is no continuous animation.

Refresh the recorded public snapshot and native TRAIN meshes:

```bash
python web/scripts/snapshot.py
```

This requires the local artifacts; serving the checked-in site does not. The
snapshot is intentionally not live telemetry. Completed sequential native
measurements are labelled separately from ongoing retention/batch evaluations.
The website does not run CAD commands in a visitor's browser.
