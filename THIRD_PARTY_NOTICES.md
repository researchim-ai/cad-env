# Third-party notices and license scope

The root `LICENSE` contains Apache License 2.0. Independent external applications,
Python dependencies, downloaded fonts and model weights retain their own terms.
This file records scope and does not relicense any third-party work.

## External technical font

Font binaries, editable font sources and copied GNU license texts are no longer
bundled in this repository. `python -m cad_env.drawings.font_cache --install`
downloads pinned, SHA-256-verified files into the user cache outside the checkout:

* `osifont-lgpl3fe.ttf`, stored in the cache as `osifont.ttf`: embedded LGPLv3
  notice with font exception.
* The corresponding editable upstream SFD source, with its own GPLv3/font-exception
  notice, upstream README, and full GPL/LGPL texts.

Author: Zefram Cochrane / hikikomori82 and upstream contributors.
Upstream: https://github.com/hikikomori82/osifont
Pinned commit: `afbc9134f90733eb5eea0a1c70c56a6d5dd5b758`.
The download manifest/URLs and hashes are in `cad_env/drawings/font_cache.py`.

New drawing bundles include vector glyph outlines and `font-provenance.json`,
not the font binary/source. The document exception permits the specified font use
without itself imposing GPL on the document. Downloading externally does not
waive the font license, copyright/attribution or other applicable obligations.
Older generated bundles keep their existing notices and source snapshots; this
change does not relicense or rewrite old artifacts.

## Runtime applications and dependencies

Supported CAD runtimes are FreeCAD and CadQuery. Their applications and transitive
CAD/Qt/solver dependencies have their own terms. FreeCAD is LGPL-licensed;
CadQuery's own code is Apache 2.0. Do not label every binary in an environment or
container Apache merely because the project source is Apache.

The former Blender integration, adapters, flags and runtime installation
instructions have been removed. No new license was chosen for those removed
adapters. Historical local runs/source snapshots are not part of the current
source distribution and are excluded from Git together with `artifacts/`.

## Data and models

The new industrial parts are original procedural nominal geometry. Manufacturer
PDFs, catalogue graphics, STEP models and full standards texts are not bundled.
The 6200 bearing dimensions are referenced factual boundary dimensions, attributed
in task provenance. A URL is not permission to import additional vendor material.

Qwen3.5-4B is Apache 2.0 upstream. Standalone weight releases must preserve its
actual license and applicable notices. Local weights, datasets, environments and
runtime overlays are ignored in Git; manually assembled archives/images need
review of their actual contents. See `docs/LICENSE_AUDIT.md` for remaining scope.
