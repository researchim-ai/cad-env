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

## Languages

The landing page, guide, updates and drawing article support English and Russian through the EN/RU
switch. The preference persists in localStorage across navigation and reloads;
`?lang=en` and `?lang=ru` provide direct links. Dynamic results, part names,
control labels, messages and metadata are localized. CAD JSON and commands
retain their original identifiers. Bundled original technical Markdown documents
are in English and their links are marked EN.

Translations are maintained in `translations.json` (page copy) and `i18n.js`
(dynamic UI and accessibility labels). All pages load the shared locale before
the application. No translation service or network dependency is required.

## Engineering-sheet design

The landing page uses a paper drawing grid, a navy model-space viewport and
high-visibility geometry. Feature history is derived from the selected reference
trajectory; the displayed bounding box comes from the actual mesh coordinates.
The four-step protocol explorer (Observe / Act / Execute / Verify) is an
explicitly illustrative interaction diagram, not a running CAD or RL backend.
Its selection and text update immediately in both languages. Existing recorded
scores remain labelled as checkpoint-500 corpus results.

`workspace.js` implements the protocol explorer. `style.css` covers desktop and
mobile layouts as well as the shared field guide. The site remains buildless
and has no external runtime dependencies.

## Project updates and drawing examples

`blog.html` lists dated updates; `drawing-update.html` is the first long-form post.
It explains paired ISO/GOST draft profiles, multi-sheet VLM input and SFT exports.
The page includes three generated TRAIN parts, six full PDF drawing sets and a
real feature-table crop. The source models are the completed mechanical pilot,
not model predictions. Full standards conformance is explicitly unconfirmed.

Refresh the static drawing examples from verified local TRAIN artifacts:

```bash
python web/scripts/export_drawings.py
```

This copies only PNG/PDF documents and provenance into `assets/drawings`; no
font binary/source or model weights are copied. The font's upstream attribution
and document exception are recorded in `provenance.json`. The generated example
PDFs use vector glyph outlines. The site's typography uses system fonts.

`journal.css` covers the update pages and homepage news section. `site.js` handles
navigation/copy controls on the guide and journal pages. The language query is
preserved between pages even when localStorage is unavailable.

The homepage distinguishes the completed 20-part drawing SFT pilot from the
earlier checkpoint-500 experiment. In both comparisons, the starting checkpoint
had already received CAD training. Neither is advertised as a raw-model score.
`assets/drawing-pilot-results.json` records the drawing pilot counts and scope.

The bilingual report `tuning-results.html` describes joint SFT on paired ISO/GOST representations of the same tasks. It shows two audited student designs per difficulty level, individual tuned/baseline FCStd files, and the exact ordered model input crops for both profiles. The input viewer switches ISO/GOST, reports success per representation and lets readers browse every crop. `python web/scripts/export_solutions.py` copies inputs only after checking them against logged request hashes; `--reuse-previews` reuses unchanged renders. No ZIP downloads are generated.

Report benchmark quality over the 90 parent tasks separately for ISO and GOST (30 tasks per difficulty), with paired agreement if helpful. Do not sum representation attempts into a combined task success rate.

Page footers must not display automatic update timestamps or snapshot stamps.
Keep recording dates in experiment data and dated posts; do not inject them into
the shared footer. `scripts/check_site.py` checks this constraint.
