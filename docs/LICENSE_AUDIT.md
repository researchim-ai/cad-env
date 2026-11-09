# License-scope review — 2026-11-09 follow-up

The initial audit identified two Blender Python adapters and a bundled technical
font as items requiring explicit license scope. At the user's request, the
Blender integration was removed and fonts were moved to an explicit external
installation. No source code was relicensed and the root Apache 2.0 license was
not changed.

## Changes completed

* Removed `cad_env/multicad/blender_engine.py`,
  `cad_env/solidcad/blender_engine.py` and the mesh grader used only by that backend.
* Removed runtime branches, constructor arguments, command flags, generation
  defaults, configuration entries and backend tests for the removed engine.
  Both retained backends reject the old engine name. Documentation and both
  website languages advertise only FreeCAD and CadQuery.
* Removed font binaries, editable font source and copied license texts from
  `cad_env/drawings/assets`. Git ignore rules exclude accidental font additions.
* Added `cad_env.drawings.font_cache`: explicit download during installation,
  fixed upstream commit and SHA-256 for every file, atomic replacement, verified
  cache reuse, editable source and licenses retained alongside the font.
* New exports snapshot our Python code and preserve hashed font provenance;
  they do not copy font TTF/SFD/license assets. The pinned font itself and its
  rendered glyph outlines remain unchanged. Existing generated corpora and their
  original attribution/snapshots are not rewritten during active benchmarks.

The default cache is `$XDG_CACHE_HOME/cad-env/fonts/<commit>` or
`~/.cache/cad-env/fonts/<commit>`. `CAD_ENV_CACHE` changes the cache parent.
`make install` and `make install-drawings` run the installer. Direct dependency
installation is followed by `python -m cad_env.drawings.font_cache --install`.
Rendering never silently downloads/substitutes a font. Installation was exercised
against the real upstream files, including license texts, and verified all hashes.

## Remaining license boundaries

External installation changes distribution contents, not the license of the
installed component. Osifont retains its LGPL/GPL and font-exception terms.
[Upstream](https://github.com/hikikomori82/osifont) documents the font variants;
[SPDX](https://spdx.org/licenses/Font-exception-2.0.html) records the document
exception. Generated document outlines and the font software are distinct.

The source tree can retain its Apache 2.0 license while separately installed
applications and dependencies keep theirs. [FreeCAD](https://github.com/FreeCAD/FreeCAD)
is LGPL; [CadQuery](https://github.com/CadQuery/cadquery/blob/master/README.md)
is Apache for its own code, with separate dependency licenses. Any redistributed
application binaries, Python wheels, containers and caches need their own notices
and applicable source/redistribution obligations. Do not ship the external cache
inside an Apache-only source archive.

The earlier Blender finding followed the [Foundation FAQ](https://www.blender.org/support/faq/).
It is addressed in the current tree by removing that integration, not by claiming
a subprocess universally cancels GPL obligations. Historical Git commits and
local run archives are not deleted or relicensed by removal from the working tree.

The new part generator creates original nominal geometry. Vendor PDFs/CAD and
full ISO/GOST texts were not imported. The 6200 bearing's 10/30/9 mm boundary
facts are attributed to [NSK](https://www.nsk.com/my-en/engineering/products/bearings/ball-bearings/deep-groove-ball-bearings/single-row-deep-groove-ball-bearings/6200-apn.html).
Real external drawings still require an explicit source/permission review.

[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/main) is Apache 2.0.
Its original local checkpoint contains LICENSE; the earlier merged pilot directory
lacks it. Before publishing that standalone model, preserve the upstream license
and applicable notices. Active model identities were not modified in this change.

## Audit limits

The initial scan reviewed 295 tracked/nonignored source candidates and 634 local
package metadata records (587 distinct name/version pairs). The inventory remains
in `artifacts/license-audit-2026-11-09/inventory.json`. Installed GPL/proprietary
packages in ignored environments are not necessarily project imports or bundled
components. Rights to the original project illustration were not independently
authenticated. This is a targeted implementation/distribution-scope check, not a
complete legal/source audit of every embedded transitive binary.

Review final staged/archive contents before publishing a new distribution. See
`THIRD_PARTY_NOTICES.md` for the current scope; no blanket Apache-only assertion is
made about arbitrary model, data, font-cache or binary bundles.

Follow-up verification: 35 retained-engine tests, 33 drawing/generator tests,
three cache-integrity tests and a removed-engine rejection test passed. A targeted
native ISO/GOST re-export also passed with no TTF/SFD in its bundles. Built and
inspected a wheel: it contains the installer, but no font binaries/sources or
removed runtime adapters. EN/RU website content advertises the two retained engines.
