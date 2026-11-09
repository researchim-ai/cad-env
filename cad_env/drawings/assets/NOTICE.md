# External technical font

No font binaries, editable font sources or copied license texts are bundled here.
Run `python -m cad_env.drawings.font_cache --install` to download the pinned
osifont variant, corresponding editable source and license texts into the user
cache. The installer verifies SHA-256 for every file. Runtime rendering is offline
and refuses missing/corrupted font files rather than silently substituting fonts.

The default cache is `$XDG_CACHE_HOME/cad-env/fonts/<commit>` or
`~/.cache/cad-env/fonts/<commit>`; `CAD_ENV_CACHE` overrides the cache parent.
The font retains its LGPLv3/font-exception terms and upstream attribution.
Generated documents contain glyph outlines and a `font-provenance.json`, not the
font binary. Downloading externally does not remove the font's license obligations.
