# Marine shader oracle fixtures

`marine.vs` / `marine.fs` are the composed `MARINE_VS` / `MARINE_FS` source strings of
`renderer/marine_shader.py`, pinned by
`tests/test_lit3d_extraction.py::test_marine_shader_byte_identical_to_oracle` so that a
refactor of the shared lit-3D GLSL (`renderer/lit3d.py`) cannot change the unit shader
unnoticed.

A DELIBERATE shader change re-captures them in the same commit as the change, with the
reason in the commit message. The sources are plain module-level strings, so no GL context
is needed. From the repo root:

    C:/Users/steen/anaconda3/python.exe -c "import sys; sys.path[:0]=['.','src']; from pathlib import Path; from renderer import marine_shader as ms; d=Path('tests/_lit3d_marine_shader_golden'); (d/'marine.vs').write_text(ms.MARINE_VS, encoding='utf-8', newline='\n'); (d/'marine.fs').write_text(ms.MARINE_FS, encoding='utf-8', newline='\n')"

Re-captures: 2026-09-06 (#60 P1, the oracle); 2026-10-04 (#33, the rim takes the albedo,
`u_rim_albedo`); 2026-10-04 (#33, the gloss highlight: the albedo's alpha is the gloss mask,
`u_gloss_strength` / `u_gloss_shininess`); 2026-10-19
(#93 graphics G0 P1, one HDR world RT + one tone-map: the marine writes linear HDR into the
RGBA16F scene RT; its `aces_tonemap` + `linear_to_srgb` output lines and the two helpers left
`_COMMON_GLSL` for `WorldComposite.resolve()`).; 2026-10-20 (#93 graphics G2, the smoke cast shadow: `_COMMON_GLSL` carries
`_SMOKE_SHADOW_GLSL`, the one copy of the shadow function and its uniforms, and
`_FIELD_SAMPLE_GLSL` multiplies `incoming_rgb` by `smokeShadow(...)` at the fragment's own
height); 2026-10-27 (#114 lit 3D units tier 1: with `[render.unit_shading] body_light =
"unit"` the marine's own `main()` lights the whole body from the unit's one light read --
per-unit uniforms `u_body_light` / `u_body_dir_rgb` / `u_body_iso_rgb` / `u_body_to_light`, a
balanced share weighted by the azimuth-averaged half-Lambert, the smoke cast shadow re-taken
per fragment toward the read's light; the shared `_COMMON_GLSL` / `_FIELD_SAMPLE_GLSL` are
unchanged and still embedded verbatim); 2026-10-27 (#114, the light's shape as settings: `u_wrap` --
the diffuse term is the wrapped Lambert `max(0, (N.L + w)/(1 + w))`, w 1 = the old half-Lambert,
and the balanced share's weight wraps alike -- with `u_light_z` now set per frame from
`[render.unit_shading] light_height` / `wrap`).
