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
`u_rim_albedo`).
