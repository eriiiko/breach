# Bounded compressed-sleeve study

Completed: 2026-10-05. This experiment is **separate from the accepted character source**. Both authorized candidates are complete. **Implementation and parent visual review agree: reject rollout and retain the accepted cleanup baseline.** Neither study clearly improves the complete sleeve. No third candidate or whole-character change was made.

The cleanup baseline is commit `a298b68`, with source SHA-256 `75b74ff12e5d8073ca8e10253c57d21b8a5de31b28390773244f64db3c3140e9`. `baseline_files.json` records hashes of its source, validation, documentation and final previews; the study checks these before and after running. The baseline remains unchanged.

Candidate 1 uses a single continuous sleeve with ivory/charcoal material regions, 13% longitudinal cloth compression, bending stiffness 0.045 and pressure 0.6. The compressed result is applied to editable geometry and its seams follow that final surface. Its saved mesh contains 36,994 vertices and 73,984 triangles, with no live cloth modifier, shape key or external simulation cache. Endpoint error is zero at the pinned solver vertices. See `candidate_1_metrics.json` for measured values. The archived setup error was fixed by reusing the accepted source's studio and cameras.

Candidate 1 visual verdict after inspecting both views: **retain as a study, reject for whole-sleeve rollout**. The ivory folds look more natural and avoid the baseline's sampled cord ridges, but too much of the sleeve is smooth, the remaining folds are broad and puffy, the black elbow flex loses its definition, and an ivory cuff interior is exposed in the side view.

Candidate 2 restores the accepted elbow flex detail and covers the exposed cuff binding. Its individual creases are shorter and avoid the baseline's serrated cord ridges, but they cover too much of both ivory sections. This dense crumpling reads as thin fabric rather than padded EVA, with insufficient calmer planes between joint compression zones. A narrow dark nick also remains beside the lower elbow plate in the front view. **Reject rollout.** The parent independently inspected both views and agrees. The accepted baseline keeps its known long diagonal crease limitation; this bounded study did not establish a better complete replacement.

| Applied sleeve mesh | Vertices | Triangles | Largest pinned-vertex solver error |
|---|---:|---:|---:|
| Candidate 1 | 36,994 | 73,984 | 0 m |
| Candidate 2 | 78,530 | 157,056 | 0.000000123 m |

Counts are for each candidate sleeve only. The rest of the original character is retained but hidden in each study file. No character decimation, UV work, rigging or game export was performed. Both study surfaces are editable applied geometry; neither needs a live cloth modifier or external cache. `reopened_validation.json` records the separate saved-file audit and the comparison settings. Seams trace each applied solver surface; no original analytic fold field is used for their placement.

Review pairs use identical cameras, lighting, ivory fabric shader and render settings: 1000 x 1400 pixels, Cycles, 96 samples. The material library is unchanged. Candidate 1's cuff used the charcoal textile assignment; candidate 2 restores the accepted graphite elastomer assignment there. `baseline_front.png` and `baseline_side.png` are actual renders of the accepted right arm; corresponding candidate PNGs show its replacement. These are isolated-sleeve comparisons, not final character renders. Open `index.html` or `comparison.png` to compare all six images at equal scale.

Rebuild a study candidate from this worktree (replace `1` with `2` for the second; this overwrites that candidate's experiment outputs and resets its metrics verdict):

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python 'C:/Users/steen/projects/breach/.claude/worktrees/33-agent-blockout/art/characters/space_marine/experiments/compressed_cloth/study_cloth.py' -- --candidate 1
```

Candidate 2 was specified and committed before its run: 18% compression, bending stiffness 0.014, pressure 0.4, 96 circumferential samples, maximum 3 mm axial spacing and solver quality 12. Weak local sewing restraints distribute compression along the existing seam paths. The accepted elbow flex shape is resampled into the pinned black region with shared boundary rings; the white sections reach the original elbow transitions more closely. The cuff extends to the accepted seal's lower extent and uses the original graphite elastomer material. Candidate 2's saved camera/render settings were normalized after rendering, without changing its geometry. The baseline and candidate 1 outputs were retained unchanged.

The solver is an authoring approximation: low gravity, pressure support, pinned seals and sewing restraints, with collisions disabled. It is not a material-accurate EVA cloth model or runtime simulation. All output is authored geometry after application. Do not use the main character generator to reproduce this experiment, and do not overwrite the accepted source or its previews.

Read-only reopen audit and comparison composition:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python 'C:/Users/steen/projects/breach/.claude/worktrees/33-agent-blockout/art/characters/space_marine/experiments/compressed_cloth/inspect_study.py'
& 'C:/Users/steen/anaconda3/python.exe' 'C:/Users/steen/projects/breach/.claude/worktrees/33-agent-blockout/art/characters/space_marine/experiments/compressed_cloth/make_comparison.py'
```
