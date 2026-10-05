# Bounded compressed-sleeve study

Checkpoint: 2026-10-05. This experiment is **separate from the accepted character source**. The authorized study allows at most two candidates, with a parent visual decision before any application to the character. Candidate 1 is complete; candidate 2 is pending. No automatic rollout is authorized.

The cleanup baseline is commit `a298b68`, with source SHA-256 `75b74ff12e5d8073ca8e10253c57d21b8a5de31b28390773244f64db3c3140e9`. `baseline_files.json` records hashes of its source, validation, documentation and final previews; the study checks these before and after running. The baseline remains unchanged.

Candidate 1 uses a single continuous sleeve with ivory/charcoal material regions, 13% longitudinal cloth compression, bending stiffness 0.045 and pressure 0.6. The compressed result is applied to editable geometry and its seams follow that final surface. Its saved mesh contains 36,994 vertices and 73,984 triangles, with no live cloth modifier, shape key or external simulation cache. Endpoint error is zero at the pinned solver vertices. See `candidate_1_metrics.json` for measured values. The archived setup error was fixed by reusing the accepted source's studio and cameras.

Visual verdict after inspecting both views: **retain as a study, reject for whole-sleeve rollout**. The ivory folds look more natural and avoid the baseline's sampled cord ridges, but too much of the sleeve is smooth, the remaining folds are broad and puffy, the black elbow flex loses its definition, and an ivory cuff interior is exposed in the side view. Candidate 2 will be the final attempt: smaller localized compression folds, meaningful elbow flex detail and a clean cuff connection, while preserving the padded silhouette.

Review pairs use identical camera, lighting, material and render settings: 1000 x 1400 pixels, Cycles, 96 samples. `baseline_front.png` and `baseline_side.png` are actual renders of the accepted right arm; `candidate_1_front.png` and `candidate_1_side.png` show its replacement under those settings. The rest of the source is retained but hidden in `candidate_1.blend` for context and self-contained materials. These are isolated-sleeve comparisons, not final character renders.

Rebuild a study candidate from this worktree (this overwrites only that candidate's experiment files):

```powershell
& 'C:/Program Files/Blender Foundation/Blender 4.5/blender.exe' --background --python 'C:/Users/steen/projects/breach/.claude/worktrees/33-agent-blockout/art/characters/space_marine/experiments/compressed_cloth/study_cloth.py' -- --candidate 1
```

Candidate 2 is specified before its run: 18% compression, bending stiffness 0.014, pressure 0.4, 96 circumferential samples, maximum 3 mm axial spacing and solver quality 12. Weak local sewing restraints distribute compression along the existing seam paths. The accepted elbow flex shape is resampled into the pinned black region with shared boundary rings; the white sections reach the original elbow transitions more closely. The cuff extends to the accepted seal's lower extent and uses the original graphite elastomer material. The baseline and candidate 1 files will be retained unchanged. Do not use the main character generator to reproduce this experiment, and do not overwrite the accepted source or its previews.
