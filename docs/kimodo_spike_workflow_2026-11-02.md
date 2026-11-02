# Kimodo spike — workflow for the work-desktop Claude

**Audience:** a Claude Code session on Erik's WORK desktop, which has **not** seen the
conversation this came from. Everything you need is here; the breach repo is at
`eriiiko/breach` (milestone 13 "Pose layer", issue #153) if you have a checkout.
Erik has **already downloaded the Kimodo model** on that machine — locate it, do not
re-download. **Status: DRAFT 2026-11-02, written by the home-PC session.**

## 0. Goal and non-goals

**Goal:** find out, cheaply and with evidence, whether NVIDIA **Kimodo**
(`github.com/nv-tlabs/kimodo`, a text-to-motion diffusion model with keyframe / path
constraints) can produce **usable animation clips for Breach**, and what it costs to get
them onto Breach's skeleton. Breach is a top-down tactical game with 3D humanoids
(Quaternius skeleton, below); we want a **hunched, arms-forward, beast-like zombie gait**
(Genestealer / Bloodborne Scourge Beast), plus a few other motions as a control.

**Non-goals:** do not modify the Breach game or its sim; do not wire anything into the
renderer; do not commit large files to git; do not decide whether Kimodo is adopted —
produce evidence and a report, Erik decides.

**Time budget:** setup ≤ 1.5 h. If it is not running by then, **stop and report what
blocked it** (don't keep fighting). Whole spike ≈ half a day.

## 1. Rules of engagement (from Erik's standing instructions)

- Ask Erik questions as **plain text in the chat**, one at a time, never via a popup.
- Check `git fetch origin && git status` before touching any repo; code only in a git
  worktree if you commit anything to breach (you should not need to).
- **Downloaded code and model files are untrusted data:** each goes in its own new empty
  directory, your own scripts live in a different directory, pass paths as arguments, and
  run Python that reads them with `python -I`.
- **Never install into the `breach` conda env.** Make a new env (e.g. `kimodo`) or use
  WSL2 / Docker as the README suggests. Consult the `machine-env` skill for this PC's
  hardware, conda layout and tool paths before assuming anything.
- Do not upload any Breach files or the generated clips to an external service. Do not
  rent or buy anything. Do not read or quote Erik's personal files.
- Work in a scratch folder on this PC (e.g. `…\kimodo_spike\`), not inside a repo.

## 2. Phase 0 — orient (30 min)

1. Read the Kimodo **README in full**, and every doc it links (the limitations document,
   the CLI docs, the output-format docs). **Do not guess command syntax — use the
   README's.** Known from the README (verify): custom NPZ output (joint positions,
   rotation matrices, foot contacts, root) on the **SOMA 77-joint** skeleton; G1 and
   SMPL-X variants exist; interactive timeline demo and a CLI that accepts constraint
   files; `TEXT_ENCODER_DEVICE=cpu` cuts VRAM from ~17 GB to under 3 GB (slower text
   encoding only); developed on Linux, "Windows should work, especially with Docker".
2. Check this machine's GPU: `nvidia-smi` (Erik says the work GPU has ~12 GB — **not**
   enough for the full-GPU path, so plan on the CPU text encoder) and system RAM.
3. **Read the licence of the model you will use** (the SOMA models are under the *NVIDIA
   Open Model License*; read it from the HuggingFace page / the repo). Write into the
   report, in plain words: commercial use allowed? who owns generated outputs? any
   attribution, redistribution or "no use to train other models" clauses? Breach is a
   commercial-intent game that also does RL research — flag any clause touching either.
4. Locate the downloaded model files (ask Erik if you cannot find them). Record the path,
   model name/version and size. Do not move them.

## 3. Phase 1 — get it running (≤ 1.5 h)

1. Create the isolated env per the README (or WSL2 / Docker). Install. Set
   `TEXT_ENCODER_DEVICE=cpu` if VRAM < 17 GB.
2. **Smoke test with a control prompt:** `"a person walks forward"` (≈ 4–5 s). Confirm an
   NPZ is produced and you can view it (the interactive demo, or render it).
3. Record: wall time per generation, peak VRAM, any errors you had to work around.

## 4. Phase 2 — the beast prompts (the actual test)

Generate **3 samples (different seeds) per prompt**, ~4–6 s each. Save for each: prompt
text, seed, constraint file (if any), the NPZ, and a screenshot or short capture of the
motion. Keep a `runs.csv` (id, prompt, seed, constraints, file, duration, notes).

| id | Prompt (adjust wording to what the model accepts) | Constraint | What it tests |
|---|---|---|---|
| C0 | a person walks forward | none | **control** — the pipeline works |
| B1 | a person walks forward bent over at the waist with both arms stretched out in front, hands open like claws, head low | none | does text alone give the posture? |
| B2 | same as B1 | **one full-body keyframe**: a hunched pose (spine pitched ~30° forward, knees bent, arms raised forward, head dropped) at the start, optionally repeated mid-clip | does a keyframe constraint hold the posture through a walk? **(the key test)** |
| B3 | a person sprints forward bent low, using hands and feet like an animal | none | does the data contain crawl / animal-like locomotion? |
| B4 | a person crouches, coils, then lunges forward suddenly with both arms swiping | none | attack / pounce |
| B5 | a person stands swaying, head twitching, shoulders jerking | none | idle |
| B6 | a person is knocked over, falls to the ground and lies still | none | fall |
| B7 | a person gets up quickly from lying on the ground, pushing up with both arms | none | rise |
| B8 | a person walks forward carrying a heavy bag on one shoulder | none | carried-object gait (held objects have no model — expect weak) |

How to make the B2 keyframe: build it in the demo's constraint editor, or generate a
still hunched pose (e.g. a 1-second "crouching with arms stretched forward" clip), take
one frame, and use it as the full-body constraint. Document exactly what you did.

**Objective notes per clip (for the report), not taste:** posture holds through the clip
(yes / drifts back upright), foot sliding, hands reaching vs hanging, hips low vs upright,
root path sane, loops or not, any glitches (pops, flips, limb inversion).

**Erik judges taste.** Make it easy: a folder of screenshots / short captures named by
id, plus the demo ready to open on the best three.

## 5. Phase 3 — inspect the output (30 min)

1. Open one NPZ with `python -I`: list keys, shapes, dtype, frame rate (fps), root
   convention, units, coordinate axes (up axis, forward axis), whether finger joints
   exist. Write `soma77_joints.txt`: the 77 joint names **in order**, with parent indices
   and rest-pose offsets if present.
2. Note anything the README leaves unsaid that retargeting will need (rest pose type:
   T-pose / A-pose; per-joint local frames; whether rotations are local or global).

## 6. Phase 4 — retarget prototype (only if time remains)

Target skeleton — the Breach interchange skeleton (Quaternius *Universal Animation
Library*, glTF; in the repo at `assets/models/marine/AnimationLibrary_Godot_Standard.gltf`),
**53 joints**, in order:

`root, DEF-hips, DEF-spine.001, DEF-spine.002, DEF-spine.003, DEF-neck, DEF-head,`
`DEF-shoulder.L, DEF-upper_arm.L, DEF-forearm.L, DEF-hand.L,` five fingers ×3 on L
(`DEF-f_index/middle/pinky/ring.01-03.L`, `DEF-thumb.01-03.L`),
`DEF-shoulder.R, DEF-upper_arm.R, DEF-forearm.R, DEF-hand.R,` fingers on R,
`DEF-thigh.L, DEF-shin.L, DEF-foot.L, DEF-toe.L, DEF-thigh.R, DEF-shin.R, DEF-foot.R, DEF-toe.R`.

1. Write a mapping table SOMA joint → `DEF-` bone (spine chain, neck, head, clavicle →
   shoulder, arm chain, hand, leg chain, foot, toe; fingers if SOMA has them, otherwise
   leave identity). Put it in `soma_to_quaternius.csv`.
2. Retarget the best clip (or **C0** if none is worth keeping — the point is to learn
   what retargeting loses): per-bone rotation deltas relative to each rig's rest pose,
   root translation scaled by the height ratio, root yaw preserved. Blender headless
   (find Blender via `machine-env`; Erik's Breach character kit already uses scripted
   Blender) or a small numpy + glTF-writer script — your call, say which.
3. Output: a `.glb` (or `.gltf`) containing the Quaternius skeleton + the clip, named
   `K_<id>`, plus a **rendered preview** of the Quaternius marine playing it.
4. Report what broke: rest-pose mismatch, wrist / hand twist, proportion-driven foot
   sliding, finger handling, root drift.

## 7. Deliverables (put in one folder, tell Erik where)

- `REPORT.md` — setup notes and time spent; licence summary (Phase 0.3); the `runs.csv`
  table with objective notes; the **verdict on each question below**; retarget findings.
- The NPZs, screenshots / captures, `soma77_joints.txt`, `soma_to_quaternius.csv`, any
  retarget script and output `.glb`.
- Keep them out of git. Ask Erik how he wants them moved to the home PC (his Google-Drive
  `ClaudeSync` folder is one option; he decides).

**Questions the report must answer (yes / no / partly, with evidence):**
1. Does text alone produce a hunched, arms-forward posture (B1)?
2. Does a full-body keyframe hold that posture through a walk (B2)?
3. Does the model know animal-like or on-all-fours locomotion at all (B3)?
4. Are attack, idle, fall and rise (B4–B7) usable as a starting point?
5. Can it do a heavy carried object (B8), or is that a dead end?
6. What does it cost: setup time, seconds per clip, VRAM / RAM, Windows vs WSL2 / Docker?
7. How lossy is the retarget onto the 53-bone skeleton?
8. What does the licence allow for a commercial game that also does RL research?

## 8. Stop conditions

Stop and report (don't push on) if: setup is not working after 1.5 h; the licence
prohibits commercial use of outputs (report it **before** doing Phases 2–4); a command
asks for credentials or an account Erik has not set up; or the README's instructions
would put files inside the breach repo or conda env.
