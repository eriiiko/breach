# Sound from the sim — project folder (START HERE)

> **Status: PARKED (2026-07-12).** Idea captured, refined, and prior-art-scouted; **no research
> pass run in full, no code written.** Erik is closing other arcs first and will pick this up
> later. This README is the entry point — read it, then dive into the two companion docs:
>
> | File | What it is |
> |---|---|
> | `README.md` (this) | Status, everything we learned, and the ordered next steps. |
> | `research_brief.md` | The detailed design + the refined question for a deep-research pass. |
> | `papers.md` | Annotated bibliography from the 2026-07-12 scout — the ~25 sources of value. |
>
> Project memory also carries a one-paragraph pointer (`sound-ml-track`).

---

## 1. What this is, in three sentences

Breach's physics engine already computes a live pressure field (`P = C·N·T`, ideal gas). **Sound
is pressure fluctuation**, so the engine's per-tick pressure transient `|P − P_prev|` at the
listener is, for free, a coarse acoustic signal that already respects the ship's real, dynamic,
destructible geometry. The idea: **sample a small neighbourhood around the listener and map it —
analytically, then with a small neural network — to DSP parameters (EQ, gain, muffle, reverb send,
pan) applied to prerecorded samples**, instead of generating audio waveforms with a net.

## 2. Everything we learned (2026-07-12 session)

### 2a. The engine changed under the idea — for the better

The original idea (2026-07-07) was built on `wave_p`, a dedicated acoustic wave field. **The EOS
refactor retired `wave_p`.** There is now one derived pressure `P = C·N_total·T`, advanced by a
Kwatra semi-implicit solver, materialized once per tick as a deterministic Q16.16 field. The old
`gmap.wave_p` buffer is recycled as `P_prev`, and the engine already computes `|P − P_prev|` every
tick (it drives the water-ripple splash and the over-pressure burst trigger). So the acoustic
oracle didn't vanish — it became **more physically honest** (a real pressure transient, not a
bolted-on zero-mean field). Erik's verdict: an improvement.

- **Sampling rate is a non-issue.** (An early worry, corrected.) Nothing was ever sampled at
  "600 Hz" — that was a hypothetical. The pressure field carries no genuine high-frequency content,
  so the ~12 Hz tick rate captures its (inherently slow) envelope faithfully.
- **Two real consequences**, both properties of the sim, not the sampling: reverb must be
  **parametric** (send + decay), never convolution-from-a-sim-IR; and cross-wall muffling must be
  **approximated** (the engine's through-wall wave transmission "4b" is deferred, so the front does
  not cross walls, only diffracts through openings).

### 2b. The design: teacher–student distillation (Erik's, 2026-07-12)

The make-or-break question the brief had flagged — *what is the training target?* — is answered:

- **Teacher:** a high-resolution offline acoustic simulation (proper FDTD/wave) = ground truth for
  true attenuation / reverb / occlusion.
- **Student:** a small net reading only the **cheap live coarse features** the game already computes,
  trained to reproduce the teacher's perceptual params — i.e. it learns the **correction** from
  cheap-game-acoustics to true-acoustics ("adjust the sound to match the true simulation").
- **Infinite training data** is Breach's structural edge: the deterministic engine emits unlimited
  (geometry, event) scenarios; the teacher labels each once, offline. Same "engine as data
  generator" muscle as the self-play training infra.
- **Crucial design choice:** the student conditions on the **cheap live field, not the geometry** —
  that is what makes it both novel and runtime-cheap.

### 2c. The prior-art verdict (16 targeted searches — see `papers.md`)

1. **As a game-audio tool, the idea is derivative — so don't frame it that way** (Erik's steer, and
   the scout confirms it). **Planeverb** (Microsoft, CGF 2020) already does live 2D wave sim →
   perceptual params for dynamic scenes, open-source; **Project Acoustics** is the precomputed-static
   version; **Learning Acoustic Scattering Fields** already *learns* dynamic propagation.
2. **The teacher–student method is PROVEN and common — feasibility is de-risked.** Distilling an
   expensive acoustic solver into a cheap net gets 67–1000× speedups across many papers (Alguacil
   AIAA'22, FNO/neural-operator surrogates, NAT'25, SOAF'24). We are not betting on an unproven method.
3. **Breach's genuinely open cell is the *configuration*, not the method:** a *learned
   parameter-mapping sonification of a live, coarse, gameplay compressible-flow field onto
   DSP-controlled prerecorded samples*, on dynamic geometry. Nobody reuses the game's own physics as
   the acoustic feature. **Frame it as audio-ML / acoustics / parameter-mapping sonification**, not
   game audio. (The Sonification Handbook ch.15 is the cleanest conceptual home.)

## 3. The plan (three rungs)

- **Rung 0 — analytic map, no ML.** Listener-neighbourhood transient + `∇P` → hand-tuned DSP params.
  Ships game value immediately; doubles as the **stealth-layer detector** for the reserved `loudness`
  weapon stat (propagation-based AI hearing off the deterministic `P` field).
- **Rung 1 — the teacher–student net** (§2b). The portfolio-paper-shaped rung.
- **Rung 2 — learned spatial/directional acoustics** (neural operator / learned IR). Default out of
  scope; the coarse control rate makes IR-flavored variants ill-fitting.

## 4. What to do next (in order, when this is un-parked)

0. **The make-or-break validation experiment (cheap, in-house — do this FIRST).** On one ship
   scenario, build a toy high-res acoustic teacher, extract ground-truth attenuation/decay, and test
   whether the coarse `|P − P_prev|` neighbourhood features can predict it. The game field and the
   teacher are *different physics* (compressible flow vs linear acoustics) — this is **the** risk, and
   it gates everything. If the coarse field predicts the teacher well, rung 1 is real; if not, rung 0
   is the answer and rung 1 dies. This is a small standalone script, not a token spend.
1. **Build the audio playback layer** (a prerequisite nobody mentioned). Breach has **no audio at
   all** yet — rung 0 needs a mundane sample-load / mixer / pan / filter foundation first. (Erik has
   not downloaded any samples yet; that is this prerequisite, not a blocker for the research.)
2. **(Optional) the deep-research pass** — now *deferrable*, since the scout already answered "is the
   cell open?" (yes) and "does the method work?" (yes). If run, **partition it** (~3M tokens; it blew
   Erik's 5 h window twice) and aim it only at the three open sub-questions in `research_brief.md` §6
   Q4: (a) teacher solver + ground-truth parameterization, (b) whether a 2024–25 surrogate paper
   already fills the cell, (c) the validation-experiment design.
3. **Rung 0 implementation** — attaches to Phase 2 game rules (stealth) + a beauty track.
4. **Rung 1 implementation** — only after rung 0 and a green validation experiment.

## 5. The one risk to remember

**Does the coarse gameplay pressure field carry enough signal to predict a true high-res acoustic
simulation?** Everything hinges on this, and it is answered by experiment #0 above, not by more
reading. Bet: yes for envelope-level features (occlusion, room size, venting are low-frequency and
present in the coarse field), unknown for anything finer.

## 6. Where it sits in the roadmap

Not yet in `docs/roadmap_2026-07.md` (awaiting Erik's sign-off). Natural home: **Phase 2** — rung 0
with the game-rules/stealth work (where `loudness` finally gets a consumer), the audible half as a
beauty track alongside the black-body emitter. Junior to the open arcs Erik is closing first.
