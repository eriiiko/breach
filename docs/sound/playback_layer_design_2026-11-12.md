# Sound playback layer — design (2026-11-12)

> **Status: PROPOSAL, awaiting Erik.** No game code written. The audition tool
> (`tools/sound_audition.py`, branch `sound-audition`) proved the carbine's
> choke scheme by ear; this is the layer that puts sounds in the game.
> Companion: `README.md` (the parked pressure-field idea — this layer is its
> listed prerequisite, and section 8 says how the two meet).

## 1. What it is

A presentation layer beside the renderer: it **reads** the sim's per-tick
events and state, and **plays** prepared samples through raylib's audio
device. It writes nothing to the sim, runs only in `main.py` (never in
`run_match`, the RL farm or a test), and no digest moves.

Decided so far (Erik, 2026-11-12):

- Sounds are AI-generated clips (Krea Seed Audio now; local models later).
- The carbine is `carbine_corridor_single` at 12 rounds a second by the
  **choke scheme**: every shot plays the whole sample; a new shot fades the
  previous one out in a few ms; the last shot's reverb rings out.
- The frag grenade is `frag_explosion_steel_room`.
- The vent fan loop belongs to the **vent entities**.

## 2. Shape: decide / play, like the UI split

| Part | File | Does |
|---|---|---|
| Model (pure, no raylib) | `audio/model.py` | Turns events + state + listener into **voice commands** (start cue X at gain g, pan p, pitch q; choke voice V; set loop L's gain). All decisions live here; it is tested headless. |
| Bank + door | `audio/bank.py` | Reads `[sound]` from config through a refusing door; maps a cue name to its sample files. |
| Device (dumb) | `audio/device.py` | Owns raylib's audio device, the loaded sounds and the voice pool; executes the commands; decides nothing. |

`main.py` calls it at the two places it already calls the renderer:

- after **every** sim step, beside `renderer.consume_events(...)`:
  `audio.consume_events(sim.tick_events, sim_tick)` — one-shots;
- once per frame: `audio.update(state, listener, dt, paused)` — loops,
  fades, listener movement.

## 3. Cues are data

A **cue** is a named row; nothing in code names a file.

```toml
[sound]
enabled = true
master = 0.8
max_voices = 24
pan_width_m = 12.0        # a source this far to the side is fully left / right

[sound.cues.carbine_shot]
files = ["carbine/corridor_1.wav"]   # several = picked at random, no repeat
gain = 1.0
pitch_jitter = 0.03
gain_jitter = 0.06
choke = "emitter"          # a new one from the same unit fades its previous
choke_fade_ms = 12
reach = "loudness"         # how far it carries: see section 5

[sound.cues.frag_explosion]
files = ["explosion/frag_steel_room.wav"]
reach = "loudness"

[sound.cues.vent_fan]
files = ["loops/vent_fan.wav"]
loop = true
reach_m = 6.0
max_emitters = 4           # only the nearest few play

[sound.bindings.shot]       # weapon id -> cue
k5_carbine = "carbine_shot"

[sound.bindings.explosion]  # payload row -> cue
grenade_frag = "frag_explosion"
```

- The door refuses a missing file, an unknown cue in a binding, an unknown
  weapon or payload id, and any unknown key, by name — as every other table.
- A weapon or payload **without** a binding is silent, not an error (so
  sounds arrive one at a time).
- Samples live tracked in `assets/sounds/`, written by the preparation tool
  (`tools/sound_clips.py`: trimmed, normalized WAV). A `SOURCES.md` there
  records, per file, the generator and the prompt.

## 4. What triggers what

**One-shots, from events** (all exist today, all render-only or already
consumed by the renderer):

| Event | Cue from | Notes |
|---|---|---|
| `ShotFiredEvent` with `launch` | the shooter's weapon id → `bindings.shot` | choke per shooter |
| `ExplosionEvent` | its payload row → `bindings.explosion` | needs the row's name on the event (below) |
| `ReloadEvent` | weapon id → `bindings.reload` | |
| `MeleeEvent`, `ThrowEvent`, `PlantEvent`, `StaggerEvent` | one binding each | later patches |
| `UnitHitEvent` / `UnitKilledEvent` | by role (marine / zombie) and cause | later |
| `DoorDestroyedEvent`, `WallDestroyedEvent` | by material | later |

**One small sim-side addition:** `ExplosionEvent` carries `kind`
("grenade" / "door_explosive") but not which payload row fired, so a frag
and a C4 cannot get different sounds. Proposed: one more render-only field,
`payload: str` (the row's name), filled where the event is built. The event
is outside the digest (as its `strength` and `flash` fields are), so nothing
hashes differently.

**Loops, from state** (read each frame, no events needed):

| Emitter | Cue | Plays while |
|---|---|---|
| vent entity | `vent_fan` | the vent is alive and wired to a duct |
| fire | `fire_small` / `fire_large` | burning tiles near the listener; ONE pair of voices for the whole fire, their gains following how much burns within earshot (never a voice per tile) |
| heat source (engine), props, TVs | a `sound` column on their row | later patches |

Footsteps are their own patch: the pose layer already measures each gait
clip's phase, so a step sound fires when a drawn unit's phase crosses a foot
contact, with the cue chosen by the floor material under it.

## 5. Where you hear it from, and how far it carries

- **Listener:** the possessed unit under direct control; otherwise the
  camera's centre.
- **Gain:** falls with distance from the listener to 0 at the cue's reach.
- **Pan:** by the source's horizontal offset on screen (right of the
  listener = right speaker), full at `pan_width_m`.
- **No occlusion in v1:** a shot behind a wall sounds like one in the open.
  Section 8 is where that gets fixed.

**Proposal — `reach = "loudness"`:** weapon and payload rows already carry
`loudness`, the hearing radius in metres that wakes zombies (carbine 24 m).
A cue with `reach = "loudness"` uses that same number as the distance at
which the player stops hearing it. Then what you hear is what the horde
hears: if you can hear a shot, so can a zombie at your position. The mix
dial stays the cue's `gain`; a cue may instead give its own `reach_m`.
**This is the one decision in this document I want Erik's yes or no on.**

## 6. Voices

- A fixed pool (`max_voices`, 24). When it is full, the quietest voice is
  stolen; loops are never stolen by one-shots of lower gain.
- **Choke groups:** `choke = "emitter"` fades the previous voice of the same
  cue from the same unit — two marines firing do not cut each other off.
- **Loops:** only the `max_emitters` nearest emitters of a cue play; gains
  ramp over ~100 ms so a vent coming into range does not pop in.
- **Variation:** file choice, pitch and gain jitter come from the layer's own
  seeded generator — never the sim's.

## 7. Time

- One-shots start on the frame their sim step ran. Several steps in one
  frame all sound (the same reason `consume_events` runs per step).
- **Paused:** playing one-shots finish; loops fade to silence and resume.
- **Slow motion (`[display] sim_speed`):** v1 plays samples at normal pitch.
- **Replay seek / fast-forward:** events of undrawn ticks are dropped, so a
  seek does not play a thousand shots at once.

## 8. How the parked pressure-field idea plugs in later

Every voice's gain already passes through one function,
`acoustics(listener, source) -> gain` (v1: distance only). Rung 0 of the
parked design replaces that function with one that reads the pressure
transient and the geometry, and adds a second output, **muffle**.

raylib's simple sound playback has no filter. Two routes for muffle, to be
chosen then, not now: (a) each cue ships a pre-filtered "muffled" copy and
the layer crossfades between the two; (b) the layer mixes its own stream.
The same choice decides whether the baked-in corridor reverb of today's
carbine sample is replaced by a dry shot plus a per-room tail.

## 9. Known limits of v1

- The choke fade is stepped per frame, not per audio sample. It was not
  audibly a problem in the audition tool's first listen; if clicks appear
  under load, route (b) above fixes it.
- No occlusion, no reverb change per room, no Doppler, no vacuum silence.
  (Sound in a decompressed room is a nice later rule: gain scaled by the
  air density at the source and at the listener — one term in `acoustics`.)
- WEGO / onephase paths get whatever their events give; not tuned.

## 10. Patches

| # | Content | Gate |
|---|---|---|
| S1 | The layer (model, bank + door, device), `[sound]`, listener, gain, pan, voices, choke; carbine shot, frag explosion, reload; `ExplosionEvent.payload`; `assets/sounds/` + `SOURCES.md` | model tests headless; E2E driver run with sound on exits clean; **HUMAN-TEST** (Erik plays) |
| S2 | Loops: vent fans, the fire pair | HUMAN-TEST |
| S3 | Footsteps from gait phase, by floor material; melee, throw, plant, hits, deaths, doors | HUMAN-TEST |
| S4 | Sounds on props and heat sources (a `sound` column); TV programmes (generated speech + music) | HUMAN-TEST |
| later | Occlusion and muffle from the pressure field (the parked rung 0) | its own design |

Every patch is feel-adjacent, so none auto-merges.

## 11. Statements Erik may veto

1. The layer lives in a new top-level `audio/` package, beside `renderer/`.
2. Sounds are bound by weapon id / payload row in `[sound.bindings]`, not as
   columns on the sim's weapon rows (the held-props precedent: render data
   stays out of sim tables).
3. An unbound weapon or payload is silent, not refused.
4. Fire is one pair of voices for the whole fire, not a voice per tile.
5. Vents are capped to the nearest four audible at once.
6. Tracked WAVs in `assets/sounds/` (the whole basic set is a few MB).

## Systems

**Existing canonical systems this design uses**

- Events (`simulation/events.py`) and the per-step consume seam in `main.py`
  — the layer is a second consumer beside `GameRenderer.consume_events`.
- Config (`CFG`, refusing doors) — every dial and binding.
- Weapon / payload `loudness` (the Noise row) — the reach, if approved.
- Entity system (vents; later heat sources, props) — loop emitters read
  entity state, never a second registry.
- Pose layer gait phase (`pose_locomotion`) — footsteps (S3).
- E2E driver — the reproduction and the smoke gate.
- `tools/sound_clips.py` — the one clip-preparation path.

**New systems it creates (draft rules)**

- **Sound layer** (`audio/model.py` pure / `audio/device.py` dumb): the only
  audio path; decisions in the model as data, the device decides nothing;
  reads events and state, writes no sim state, never imported headless.
- **Cue table** (`[sound.cues.*]`, `[sound.bindings.*]`): a sound is a cue
  row and a binding row — never a file name or a per-weapon branch in code.
- **Acoustics seam** (`audio/model.py::acoustics`): every voice's gain (and
  later muffle) comes from this one function; a new propagation effect is a
  term here, never a second path.
- **Sound assets** (`assets/sounds/` + `SOURCES.md`): every sample is
  written by `tools/sound_clips.py` and has its generator and prompt
  recorded.
