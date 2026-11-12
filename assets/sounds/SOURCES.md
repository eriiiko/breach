# Sound samples — where each came from

Every sample here is AI-generated and written by
`tools/sound_audition.py --export SRC DST [--shot N]` (trimmed to its first
transient, normalized; `--shot 0` = the whole clip). The source clips are
kept outside the repo history, in `docs/sound/prototypes/seed_audio/`.

Generator: **Seed Audio 1.0** (`bytedance/seed-audio-1.0`) through Krea,
2026-11-12. The model takes only a text prompt.

| Sample | Source clip | Export | Prompt |
|---|---|---|---|
| `carbine/corridor_1.wav` | `carbine_corridor_single.mp3` | `--shot 1` | Sound effect, no speech, no music: a single gunshot from a short military carbine fired indoors in a steel spaceship corridor. Sharp loud crack, punchy low thump, short metallic ring and a brief tight reverb tail. Close microphone, clean recording. |
| `carbine/reload_1.wav` | `carbine_reload.mp3` | `--shot 0 --squeeze` (7.1 s -> 2.4 s: silences shortened, nothing dropped) | Sound effect only, no voice, no music: reloading a military carbine, recorded very close and dry. Magazine release click, empty magazine sliding out, new magazine pushed in with a firm clack, then the bolt racked with a sharp metallic snap. |
| `explosion/frag_steel_room.wav` | `frag_explosion_steel_room.mp3` | `--shot 0` | Sound effect only, no voice, no music: a single fragmentation grenade exploding inside a steel room on a spaceship. Violent sharp blast, deep bass boom, metal walls rattling, debris and shrapnel pinging off steel, a short heavy metallic reverb tail fading out. |

Chosen by Erik, 2026-11-12 (the carbine at 12 rounds a second by the choke
scheme; the explosion "amazing for the grenade").
