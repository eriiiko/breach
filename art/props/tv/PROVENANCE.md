# CRT televisions — provenance

Generated 2026-11-02 with Krea (Erik's Individual Pro plan) for Breach's
interior props, then made game-ready by `scripts/tv_gameready.py`.

## Sources (not in git: 23–84 MB each)

Raw generator output lives untracked in the main checkout's
`art/props/tv/krea/`. Every model got the same prompt:

> A retro 1970s CRT television set for a worn industrial spaceship crew
> lounge. Thick chunky boxy cabinet in scuffed off-white and grey plastic with
> metal trim. The front glass screen is a rounded-corner rectangle that bulges
> outward, convex curved glass, dark and switched off. Thick bezel around the
> screen, a narrow side panel with two rotary knobs and a small speaker
> grille. The back is shallow and tapered, not a deep tube. Single object,
> standing on a flat base, game prop.

| game asset | source file | route | model, settings | cost (CU) |
|---|---|---|---|---|
| `tv_crt_a` (**the chosen one**) | `A_text_hunyuan.glb` | text → 3D | Hunyuan3D 3.1 Pro, face_count 1.5M, PBR | 500 |
| `tv_crt_b` | `A_text_tripo.glb` | text → 3D | Tripo H3.1, face_limit 2M | 148 |
| `tv_crt_c` | `A_text_rodin.glb` | text → 3D | Rodin 2.5 | 296 |
| `tv_crt_d` | `B_img_hunyuan.glb` | image → 3D | Hunyuan3D 3.1 Pro, 1.5M, PBR, from `concept_nbpro.jpg` | 500 |
| `tv_crt_e` | `B_img_tripo.glb` | image → 3D | Tripo H3.1, 2M, from `concept_nbpro.jpg` | 148 |

`concept_nbpro.jpg`: Nano Banana Pro, 2K, a product-render prompt of the same TV.

## Game-ready commands

    blender -b -P scripts/tv_gameready.py -- <src>.glb assets/models/props/krea_tv/<name>/<name>.glb art/props/tv/previews <flags>

| asset | flags |
|---|---|
| `tv_crt_a` | `--rect -0.200 0.110 0.107 0.338 --round 4` |
| `tv_crt_b` | `--front +x` (screen found automatically) |
| `tv_crt_c` | `--front -y --rect -0.143 0.131 0.145 0.414 --round 4` |
| `tv_crt_d` | `--front +x --rect -0.198 0.103 0.040 0.278 --round 4` |
| `tv_crt_e` | `--front +x --rect -0.191 0.126 0.055 0.323 --round 4` |

Each result: ~4,000-triangle body (material `tv_body`, 1024² baked albedo)
plus our own screen glass (material `tv_screen`, UVs 0..1 over the glass),
0.50 m wide, screen facing −Y (glTF +Z), standing on the floor.
