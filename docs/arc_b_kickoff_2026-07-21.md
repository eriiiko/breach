# Arc B kickoff — SignalBus, sensors, pump, automatic airlock

> Ready-to-paste session prompt (written at the S8a Path-A close, 2026-07-21).
> Arc B's gate condition — the physics close-out — is now MET (S8a complete,
> merge `93a014c`; physics engine v1 CLOSED). Paste the block below into a
> fresh session. Workflow: autonomous-patch-workflow; the design is ALREADY
> LOCKED (2026-07-18) — this arc is build-to-spec, with design-gates only
> where the locked docs left explicit open points.

---

You are building **Arc B — the logic layer** of the entity system on the
breach project. The design is LOCKED and errata'd as-built:
`docs/entity_system_design_2026-07-18.md` (canon model; Arc B section ~line
342) + `docs/level_editor_v3_design_2026-07-18.md` (view) + canon chapter
`docs/architecture/engine/16_entity_system.md` (§ "Arc B — logic": SignalBus
+ slot 9e; sensors sample → logic sweep in id order). Arc A (the entity
foundation, A1–A9) is merged and Erik-blessed; doors run on the dev latch
awaiting Arc B's real drivers.

**Scope (from the locked design):** SignalBus dataflow over integer signals
(never a scripting language) · sensors (sample sites → signal values) · the
logic sweep in member-id order · the pump · the automatic airlock (the
while-held `close` input replacing the dev latch) · `button`/terminal
classes. Units are NOT signal endpoints (design §).

**THE BINDING CONTRACT — §5a sensor-gather**
(`docs/cuda_s8a_residency_spec_2026-07-19.md` §5a; Arc B is gated on it):
- Sensor sample sites are STATIC per level. At level load, build the site
  index `(site → tile, field-channel)`.
- Sensors read through ONE accessor: the host mirror on the CPU path, and
  (when the resident gather kernel lands) a `(n_sites × n_channels)` int32
  gather buffer on the resident path. Deterministic site order (id order),
  Q16.16 values, NO dequantize in the sim path.
- **No new consumer may depend on per-tick full-field streaming.** The Q4
  full mirror is transitional; the gather buffer is what survives into
  batched training. Build the accessor seam NOW even though today it reads
  the mirror — Arc B code must not care which backing it has.
- The resident gather KERNEL itself may land in Arc B or be stubbed to the
  mirror accessor — implementer's choice; the interface is what is binding.
  (S8a facts if you build the kernel: mirror is authoritative at tick entry;
  the resident tick's post-EOS D2H is in `_step_resident` step 6 — the
  gather would ride that sync point. ★ `is_ambient` is not static; check
  structural-edit writers before caching anything per-level that combat can
  mutate.)

**Determinism (iron rules):** sensors/logic are sim state — synced, Q16.16
integer only, digest-gated. Logic sweep order is member-id order (canon).
Timers/edges must be integer tick counts. Any new synced state joins the
recorder + digest surface. Dead sensors fail-deadly (design decision,
entity_system_design ~line 297).

**Environment (Lenovo/Ada):** python =
`C:/Users/steen/miniconda3/envs/data/python.exe` (NOT `conda run`); pytest =
`pytest tests -q`; builds `cpp/build_cpu_data.bat` (+ `build_cuda_lenovo.bat`
if the gather kernel lands). Worktree per the concurrent-agent rules; commit
design/impl docs to the branch before spawning worktree agents. ⚠ After any
merge, rebuild the main tree's `.pyd`s.

**Gates:** unit tests per patch + the existing goldens byte-untouched (no
re-baseline — new logic must be dormant on existing levels) + an E2E fixture
level exercising sensor → logic → airlock (extend `levels/` the A-arc way).
The automatic airlock is feel-adjacent at the margin (door timing) —
HUMAN-TEST gate before merge of the airlock behavior patch; mechanical
digest-gated patches may auto-merge on green only if Erik pre-authorizes.

**Arc riders on the books (do NOT absorb):** baker `[art]`/`[bake]`
writeback (fold at Arc C) · `bake_demo` legacy-form migration (Arc C) ·
editor UX panes/wand/wiring (Arc C) · AI tilesets (parked behind Arc C).
