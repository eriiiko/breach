# Q3. Isaac Gym / Isaac Lab tensor API: step/reset contract, masks, randomisation, host syncs

Raw research note, 2026-11-07, for #29.

## Sources
- Isaac Gym tensor API docs (mirror of the Preview docs):
  https://lequn-f.github.io/isaacgym/programming/tensors.html
- Isaac Lab `DirectRLEnv` source (read):
  https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/direct_rl_env.py
- Isaac Lab envs API: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html
- Isaac Lab event terms / manager (via search):
  https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/mdp/events.html ,
  https://isaac-sim.github.io/IsaacLab/v2.1.0/_modules/isaaclab/managers/event_manager.html
- Isaac Lab reproducibility page:
  https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html

## Isaac Gym (2021) tensor API
- "Global" tensors hold one quantity for all actors of all envs. They are COPIES of engine state,
  refreshed explicitly (`refresh_*_tensor`, once per step after `simulate`), "so no implicit
  copying is ever done"; with the GPU pipeline "the data is never copied to the host".
- Partial reset: `set_actor_root_state_tensor_indexed(sim, state, indices, n)` (and the DOF
  variants) apply the full-size tensor only to the actors listed in an int32 index tensor; the
  others are untouched. On the GPU pipeline each setter "should be called only once per step",
  and refreshes must come before setters.
- `prepare_sim` runs after all envs are built: the actor layout is fixed from then on.

## Isaac Lab (current) `DirectRLEnv.step()` order (source, verified)
1. `_pre_physics_step(action)` (after optional action noise);
2. `for _ in range(decimation): _apply_action(); sim.step()` (control at the policy dt, physics
   at the sim dt);
3. `episode_length_buf += 1`;
4. `reset_terminated, reset_time_outs = _get_dones()`; `reset_buf = terminated | time_outs`;
5. `reward_buf = _get_rewards()`;
6. `reset_env_ids = _reset_envs_from_buffer()` -> `_reset_idx(env_ids)`, which also applies the
   event manager's `"reset"` terms (randomisation at reset);
7. `event_manager.apply(mode="interval", dt=step_dt)`;
8. `obs_buf = _compute_observations()`;
9. return `(obs, reward, terminated, truncated, extras)`: same-step autoreset
   (`autoreset_mode = SAME_STEP`; `compute_final_obs` optionally keeps the terminal observation
   before the reset overwrites it).
- HOST SYNC, documented in the source: the default `_reset_envs_from_buffer` uses
  `reset_buf.nonzero()`, and "CUDA environments synchronize while determining the dynamic output
  size of torch.Tensor.nonzero"; a task "with backend-native mask reset support" may override the
  hook and return None to avoid the sync. So the one sync per step Isaac Lab still has by default
  is building the env-id list for resets, and the documented fix is a MASK-based reset.
- Domain randomisation = event terms with mode `"startup"` (once), `"reset"` (inside
  `_reset_idx`, for the reset env ids) and `"interval"` (per-env timers); e.g.
  `randomize_rigid_body_material` samples per-env friction / restitution into device arrays. The
  reproducibility page advises randomising physics materials only at set-up, because changing
  parameters at run time lets GPU scheduling reorder float operations.
- Determinism (reproducibility page): same hardware, same Isaac Sim / PhysX version, seed set in
  the env config; rigid bodies only; results "can vary across different hardware
  configurations" (float rounding).

## Contract summary (what a Breach design is measured against)
- State lives in device tensors with a leading env axis; the learner reads them in place.
- One `step(actions)` per policy decision; physics runs `decimation` sub-steps inside it.
- Termination and truncation are separate per-env bool tensors; resets happen inside `step` for
  the done envs (same-step autoreset); randomisation is a per-reset event; the observations
  returned are post-reset (terminal observation optional).
- The only host sync left in the default loop is computing reset ids; masks remove it.
