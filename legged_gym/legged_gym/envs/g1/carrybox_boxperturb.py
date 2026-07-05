import math

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate

from .carrybox import LeggedRobot as CarryBox


class LeggedRobot(CarryBox):
    """Carry-box task with a single, gated force pulse at the free box COM."""

    _DIRECTION_IDS = {
        "+box_x": 0,
        "-box_x": 1,
        "+box_y": 2,
        "-box_y": 3,
        "-z_world": 4,
    }

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        device = self.device

        # This tensor is intentionally separate from the legacy robot disturbance.
        self.box_perturb_force_tensor = torch.zeros_like(self.disturbance)
        self.box_perturb_peak_force_world = torch.zeros((n, 3), device=device)
        self.box_perturb_direction_world = torch.zeros((n, 3), device=device)
        self.box_perturb_peak_force_N = torch.zeros(n, device=device)
        self.box_perturb_beta = torch.zeros(n, device=device)
        self.box_perturb_elapsed_physics_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_remaining_physics_steps = torch.zeros(n, dtype=torch.long, device=device)

        self.confirmed_carry_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_decision_made_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_event_count_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_direction_id_buf = torch.full((n,), -1, dtype=torch.long, device=device)
        self.box_perturb_mass_kg = torch.zeros(n, device=device)
        self.box_perturb_cap_used_buf = torch.zeros(n, dtype=torch.bool, device=device)

        self.box_perturb_recovery_active_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_recovery_confirmed_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_recovery_elapsed_policy_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_recovery_success_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_recovery_done_buf = torch.zeros(n, dtype=torch.bool, device=device)

        # Run-level counters provide stable rollout metrics even when no episode ends.
        self._perturb_total_decisions = torch.zeros((), device=device)
        self._perturb_total_events = torch.zeros((), device=device)
        self._perturb_total_completed_episodes = torch.zeros((), device=device)
        self._perturb_total_force_peak_N = torch.zeros((), device=device)
        self._perturb_total_beta = torch.zeros((), device=device)
        self._perturb_total_mass_kg = torch.zeros((), device=device)
        self._perturb_total_cap_used = torch.zeros((), device=device)
        self._perturb_total_recoveries = torch.zeros((), device=device)
        self._perturb_total_recovery_successes = torch.zeros((), device=device)

        assert self.box_perturb_force_tensor.shape == self.contact_forces.shape
        assert int(self.box_net_contact_force_index) == 2, (
            "CarryBox actor creation order changed; expected box rigid-body index 2, got "
            f"{int(self.box_net_contact_force_index)}"
        )
        self.debug_viz = bool(self.cfg.box_perturbation.debug_draw_force)

    def step(self, actions):
        """Apply the pulse before every physics simulate call in the decimation loop."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self._apply_box_perturbation_force()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        termination_ids, termination_privileged_obs, amp_obs_buf = self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_privileged_obs,
            amp_obs_buf,
        )

    def _apply_box_perturbation_force(self):
        """Apply a midpoint-sampled half-sine force at the box rigid-body COM."""
        self.box_perturb_force_tensor.zero_()
        active = (
            bool(self.cfg.box_perturbation.enabled)
            & (self.box_perturb_remaining_physics_steps > 0)
        )
        if torch.any(active):
            pulse_steps = self._pulse_physics_steps()
            tau_fraction = (
                self.box_perturb_elapsed_physics_steps[active].float() + 0.5
            ) / float(pulse_steps)
            profile = torch.sin(math.pi * tau_fraction)
            force = self.box_perturb_peak_force_world[active] * profile.unsqueeze(1)
            self.box_perturb_force_tensor[
                active, int(self.box_net_contact_force_index), :
            ] = force

        # ENV_SPACE has world-aligned axes for these untranslated/unrotated env frames.
        # The rigid-body tensor API acts at each body COM and supplies no torque.
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            forceTensor=gymtorch.unwrap_tensor(self.box_perturb_force_tensor),
            space=gymapi.CoordinateSpace.ENV_SPACE,
        )

        if torch.any(active):
            self.box_perturb_elapsed_physics_steps[active] += 1
            self.box_perturb_remaining_physics_steps[active] -= 1

    def _update_carry_phase(self):
        # Base post_physics_step calls this after refreshing state/contact tensors.
        super()._update_carry_phase()
        self._update_box_perturbation_state()

    def _update_box_perturbation_state(self):
        cfg = self.cfg.box_perturbation
        if not bool(cfg.enabled):
            self._clear_box_perturbation_state_for_gate()
            self.extras["perturb"] = self._build_perturb_log_info()
            return

        self.confirmed_carry_streak[:] = torch.where(
            self.confirmed_carry_buf,
            self.confirmed_carry_streak + 1,
            torch.zeros_like(self.confirmed_carry_streak),
        )

        self._update_recovery_state()
        self._log_applied_force_debug()

        eligible = (
            (self.confirmed_carry_streak >= int(cfg.stable_confirmed_carry_policy_steps))
            & ~self.box_perturb_decision_made_buf
            & (self.box_perturb_event_count_buf < int(cfg.max_events_per_episode))
        )
        if torch.any(eligible):
            self.box_perturb_decision_made_buf[eligible] = True
            self._perturb_total_decisions += eligible.float().sum()
            probability = 1.0 if bool(cfg.debug_force_event) else self._stage_probability()
            sampled = torch.rand(self.num_envs, device=self.device) < probability
            trigger_ids = torch.nonzero(eligible & sampled, as_tuple=False).flatten()
            if trigger_ids.numel() > 0:
                self._schedule_box_perturbation(trigger_ids)

        self.extras["perturb"] = self._build_perturb_log_info()

    def _clear_box_perturbation_state_for_gate(self):
        """Make runtime gate-off transitions immediate and free of force leakage."""
        self.box_perturb_force_tensor.zero_()
        self.box_perturb_peak_force_world.zero_()
        self.box_perturb_direction_world.zero_()
        self.box_perturb_peak_force_N.zero_()
        self.box_perturb_beta.zero_()
        self.box_perturb_mass_kg.zero_()
        self.box_perturb_elapsed_physics_steps.zero_()
        self.box_perturb_remaining_physics_steps.zero_()
        self.confirmed_carry_streak.zero_()
        self.box_perturb_decision_made_buf.zero_()
        self.box_perturb_event_count_buf.zero_()
        self.box_perturb_direction_id_buf.fill_(-1)
        self.box_perturb_cap_used_buf.zero_()
        self.box_perturb_recovery_active_buf.zero_()
        self.box_perturb_recovery_confirmed_streak.zero_()
        self.box_perturb_recovery_elapsed_policy_steps.zero_()
        self.box_perturb_recovery_success_buf.zero_()
        self.box_perturb_recovery_done_buf.zero_()

    def _log_applied_force_debug(self):
        cfg = self.cfg.box_perturbation
        interval = int(cfg.debug_force_log_interval_policy_steps)
        if not bool(cfg.debug_draw_force) or interval <= 0:
            return
        if self.common_step_counter % interval != 0:
            return
        applied = self.box_perturb_force_tensor[
            :, int(self.box_net_contact_force_index), :
        ]
        active_ids = torch.nonzero(
            torch.linalg.vector_norm(applied, dim=-1) > 1.0e-6,
            as_tuple=False,
        ).flatten()
        if active_ids.numel() == 0:
            return
        env_id = int(active_ids[0].item())
        force = applied[env_id]
        magnitude = torch.linalg.vector_norm(force)
        print(
            "[Box perturb applied] "
            f"policy_step={self.common_step_counter} env={env_id} "
            f"force_world_N={force.detach().cpu().tolist()} "
            f"magnitude_N={float(magnitude):.6f} "
            f"peak_N={float(self.box_perturb_peak_force_N[env_id]):.6f}"
        )

    def _schedule_box_perturbation(self, env_ids):
        cfg = self.cfg.box_perturbation
        stage_name = self._stage_name()
        stage_cfg = cfg.stages[stage_name]
        direction_names = stage_cfg["directions"]
        sampled_indices = torch.randint(
            len(direction_names), (env_ids.numel(),), device=self.device
        )

        direction_world = torch.zeros((env_ids.numel(), 3), device=self.device)
        beta = torch.zeros(env_ids.numel(), device=self.device)
        direction_ids = torch.full(
            (env_ids.numel(),), -1, dtype=torch.long, device=self.device
        )
        box_quat = self.box_states[env_ids, 3:7]
        for index, direction_name in enumerate(direction_names):
            mask = sampled_indices == index
            if not torch.any(mask):
                continue
            direction_ids[mask] = self._DIRECTION_IDS[direction_name]
            axis_name = "z" if direction_name == "-z_world" else direction_name[-1]
            low, high = stage_cfg["beta"][axis_name]
            beta[mask] = float(low) + (float(high) - float(low)) * torch.rand(
                int(mask.sum().item()), device=self.device
            )
            if direction_name == "-z_world":
                direction_world[mask, 2] = -1.0
            else:
                local_axis = torch.zeros((int(mask.sum().item()), 3), device=self.device)
                component = 0 if axis_name == "x" else 1
                sign = 1.0 if direction_name[0] == "+" else -1.0
                local_axis[:, component] = sign
                direction_world[mask] = quat_rotate(box_quat[mask], local_axis)

        mass = self.box_masses[env_ids]
        uncapped_peak = beta * mass * 9.81
        cap = cfg.force_peak_cap_N
        if cap is None:
            peak = uncapped_peak
            cap_used = torch.zeros_like(peak, dtype=torch.bool)
        else:
            peak = torch.clamp(uncapped_peak, max=float(cap))
            cap_used = uncapped_peak > float(cap)

        self.box_perturb_direction_world[env_ids] = direction_world
        self.box_perturb_peak_force_N[env_ids] = peak
        self.box_perturb_peak_force_world[env_ids] = direction_world * peak.unsqueeze(1)
        self.box_perturb_beta[env_ids] = beta
        self.box_perturb_mass_kg[env_ids] = mass
        self.box_perturb_cap_used_buf[env_ids] = cap_used
        self.box_perturb_direction_id_buf[env_ids] = direction_ids
        self.box_perturb_elapsed_physics_steps[env_ids] = 0
        self.box_perturb_remaining_physics_steps[env_ids] = self._pulse_physics_steps()
        self.box_perturb_event_count_buf[env_ids] += 1
        self.box_perturb_recovery_success_buf[env_ids] = False

        count = float(env_ids.numel())
        self._perturb_total_events += count
        self._perturb_total_force_peak_N += peak.sum()
        self._perturb_total_beta += beta.sum()
        self._perturb_total_mass_kg += mass.sum()
        self._perturb_total_cap_used += cap_used.float().sum()

        if bool(cfg.debug_force_event):
            first = int(env_ids[0].item())
            print(
                "[Box perturb debug] "
                f"stage={stage_name} env={first} direction_id={int(direction_ids[0])} "
                f"beta={float(beta[0]):.4f} mass_kg={float(mass[0]):.4f} "
                f"peak_N={float(peak[0]):.4f} cap_used={bool(cap_used[0])}"
            )

    def _update_recovery_state(self):
        had_event = self.box_perturb_event_count_buf > 0
        pulse_finished = (
            had_event
            & (self.box_perturb_elapsed_physics_steps >= self._pulse_physics_steps())
            & ~self.box_perturb_recovery_active_buf
            & ~self.box_perturb_recovery_done_buf
        )
        self.box_perturb_recovery_active_buf[pulse_finished] = True
        self.box_perturb_recovery_elapsed_policy_steps[pulse_finished] = 0
        self.box_perturb_recovery_confirmed_streak[pulse_finished] = 0

        active = self.box_perturb_recovery_active_buf
        if not torch.any(active):
            return
        self.box_perturb_recovery_elapsed_policy_steps[active] += 1
        self.box_perturb_recovery_confirmed_streak[:] = torch.where(
            active & self.confirmed_carry_buf,
            self.box_perturb_recovery_confirmed_streak + 1,
            torch.zeros_like(self.box_perturb_recovery_confirmed_streak),
        )

        success = active & (
            self.box_perturb_recovery_confirmed_streak
            >= int(self.cfg.box_perturbation.recovery_confirmed_carry_steps)
        )
        if torch.any(success):
            self.box_perturb_recovery_success_buf[success] = True
            self.box_perturb_recovery_done_buf[success] = True
            self.box_perturb_recovery_active_buf[success] = False
            self._perturb_total_recoveries += success.float().sum()
            self._perturb_total_recovery_successes += success.float().sum()

        timeout = self.box_perturb_recovery_active_buf & (
            self.box_perturb_recovery_elapsed_policy_steps
            >= self._recovery_policy_steps()
        )
        if torch.any(timeout):
            self.box_perturb_recovery_active_buf[timeout] = False
            self.box_perturb_recovery_done_buf[timeout] = True
            self._perturb_total_recoveries += timeout.float().sum()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        has_buffers = hasattr(self, "box_perturb_event_count_buf")
        if has_buffers:
            interrupted_recovery = self.box_perturb_recovery_active_buf[env_ids]
            self._perturb_total_recoveries += interrupted_recovery.float().sum()
            self._perturb_total_completed_episodes += float(len(env_ids))
            episode_events = self.box_perturb_event_count_buf[env_ids].float()
            episode_success = self.box_perturb_recovery_success_buf[env_ids].float()

        super().reset_idx(env_ids)
        if not has_buffers:
            return

        self.extras["episode"]["Perturb/events_per_episode"] = episode_events.mean()
        event_denominator = torch.clamp(episode_events.sum(), min=1.0)
        self.extras["episode"]["Perturb/recovery_success_rate"] = (
            (episode_success * episode_events).sum() / event_denominator
        )

        for name in (
            "box_perturb_force_tensor",
            "box_perturb_peak_force_world",
            "box_perturb_direction_world",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "box_perturb_peak_force_N",
            "box_perturb_beta",
            "box_perturb_mass_kg",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "box_perturb_elapsed_physics_steps",
            "box_perturb_remaining_physics_steps",
            "confirmed_carry_streak",
            "box_perturb_event_count_buf",
            "box_perturb_recovery_confirmed_streak",
            "box_perturb_recovery_elapsed_policy_steps",
        ):
            getattr(self, name)[env_ids] = 0
        for name in (
            "box_perturb_decision_made_buf",
            "box_perturb_cap_used_buf",
            "box_perturb_recovery_active_buf",
            "box_perturb_recovery_success_buf",
            "box_perturb_recovery_done_buf",
        ):
            getattr(self, name)[env_ids] = False
        self.box_perturb_direction_id_buf[env_ids] = -1

    def _pulse_physics_steps(self):
        return max(
            1,
            int(
                round(
                    float(self.cfg.box_perturbation.pulse_duration_s)
                    / float(self.sim_params.dt)
                )
            ),
        )

    def _recovery_policy_steps(self):
        policy_dt = float(self.sim_params.dt) * int(self.cfg.control.decimation)
        return max(
            1,
            int(round(float(self.cfg.box_perturbation.recovery_window_s) / policy_dt)),
        )

    def _stage_name(self):
        cfg = self.cfg.box_perturbation
        if cfg.manual_stage_override is not None:
            stage_name = str(cfg.manual_stage_override)
            if stage_name not in cfg.stage_start_policy_steps:
                raise ValueError(f"Unknown box perturbation stage: {stage_name}")
            return stage_name
        if cfg.schedule_mode != "staged_policy_steps":
            raise ValueError(f"Unsupported perturbation schedule mode: {cfg.schedule_mode}")
        stage_name = "C1"
        for candidate, start in sorted(
            cfg.stage_start_policy_steps.items(), key=lambda item: item[1]
        ):
            if self.common_step_counter >= int(start):
                stage_name = candidate
        return stage_name

    def _stage_id(self):
        return int(self._stage_name()[1:])

    def _stage_probability(self):
        return float(self.cfg.box_perturbation.stages[self._stage_name()]["probability"])

    @staticmethod
    def _safe_ratio(numerator, denominator):
        return torch.where(
            denominator > 0.0,
            numerator / torch.clamp(denominator, min=1.0),
            torch.zeros_like(numerator),
        )

    def _build_perturb_log_info(self):
        events = self._perturb_total_events
        recoveries = self._perturb_total_recoveries
        return {
            "enabled": torch.tensor(
                float(bool(self.cfg.box_perturbation.enabled)), device=self.device
            ),
            "stage_id": torch.tensor(float(self._stage_id()), device=self.device),
            "event_trigger_rate": self._safe_ratio(
                events, self._perturb_total_decisions
            ),
            "events_per_episode": self._safe_ratio(
                events, self._perturb_total_completed_episodes
            ),
            "force_peak_N_mean": self._safe_ratio(
                self._perturb_total_force_peak_N, events
            ),
            "beta_mean": self._safe_ratio(self._perturb_total_beta, events),
            "mass_kg_mean": self._safe_ratio(self._perturb_total_mass_kg, events),
            "force_cap_usage_rate": self._safe_ratio(
                self._perturb_total_cap_used, events
            ),
            "recovery_success_rate": self._safe_ratio(
                self._perturb_total_recovery_successes, recoveries
            ),
            "active_pulse_fraction": (
                self.box_perturb_remaining_physics_steps > 0
            ).float().mean(),
            "recovery_window_fraction": self.box_perturb_recovery_active_buf.float().mean(),
        }

    def _draw_debug_vis(self):
        """Draw the latest commanded external force as an arrow at the box COM."""
        super()._draw_debug_vis()
        cfg = self.cfg.box_perturbation
        if not bool(cfg.debug_draw_force):
            return

        applied = self.box_perturb_force_tensor[
            :, int(self.box_net_contact_force_index), :
        ]
        scale = float(cfg.debug_force_draw_scale_m_per_N)
        max_envs = min(self.num_envs, int(cfg.debug_force_draw_max_envs))
        color = np.asarray([1.0, 0.25, 0.0], dtype=np.float32)

        for env_id in range(max_envs):
            force = applied[env_id].detach().cpu().numpy()
            magnitude = float(np.linalg.norm(force))
            if magnitude <= 1.0e-6:
                continue

            direction = force / magnitude
            start = self.box_states[env_id, :3].detach().cpu().numpy()
            end = start + force * scale

            reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(np.dot(direction, reference))) > 0.9:
                reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            side = np.cross(direction, reference)
            side /= max(float(np.linalg.norm(side)), 1.0e-6)

            shaft_length = magnitude * scale
            head_length = min(
                float(cfg.debug_force_arrow_head_length_m),
                max(0.25 * shaft_length, 0.01),
            )
            head_width = 0.5 * head_length
            head_base = end - direction * head_length
            left = head_base + side * head_width
            right = head_base - side * head_width

            vertices = np.stack(
                (start, end, end, left, end, right), axis=0
            ).astype(np.float32).reshape(3, 6)
            colors = np.repeat(color.reshape(1, 3), 3, axis=0)
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                3,
                vertices,
                colors,
            )
