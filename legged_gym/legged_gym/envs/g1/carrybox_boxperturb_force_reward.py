import warnings

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

from .carrybox import LeggedRobot as CarryBox
from .hand_box_force import (
    decompose_force,
    estimate_box_face_normal_local,
    force_closure_residual,
)


class LeggedRobot(CarryBox):
    """Simulator-only force-quality shaping layered on the carry-box task."""

    _FORCE_CONTEXTS = {
        "confirmed": lambda self: self.confirmed_carry_buf,
        "nominal": lambda self: self.confirmed_carry_buf & ~self.box_perturb_active_buf & ~self.perturb_recovery_window_buf,
        "perturb_active": lambda self: self.confirmed_carry_buf & self.box_perturb_active_buf,
        "recovery": lambda self: self.confirmed_carry_buf & self.perturb_recovery_window_buf,
    }

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        device = self.device
        self.box_perturb_active_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.perturb_recovery_window_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_direction_id_buf = torch.full((n,), -1, dtype=torch.long, device=device)
        self._force_debug_steps_left = torch.zeros(n, dtype=torch.long, device=device)

        self.force_face_locked_buf = torch.zeros((n, 2), dtype=torch.bool, device=device)
        self.force_face_id_buf = torch.full((n, 2), -1, dtype=torch.long, device=device)
        self.force_normal_local_buf = torch.zeros((n, 2, 3), device=device)
        self.force_normal_sign_buf = torch.ones((n, 2), device=device)
        self.force_sign_sum_buf = torch.zeros((n, 2), device=device)
        self.force_sign_count_buf = torch.zeros((n, 2), device=device)
        self.force_sign_positive_count_buf = torch.zeros((n, 2), device=device)
        self.force_verified_sample_count_buf = torch.zeros((n, 2), device=device)
        self.force_sign_verified_buf = torch.zeros((n, 2), dtype=torch.bool, device=device)
        self.force_reward_valid_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.force_reward_warned_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.force_reward_disabled_buf = torch.zeros(n, dtype=torch.bool, device=device)

        self.force_normal_world_buf = torch.zeros((n, 2, 3), device=device)
        self.force_raw_world_buf = torch.zeros((n, 2, 3), device=device)
        self.force_normal_component_world_buf = torch.zeros((n, 2, 3), device=device)
        self.force_tangential_world_buf = torch.zeros((n, 2, 3), device=device)
        self.force_fn_signed_buf = torch.zeros((n, 2), device=device)
        self.force_fn_ema_buf = torch.zeros((n, 2), device=device)
        self.force_ft_ema_buf = torch.zeros((n, 2), device=device)
        self.force_fn_over_mg_buf = torch.zeros((n, 2), device=device)
        self.force_rho_buf = torch.zeros((n, 2), device=device)
        self.force_normal_load_asymmetry_buf = torch.zeros(n, device=device)
        self.force_closure_residual_buf = torch.zeros(n, device=device)
        self.force_normal_reward_buf = torch.zeros(n, device=device)
        self.force_shear_penalty_buf = torch.zeros(n, device=device)
        self.force_overcompression_penalty_buf = torch.zeros(n, device=device)
        self.force_calibration_sum_buf = torch.zeros((n, 5), device=device)
        self.force_calibration_count_buf = torch.zeros(n, device=device)

    def _update_carry_phase(self):
        super()._update_carry_phase()
        self._update_debug_c1_event()
        self._update_force_quality()

    def _update_debug_c1_event(self):
        cfg = self.cfg.force_reward
        if not bool(cfg.debug_force_c1):
            return
        start = self.confirmed_carry_buf & (self._force_debug_steps_left == 0)
        # A single deterministic event per episode is marked by direction id >= 0.
        start &= self.box_perturb_direction_id_buf < 0
        self._force_debug_steps_left[start] = int(cfg.debug_event_steps)
        self.box_perturb_direction_id_buf[start] = int(cfg.debug_direction_id)
        active = self._force_debug_steps_left > 0
        self.box_perturb_active_buf[:] = active
        self.perturb_recovery_window_buf.zero_()
        self._force_debug_steps_left[active] -= 1

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        cfg = self.cfg.force_reward
        if not bool(cfg.debug_force_c1) or not torch.any(self.box_perturb_active_buf):
            return
        force = torch.zeros_like(self.disturbance)
        direction = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        ids = self.box_perturb_active_buf.nonzero(as_tuple=False).flatten()
        force[ids, self.box_net_contact_force_index, :] = float(cfg.debug_force_n) * direction
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(force), None, gymapi.CoordinateSpace.ENV_SPACE
        )

    def _lock_box_face_normals(self):
        new_lock = self.confirmed_carry_buf.unsqueeze(1) & ~self.force_face_locked_buf
        if not torch.any(new_lock):
            return
        hand_pos = torch.stack(
            (
                self.rigid_body_states[:, self.left_hand_net_contact_force_index, 0:3],
                self.rigid_body_states[:, self.right_hand_net_contact_force_index, 0:3],
            ), dim=1,
        )
        relative_world = hand_pos - self.box_states[:, None, 0:3]
        box_quat = self.box_states[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        relative_local = quat_rotate_inverse(box_quat, relative_world.reshape(-1, 3)).reshape(-1, 2, 3)
        normal_local, face_id = estimate_box_face_normal_local(
            relative_local, self._box_size[:, None, :], self.cfg.force_reward.eps
        )
        self.force_face_id_buf[new_lock] = face_id[new_lock]
        self.force_normal_local_buf[new_lock] = normal_local[new_lock]
        self.force_face_locked_buf[new_lock] = True

    @staticmethod
    def _bounded_huber(x, delta):
        x = torch.clamp(x, min=0.0)
        value = torch.where(x <= delta, 0.5 * x.square(), delta * (x - 0.5 * delta))
        return torch.clamp(value, 0.0, 1.0)

    def _update_force_quality(self):
        cfg = self.cfg.force_reward
        eps = float(cfg.eps)
        self._lock_box_face_normals()

        raw = torch.stack(
            (
                self.contact_forces[:, self.left_hand_net_contact_force_index, :],
                self.contact_forces[:, self.right_hand_net_contact_force_index, :],
            ), dim=1,
        )
        self.force_raw_world_buf[:] = raw
        q = self.box_states[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        n_world = quat_rotate(q, self.force_normal_local_buf.reshape(-1, 3)).reshape(-1, 2, 3)

        verify_mask = self.confirmed_carry_buf.unsqueeze(1) & self.force_face_locked_buf & ~self.force_sign_verified_buf
        outward_dot = torch.sum(raw * n_world, dim=-1)
        self.force_sign_sum_buf += torch.where(verify_mask, outward_dot, torch.zeros_like(outward_dot))
        self.force_sign_count_buf += verify_mask.float()
        ready = verify_mask & (self.force_sign_count_buf >= int(cfg.sign_verification_samples))
        self.force_normal_sign_buf[ready] = torch.where(self.force_sign_sum_buf[ready] < 0.0, -1.0, 1.0)
        self.force_sign_verified_buf[ready] = True

        n_world = n_world * self.force_normal_sign_buf.unsqueeze(-1)
        self.force_normal_world_buf[:] = n_world
        fn_signed, fn, fn_vector, tangential, ft = decompose_force(raw, n_world)
        self.force_fn_signed_buf[:] = fn_signed
        self.force_normal_component_world_buf[:] = fn_vector
        self.force_tangential_world_buf[:] = tangential

        positive = self.confirmed_carry_buf.unsqueeze(1) & (fn_signed > 0.0)
        verified_sample = self.confirmed_carry_buf.unsqueeze(1) & self.force_sign_verified_buf
        self.force_sign_positive_count_buf += positive.float()
        self.force_verified_sample_count_buf += verified_sample.float()
        verified_count = torch.clamp(self.force_verified_sample_count_buf, min=1.0)
        positive_fraction = self.force_sign_positive_count_buf / verified_count
        both_verified = torch.all(self.force_sign_verified_buf, dim=1)
        enough_verified = torch.all(
            self.force_verified_sample_count_buf >= int(cfg.sign_verification_samples), dim=1
        )
        valid_fraction = torch.all(positive_fraction >= float(cfg.min_positive_normal_fraction), dim=1)
        self.force_reward_valid_buf[:] = (
            both_verified & enough_verified & valid_fraction & ~self.force_reward_disabled_buf
        )
        unstable = both_verified & ~valid_fraction & (
            torch.min(self.force_verified_sample_count_buf, dim=1).values >= int(cfg.warning_check_samples)
        )
        new_warning = unstable & ~self.force_reward_warned_buf
        if torch.any(new_warning):
            warnings.warn(
                f"Disabling force rewards for {int(new_warning.sum())} envs: low positive-normal fraction.",
                RuntimeWarning,
            )
            self.force_reward_warned_buf[new_warning] = True
            self.force_reward_disabled_buf[new_warning] = True

        policy_dt = float(self.sim_params.dt) * int(self.cfg.control.decimation)
        alpha = float(np.exp(-policy_dt / float(cfg.force_filter_tau_s)))
        filter_gate = self.confirmed_carry_buf.unsqueeze(1)
        self.force_fn_ema_buf[:] = torch.where(
            filter_gate, alpha * self.force_fn_ema_buf + (1.0 - alpha) * fn, torch.zeros_like(fn)
        )
        self.force_ft_ema_buf[:] = torch.where(
            filter_gate, alpha * self.force_ft_ema_buf + (1.0 - alpha) * ft, torch.zeros_like(ft)
        )
        box_weight = self.box_masses.unsqueeze(1) * 9.81 + eps
        self.force_fn_over_mg_buf[:] = self.force_fn_ema_buf / box_weight
        self.force_rho_buf[:] = torch.clamp(
            self.force_ft_ema_buf / (self.force_fn_ema_buf + eps), 0.0, float(cfg.rho_max)
        )
        self.force_normal_load_asymmetry_buf[:] = (
            torch.abs(self.force_fn_ema_buf[:, 0] - self.force_fn_ema_buf[:, 1])
            / (self.force_fn_ema_buf.sum(dim=1) + eps)
        )

        box_force = self.contact_forces[:, self.box_net_contact_force_index, :]
        self.force_closure_residual_buf[:] = force_closure_residual(raw, box_force, eps)

        confirmed_gate = self.confirmed_carry_buf & self.force_reward_valid_buf
        event_gate = confirmed_gate & (self.box_perturb_active_buf | self.perturb_recovery_window_buf)
        fn_norm = self.force_fn_over_mg_buf
        margin = torch.clamp(
            (fn_norm - float(cfg.f_n_min)) / max(float(cfg.f_n_sat) - float(cfg.f_n_min), eps), 0.0, 1.0
        )
        risk = torch.relu(self.force_rho_buf - float(cfg.rho_safe))
        over = torch.relu(fn_norm - float(cfg.f_n_max))
        self.force_normal_reward_buf[:] = event_gate.float() * 0.5 * margin.sum(dim=1)
        self.force_shear_penalty_buf[:] = event_gate.float() * 0.5 * self._bounded_huber(risk, float(cfg.huber_delta)).sum(dim=1)
        self.force_overcompression_penalty_buf[:] = confirmed_gate.float() * 0.5 * self._bounded_huber(over, float(cfg.huber_delta)).sum(dim=1)
        calibration_values = torch.cat(
            (self.force_fn_over_mg_buf, self.force_rho_buf, self.force_normal_load_asymmetry_buf.unsqueeze(1)), dim=1
        )
        self.force_calibration_sum_buf += torch.where(
            self.confirmed_carry_buf.unsqueeze(1), calibration_values, torch.zeros_like(calibration_values)
        )
        self.force_calibration_count_buf += self.confirmed_carry_buf.float()
        self.extras["force"] = self._build_force_log_info(positive_fraction)

        if bool(cfg.debug_force_c1) and self.common_step_counter % int(cfg.debug_log_interval) == 0:
            i = 0
            print(
                "[Force debug] "
                f"face={self.force_face_id_buf[i].tolist()} raw={raw[i].tolist()} "
                f"normal={self.force_normal_component_world_buf[i].tolist()} "
                f"tangent={tangential[i].tolist()} rho={self.force_rho_buf[i].tolist()} "
                f"confirmed={bool(confirmed_gate[i])} event={bool(event_gate[i])}"
            )

    def _masked_stats(self, values, mask):
        selected = values[mask]
        if selected.numel() == 0:
            zero = torch.zeros((), device=self.device)
            return zero, zero, zero
        return selected.mean(), torch.quantile(selected.float(), 0.95), selected.max()

    def _build_force_log_info(self, positive_fraction):
        info = {}
        confirmed = self.confirmed_carry_buf
        metrics = {
            "fn_left_over_mg": self.force_fn_over_mg_buf[:, 0],
            "fn_right_over_mg": self.force_fn_over_mg_buf[:, 1],
            "rho_left": self.force_rho_buf[:, 0],
            "rho_right": self.force_rho_buf[:, 1],
            "normal_load_asymmetry": self.force_normal_load_asymmetry_buf,
            "closure_residual": self.force_closure_residual_buf,
            "normal_reward": self.force_normal_reward_buf,
            "shear_risk_penalty": self.force_shear_penalty_buf,
            "overcompression_penalty": self.force_overcompression_penalty_buf,
        }
        for context, mask_fn in self._FORCE_CONTEXTS.items():
            mask = mask_fn(self)
            for name, values in metrics.items():
                mean, p95, maximum = self._masked_stats(values, mask)
                info[f"{context}/{name}_mean"] = mean
                info[f"{context}/{name}_p95"] = p95
                if name == "closure_residual":
                    info[f"{context}/{name}_max"] = maximum
            info[f"{context}/positive_normal_fraction_left"] = self._masked_stats(
                (self.force_fn_signed_buf[:, 0] > 0.0).float(), mask
            )[0]
            info[f"{context}/positive_normal_fraction_right"] = self._masked_stats(
                (self.force_fn_signed_buf[:, 1] > 0.0).float(), mask
            )[0]
            info[f"{context}/force_reward_valid_fraction"] = self._masked_stats(
                self.force_reward_valid_buf.float(), mask
            )[0]
            for hand, index in (("left", 0), ("right", 1)):
                for face in range(6):
                    face_fraction = ((self.force_face_id_buf[:, index] == face) & mask).float().sum()
                    info[f"{context}/normal_face_id_{hand}_distribution/face_{face}"] = (
                        face_fraction / torch.clamp(mask.float().sum(), min=1.0)
                    )
        # Unprefixed required tags refer to all confirmed-carry samples.
        for name, values in metrics.items():
            mean, p95, maximum = self._masked_stats(values, confirmed)
            info[f"{name}_mean"] = mean
            info[f"{name}_p95"] = p95
            if name == "closure_residual":
                info[f"{name}_max"] = maximum
        verified_envs = torch.all(self.force_sign_verified_buf, dim=1)
        info["positive_normal_fraction_left"] = self._masked_stats(positive_fraction[:, 0], verified_envs)[0]
        info["positive_normal_fraction_right"] = self._masked_stats(positive_fraction[:, 1], verified_envs)[0]
        for hand, index in (("left", 0), ("right", 1)):
            for face in range(6):
                info[f"normal_face_id_{hand}_distribution/face_{face}"] = (
                    (self.force_face_id_buf[:, index] == face) & confirmed
                ).float().sum() / torch.clamp(confirmed.float().sum(), min=1.0)
        info["force_reward_valid_fraction"] = self.force_reward_valid_buf.float().mean()
        info["normal_reward_mean"] = self.force_normal_reward_buf.mean()
        info["shear_risk_penalty_mean"] = self.force_shear_penalty_buf.mean()
        info["overcompression_penalty_mean"] = self.force_overcompression_penalty_buf.mean()
        info["signed_normal_mean_left"] = self._masked_stats(self.force_fn_signed_buf[:, 0], confirmed)[0]
        info["signed_normal_mean_right"] = self._masked_stats(self.force_fn_signed_buf[:, 1], confirmed)[0]
        for direction in range(6):
            mask = confirmed & (self.box_perturb_direction_id_buf == direction)
            info[f"direction_{direction}/sample_fraction"] = mask.float().mean()
            for name in ("fn_left_over_mg", "fn_right_over_mg", "rho_left", "rho_right", "normal_load_asymmetry"):
                info[f"direction_{direction}/{name}_mean"] = self._masked_stats(metrics[name], mask)[0]
        calibrated = confirmed & self.force_reward_valid_buf
        fn_samples = self.force_fn_over_mg_buf[calibrated].reshape(-1)
        rho_samples = self.force_rho_buf[calibrated].reshape(-1)
        if fn_samples.numel() > 0:
            info["calibration_recommendation/f_n_min"] = 0.8 * torch.quantile(fn_samples, 0.20)
            info["calibration_recommendation/f_n_sat"] = torch.quantile(fn_samples, 0.70)
            info["calibration_recommendation/f_n_max"] = 1.25 * torch.quantile(fn_samples, 0.95)
            info["calibration_recommendation/rho_safe"] = 1.10 * torch.quantile(rho_samples, 0.90)
        return info

    def _force_rewards_enabled(self):
        cfg = self.cfg.force_reward
        return bool(cfg.enabled) and not bool(cfg.calibration_only)

    def _reward_normal_force_margin(self):
        return self.force_normal_reward_buf if self._force_rewards_enabled() else torch.zeros_like(self.force_normal_reward_buf)

    def _reward_tangential_compression_risk(self):
        return self.force_shear_penalty_buf if self._force_rewards_enabled() else torch.zeros_like(self.force_shear_penalty_buf)

    def _reward_normal_force_overcompression(self):
        return self.force_overcompression_penalty_buf if self._force_rewards_enabled() else torch.zeros_like(self.force_overcompression_penalty_buf)

    def reset_idx(self, env_ids):
        if len(env_ids) > 0 and hasattr(self, "force_calibration_sum_buf"):
            successful = self.success_buf[env_ids]
            successful_ids = env_ids[successful]
            successful_sum = self.force_calibration_sum_buf[successful_ids].sum(dim=0)
            successful_count = self.force_calibration_count_buf[successful_ids].sum()
        else:
            successful_sum = None
            successful_count = None
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "force_face_locked_buf"):
            return
        if successful_sum is not None:
            names = (
                "fn_left_over_mg", "fn_right_over_mg", "rho_left", "rho_right", "normal_load_asymmetry"
            )
            for index, name in enumerate(names):
                self.extras["episode"][f"Force/successful_confirmed/{name}_mean"] = (
                    successful_sum[index] / torch.clamp(successful_count, min=1.0)
                )
            self.extras["episode"]["Force/successful_confirmed/sample_count"] = successful_count
        for name in (
            "box_perturb_active_buf", "perturb_recovery_window_buf", "force_reward_valid_buf",
            "force_reward_warned_buf", "force_reward_disabled_buf",
        ):
            getattr(self, name)[env_ids] = False
        self.box_perturb_direction_id_buf[env_ids] = -1
        self._force_debug_steps_left[env_ids] = 0
        self.force_face_locked_buf[env_ids] = False
        self.force_face_id_buf[env_ids] = -1
        self.force_sign_verified_buf[env_ids] = False
        self.force_normal_sign_buf[env_ids] = 1.0
        for name in (
            "force_normal_local_buf", "force_sign_sum_buf", "force_sign_count_buf",
            "force_sign_positive_count_buf", "force_verified_sample_count_buf", "force_normal_world_buf", "force_raw_world_buf",
            "force_normal_component_world_buf", "force_tangential_world_buf", "force_fn_signed_buf",
            "force_fn_ema_buf", "force_ft_ema_buf", "force_fn_over_mg_buf", "force_rho_buf",
            "force_normal_load_asymmetry_buf", "force_closure_residual_buf", "force_normal_reward_buf",
            "force_shear_penalty_buf", "force_overcompression_penalty_buf",
            "force_calibration_sum_buf", "force_calibration_count_buf",
        ):
            getattr(self, name)[env_ids] = 0.0

    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        if not bool(self.cfg.force_reward.debug_force_c1):
            return
        scale = float(self.cfg.force_reward.debug_draw_scale)
        colors = np.asarray(((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0.5, 0)), dtype=np.float32)
        for i, env_ptr in enumerate(self.envs):
            starts = torch.stack((
                self.rigid_body_states[i, self.left_hand_net_contact_force_index, 0:3],
                self.rigid_body_states[i, self.right_hand_net_contact_force_index, 0:3],
            ))
            vectors = (
                self.force_normal_world_buf[i] / max(scale, 1.0e-6),
                self.force_raw_world_buf[i], self.force_normal_component_world_buf[i],
                self.force_tangential_world_buf[i],
            )
            for vector, color in zip(vectors, colors):
                ends = starts + vector * scale
                vertices = torch.cat((starts, ends), dim=1).detach().cpu().numpy()
                self.gym.add_lines(self.viewer, env_ptr, 2, vertices, np.repeat(color[None], 2, axis=0))
