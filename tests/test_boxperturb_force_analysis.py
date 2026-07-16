import importlib.util
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


force = _load(
    "hand_box_force",
    "legged_gym/legged_gym/envs/g1/hand_box_force.py",
)
reporting = _load(
    "boxperturb_reporting",
    "legged_gym/legged_gym/scripts/boxperturb_reporting.py",
)


def test_force_decomposition_pure_normal_tangent_and_mixed():
    normal = torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1)
    raw = torch.tensor([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [4.0, 3.0, 0.0]])
    signed, fn, _, tangent, ft = force.decompose_force(raw, normal)
    assert torch.allclose(signed, torch.tensor([4.0, 0.0, 4.0]))
    assert torch.allclose(fn, torch.tensor([4.0, 0.0, 4.0]))
    assert torch.allclose(ft, torch.tensor([0.0, 3.0, 3.0]))
    assert torch.allclose(tangent[2], torch.tensor([0.0, 3.0, 0.0]))


def test_force_decomposition_rotated_normal_and_compression_clamp():
    inv_sqrt_2 = 2.0 ** -0.5
    normal = torch.tensor([[inv_sqrt_2, inv_sqrt_2, 0.0], [1.0, 0.0, 0.0]])
    raw = torch.tensor([[2.0 * inv_sqrt_2, 2.0 * inv_sqrt_2, 5.0], [-3.0, 4.0, 0.0]])
    signed, fn, _, _, ft = force.decompose_force(raw, normal)
    assert torch.allclose(signed, torch.tensor([2.0, -3.0]), atol=1e-6)
    assert torch.allclose(fn, torch.tensor([2.0, 0.0]), atol=1e-6)
    assert torch.allclose(ft, torch.tensor([5.0, 5.0]), atol=1e-6)


def test_face_estimation_and_closure():
    relative = torch.tensor([[0.51, 0.1, 0.1], [0.1, -1.01, 0.1]])
    size = torch.tensor([[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]])
    normal, face_id = force.estimate_box_face_normal_local(relative, size)
    assert torch.equal(normal, torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    assert torch.equal(face_id, torch.tensor([1, 2]))
    hands = torch.tensor([[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    box = torch.tensor([[-5.0, 0.0, 0.0]])
    assert torch.allclose(force.force_closure_residual(hands, box), torch.zeros(1), atol=1e-6)


def _trace_row(phase, value, valid=1, impulse=0.0):
    row = {
        "phase": phase, "force_decomposition_valid": valid,
        "force_closure_residual": 0.1, "left_rho_raw": 0.2,
        "right_rho_raw": 0.3, "normal_load_asymmetry": 0.1,
        "force_impulse_Ns": impulse, "force_uncapped_peak_N": 12.0,
        "force_peak_N": 10.0, "force_cap_used": 1,
        "perturb_direction_world_x": 1.0, "perturb_direction_world_y": 0.0,
        "perturb_direction_world_z": 0.0,
    }
    for key in reporting.FORCE_METRICS:
        row[key] = value
    return row


def test_trace_summary_baseline_delta_validity_and_outputs(tmp_path):
    trace = [_trace_row("pre", 2.0) for _ in range(40)]
    trace += [_trace_row("pulse", 5.0, valid=index % 2, impulse=0.01 * index) for index in range(4)]
    trace += [_trace_row("recovery", 3.0) for _ in range(2)]
    summary = reporting.summarize_force_trace(trace)
    assert summary["left_fn_raw_N_pre_mean"] == 2.0
    assert summary["left_fn_raw_N_pulse_mean"] == 5.0
    assert summary["left_fn_raw_N_pulse_delta_from_pre"] == 3.0
    assert summary["pulse_force_valid_fraction"] == 0.5
    assert summary["force_baseline_unavailable"] == 0
    trial = {
        "model": "m", "direction": "+box_x", "requested_beta": 0.25,
        "precondition_success": 1, "recovery_success": 1,
        "pulse_hold_retention": 1.0, "post_confirmed_ratio": 1.0,
        **summary,
    }
    reporting.write_run_files(str(tmp_path), [trial], trace, {"physics_dt_s": 0.005})
    for name in ("force_trace.csv", "trials.csv", "summary.csv", "run_metadata.json"):
        assert (tmp_path / name).is_file()
    assert json.loads((tmp_path / "run_metadata.json").read_text())["physics_dt_s"] == 0.005


def test_half_sine_midpoint_peak_cap_and_impulse():
    dt, duration, mass, beta, cap = 0.005, 0.1, 4.0, 0.75, 10.0
    steps = round(duration / dt)
    uncapped = beta * mass * 9.81
    peak = min(uncapped, cap)
    samples = [peak * math.sin(math.pi * (k + 0.5) / steps) for k in range(steps)]
    expected_impulse = 2.0 * peak * duration / math.pi
    assert uncapped > cap and peak == cap
    assert abs(sum(samples) * dt - expected_impulse) < 0.002


def test_offset_force_torque_formula_and_boundary_summary():
    r = torch.tensor([0.0, 0.15, 0.05])
    f = torch.tensor([10.0, 0.0, 0.0])
    assert torch.allclose(torch.cross(r, f, dim=0), torch.tensor([0.0, 0.5, -1.5]))
    trials = [
        {
            "model": "m", "direction": "+box_x", "requested_beta": 0.5,
            "force_point_mode": "box_surface_grid", "force_point_label": "face_upper",
            "pulse_duration_s": 0.1, "pulse_profile": "half_sine",
            "precondition_success": 1, "termination": 0, "post_confirmed_ratio": 1.0,
            "recovery_success": 1, "pulse_force_valid_fraction": 1.0,
            "hand_box_rel_speed_peak_mps": 0.2,
        },
        {
            "model": "m", "direction": "+box_x", "requested_beta": 1.0,
            "force_point_mode": "box_surface_grid", "force_point_label": "face_upper",
            "pulse_duration_s": 0.1, "pulse_profile": "half_sine",
            "precondition_success": 1, "termination": 1, "post_confirmed_ratio": 0.0,
            "recovery_success": 0, "pulse_force_valid_fraction": 0.0,
            "hand_box_rel_speed_peak_mps": 2.0,
        },
    ]
    rows = reporting.build_boundary_rows(trials)
    assert len(rows) == 1
    assert rows[0]["max_pass_beta"] == 0.5
    assert rows[0]["min_primary_failure_beta"] == 1.0
    assert rows[0]["min_degradation_beta"] == 1.0


def test_box_following_direction_math():
    yaw = math.pi / 2.0
    rot_z = torch.tensor(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    box_x_local = torch.tensor([1.0, 0.0, 0.0])
    world_z = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(rot_z @ box_x_local, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(world_z, torch.tensor([0.0, 0.0, 1.0]))


def test_six_direction_registry_and_observation_dimensions_are_unchanged():
    perturb_source = (ROOT / "legged_gym/legged_gym/envs/g1/carrybox_boxperturb.py").read_text()
    config_source = (ROOT / "legged_gym/legged_gym/envs/g1/carrybox_boxperturb_resume_config.py").read_text()
    expected_order = (
        '"+box_x"', '"-box_x"', '"+box_y"', '"-box_y"',
        '"+z_world"', '"-z_world"',
    )
    debug_block = config_source.split("debug_sweep_directions = (", 1)[1].split(")", 1)[0]
    assert all(token in perturb_source for token in expected_order)
    assert [debug_block.index(token) for token in expected_order] == sorted(
        debug_block.index(token) for token in expected_order
    )
    assert "num_actor_obs = 738" in config_source
    assert "num_privileged_obs = 143" in config_source
    assert "num_interaction_priv_obs = 17" in config_source
    assert "apply_rigid_body_force_at_pos_tensors" in perturb_source
    assert "external_torque_world" in perturb_source
    assert "set_evaluation_long_range_goal" in perturb_source
    assert "box_perturb_direction_local" in perturb_source
    assert "box_perturb_direction_is_world" in perturb_source
    assert "direction_world = self._current_perturb_direction_world(active)" in perturb_source
    assert "force = peak_force_world * profile.unsqueeze(1)" in perturb_source


def test_debug_force_visualization_matches_falcon_bundle_style():
    perturb_source = (ROOT / "legged_gym/legged_gym/envs/g1/carrybox_boxperturb.py").read_text()
    config_source = (ROOT / "legged_gym/legged_gym/envs/g1/carrybox_boxperturb_resume_config.py").read_text()
    draw_block = perturb_source.split("    def _draw_debug_vis(self):", 1)[1]

    assert "debug_force_draw_scale_m_per_N = 0.12" in config_source
    assert "debug_force_bundle_line_count = 20" in config_source
    assert "debug_force_bundle_jitter_m = 0.01" in config_source
    assert "[0.851, 0.144, 0.07]" in draw_block
    assert "start = draw_point[env_id]" in draw_block
    assert "end = start + force * scale" in draw_block
    assert "np.random.random((line_count, 3))" in draw_block
    assert "debug_force_arrow_head_length_m" not in config_source
    assert "debug_force_arrow_shaft_width_m" not in config_source
    assert "debug_force_point_marker_size_m" not in config_source
    assert "head_base" not in draw_block
    assert "marker_segments" not in draw_block
    assert "torch.maximum(" in perturb_source
    assert "self.box_perturb_debug_draw_force_N[env_ids] = 0.0" in perturb_source


def test_boundary_evaluator_viewer_enables_force_visualization():
    source = (ROOT / "legged_gym/legged_gym/scripts/evaluate_carrybox_boxperturb.py").read_text()

    assert "headless=not parsed.viewer" in source
    assert "perturb.debug_draw_force = bool(parsed.viewer)" in source
    assert 'parser.add_argument("--viewer", action="store_true", default=False)' in source


def test_boundary_evaluator_none_force_cap_clears_config_default():
    source = (ROOT / "legged_gym/legged_gym/scripts/evaluate_carrybox_boxperturb.py").read_text()

    assert 'str(parsed.force_peak_cap_N).lower() == "none"' in source
    assert "env.cfg.box_perturbation.force_peak_cap_N = None" in source


def test_evaluator_reset_uses_single_history_commit_and_pre_step_goal_hook():
    evaluator_source = (
        ROOT / "legged_gym/legged_gym/scripts/evaluate_carrybox_boxperturb.py"
    ).read_text()
    perturb_source = (
        ROOT / "legged_gym/legged_gym/envs/g1/carrybox_boxperturb.py"
    ).read_text()

    run_trial = evaluator_source.split("def _run_trial(", 1)[1].split(
        "\ndef _write_csv", 1
    )[0]
    goal_sampler = perturb_source.split(
        "    def _set_evaluation_long_range_goals(", 1
    )[1].split("    def set_evaluation_long_range_goal(", 1)[0]
    reset_hook = perturb_source.split("    def _reset_task(self, env_ids):", 1)[1].split(
        "    def _set_evaluation_long_range_goals(", 1
    )[0]

    assert "obs, _ = env.reset()" in run_trial
    assert "env.get_observations()" not in run_trial
    assert "env.compute_observations()" not in run_trial
    assert "initial_physics_fingerprint = _tensor_fingerprint(" in run_trial
    assert "reset_actor_obs_fingerprint = _tensor_fingerprint(obs[0])" in run_trial
    assert "super()._reset_task(env_ids)" in reset_hook
    assert "self._set_evaluation_long_range_goals(" in reset_hook
    assert "self.compute_observations()" not in goal_sampler
    assert '"goal_before_zero_action_single_commit"' in perturb_source


def test_play_visual_sweep_uses_reset_observation_without_goal_resampling():
    source = (ROOT / "legged_gym/legged_gym/scripts/play.py").read_text()
    sweep = source.split("def run_boxperturb_visual_sweep(", 1)[1].split(
        "\ndef play(", 1
    )[0]
    goal_reader = source.split(
        "def _initial_goal_distance_after_reset(", 1
    )[1].split("\ndef load_actor_only_for_inference", 1)[0]

    assert "obs, _ = env.reset()" in sweep
    assert "_initial_goal_distance_after_reset(env, args)" in sweep
    assert "env.get_observations()" not in sweep
    assert "set_evaluation_long_range_goal" not in sweep
    assert "env.compute_observations()" not in goal_reader
    assert "env.evaluation_initial_goal_distance_xy[0]" in goal_reader


def test_perturb_termination_does_not_require_disabled_long_range_buffers():
    source = (ROOT / "legged_gym/legged_gym/envs/g1/carrybox_boxperturb.py").read_text()
    check_block = source.split("    def check_termination(self):", 1)[1].split(
        "    def reset_idx(self, env_ids):", 1
    )[0]
    assert "super().check_termination()" in check_block
    assert "self.carry_success_buf" not in check_block
    assert 'hasattr(self, "carry_drop_failure_buf")' in source
