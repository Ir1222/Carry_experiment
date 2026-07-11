import os
from datetime import datetime
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, export_policy_as_jit, export_jit_to_onnx, load_onnx_policy, task_registry, set_seed

import numpy as np
import torch

from legged_gym.scripts.boxperturb_reporting import (
    aggregate_trials,
    summarize_force_trace,
    write_run_files,
)


def _parse_csv_floats(value, default):
    if value is None:
        return tuple(default)
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_csv_strings(value, default):
    if value is None:
        return tuple(default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_float_pair(value, default):
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected exactly two comma-separated values, got: {value}")
    return tuple(values)


def _parse_force_cap(value, default):
    if value is None:
        return default
    if str(value).lower() == "none":
        return None
    return float(value)


def _build_force_point_specs(modes, labels):
    specs = []
    for mode in modes:
        if mode == "com":
            specs.append(("com", "com"))
        elif mode == "box_surface_grid":
            specs.extend((mode, label) for label in labels)
        elif mode == "box_surface_random":
            specs.append((mode, "random"))
        else:
            raise ValueError(f"Unknown force point mode: {mode}")
    return specs


def _apply_eval_goal_if_requested(env, args):
    if getattr(args, "eval_goal_mode", "default") == "default":
        env.compute_observations()
        return 0.0
    if args.eval_goal_mode != "long_range":
        raise ValueError(f"Unknown eval goal mode: {args.eval_goal_mode}")
    return env.set_evaluation_long_range_goal(
        distance_range=_parse_float_pair(args.eval_goal_distance_range, (4.0, 8.0)),
        bearing_offset_deg=_parse_float_pair(args.eval_goal_bearing_offset_deg, (15.0, 75.0)),
        env_id=0,
    )


def load_actor_only_for_inference(ppo_runner, checkpoint_path, device):
    """Load a compatible actor while ignoring a training-only critic mismatch."""
    if checkpoint_path is None:
        raise ValueError("CarryBox play requires --resume_path to load the inference actor.")

    checkpoint_path = os.path.expanduser(
        checkpoint_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"CarryBox checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = checkpoint["model_state_dict"]
    actor_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key == "std" or key.startswith("actor.")
    }
    current_state = ppo_runner.alg.actor_critic.state_dict()
    shape_mismatches = {}
    for key, value in actor_state.items():
        if key not in current_state:
            shape_mismatches[key] = (tuple(value.shape), None)
        elif value.shape != current_state[key].shape:
            shape_mismatches[key] = (
                tuple(value.shape), tuple(current_state[key].shape)
            )
    if shape_mismatches:
        raise RuntimeError(
            "CarryBox actor is incompatible with the current policy: "
            f"{shape_mismatches}"
        )

    incompatible = ppo_runner.alg.actor_critic.load_state_dict(
        actor_state, strict=False
    )
    missing_non_critic = [
        key for key in incompatible.missing_keys if not key.startswith("critic.")
    ]
    if missing_non_critic or incompatible.unexpected_keys:
        raise RuntimeError(
            "Actor-only checkpoint load was incomplete: "
            f"missing={missing_non_critic}, unexpected={incompatible.unexpected_keys}"
        )

    checkpoint_critic_dim = checkpoint_state["critic.0.weight"].shape[1]
    current_critic_dim = current_state["critic.0.weight"].shape[1]
    actor_input_dim = checkpoint_state["actor.0.weight"].shape[1]
    print(
        f"Loaded CarryBox inference actor from: {checkpoint_path} "
        f"(actor_input={actor_input_dim}, "
        f"checkpoint_critic={checkpoint_critic_dim}, "
        f"current_critic={current_critic_dim}; critic intentionally skipped)"
    )


def print_carry_phase_debug(env, dones, step, env_id=0):
    """Print the CarryBox carry-phase decision and every signal used by it."""
    # Carry phase detection: this block is used to detect and print the current carry phase.
    if bool(dones[env_id].item()):
        return

    cfg = env.cfg.carry_phase
    support_height = torch.maximum(
        torch.tensor(cfg.support_height, dtype=torch.float, device=env.device),
        env.platform_pos[env_id, 2] + 0.5 * env._platform_height,
    )
    box_bottom_height = (
        env.box_states[env_id, 2] - 0.5 * env._box_size[env_id, 2]
    )
    box_rel_lin_speed = torch.linalg.vector_norm(
        env.box_states[env_id, 7:10] - env.root_states[env_id, 7:10]
    )
    box_ang_speed = torch.linalg.vector_norm(env.box_states[env_id, 10:13])
    left_hand_force = torch.linalg.vector_norm(
        env.contact_forces[env_id, env.left_hand_net_contact_force_index, :]
    )
    right_hand_force = torch.linalg.vector_norm(
        env.contact_forces[env_id, env.right_hand_net_contact_force_index, :]
    )

    height_mask = env.box_clearance_buf[env_id] > cfg.clearance_on
    static_mask = (
        (box_rel_lin_speed < cfg.max_box_rel_lin_vel)
        & (box_ang_speed < cfg.max_box_ang_vel)
        if cfg.use_static_check
        else torch.tensor(True, device=env.device)
    )
    left_contact = left_hand_force > cfg.contact_force_threshold
    right_contact = right_hand_force > cfg.contact_force_threshold
    both_contact = left_contact & right_contact

    print(
        f"[CarryPhaseDetector] step={step} env={env_id} "
        f"carry={int(env.carry_phase_buf[env_id].item())} "
        f"confirmed={int(env.confirmed_carry_buf[env_id].item())} "
        f"height_mask={int(height_mask.item())} "
        f"static_mask={int(static_mask.item())} "
        f"both_contact={int(both_contact.item())} "
        f"clearance={env.box_clearance_buf[env_id].item():.3f}/>{cfg.clearance_on:.3f}m "
        f"box_bottom={box_bottom_height.item():.3f}m "
        f"support={support_height.item():.3f}m "
        f"rel_lin_speed={box_rel_lin_speed.item():.3f}/<{cfg.max_box_rel_lin_vel:.3f}m/s "
        f"ang_speed={box_ang_speed.item():.3f}/<{cfg.max_box_ang_vel:.3f}rad/s "
        f"left_force={left_hand_force.item():.2f}N "
        f"right_force={right_hand_force.item():.2f}N "
        f"contact_threshold={cfg.contact_force_threshold:.2f}N "
        f"batch_carry_ratio={env.carry_phase_buf.float().mean().item():.2f} "
        f"batch_confirmed_ratio={env.confirmed_carry_buf.float().mean().item():.2f}"
    )


def _disable_boxperturb_eval_randomization(env_cfg):
    env_cfg.env.test = False
    env_cfg.noise.add_noise = False
    env_cfg.asset.box.reset_mode = "random"
    env_cfg.asset.box.skill_init_prob = [0.0, 0.0, 1.0, 0.0]
    for name in (
        "randomize_actuation_offset",
        "randomize_motor_strength",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_link_mass",
        "randomize_friction",
        "randomize_restitution",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "disturbance",
        "delay",
        "push_robots",
    ):
        setattr(env_cfg.domain_rand, name, False)


def run_boxperturb_visual_sweep(env, policy, args):
    cfg = env.cfg.box_perturbation
    betas = _parse_csv_floats(args.force_sweep_betas, cfg.debug_sweep_beta_values)
    directions = _parse_csv_strings(args.force_sweep_directions, cfg.debug_sweep_directions)
    point_specs = _build_force_point_specs(
        _parse_csv_strings(args.force_sweep_point_modes, ("com",)),
        _parse_csv_strings(args.force_sweep_point_labels, ("face_center",)),
    )
    pulse_durations = _parse_csv_floats(args.force_sweep_pulse_durations, (cfg.pulse_duration_s,))
    pulse_profiles = _parse_csv_strings(args.force_sweep_pulse_profiles, (cfg.pulse_profile,))
    force_cap = _parse_force_cap(args.force_peak_cap_N, cfg.force_peak_cap_N)
    eval_goal_distance_range = _parse_float_pair(args.eval_goal_distance_range, (4.0, 8.0))
    eval_goal_bearing_offset_deg = _parse_float_pair(args.eval_goal_bearing_offset_deg, (15.0, 75.0))
    unknown = [name for name in directions if name not in env._DIRECTION_IDS]
    if unknown:
        raise ValueError(f"Unknown force sweep directions: {unknown}")
    output_dir = args.boxperturb_output_dir or os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "boxperturb_play", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    seed = 1 if args.seed is None else int(args.seed)
    precondition_steps = int(
        round(float(cfg.evaluation_precondition_timeout_s) / env.dt)
    )
    recovery_steps = env._recovery_policy_steps()
    post_steps = int(round(float(cfg.evaluation_post_window_s) / env.dt))
    total = len(betas) * len(directions) * len(point_specs) * len(pulse_durations) * len(pulse_profiles)
    trials = []
    traces = []

    test_cells = [
        (beta, direction, point_mode, point_label, duration, profile)
        for beta in betas
        for direction in directions
        for point_mode, point_label in point_specs
        for duration in pulse_durations
        for profile in pulse_profiles
    ]
    for test_index, (beta, direction, point_mode, point_label, duration, profile) in enumerate(test_cells, start=1):
        set_seed(seed)
        obs, _ = env.reset()
        initial_goal_distance = _apply_eval_goal_if_requested(env, args)
        obs = env.get_observations()
        force_point_box, resolved_point_label = env.resolve_force_point_box(
            direction, point_mode, point_label, env_id=0
        )
        env.begin_box_perturb_trace(
            {
                "model": "play",
                "seed": seed,
                "direction": direction,
                "requested_beta": float(beta),
                "force_point_mode": point_mode,
                "force_point_label": resolved_point_label,
                "pulse_duration_s": float(duration),
                "pulse_profile": profile,
                "initial_goal_distance_xy_m": initial_goal_distance,
                "eval_goal_mode": args.eval_goal_mode,
                "test_index": test_index,
            },
            verbose=args.verbose_force_trace,
        )
        precondition_ok = False
        dones = torch.zeros(1, device=env.device)
        for _ in range(precondition_steps):
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _, _ = env.step(actions.detach())
            if bool(dones[0].item()):
                break
            if int(env.confirmed_carry_streak[0].item()) >= int(
                cfg.stable_confirmed_carry_policy_steps
            ):
                precondition_ok = True
                break

        if not precondition_ok:
            trace = env.end_box_perturb_trace()
            trial = {
                "model": "play", "seed": seed, "direction": direction,
                "requested_beta": float(beta), "precondition_success": 0,
                "force_point_mode": point_mode,
                "force_point_label": resolved_point_label,
                "pulse_duration_s": float(duration),
                "pulse_profile": profile,
                "initial_goal_distance_xy_m": initial_goal_distance,
                "goal_distance_xy_m": float(env.object2goal_dist_xy[0].item()),
                "event_triggered": 0, "peak_force_N": float("nan"),
                "recovery_success": float("nan"), "termination_reason": env.box_perturb_last_termination_reason[0],
                "trace_rows": len(trace), **summarize_force_trace(trace),
            }
            trials.append(trial)
            traces.extend(trace)
            continue

        peak = env.schedule_explicit_box_perturbation(
            direction,
            beta,
            env_id=0,
            force_point_box=force_point_box,
            pulse_duration_s=float(duration),
            pulse_profile=profile,
            force_peak_cap_N=force_cap,
        )
        pulse_finished = False
        after_pulse_steps = 0
        recovery_streak = 0
        recovery_success = False
        recovery_time_steps = -1
        pulse_confirmed = []
        post_confirmed = []
        goal_start = float(env.object2goal_dist_xy[0].item())
        termination_reason = ""

        while after_pulse_steps < recovery_steps + post_steps:
            if pulse_finished and after_pulse_steps >= recovery_steps:
                env.set_box_perturb_trace_phase("post")
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _, _ = env.step(actions.detach())
            if not pulse_finished:
                pulse_confirmed.append(float(env.confirmed_carry_buf[0].item()))
                if int(env.box_perturb_remaining_physics_steps[0].item()) == 0:
                    pulse_finished = True
                    env.set_box_perturb_trace_phase("recovery")
            else:
                after_pulse_steps += 1
                if after_pulse_steps <= recovery_steps:
                    if bool(env.confirmed_carry_buf[0].item()):
                        recovery_streak += 1
                    else:
                        recovery_streak = 0
                    if not recovery_success and recovery_streak >= int(
                        cfg.recovery_confirmed_carry_steps
                    ):
                        recovery_success = True
                        recovery_time_steps = after_pulse_steps
                else:
                    post_confirmed.append(float(env.confirmed_carry_buf[0].item()))
            if bool(dones[0].item()):
                termination_reason = env.box_perturb_last_termination_reason[0]
                break

        goal_end = (
            float(env.object2goal_dist_xy[0].item())
            if not termination_reason
            else float("nan")
        )
        trace = env.end_box_perturb_trace()
        pulse_hold = float(np.mean(pulse_confirmed)) if pulse_confirmed else 0.0
        post_ratio = float(np.mean(post_confirmed)) if post_confirmed else 0.0
        recovery_time_s = (
            recovery_time_steps * env.dt if recovery_time_steps >= 0 else float("nan")
        )
        trial = {
            "model": "play", "seed": seed, "direction": direction,
            "requested_beta": float(beta), "precondition_success": 1,
            "force_point_mode": point_mode,
            "force_point_label": resolved_point_label,
            "pulse_duration_s": float(duration),
            "pulse_profile": profile,
            "initial_goal_distance_xy_m": initial_goal_distance,
            "goal_distance_xy_m": goal_end,
            "event_triggered": 1, "peak_force_N": peak,
            "pulse_hold_retention": pulse_hold,
            "recovery_success": int(recovery_success),
            "recovery_time_s": recovery_time_s,
            "post_confirmed_ratio": post_ratio,
            "goal_progress_m": goal_start - goal_end,
            "termination": int(bool(termination_reason)),
            "termination_reason": termination_reason,
            "trace_rows": len(trace), **summarize_force_trace(trace),
        }
        trials.append(trial)
        traces.extend(trace)

    metadata = {
        "mode": "play", "checkpoint": args.resume_path, "seed": seed,
        "coordinate_convention": {
            "box_xy": "box-local axes recomputed from the current box orientation every physics substep",
            "world_z": "gravity-aligned world axis",
            "force": "world-frame N", "impulse": "N s",
        },
        "physics_dt_s": float(env.sim_params.dt), "policy_dt_s": float(env.dt),
        "pulse_profiles": list(pulse_profiles),
        "pulse_durations_s": list(pulse_durations),
        "force_peak_cap_N": force_cap,
        "force_point_specs": [{"mode": mode, "label": label} for mode, label in point_specs],
        "eval_goal_mode": args.eval_goal_mode,
        "eval_goal_distance_range": list(eval_goal_distance_range),
        "eval_goal_bearing_offset_deg": list(eval_goal_bearing_offset_deg),
        "directions": list(directions), "betas": list(betas),
        "force_decomposition": "hand rigid-body net contact force projected on estimated locked box-face normal",
        "ema_tau_s": 0.04, "baseline_valid_physics_substeps": 40,
        "force_sign_verification_samples": int(cfg.force_sign_verification_samples),
        "force_closure_residual_max": float(cfg.force_closure_residual_max),
    }
    write_run_files(output_dir, trials, traces, metadata, aggregate_trials(trials))
    print(f"[SweepComplete] trials={len(trials)} trace_rows={len(traces)} output_dir={os.path.abspath(output_dir)}")
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.env.test = True
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = True
    actor_only_tasks = ('carrybox', 'carrybox_boxperturb_resume')

    # carrybox
    if args.task == 'carrybox':
        env_cfg.asset.box.random_props = False
        env_cfg.asset.box.reset_mode = 'default'
        env_cfg.env.episode_length_s = 10
        # Play only needs the actor. Load it separately so old 126-D critic
        # checkpoints can run with the current 143-D carry-phase environment.
        train_cfg.runner.resume = False
    if args.task == 'carrybox_boxperturb_resume':
        env_cfg.box_perturbation.enabled = not args.disable_box_perturb
        env_cfg.box_perturbation.debug_force_event = args.debug_force_event
        # Sweep scheduling is driven explicitly below so every cell gets a fresh reset.
        env_cfg.box_perturbation.debug_sweep_enabled = False
        env_cfg.box_perturbation.debug_draw_force = (
            args.debug_force_event or args.debug_force_sweep
        )
        if args.debug_force_sweep:
            env_cfg.env.num_envs = 1
            env_cfg.env.episode_length_s = float(args.eval_episode_length_s)
            env_cfg.box_perturbation.evaluation_mode = True
            env_cfg.box_perturbation.evaluation_manual_schedule = True
            env_cfg.box_perturbation.evaluation_trace_enabled = True
            env_cfg.box_perturbation.evaluation_goal_mode = args.eval_goal_mode
            env_cfg.box_perturbation.evaluation_goal_distance_range = _parse_float_pair(args.eval_goal_distance_range, (4.0, 8.0))
            env_cfg.box_perturbation.evaluation_goal_bearing_offset_deg = _parse_float_pair(args.eval_goal_bearing_offset_deg, (15.0, 75.0))
            env_cfg.box_perturbation.evaluation_precondition_timeout_s = float(args.eval_precondition_timeout_s)
            env_cfg.box_perturbation.evaluation_verbose_substeps = (
                args.verbose_force_trace
            )
            env_cfg.box_perturbation.evaluation_ignore_task_success_reset = True
            _disable_boxperturb_eval_randomization(env_cfg)
        # The bundled nominal checkpoint has a 738-D actor but a legacy 126-D
        # critic. Evaluation needs only the compatible actor.
        train_cfg.runner.resume = False
    # sitdown
    if args.task == 'sitdown' or args.task == 'liedown':
        env_cfg.asset.chair.random_size = False
        env_cfg.asset.chair.reset_mode = 'default'
    # styleloco
    if args.task == 'styleloco_dinosaur' or args.task == 'styleloco_highknee':
        env_cfg.terrain.mesh_type = 'plane'
        env_cfg.terrain.num_rows = 3
        env_cfg.terrain.num_cols = 3
    
    if args.play_dataset:
        train_cfg.runner.resume = False
        env_cfg.viewer.pos = [-5, -5, 4]
        env_cfg.viewer.lookat = [0, 0, 2.]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if args.task in actor_only_tasks and not args.play_dataset:
        load_actor_only_for_inference(
            ppo_runner, args.resume_path, device=env.device
        )
    policy = ppo_runner.get_inference_policy(device=env.device)

    if args.task == 'carrybox_boxperturb_resume' and args.debug_force_sweep:
        run_boxperturb_visual_sweep(env, policy, args)
        return
    
    # export policy as a jit & onnx module (used to run it from C++)
    if EXPORT_POLICY:
        policy_name = 'policy_name'
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, policy_name)
        print('Exported policy as jit script to: ', path)

        jit_path = os.path.join(path, f'{policy_name}.pt')
        jit_model = torch.jit.load(jit_path)
        dummy_input = torch.randn(1, obs.shape[1], device='cpu')
        onnx_path = os.path.join(path, f'{policy_name}.onnx')
        export_jit_to_onnx(jit_model, onnx_path, dummy_input)
        policy = load_onnx_policy(onnx_path)

    for i in range(10*int(env.max_episode_length)):
        env.commands[:, 0] = 0.8
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        if args.play_dataset:
            env.play_dataset_step(i)
        else:
            obs, _, rews, dones, infos, _, _, amp_state = env.step(actions.detach())

            # Carry phase detection: print the detector result while replaying carrybox.pt.
            if args.task == 'carrybox' and CARRY_PHASE_DEBUG and i % CARRY_PHASE_DEBUG_INTERVAL == 0:
                print_carry_phase_debug(env, dones, i, env_id=CARRY_PHASE_DEBUG_ENV_ID)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    CARRY_PHASE_DEBUG = True
    CARRY_PHASE_DEBUG_INTERVAL = 25
    CARRY_PHASE_DEBUG_ENV_ID = 0
    args = get_args()
    play(args)
