import argparse
import copy
import csv
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean, stdev
from types import SimpleNamespace

import isaacgym  # noqa: F401
from isaacgym import gymapi
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.scripts.boxperturb_reporting import (
    FORCE_METRICS,
    print_trial_terminal,
    summarize_force_trace,
    write_run_files,
)


TASK = "carrybox_boxperturb_resume"
DEFAULT_DIRECTIONS = ("+box_x", "-box_x", "+box_y", "-box_y", "+z_world", "-z_world")
DEFAULT_BETAS = (0.10, 0.25, 0.50, 0.75)


def _make_gym_args(parsed):
    use_gpu = parsed.sim_device.startswith("cuda")
    return SimpleNamespace(
        task=TASK,
        resume=False,
        resume_path=None,
        experiment_name=None,
        run_name="carrybox_boxperturb_ab",
        load_run=None,
        checkpoint=None,
        exptid=None,
        resumeid=None,
        headless=not parsed.viewer,
        horovod=False,
        rl_device=parsed.rl_device,
        sim_device=parsed.sim_device,
        device=parsed.rl_device,
        num_envs=1,
        seed=parsed.seeds[0],
        max_iterations=None,
        play_dataset=False,
        disable_box_perturb=False,
        debug_force_event=False,
        debug_force_sweep=False,
        verbose_force_trace=parsed.verbose_force_trace,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=use_gpu,
        use_gpu_pipeline=use_gpu,
        pipeline="gpu" if use_gpu else "cpu",
        subscenes=0,
        num_threads=10,
        slices=None,
        graphics_device_id=0,
        sim_device_id=0,
        compute_device_id=0,
    )


def _configure_eval(env_cfg):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 120
    env_cfg.env.test = False
    env_cfg.noise.add_noise = False
    env_cfg.asset.box.reset_mode = "random"
    env_cfg.asset.box.skill_init_prob = [0.0, 0.0, 1.0, 0.0]
    env_cfg.asset.box.random_size = False
    env_cfg.asset.box.random_density = False
    env_cfg.asset.box.density_default = 50.0
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
    perturb = env_cfg.box_perturbation
    perturb.enabled = True
    perturb.debug_force_event = False
    perturb.debug_sweep_enabled = False
    perturb.debug_draw_force = False
    perturb.evaluation_mode = True
    perturb.evaluation_manual_schedule = True
    perturb.evaluation_trace_enabled = True
    perturb.evaluation_ignore_task_success_reset = True
    return env_cfg


def _set_trial_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_actor_only(runner, checkpoint_path, device, label):
    path = os.path.expanduser(
        checkpoint_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint for {label} not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model_state_dict"]
    actor_state = {
        key: value
        for key, value in state.items()
        if key == "std" or key.startswith("actor.")
    }
    current = runner.alg.actor_critic.state_dict()
    mismatches = {
        key: (tuple(value.shape), tuple(current[key].shape) if key in current else None)
        for key, value in actor_state.items()
        if key not in current or value.shape != current[key].shape
    }
    if mismatches:
        raise RuntimeError(f"Actor checkpoint mismatch for {label}: {mismatches}")
    incompatible = runner.alg.actor_critic.load_state_dict(actor_state, strict=False)
    missing_non_critic = [
        key for key in incompatible.missing_keys if not key.startswith("critic.")
    ]
    if missing_non_critic or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Incomplete actor load for {label}: missing={missing_non_critic}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    actor_dim = int(state["actor.0.weight"].shape[1])
    critic_dim = int(state["critic.0.weight"].shape[1])
    if actor_dim != 738:
        raise RuntimeError(f"{label} actor input is {actor_dim}, expected 738")
    print(
        f"[Checkpoint] label={label} path={path} actor_input={actor_dim} "
        f"checkpoint_critic={critic_dim} current_critic=143 critic_skipped=True"
    )
    return path, actor_dim, critic_dim


def _finite_values(rows, key):
    values = []
    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            values.append(value)
    return values


def _mean(rows, key):
    values = _finite_values(rows, key)
    return float(np.mean(values)) if values else float("nan")


def _max(rows, key):
    values = _finite_values(rows, key)
    return max(values) if values else float("nan")


def _rms(rows, keys):
    values = []
    for row in rows:
        values.extend(float(row[key]) for key in keys)
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _run_trial(env, policy, model_label, checkpoint_path, seed, direction, beta, verbose):
    cfg = env.cfg.box_perturbation
    _set_trial_seed(int(seed))
    obs, _ = env.reset()
    metadata = {
        "model": model_label,
        "checkpoint": checkpoint_path,
        "seed": int(seed),
        "direction": direction,
        "requested_beta": float(beta),
    }
    env.begin_box_perturb_trace(metadata, verbose=verbose)
    precondition_steps = int(
        round(float(cfg.evaluation_precondition_timeout_s) / env.dt)
    )
    recovery_steps = env._recovery_policy_steps()
    post_steps = int(round(float(cfg.evaluation_post_window_s) / env.dt))
    threshold = int(cfg.stable_confirmed_carry_policy_steps)
    dones = torch.zeros(1, device=env.device)
    precondition_ok = False
    max_abs_roll = 0.0
    max_abs_pitch = 0.0

    for _ in range(precondition_steps):
        actions = policy(obs.detach())
        obs, _, _, dones, _, _, _, _ = env.step(actions.detach())
        max_abs_roll = max(max_abs_roll, abs(float(env.roll[0].item())))
        max_abs_pitch = max(max_abs_pitch, abs(float(env.pitch[0].item())))
        if bool(dones[0].item()):
            break
        if int(env.confirmed_carry_streak[0].item()) >= threshold:
            precondition_ok = True
            break

    base_summary = {
        **metadata,
        "precondition_success": int(precondition_ok),
        "event_triggered": 0,
        "peak_force_N": float("nan"),
        "pulse_hold_retention": float("nan"),
        "pulse_bimanual_contact_retention": float("nan"),
        "recovery_success": float("nan"),
        "recovery_time_s": float("nan"),
        "termination": 0,
        "termination_reason": env.box_perturb_last_termination_reason[0],
        "drop_failure": 0,
        "fall_failure": 0,
        "post_confirmed_ratio": float("nan"),
        "goal_progress_m": float("nan"),
        "max_abs_roll_rad": max_abs_roll,
        "max_abs_pitch_rad": max_abs_pitch,
    }
    if not precondition_ok:
        trace = env.end_box_perturb_trace()
        reason = env.box_perturb_last_termination_reason[0]
        reason_lower = reason.lower()
        base_summary.update(
            termination=int(bool(reason)),
            termination_reason=reason,
            drop_failure=int("drop" in reason_lower or "box_tilt" in reason_lower),
            fall_failure=int(
                any(
                    token in reason_lower
                    for token in ("head_low", "base_low", "base_tilt")
                )
            ),
        )
        base_summary["trace_rows"] = len(trace)
        base_summary.update(summarize_force_trace(trace))
        return base_summary, trace

    goal_start = float(env.object2goal_dist_xy[0].item())
    peak = env.schedule_explicit_box_perturbation(direction, beta, env_id=0)
    pulse_finished = False
    after_pulse_steps = 0
    recovery_streak = 0
    recovery_success = False
    recovery_time_steps = -1
    pulse_confirmed = []
    pulse_bimanual = []
    post_confirmed = []
    termination_reason = ""

    while after_pulse_steps < recovery_steps + post_steps:
        if pulse_finished and after_pulse_steps >= recovery_steps:
            env.set_box_perturb_trace_phase("post")
        actions = policy(obs.detach())
        obs, _, _, dones, _, _, _, _ = env.step(actions.detach())
        max_abs_roll = max(max_abs_roll, abs(float(env.roll[0].item())))
        max_abs_pitch = max(max_abs_pitch, abs(float(env.pitch[0].item())))
        if not pulse_finished:
            pulse_confirmed.append(float(env.confirmed_carry_buf[0].item()))
            pulse_bimanual.append(float(env.both_hand_contact_buf[0].item()))
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
    pre_rows = [row for row in trace if row["phase"] == "pre" and row["confirmed_carry"]]
    pre_rows = pre_rows[-40:]
    pulse_rows = [row for row in trace if row["phase"] == "pulse"]
    recovery_rows = [row for row in trace if row["phase"] == "recovery"]
    post_rows = [row for row in trace if row["phase"] == "post"]
    response_rows = pulse_rows + recovery_rows
    left_pre = _mean(pre_rows, "left_hand_on_box_proxy_norm_N")
    right_pre = _mean(pre_rows, "right_hand_on_box_proxy_norm_N")
    left_pulse = _mean(pulse_rows, "left_hand_on_box_proxy_norm_N")
    right_pulse = _mean(pulse_rows, "right_hand_on_box_proxy_norm_N")
    reason_lower = termination_reason.lower()

    base_summary.update(
        event_triggered=1,
        peak_force_N=peak,
        pulse_hold_retention=float(np.mean(pulse_confirmed)) if pulse_confirmed else 0.0,
        pulse_bimanual_contact_retention=float(np.mean(pulse_bimanual)) if pulse_bimanual else 0.0,
        recovery_success=int(recovery_success),
        recovery_time_s=(
            recovery_time_steps * env.dt if recovery_time_steps >= 0 else float("nan")
        ),
        termination=int(bool(termination_reason)),
        termination_reason=termination_reason,
        drop_failure=int("drop" in reason_lower or "box_tilt" in reason_lower),
        fall_failure=int(
            any(token in reason_lower for token in ("head_low", "base_low", "base_tilt"))
        ),
        post_confirmed_ratio=float(np.mean(post_confirmed)) if post_confirmed else 0.0,
        goal_progress_m=(goal_start - goal_end if math.isfinite(goal_end) else float("nan")),
        max_abs_roll_rad=max_abs_roll,
        max_abs_pitch_rad=max_abs_pitch,
        external_impulse_Ns=sum(row["f_ext_norm_N"] for row in pulse_rows)
        * float(env.sim_params.dt),
        left_hand_pre_mean_N=left_pre,
        right_hand_pre_mean_N=right_pre,
        left_hand_pulse_mean_N=left_pulse,
        right_hand_pulse_mean_N=right_pulse,
        left_hand_pulse_peak_N=_max(pulse_rows, "left_hand_on_box_proxy_norm_N"),
        right_hand_pulse_peak_N=_max(pulse_rows, "right_hand_on_box_proxy_norm_N"),
        left_hand_recovery_mean_N=_mean(recovery_rows, "left_hand_on_box_proxy_norm_N"),
        right_hand_recovery_mean_N=_mean(recovery_rows, "right_hand_on_box_proxy_norm_N"),
        left_hand_force_delta_N=left_pulse - left_pre,
        right_hand_force_delta_N=right_pulse - right_pre,
        resistive_hand_force_mean_N=_mean(pulse_rows, "resistive_hand_force_N"),
        resistive_hand_force_peak_N=_max(pulse_rows, "resistive_hand_force_N"),
        hand_load_asymmetry_mean=_mean(pulse_rows, "hand_load_asymmetry"),
        hand_box_rel_speed_rms_mps=_rms(
            response_rows,
            ("left_hand_box_rel_speed_mps", "right_hand_box_rel_speed_mps"),
        ),
        hand_box_rel_speed_peak_mps=max(
            _max(response_rows, "left_hand_box_rel_speed_mps"),
            _max(response_rows, "right_hand_box_rel_speed_mps"),
        ),
        pairwise_left_normal_mean_N=_mean(pulse_rows, "left_pair_normal_lambda_N"),
        pairwise_right_normal_mean_N=_mean(pulse_rows, "right_pair_normal_lambda_N"),
        pairwise_left_contact_count_mean=_mean(pulse_rows, "left_pair_count"),
        pairwise_right_contact_count_mean=_mean(pulse_rows, "right_pair_count"),
        pairwise_proxy_unmatched_fraction=(
            float(
                np.mean(
                    [
                        int(
                            (row["left_contact"] and row["left_pair_count"] == 0)
                            or (row["right_contact"] and row["right_pair_count"] == 0)
                        )
                        for row in pulse_rows
                        if row["left_pair_count"] >= 0 and row["right_pair_count"] >= 0
                    ]
                )
            )
            if any(
                row["left_pair_count"] >= 0 and row["right_pair_count"] >= 0
                for row in pulse_rows
            )
            else float("nan")
        ),
        trace_rows=len(trace),
        pre_trace_rows=len(pre_rows),
        pulse_trace_rows=len(pulse_rows),
        recovery_trace_rows=len(recovery_rows),
        post_trace_rows=len(post_rows),
    )
    base_summary.update(summarize_force_trace(trace))
    return base_summary, trace


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(trials):
    groups = defaultdict(list)
    for trial in trials:
        groups[(trial["model"], trial["direction"], trial["requested_beta"])].append(trial)
    metric_names = (
        "pulse_hold_retention",
        "pulse_bimanual_contact_retention",
        "recovery_time_s",
        "post_confirmed_ratio",
        "goal_progress_m",
        "hand_box_rel_speed_rms_mps",
        "hand_box_rel_speed_peak_mps",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
        "left_hand_pulse_mean_N",
        "right_hand_pulse_mean_N",
        "resistive_hand_force_mean_N",
        "hand_load_asymmetry_mean",
        "force_valid_fraction",
        "force_closure_residual_pulse_mean",
        "left_rho_pulse_mean",
        "right_rho_pulse_mean",
        "normal_load_asymmetry_pulse_mean",
    ) + tuple(
        f"{key}_{suffix}"
        for key in FORCE_METRICS
        for suffix in ("pre_mean", "pulse_mean", "pulse_peak", "pulse_delta_from_pre")
    )
    summary = []
    for (model, direction, beta), rows in sorted(groups.items()):
        valid = [row for row in rows if row["precondition_success"]]
        item = {
            "model": model,
            "direction": direction,
            "beta": beta,
            "trials": len(rows),
            "precondition_success_rate": sum(row["precondition_success"] for row in rows) / len(rows),
            "conditional_recovery_success_rate": (
                sum(row["recovery_success"] for row in valid) / len(valid)
                if valid
                else float("nan")
            ),
            "termination_rate": sum(row["termination"] for row in valid) / len(valid)
            if valid
            else float("nan"),
            "drop_failure_rate": sum(row["drop_failure"] for row in valid) / len(valid)
            if valid
            else float("nan"),
            "fall_failure_rate": sum(row["fall_failure"] for row in valid) / len(valid)
            if valid
            else float("nan"),
        }
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in valid
                if metric in row and math.isfinite(float(row[metric]))
            ]
            item[f"{metric}_mean"] = mean(values) if values else float("nan")
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def _paired_comparison(trials, baseline_label, interaction_label):
    by_key = {
        (row["model"], row["direction"], row["requested_beta"], row["seed"]): row
        for row in trials
    }
    metrics = (
        "recovery_success",
        "pulse_hold_retention",
        "post_confirmed_ratio",
        "goal_progress_m",
        "hand_box_rel_speed_rms_mps",
        "max_abs_roll_rad",
        "max_abs_pitch_rad",
    )
    result = {
        "positive_difference_means_interaction_minus_baseline": True,
        "higher_is_better": [
            "recovery_success",
            "pulse_hold_retention",
            "post_confirmed_ratio",
            "goal_progress_m",
        ],
        "lower_is_better": [
            "hand_box_rel_speed_rms_mps",
            "max_abs_roll_rad",
            "max_abs_pitch_rad",
        ],
        "notes": "Five seeds per cell are exploratory; no strong significance claim.",
        "cells": [],
    }
    cells = sorted(
        {(row["direction"], row["requested_beta"]) for row in trials}
    )
    for direction, beta in cells:
        cell = {"direction": direction, "beta": beta, "paired_seed_count": 0}
        differences = defaultdict(list)
        for seed in sorted({row["seed"] for row in trials}):
            a = by_key.get((baseline_label, direction, beta, seed))
            b = by_key.get((interaction_label, direction, beta, seed))
            if not a or not b or not a["precondition_success"] or not b["precondition_success"]:
                continue
            cell["paired_seed_count"] += 1
            for metric in metrics:
                av = float(a[metric])
                bv = float(b[metric])
                if math.isfinite(av) and math.isfinite(bv):
                    differences[metric].append(bv - av)
        for metric, values in differences.items():
            cell[f"{metric}_paired_mean_difference"] = mean(values)
        result["cells"].append(cell)
    return result


def main(parsed):
    os.makedirs(parsed.output_dir, exist_ok=True)
    gym_args = _make_gym_args(parsed)
    env_cfg, train_cfg = task_registry.get_cfgs(TASK)
    env_cfg = _configure_eval(copy.deepcopy(env_cfg))
    train_cfg = copy.deepcopy(train_cfg)
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(TASK, args=gym_args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, args=gym_args, train_cfg=train_cfg, log_root=None
    )
    if env.num_envs != 1 or env.num_obs != 738 or env.num_privileged_obs != 143:
        raise AssertionError(
            (env.num_envs, env.num_obs, env.num_privileged_obs)
        )

    checkpoints = (
        (parsed.baseline_label, parsed.baseline_checkpoint),
        (parsed.interaction_label, parsed.interaction_checkpoint),
    )
    trials = []
    traces = []
    for label, checkpoint in checkpoints:
        resolved_path, _, _ = _load_actor_only(runner, checkpoint, env.device, label)
        policy = runner.get_inference_policy(device=env.device)
        for beta in parsed.betas:
            for direction in parsed.directions:
                for seed in parsed.seeds:
                    trial, trace = _run_trial(
                        env,
                        policy,
                        label,
                        resolved_path,
                        seed,
                        direction,
                        beta,
                        parsed.verbose_force_trace,
                    )
                    trials.append(trial)
                    traces.extend(trace)
                    print_trial_terminal(trial, prefix=f"AB:{label}")

    summary = _aggregate(trials)
    comparison = _paired_comparison(
        trials, parsed.baseline_label, parsed.interaction_label
    )
    metadata = {
        "mode": "batch_ab", "checkpoints": {
            parsed.baseline_label: parsed.baseline_checkpoint,
            parsed.interaction_label: parsed.interaction_checkpoint,
        },
        "seeds": list(parsed.seeds), "directions": list(parsed.directions),
        "betas": list(parsed.betas),
        "coordinate_convention": {
            "box_xy": "box-local axes rotated to world at schedule time",
            "world_z": "gravity-aligned world axis", "force": "world-frame N", "impulse": "N s",
        },
        "physics_dt_s": float(env.sim_params.dt), "policy_dt_s": float(env.dt),
        "pulse_profile": "midpoint-sampled half-sine",
        "pulse_duration_s": float(env.cfg.box_perturbation.pulse_duration_s),
        "force_peak_cap_N": env.cfg.box_perturbation.force_peak_cap_N,
        "force_decomposition": "hand rigid-body net contact force projected on estimated locked box-face normal",
        "ema_tau_s": 0.04, "baseline_valid_physics_substeps": 40,
        "force_sign_verification_samples": int(env.cfg.box_perturbation.force_sign_verification_samples),
        "force_closure_residual_max": float(env.cfg.box_perturbation.force_closure_residual_max),
    }
    write_run_files(parsed.output_dir, trials, traces, metadata, summary)
    with open(
        os.path.join(parsed.output_dir, "comparison.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(
        f"[ABComplete] trials={len(trials)} trace_rows={len(traces)} "
        f"output_dir={os.path.abspath(parsed.output_dir)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--interaction_checkpoint", required=True)
    parser.add_argument("--baseline_label", default="builtin")
    parser.add_argument("--interaction_label", default="interaction_priv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--directions", nargs="+", default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--betas", nargs="+", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument("--output_dir", default="logs/boxperturb_ab")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--verbose_force_trace", action="store_true", default=False)
    main(parser.parse_args())
