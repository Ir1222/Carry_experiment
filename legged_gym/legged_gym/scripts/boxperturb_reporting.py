"""Shared CSV and terminal reporting for carry-box perturbation evaluation."""

import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev


FORCE_METRICS = (
    "left_fn_raw_N", "right_fn_raw_N", "left_ft_raw_N", "right_ft_raw_N",
    "left_fn_ema_N", "right_fn_ema_N", "left_ft_ema_N", "right_ft_ema_N",
)


def _finite(rows, key):
    return [
        float(row[key]) for row in rows
        if key in row and math.isfinite(float(row[key]))
    ]


def _stat(rows, key, operation="mean"):
    values = _finite(rows, key)
    if not values:
        return float("nan")
    return max(values) if operation == "max" else sum(values) / len(values)


def summarize_force_trace(trace):
    """Return per-trial pre/pulse/recovery response metrics from raw trace rows."""
    pre = [r for r in trace if r.get("phase") == "pre" and r.get("force_decomposition_valid") == 1][-40:]
    pulse = [r for r in trace if r.get("phase") == "pulse"]
    recovery = [r for r in trace if r.get("phase") == "recovery"]
    response = {}
    for key in FORCE_METRICS:
        pre_mean = _stat(pre, key)
        pulse_mean = _stat(pulse, key)
        response[f"{key}_pre_mean"] = pre_mean
        response[f"{key}_pulse_mean"] = pulse_mean
        response[f"{key}_pulse_peak"] = _stat(pulse, key, "max")
        response[f"{key}_recovery_mean"] = _stat(recovery, key)
        response[f"{key}_pulse_delta_from_pre"] = (
            pulse_mean - pre_mean
            if math.isfinite(pre_mean) and math.isfinite(pulse_mean)
            else float("nan")
        )
    all_response = pulse + recovery
    response.update(
        force_valid_fraction=(
            sum(int(r.get("force_decomposition_valid", 0)) for r in all_response) / len(all_response)
            if all_response else float("nan")
        ),
        pulse_force_valid_fraction=(
            sum(int(r.get("force_decomposition_valid", 0)) for r in pulse) / len(pulse)
            if pulse else float("nan")
        ),
        force_baseline_sample_count=len(pre),
        force_baseline_unavailable=int(len(pre) < 40),
        force_closure_residual_pulse_mean=_stat(pulse, "force_closure_residual"),
        left_rho_pulse_mean=_stat(pulse, "left_rho_raw"),
        right_rho_pulse_mean=_stat(pulse, "right_rho_raw"),
        normal_load_asymmetry_pulse_mean=_stat(pulse, "normal_load_asymmetry"),
        external_impulse_Ns=(max(_finite(trace, "force_impulse_Ns"), default=0.0)),
    )
    if pulse:
        last = pulse[-1]
        for key in (
            "force_uncapped_peak_N", "force_peak_N", "force_cap_used",
            "perturb_direction_world_x", "perturb_direction_world_y", "perturb_direction_world_z",
        ):
            response[key] = last.get(key, float("nan"))
    return response


def print_trial_terminal(trial, prefix="BoxPerturb"):
    direction = trial.get("direction", "unknown")
    beta = float(trial.get("requested_beta", float("nan")))
    print(
        f"[{prefix}:Perturb] direction={direction} beta={beta:.3f} "
        f"world=({float(trial.get('perturb_direction_world_x', float('nan'))):.4f},"
        f"{float(trial.get('perturb_direction_world_y', float('nan'))):.4f},"
        f"{float(trial.get('perturb_direction_world_z', float('nan'))):.4f}) "
        f"peak={float(trial.get('peak_force_N', trial.get('force_peak_N', float('nan')))):.4f}N"
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_trials(trials):
    groups = defaultdict(list)
    for row in trials:
        groups[(row["model"], row["direction"], row["requested_beta"])].append(row)
    metrics = (
        "recovery_success", "pulse_hold_retention", "post_confirmed_ratio",
        "force_valid_fraction", "force_closure_residual_pulse_mean",
    ) + tuple(
        f"{key}_{suffix}"
        for key in FORCE_METRICS
        for suffix in ("pre_mean", "pulse_mean", "pulse_peak", "pulse_delta_from_pre")
    )
    result = []
    for (model, direction, beta), rows in sorted(groups.items()):
        item = {"model": model, "direction": direction, "beta": beta, "trials": len(rows)}
        item["precondition_success_rate"] = sum(int(r.get("precondition_success", 0)) for r in rows) / len(rows)
        for metric in metrics:
            values = _finite(rows, metric)
            item[f"{metric}_mean"] = mean(values) if values else float("nan")
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def write_run_files(output_dir, trials, traces, metadata, summary=None):
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "force_trace.csv"), traces)
    write_csv(os.path.join(output_dir, "trials.csv"), trials)
    write_csv(os.path.join(output_dir, "summary.csv"), summary or aggregate_trials(trials))
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=True)
