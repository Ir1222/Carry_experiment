#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
EVAL=legged_gym/legged_gym/scripts/evaluate_carrybox_boxperturb.py
BASE=legged_gym/resources/ckpt/carrybox.pt
MODEL555=${MODEL555:-/home/han/pyx/Carry_experiment/legged_gym/logs/amp_carrybox/Jul06_19-28-33_critic_143_resume_from_41000/model_55500.pt}
ROOT=${OUTPUT_ROOT:-logs/boxperturb_compare/rerun_after_single_history_commit}
SEEDS=(1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032)

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume_eval)
fi

run_case() {
  local output_dir=$1
  shift
  "$PYTHON" "$EVAL" \
    --baseline_checkpoint "$BASE" \
    --interaction_checkpoint "$MODEL555" \
    --baseline_label carrybox_builtin \
    --interaction_label critic143_model55500 \
    --output_dir "$ROOT/$output_dir" \
    --rl_device cuda:0 --sim_device cuda:0 \
    --seeds "${SEEDS[@]}" \
    --directions +box_x \
    --force_peak_cap_N none \
    "$@" "${RESUME_ARGS[@]}"
}

run_case 00_nominal_default \
  --eval_goal_mode default \
  --eval_episode_length_s 120 --eval_precondition_timeout_s 5 \
  --betas 0 --force_point_modes com \
  --pulse_durations 0.10 --pulse_profiles half_sine

run_case 01_nominal_longrange_8m \
  --eval_goal_mode long_range --eval_goal_distance_range 8.0 8.0 \
  --eval_episode_length_s 180 --eval_precondition_timeout_s 8 \
  --betas 0 --force_point_modes com \
  --pulse_durations 0.10 --pulse_profiles half_sine

run_case 10_legacy_com_dose \
  --eval_goal_mode default \
  --eval_episode_length_s 120 --eval_precondition_timeout_s 5 \
  --betas 0.10 0.25 0.50 0.75 --force_point_modes com \
  --pulse_durations 0.10 --pulse_profiles half_sine

run_case 11_longrange8m_com_dose \
  --eval_goal_mode long_range --eval_goal_distance_range 8.0 8.0 \
  --eval_episode_length_s 180 --eval_precondition_timeout_s 8 \
  --betas 0.10 0.25 0.50 0.75 --force_point_modes com \
  --pulse_durations 0.10 --pulse_profiles half_sine

run_case 20_longrange_beta2 \
  --eval_goal_mode long_range --eval_goal_distance_range 8.0 8.0 \
  --eval_episode_length_s 180 --eval_precondition_timeout_s 8 \
  --betas 0.75 2.00 --force_point_modes com \
  --pulse_durations 0.10 --pulse_profiles half_sine

run_case 30_longrange_point_torque \
  --eval_goal_mode long_range --eval_goal_distance_range 8.0 8.0 \
  --eval_episode_length_s 180 --eval_precondition_timeout_s 8 \
  --betas 0.75 --force_point_modes com box_surface_grid \
  --force_point_labels face_center face_upper face_lower face_left_edge face_right_edge \
  --pulse_durations 0.10 --pulse_profiles half_sine

run_case 40_longrange_duration_peak_matched \
  --eval_goal_mode long_range --eval_goal_distance_range 8.0 8.0 \
  --eval_episode_length_s 180 --eval_precondition_timeout_s 8 \
  --betas 0.75 --force_point_modes com \
  --pulse_durations 0.05 0.10 0.20 --pulse_profiles half_sine

run_case 41_longrange_profile_peak_matched \
  --eval_goal_mode long_range --eval_goal_distance_range 8.0 8.0 \
  --eval_episode_length_s 180 --eval_precondition_timeout_s 8 \
  --betas 0.75 --force_point_modes com \
  --pulse_durations 0.10 \
  --pulse_profiles half_sine ramp_hold jittered_half_sine
