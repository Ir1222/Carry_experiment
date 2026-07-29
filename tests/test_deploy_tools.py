from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from deploy.common.config import load_deploy_config
from deploy.common.constants import DEFAULT_DOF_POS
from deploy.common.math_utils import wxyz_to_xyzw
from deploy.common.observation import ObservationBuilder
from deploy.common.types import RobotState, TaskState
from deploy.tools.compare_observation_snapshot import compare_snapshot
from deploy.tools.make_dryrun_config import create_dryrun_config
from deploy.tools.run_udp_smoke import verify_logs


CONFIG_PATH = Path("deploy/config/g1_carrybox.yaml")


def test_dryrun_config_is_temporary_and_safety_locked(tmp_path):
    original = CONFIG_PATH.read_text(encoding="utf-8")
    output = tmp_path / "g1_carrybox_dryrun.yaml"
    created = create_dryrun_config(
        CONFIG_PATH,
        output,
        "test-g1-nic",
        validate_interface=False,
    )

    assert created == output.resolve()
    assert CONFIG_PATH.read_text(encoding="utf-8") == original
    cfg = load_deploy_config(created)
    assert cfg.project_root == Path.cwd().resolve()
    assert cfg.section("network")["interface"] == "test-g1-nic"
    assert cfg.section("simulation")["transport"] == "unitree_dds"
    assert cfg.section("safety")["dry_run"] is True
    assert float(cfg.section("control")["hardware_kp_scale"]) == 0.0


def test_dryrun_config_refuses_to_overwrite_source():
    with pytest.raises(ValueError, match="refusing to overwrite"):
        create_dryrun_config(
            CONFIG_PATH,
            CONFIG_PATH,
            "test-g1-nic",
            validate_interface=False,
        )


def test_smoke_log_verifier_checks_rates_safety_and_finiteness(tmp_path):
    policy_log = tmp_path / "policy.jsonl"
    simulator_log = tmp_path / "simulator.jsonl"
    policy_records = [
        {
            "kind": "run_metadata",
            "component": "policy",
            "model": {"profile": "model_73500"},
        }
    ] + [
        {
            "kind": "policy_step",
            "sequence": 1 + 4 * index,
            "timestamp_ns": index * 20_000_000,
            "command_armed": True,
            "command_reason": "safe",
            "episode_failed": False,
            "state_sequence_reset": False,
            "inference_time_ms": 1.0,
            "raw_action": [0.1, 0.2],
        }
        for index in range(31)
    ]
    simulator_records = [
        {
            "kind": "mujoco_step",
            "sim_time": index * 0.005,
            "wall_timestamp_ns": index * 5_000_000,
            "physics_step_stride": 1,
            "joint_pos": [0.0],
            "projected_gravity": [0.0, 0.0, -1.0],
            "joint_limit_violation": [0.0],
            "ground_contact_bodies": [],
            "contacts": [],
        }
        for index in range(121)
    ]
    policy_log.write_text(
        "".join(json.dumps(record) + "\n" for record in policy_records),
        encoding="utf-8",
    )
    simulator_log.write_text(
        "".join(json.dumps(record) + "\n" for record in simulator_records),
        encoding="utf-8",
    )

    summary = verify_logs(policy_log, simulator_log)
    assert summary.policy_hz == pytest.approx(50.0)
    assert summary.physics_hz == pytest.approx(200.0)
    assert summary.physics_wall_hz == pytest.approx(200.0)
    assert summary.safe_steps == 6
    assert summary.safe_fraction == 1.0

    unsafe_records = list(policy_records)
    unsafe = yaml.safe_load(json.dumps(policy_records[-1]))
    unsafe["command_armed"] = False
    unsafe["command_reason"] = "task state stale"
    unsafe_records[-1] = unsafe
    policy_log.write_text(
        "".join(json.dumps(record) + "\n" for record in unsafe_records),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not continuously safe"):
        verify_logs(policy_log, simulator_log)


def test_observation_snapshot_comparator_uses_policy_frame(tmp_path):
    endpoints = np.arange(15, dtype=np.float64).reshape(5, 3)
    policy_quat = np.array([1.0, 0.0, 0.0, 0.0])
    robot = RobotState(
        sequence=1,
        timestamp_ns=1,
        policy_frame_quat_wxyz=policy_quat,
        policy_frame_ang_vel=np.array([1.0, 2.0, 3.0]),
        joint_pos=np.asarray(DEFAULT_DOF_POS),
        joint_vel=np.ones(29),
        end_effector_pos_policy_frame=endpoints,
    )
    task = TaskState(
        sequence=1,
        timestamp_ns=1,
        box_pos_policy_frame=np.array([1.0, 2.0, 3.0]),
        box_quat_policy_frame_wxyz=policy_quat,
        box_size=np.array([0.3, 0.3, 0.25]),
        goal_pos_policy_frame=np.array([4.0, 5.0, 6.0]),
    )
    builder = ObservationBuilder(legacy_ankle_delay_steps=0)
    previous_action = np.ones(29) * 2.0
    builder.set_previous_action(previous_action)
    frame = builder.build_frame(robot, task)
    actor_obs = np.zeros(738, dtype=np.float32)
    actor_obs[-123:] = frame
    path = tmp_path / "snapshot.npz"
    np.savez(
        path,
        root_position_world=np.zeros(3),
        policy_frame_quat_xyzw=wxyz_to_xyzw(policy_quat),
        torso_quat_xyzw=np.array([0.0, 0.0, 1.0, 0.0]),
        policy_frame_ang_vel=robot.policy_frame_ang_vel,
        joint_pos=robot.joint_pos,
        joint_vel=robot.joint_vel,
        end_effector_pos_policy_frame=endpoints,
        previous_action=previous_action,
        box_position_world=task.box_pos_policy_frame,
        box_quat_xyzw=wxyz_to_xyzw(policy_quat),
        box_size=task.box_size,
        goal_position_world=task.goal_pos_policy_frame,
        success=np.array(False),
        current_frame=frame,
        actor_obs=actor_obs,
    )
    result = compare_snapshot(path)
    assert result["frame_max_abs_error"] == 0.0
    assert result["torso_policy_quat_difference"] > 0.0
