from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
import yaml

from deploy.common.config import load_deploy_config
from deploy.common.constants import ACTION_DIM, DEFAULT_DOF_POS
from deploy.common.types import PolicyCommand, RobotState, TaskState
from deploy.sim2sim.randomization import ScenarioSample, ScenarioSampler
from deploy.tools.evaluate_sim2sim_robustness import (
    _aggregate,
    _seed_values,
)


CONFIG_PATH = Path("deploy/config/g1_carrybox.yaml")


def _robot(sequence: int = 4) -> RobotState:
    return RobotState(
        sequence=sequence,
        timestamp_ns=123,
        policy_frame_quat_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
        policy_frame_ang_vel=np.zeros(3),
        joint_pos=np.asarray(DEFAULT_DOF_POS),
        joint_vel=np.zeros(ACTION_DIM),
        end_effector_pos_policy_frame=np.zeros((5, 3)),
    )


def _task(sequence: int = 4) -> TaskState:
    return TaskState(
        sequence=sequence,
        timestamp_ns=123,
        box_pos_policy_frame=np.asarray((1.0, 0.0, -0.4)),
        box_quat_policy_frame_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
        box_size=np.asarray((0.3, 0.3, 0.25)),
        goal_pos_policy_frame=np.asarray((2.0, 0.5, -0.5)),
    )


def test_seeded_scenarios_are_stable_distinct_and_serializable(tmp_path):
    sampler = ScenarioSampler(load_deploy_config(CONFIG_PATH))
    first = sampler.sample("light", 11)
    replay = sampler.sample("light", 11)
    different = sampler.sample("light", 12)
    assert first.fingerprint == replay.fingerprint
    assert first.fingerprint != different.fingerprint
    assert first.to_dict() == replay.to_dict()

    path = first.write(tmp_path / "scenario.json")
    loaded = ScenarioSample.read(path)
    assert loaded == first
    assert loaded.fingerprint == first.fingerprint

    data = json.loads(path.read_text(encoding="utf-8"))
    data["seed"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ScenarioSample.read(path)


def test_nominal_scenario_preserves_existing_reset():
    cfg = load_deploy_config(CONFIG_PATH)
    scenario = ScenarioSampler(cfg).sample("nominal", 123)
    simulation = cfg.section("simulation")
    np.testing.assert_allclose(
        scenario.robot_position, simulation["robot_initial_position"]
    )
    np.testing.assert_allclose(scenario.joint_position, DEFAULT_DOF_POS)
    np.testing.assert_allclose(
        scenario.box_position, simulation["box_initial_position"]
    )
    np.testing.assert_allclose(
        scenario.goal_position, simulation["goal_position"]
    )
    np.testing.assert_allclose(scenario.kp_factors, 1.0)
    np.testing.assert_allclose(scenario.kd_factors, 1.0)
    np.testing.assert_allclose(scenario.motor_strength, 1.0)
    np.testing.assert_allclose(scenario.torque_bias_fraction, 0.0)
    assert scenario.action_delay_steps == 0
    assert scenario.disturbance_force == 0.0


def test_observation_noise_is_replayable_and_does_not_mutate_truth():
    sampler = ScenarioSampler(load_deploy_config(CONFIG_PATH))
    scenario = sampler.sample("train_match", 5)
    robot = _robot()
    task = _task()
    observed_a = scenario.observed_states(robot, task)
    observed_b = scenario.observed_states(robot, task)
    np.testing.assert_allclose(observed_a[0].joint_pos, observed_b[0].joint_pos)
    np.testing.assert_allclose(
        observed_a[1].box_pos_policy_frame,
        observed_b[1].box_pos_policy_frame,
    )
    assert not np.allclose(observed_a[0].joint_pos, robot.joint_pos)
    assert not np.allclose(
        observed_a[1].goal_pos_policy_frame, task.goal_pos_policy_frame
    )
    np.testing.assert_allclose(robot.joint_pos, DEFAULT_DOF_POS)
    np.testing.assert_allclose(task.goal_pos_policy_frame, (2.0, 0.5, -0.5))

    next_observed = scenario.observed_states(_robot(5), _task(5))
    assert not np.allclose(
        observed_a[0].joint_pos, next_observed[0].joint_pos
    )


def test_randomization_config_rejects_invalid_range(tmp_path):
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    source["project_root"] = str(CONFIG_PATH.resolve().parents[2])
    source["sim2sim_randomization"]["train_match_ranges"][
        "robot_friction"
    ] = [1.0, 0.5]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="robot_friction"):
        load_deploy_config(path)


def test_seed_parser_and_aggregate_behavior():
    assert _seed_values("0:5:2") == (0, 2, 4)
    assert _seed_values("3,7") == (3, 7)
    rows = [
        {
            "model_profile": "actor",
            "outcome": "success",
            "first_failure_time": None,
            "first_failure_code": None,
            "first_failure_reason": None,
            "max_projected_gravity_xy": 0.1,
            "peak_abs_torque": 10.0,
            "max_contact_force": 100.0,
            "trace_hash": "a",
        },
        {
            "model_profile": "actor",
            "outcome": "termination",
            "first_failure_time": 2.0,
            "first_failure_code": "fall",
            "first_failure_reason": "fall",
            "max_projected_gravity_xy": 0.9,
            "peak_abs_torque": 20.0,
            "max_contact_force": 200.0,
            "trace_hash": "b",
        },
        {
            "model_profile": "actor",
            "outcome": "infrastructure_error",
            "first_failure_time": None,
            "first_failure_code": None,
            "first_failure_reason": None,
            "max_projected_gravity_xy": 0.0,
            "peak_abs_torque": 0.0,
            "max_contact_force": 0.0,
            "trace_hash": None,
        },
    ]
    result = _aggregate(rows, "actor")
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["termination_rate"] == pytest.approx(0.5)
    assert result["infrastructure_error_count"] == 1
    assert result["distinct_trace_hash_count"] == 2


def test_mujoco_randomization_reset_delay_and_torque_order():
    pytest.importorskip("mujoco")
    from deploy.sim2sim.mujoco_server import MujocoServer

    cfg = load_deploy_config(CONFIG_PATH)
    server = MujocoServer(
        cfg,
        transport="udp",
        viewer=False,
        randomization_profile="light",
        seed=17,
    )
    try:
        mass = server.model.body_mass.copy()
        inertia = server.model.body_inertia.copy()
        friction = server.model.geom_friction.copy()
        server.reset()
        np.testing.assert_allclose(server.model.body_mass, mass)
        np.testing.assert_allclose(server.model.body_inertia, inertia)
        np.testing.assert_allclose(server.model.geom_friction, friction)
        physical_bodies = server._nominal_model["body_mass"] > 0.0
        assert np.all(server.model.body_mass[physical_bodies] > 0.0)
        assert np.all(server.model.body_inertia[physical_bodies] > 0.0)

        scenario = replace(
            server.current_scenario,
            action_delay_steps=1,
            kp_factors=np.full(ACTION_DIM, 1.1).tolist(),
            kd_factors=np.full(ACTION_DIM, 0.9).tolist(),
            motor_strength=np.full(ACTION_DIM, 0.8).tolist(),
            torque_bias_fraction=np.full(ACTION_DIM, 0.01).tolist(),
        )
        server.current_scenario = scenario
        hold = server.last_command
        command_1 = PolicyCommand(
            sequence=4,
            timestamp_ns=time.monotonic_ns(),
            raw_action=np.zeros(ACTION_DIM),
            q_target=np.asarray(DEFAULT_DOF_POS) + 0.01,
            kp=np.full(ACTION_DIM, 2.0),
            kd=np.full(ACTION_DIM, 0.5),
            armed=True,
            reason="test",
        )
        command_2 = replace(command_1, sequence=8)
        server.pending_command = command_1
        server._activate_pending_command()
        assert server.last_command is hold
        server.pending_command = command_2
        server._activate_pending_command()
        assert server.last_command is command_1

        joint_pos, joint_vel = server.name_map.joint_state(server.data)
        expected = (
            (
                command_1.kp
                * 1.1
                * (command_1.q_target - joint_pos)
                - command_1.kd * 0.9 * joint_vel
            )
            * 0.8
            + 0.01 * server.robot.effort_limits
        )
        expected = np.clip(
            expected,
            -server.robot.effort_limits,
            server.robot.effort_limits,
        )
        np.testing.assert_allclose(server._apply_command(), expected)
    finally:
        server.close()
