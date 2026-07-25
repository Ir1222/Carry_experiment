from __future__ import annotations

import time

import numpy as np
import pytest

from deploy.common.config import load_deploy_config
from deploy.common.constants import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    DEFAULT_DOF_POS,
    FRAME_OBS_DIM,
    KP,
    OBSERVATION_SLICES,
)
from deploy.common.control import PDController
from deploy.common.mapping import RobotDescription
from deploy.common.math_utils import (
    quat_rotate_inverse_wxyz,
    quat_rotate_wxyz,
    quat_to_tan_norm_wxyz,
)
from deploy.common.observation import ObservationBuilder
from deploy.common.safety import SafetyGate
from deploy.common.transport import (
    pack_policy_command,
    pack_robot_state,
    pack_task_state,
    unpack_policy_command,
    unpack_robot_state,
    unpack_task_state,
)
from deploy.common.types import RobotState, TaskState
from deploy.policy.core import PolicyCore
from deploy.policy.inference import OnnxActor
from deploy.tools.export_actor import load_actor_from_checkpoint


CONFIG_PATH = "deploy/config/g1_carrybox.yaml"


def make_robot(sequence=1, timestamp_ns=None, endpoint_offset=0.0):
    endpoints = np.arange(15, dtype=np.float64).reshape(5, 3) + endpoint_offset
    return RobotState(
        sequence=sequence,
        timestamp_ns=time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
        torso_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        torso_ang_vel=np.array([1.0, 2.0, 3.0]),
        joint_pos=np.asarray(DEFAULT_DOF_POS),
        joint_vel=np.ones(ACTION_DIM),
        end_effector_pos_torso=endpoints,
    )


def make_task(sequence=1, timestamp_ns=None, success=False):
    return TaskState(
        sequence=sequence,
        timestamp_ns=time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
        box_pos_torso=np.array([1.0, 2.0, 3.0]),
        box_quat_torso_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        box_size=np.array([0.3, 0.3, 0.25]),
        goal_pos_torso=np.array([4.0, 5.0, 6.0]),
        success=success,
    )


def test_config_and_urdf_mapping():
    cfg = load_deploy_config(CONFIG_PATH)
    robot = RobotDescription.from_urdf(cfg.urdf_path)
    assert robot.joint_names[0] == "left_hip_pitch_joint"
    assert robot.joint_names[-1] == "right_wrist_yaw_joint"
    assert robot.effort_limits.shape == (29,)
    assert robot.effort_limits[1] == 139.0
    assert robot.effort_limits[4] == 35.0
    assert robot.effort_limits[-1] == 5.0


def test_quaternion_convention_and_rotation_6d():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    vector = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(quat_rotate_wxyz(identity, vector), vector)
    np.testing.assert_allclose(quat_rotate_inverse_wxyz(identity, vector), vector)
    np.testing.assert_allclose(
        quat_to_tan_norm_wxyz(identity),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_observation_shape_order_history_and_legacy_ankle_delay():
    robot0 = make_robot(endpoint_offset=0.0)
    robot1 = make_robot(sequence=2, endpoint_offset=20.0)
    task = make_task()
    builder = ObservationBuilder(legacy_ankle_delay_steps=1)
    builder.reset(robot0)
    obs0 = builder.append(robot0, task)
    assert obs0.shape == (1, ACTOR_OBS_DIM)
    np.testing.assert_array_equal(obs0[0, :-FRAME_OBS_DIM], 0.0)
    frame0 = obs0[0, -FRAME_OBS_DIM:]
    slices0 = builder.frame_slices(frame0)
    np.testing.assert_allclose(slices0["base_ang_vel"], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(slices0["projected_gravity"], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(slices0["dof_pos"], 0.0)
    np.testing.assert_allclose(slices0["dof_vel"], 0.05)
    np.testing.assert_allclose(
        slices0["task"][3:9], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )

    builder.set_previous_action(np.ones(ACTION_DIM) * 2.0)
    obs1 = builder.append(robot1, task)
    frame1 = obs1[0, -FRAME_OBS_DIM:]
    slices1 = builder.frame_slices(frame1)
    endpoints = slices1["end_effector_pos"].reshape(5, 3)
    np.testing.assert_allclose(endpoints[0:2], robot1.end_effector_pos_torso[0:2])
    np.testing.assert_allclose(endpoints[2:4], robot0.end_effector_pos_torso[2:4])
    np.testing.assert_allclose(endpoints[4], robot1.end_effector_pos_torso[4])
    np.testing.assert_allclose(slices1["previous_action"], 2.0)
    assert OBSERVATION_SLICES["task"] == (108, 123)


def test_success_masks_complete_task_observation():
    builder = ObservationBuilder()
    task_obs = builder.build_task_observation(make_task(success=True))
    np.testing.assert_array_equal(task_obs, -1.0)


def test_pd_control_matches_training_formula_and_limits():
    cfg = load_deploy_config(CONFIG_PATH)
    robot = RobotDescription.from_urdf(cfg.urdf_path)
    controller = PDController(robot)
    raw = np.ones(ACTION_DIM)
    clipped, target = controller.action_to_target(raw)
    np.testing.assert_allclose(clipped, 1.0)
    np.testing.assert_allclose(target, np.asarray(DEFAULT_DOF_POS) + 0.25)
    torque = controller.compute_torque(
        target, np.asarray(DEFAULT_DOF_POS), np.zeros(ACTION_DIM)
    )
    expected = np.minimum(np.asarray(KP) * 0.25, robot.effort_limits)
    np.testing.assert_allclose(torque, expected)


def test_udp_packet_roundtrips():
    robot = make_robot()
    task = make_task()
    cfg = load_deploy_config(CONFIG_PATH)
    description = RobotDescription.from_urdf(cfg.urdf_path)
    command = PDController(description).policy_command(
        np.linspace(-1.0, 1.0, ACTION_DIM),
        sequence=3,
        armed=True,
        reason="test",
        hardware_safe=False,
    )
    robot2 = unpack_robot_state(pack_robot_state(robot))
    task2 = unpack_task_state(pack_task_state(task))
    command2 = unpack_policy_command(pack_policy_command(command))
    np.testing.assert_allclose(robot2.joint_pos, robot.joint_pos, atol=1e-7)
    np.testing.assert_allclose(task2.box_size, task.box_size, atol=1e-7)
    np.testing.assert_allclose(command2.q_target, command.q_target, atol=1e-7)
    assert command2.armed


def test_safety_gate_is_fail_closed():
    cfg = load_deploy_config(CONFIG_PATH)
    description = RobotDescription.from_urdf(cfg.urdf_path)
    gate = SafetyGate(description)
    robot = make_robot()
    task = make_task()
    assert gate.evaluate(robot, task, armed=True, dry_run=False).allowed
    assert not gate.evaluate(robot, task, armed=False, dry_run=False).allowed
    assert not gate.evaluate(robot, task, armed=True, dry_run=True).allowed
    stale = make_task(timestamp_ns=time.monotonic_ns() - int(1e9))
    assert gate.evaluate(robot, stale, armed=True, dry_run=False).reason == "task state stale"
    invalid_quat = make_task()
    invalid_quat.box_quat_torso_wxyz[:] = 0.0
    assert (
        gate.evaluate(robot, invalid_quat, armed=True, dry_run=False).reason
        == "invalid task quaternion norm 0.000000"
    )
    invalid_robot = make_robot()
    invalid_robot.joint_pos[0] = np.nan
    assert (
        gate.evaluate(invalid_robot, task, armed=True, dry_run=False).reason
        == "robot state is non-finite"
    )
    gate.trigger_estop()
    assert (
        gate.evaluate(robot, task, armed=True, dry_run=False).reason
        == "emergency stop latched"
    )


def test_default_checkpoint_actor_contract():
    cfg = load_deploy_config(CONFIG_PATH)
    if not cfg.checkpoint_path.exists():
        pytest.skip("default checkpoint is not present")
    actor = load_actor_from_checkpoint(cfg.checkpoint_path)
    import torch

    with torch.inference_mode():
        output = actor(torch.zeros(2, ACTOR_OBS_DIM))
    assert tuple(output.shape) == (2, ACTION_DIM)
    assert torch.isfinite(output).all()


def test_exported_onnx_matches_checkpoint():
    cfg = load_deploy_config(CONFIG_PATH)
    if not cfg.checkpoint_path.exists() or not cfg.onnx_path.exists():
        pytest.skip("checkpoint or exported ONNX is not present")
    actor = load_actor_from_checkpoint(cfg.checkpoint_path)
    onnx_actor = OnnxActor(cfg.onnx_path)
    import torch

    sample = np.random.default_rng(73500).standard_normal(
        (8, ACTOR_OBS_DIM)
    ).astype(np.float32)
    with torch.inference_mode():
        expected = actor(torch.from_numpy(sample)).numpy()
    actual = onnx_actor(sample)
    assert np.max(np.abs(expected - actual)) < 1e-5


@pytest.mark.parametrize(
    ("actor_output", "timeout_ms", "expected_reason"),
    [
        (np.full((1, ACTION_DIM), np.nan), 1000.0, "non-finite"),
        (np.zeros((1, ACTION_DIM)), -1.0, "inference timeout"),
    ],
)
def test_policy_core_falls_back_on_bad_inference(
    actor_output, timeout_ms, expected_reason
):
    cfg = load_deploy_config(CONFIG_PATH)
    robot_description = RobotDescription.from_urdf(cfg.urdf_path)
    core = PolicyCore(
        lambda _: actor_output,
        ObservationBuilder(),
        PDController(robot_description),
        max_inference_time_ms=timeout_ms,
    )
    step = core.step(
        make_robot(),
        make_task(),
        command_allowed=True,
        reason="safe",
        hardware_safe=True,
    )
    assert not step.command.armed
    assert expected_reason in step.command.reason
    np.testing.assert_allclose(step.command.kp, 0.0)


def test_policy_core_never_echoes_invalid_joint_state_to_hold_command():
    cfg = load_deploy_config(CONFIG_PATH)
    robot_description = RobotDescription.from_urdf(cfg.urdf_path)
    core = PolicyCore(
        lambda _: np.zeros((1, ACTION_DIM)),
        ObservationBuilder(),
        PDController(robot_description),
    )
    invalid = make_robot()
    invalid.joint_pos[0] = np.nan
    step = core.step(
        invalid,
        make_task(),
        command_allowed=False,
        reason="robot state is non-finite",
        hardware_safe=True,
    )
    assert step.command.is_finite()
    np.testing.assert_allclose(step.command.q_target, DEFAULT_DOF_POS)


def test_mujoco_name_mapping_and_ten_second_smoke():
    pytest.importorskip("mujoco")
    from deploy.sim2sim.mujoco_server import MujocoServer

    cfg = load_deploy_config(CONFIG_PATH)
    server = MujocoServer(cfg, transport="udp", viewer=False)
    try:
        server.physics_started = True
        previous_time = server.data.time
        for _ in range(2000):
            server.last_command.timestamp_ns = time.monotonic_ns()
            server.step()
            assert server.data.time > previous_time
            previous_time = server.data.time
            assert np.isfinite(server.data.qpos).all()
            assert np.isfinite(server.data.qvel).all()
        assert server.data.time == pytest.approx(10.0)
        assert server.name_map.joint_qpos_adr.shape == (ACTION_DIM,)
        assert server.name_map.joint_dof_adr.shape == (ACTION_DIM,)
        assert server.name_map.actuator_ids.shape == (ACTION_DIM,)
    finally:
        server.close()
