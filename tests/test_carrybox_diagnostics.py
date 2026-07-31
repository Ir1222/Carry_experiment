from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deploy.common.config import load_deploy_config
from deploy.common.grasp_diagnostics import (
    GraspTracker,
    obb_bottom_height,
    point_to_obb_signed_distance,
)
from deploy.common.snapshot import CarryBoxSnapshot
from deploy.sim2sim.collision_profiles import (
    COLLISION_PROFILE_NAMES,
    CollisionProfile,
    should_filter_collision,
)
from deploy.sim2sim.mujoco_server import isaac_sector_goal
from deploy.tools.compare_actor_snapshot import compare_actor_snapshot


CONFIG_PATH = Path("deploy/config/g1_carrybox.yaml")
IDENTITY = np.asarray((1.0, 0.0, 0.0, 0.0))


def _contact(body1: str, body2: str, force: float = 2.0):
    return {
        "body1": body1,
        "body2": body2,
        "force_contact_frame": [force, 0.0, 0.0],
    }


def test_point_to_obb_signed_distance_and_bottom_height():
    size = np.asarray((2.0, 4.0, 6.0))
    assert point_to_obb_signed_distance(
        np.asarray((2.0, 0.0, 0.0)), np.zeros(3), IDENTITY, size
    ) == pytest.approx(1.0)
    assert point_to_obb_signed_distance(
        np.asarray((0.0, 0.0, 0.0)), np.zeros(3), IDENTITY, size
    ) == pytest.approx(-1.0)
    assert obb_bottom_height(
        np.asarray((0.0, 0.0, 4.0)), IDENTITY, size
    ) == pytest.approx(1.0)
    yaw_90 = np.asarray((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)))
    assert point_to_obb_signed_distance(
        np.asarray((0.0, 2.5, 0.0)), np.zeros(3), yaw_90, size
    ) == pytest.approx(1.5)


def test_grasp_tracker_requires_bimanual_dwell_clearance_and_no_substitute():
    tracker = GraspTracker()
    hands = (
        _contact("left_rubber_hand", "carry_box"),
        _contact("right_rubber_hand", "carry_box"),
    )
    for sim_time in (0.0, 0.1, 0.2):
        tracker.update(
            sim_time=sim_time,
            contacts=hands,
            box_clearance=0.06,
            left_palm_signed_distance=0.01,
            right_palm_signed_distance=0.02,
        )
    assert tracker.summary()["grasp_success"] is True
    assert tracker.summary()["max_bimanual_contact_duration"] == pytest.approx(
        0.2
    )

    substituted = GraspTracker()
    substituted.update(
        sim_time=0.0,
        contacts=(_contact("left_hip_yaw_link", "carry_box"),),
        box_clearance=0.0,
        left_palm_signed_distance=0.2,
        right_palm_signed_distance=0.2,
    )
    for sim_time in (0.1, 0.2, 0.3):
        substituted.update(
            sim_time=sim_time,
            contacts=hands,
            box_clearance=0.06,
            left_palm_signed_distance=0.0,
            right_palm_signed_distance=0.0,
        )
    assert substituted.summary()["grasp_success"] is False
    assert substituted.summary()["hip_or_torso_box_contact_steps"] == 1


def test_collision_profiles_filter_only_selected_robot_pairs():
    robot_ids = frozenset((1, 2, 3))
    excluded = frozenset((frozenset((1, 2)),))
    assert should_filter_collision(
        1,
        2,
        robot_body_ids=robot_ids,
        disable_robot_self=False,
        excluded_body_id_pairs=excluded,
    )
    assert not should_filter_collision(
        1,
        4,
        robot_body_ids=robot_ids,
        disable_robot_self=True,
        excluded_body_id_pairs=frozenset(),
    )
    assert should_filter_collision(
        1,
        3,
        robot_body_ids=robot_ids,
        disable_robot_self=True,
        excluded_body_id_pairs=frozenset(),
    )
    cfg = load_deploy_config(CONFIG_PATH)
    assert tuple(cfg.section("simulation")["collision"]["profiles"]) == (
        COLLISION_PROFILE_NAMES
    )
    for name in COLLISION_PROFILE_NAMES:
        assert (
            CollisionProfile.from_simulation_config(
                cfg.section("simulation"), name
            ).name
            == name
        )


def test_isaac_goal_sector_points_back_toward_robot():
    goal = isaac_sector_goal(
        robot_position=np.asarray((0.0, 0.0, 0.8)),
        box_position=np.asarray((1.75, 0.0, 0.335)),
        box_size=np.asarray((0.3, 0.3, 0.25)),
        distance=1.0,
        bearing_degrees=30.0,
        sampled_height=0.0,
        platform_height=0.02,
    )
    assert goal[0] < 1.75
    assert goal[1] < 0.0
    assert goal[2] == pytest.approx(0.145)


def test_snapshot_loader_validates_and_defaults_velocities(tmp_path):
    path = tmp_path / "snapshot.npz"
    np.savez(
        path,
        root_position_world=np.zeros(3),
        root_quat_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        joint_pos=np.zeros(29),
        joint_vel=np.zeros(29),
        end_effector_pos_policy_frame=np.zeros(15),
        box_position_world=np.asarray((1.0, 0.0, 0.2)),
        box_quat_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        box_size=np.asarray((0.3, 0.3, 0.25)),
        box_mass=np.asarray(1.125),
        goal_position_world=np.asarray((-1.0, 0.5, 0.2)),
        actor_obs=np.zeros(738),
        current_frame=np.zeros(123),
        previous_action=np.zeros(29),
        snapshot_phase=np.asarray("pickUp"),
    )
    snapshot = CarryBoxSnapshot.load(path)
    np.testing.assert_array_equal(snapshot.root_quaternion_wxyz, IDENTITY)
    np.testing.assert_array_equal(snapshot.root_linear_velocity_world, 0.0)
    assert snapshot.phase == "pickUp"
    assert snapshot.box_density == pytest.approx(50.0)


def test_all_three_actor_profiles_match_onnx_on_same_observation(tmp_path):
    path = tmp_path / "actor_snapshot.npz"
    np.savez(
        path,
        root_position_world=np.zeros(3),
        root_quat_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        joint_pos=np.zeros(29),
        joint_vel=np.zeros(29),
        end_effector_pos_policy_frame=np.zeros(15),
        box_position_world=np.asarray((1.0, 0.0, 0.2)),
        box_quat_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        box_size=np.asarray((0.3, 0.3, 0.25)),
        goal_position_world=np.asarray((-1.0, 0.5, 0.2)),
        actor_obs=np.linspace(-1.0, 1.0, 738, dtype=np.float32),
        current_frame=np.zeros(123),
        previous_action=np.zeros(29),
    )
    result = compare_actor_snapshot(path, CONFIG_PATH)
    assert tuple(result["profiles"]) == (
        "official_carrybox_65000",
        "model_55500",
        "model_73500",
    )
    for profile in result["profiles"].values():
        assert profile["torch_onnx_max_abs_error"] < 1e-5
