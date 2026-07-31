"""PhysHSI CarryBox MuJoCo server for DDS or local UDP sim2sim."""

from __future__ import annotations

import argparse
from collections import deque
import json
import time

import numpy as np

from deploy.common.camera import (
    CameraIntrinsics,
    project_box_to_camera,
)
from deploy.common.config import load_deploy_config
from deploy.common.constants import DEFAULT_DOF_POS, KD, KP
from deploy.common.grasp_diagnostics import (
    GraspTracker,
    obb_bottom_height,
    point_to_obb_signed_distance,
)
from deploy.common.kinematics import (
    MujocoNameMap,
    task_state_from_mujoco,
)
from deploy.common.jsonl import JsonlRecorder
from deploy.common.mapping import RobotDescription
from deploy.common.math_utils import quat_rotate_inverse_wxyz
from deploy.common.snapshot import CarryBoxSnapshot
from deploy.common.transport import (
    UdpLatestReceiver,
    UdpPublisher,
    pack_robot_state,
    pack_task_state,
    unpack_policy_command,
)
from deploy.common.types import PolicyCommand
from deploy.sim2sim.randomization import (
    PROFILE_NAMES,
    ScenarioSample,
    ScenarioSampler,
)
from deploy.sim2sim.collision_profiles import (
    COLLISION_PROFILE_NAMES,
    CollisionProfile,
    should_filter_collision,
)
from deploy.tools.build_mjcf import build_robot_mjcf


GOAL_PROFILE_NAMES = ("configured", "isaac_sector")
SOURCE_PLATFORM_PROFILE_NAMES = ("configured", "enabled", "disabled")


def isaac_sector_goal(
    *,
    robot_position: np.ndarray,
    box_position: np.ndarray,
    box_size: np.ndarray,
    distance: float,
    bearing_degrees: float,
    sampled_height: float,
    platform_height: float,
) -> np.ndarray:
    """Apply CarryBox `_reset_task()`'s default box-to-robot goal geometry."""

    robot = np.asarray(robot_position, dtype=np.float64)
    box = np.asarray(box_position, dtype=np.float64)
    size = np.asarray(box_size, dtype=np.float64)
    if robot.shape != (3,) or box.shape != (3,) or size.shape != (3,):
        raise ValueError("robot_position, box_position, and box_size must be 3-D")
    if distance < 0.0 or not 10.0 <= abs(float(bearing_degrees)) <= 80.0:
        raise ValueError("invalid Isaac goal distance or bearing")
    direction_to_robot = robot[:2] - box[:2]
    base_angle = float(
        np.arctan2(direction_to_robot[1], direction_to_robot[0])
    )
    angle = base_angle + np.deg2rad(float(bearing_degrees))
    goal_z = max(
        float(sampled_height), 0.5 * float(size[2]) + float(platform_height)
    )
    return np.asarray(
        (
            box[0] + float(distance) * np.cos(angle),
            box[1] + float(distance) * np.sin(angle),
            goal_z,
        ),
        dtype=np.float64,
    )


def _require_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required; install deploy/requirements.txt"
        ) from exc
    return mujoco


def source_platform_center(
    box_position: np.ndarray,
    box_size: np.ndarray,
    platform_size: np.ndarray,
    box_gap: float,
) -> np.ndarray:
    """Place the fixed platform below a box using the Isaac default-reset gap."""

    box_position = np.asarray(box_position, dtype=np.float64)
    box_size = np.asarray(box_size, dtype=np.float64)
    platform_size = np.asarray(platform_size, dtype=np.float64)
    if box_position.shape != (3,):
        raise ValueError("box_position must have shape (3,)")
    if box_size.shape != (3,) or np.any(box_size <= 0.0):
        raise ValueError("box_size must contain three positive values")
    if platform_size.shape != (3,) or np.any(platform_size <= 0.0):
        raise ValueError("platform_size must contain three positive values")
    if not np.isfinite(box_gap) or box_gap < 0.0:
        raise ValueError("box_gap must be finite and non-negative")
    center = box_position.copy()
    center[2] -= 0.5 * box_size[2] + float(box_gap) + 0.5 * platform_size[2]
    return center


class MujocoServer:
    def __init__(
        self,
        config,
        *,
        transport: str,
        viewer: bool,
        camera_view: str = "free",
        log_path=None,
        randomization_profile: str = "nominal",
        seed: int = 0,
        scenario_file: str | None = None,
        snapshot_file: str | None = None,
        collision_profile: str | None = None,
        action_trace_file: str | None = None,
        goal_profile: str = "configured",
        source_platform_profile: str = "configured",
        box_size_scale: tuple[float, float, float] | None = None,
    ) -> None:
        self.cfg = config
        self.sim_cfg = config.section("simulation")
        self.control_cfg = config.section("control")
        self.network_cfg = config.section("network")
        self.robot_cfg = config.section("robot")
        self.camera_cfg = config.section("camera")
        self.mujoco = _require_mujoco()
        if goal_profile not in GOAL_PROFILE_NAMES:
            raise ValueError(
                f"goal_profile must be one of {GOAL_PROFILE_NAMES}"
            )
        self.goal_profile = goal_profile
        if source_platform_profile not in SOURCE_PLATFORM_PROFILE_NAMES:
            raise ValueError(
                "source_platform_profile must be one of "
                f"{SOURCE_PLATFORM_PROFILE_NAMES}"
            )
        self.source_platform_profile = source_platform_profile
        self.box_size_scale = (
            None
            if box_size_scale is None
            else np.asarray(box_size_scale, dtype=np.float64)
        )
        if self.box_size_scale is not None and (
            self.box_size_scale.shape != (3,)
            or not np.isfinite(self.box_size_scale).all()
            or np.any(self.box_size_scale <= 0.0)
        ):
            raise ValueError("box_size_scale must contain three positive values")
        generated = config.resolve_path(self.sim_cfg["generated_robot_mjcf"])
        if not generated.exists():
            print(f"Generated robot MJCF not found; building {generated}")
            build_robot_mjcf(
                config.urdf_path,
                generated,
                joint_armature=float(self.sim_cfg.get("joint_armature", 0.01)),
                camera_config=self.camera_cfg,
            )
        self.model = self.mujoco.MjModel.from_xml_path(str(config.mjcf_path))
        self.data = self.mujoco.MjData(self.model)
        self.model.opt.timestep = float(self.sim_cfg["physics_dt"])
        self.robot = RobotDescription.from_urdf(config.urdf_path)
        self.name_map = MujocoNameMap(
            self.model,
            self.robot,
            pelvis_body=self.robot_cfg["base_body"],
            torso_body=self.robot_cfg["torso_body"],
            policy_frame_body=self.robot_cfg["policy_frame"],
            end_effector_names=tuple(self.robot_cfg["end_effectors"]),
        )
        if self.name_map.actuator_ids is None:
            raise RuntimeError("generated MJCF does not contain all 29 named actuators")
        self.transport_name = transport
        self.viewer_enabled = viewer
        if camera_view not in ("free", "d455"):
            raise ValueError(
                "camera_view must be either 'free' or 'd455'"
            )
        self.camera_view = camera_view
        self._viewer_camera_dirty = True
        # The smoke runner terminates the server process after the policy exits.
        # Flush every policy-boundary record so short evaluations cannot lose
        # all simulator data before the async writer reaches its batch size.
        self.recorder = JsonlRecorder(log_path, flush_every=1)
        self.sequence = 0
        self.reset_requested = False
        self.stop_requested = False
        self._configure_transport()
        self._configure_scene()
        self._configure_camera()
        self.collision_profile = CollisionProfile.from_simulation_config(
            self.sim_cfg, collision_profile
        )
        self._configure_collision_filter()
        self.physics_fingerprint = self._configure_physics_profile()
        self._capture_nominal_model()
        self.scenario_sampler = ScenarioSampler(config)
        self.episode_index = 0
        self.randomization_seed = int(seed)
        self.fixed_scenario = (
            None
            if scenario_file is None
            else ScenarioSample.read(config.resolve_path(scenario_file))
        )
        if self.fixed_scenario is not None:
            self.randomization_profile = self.fixed_scenario.profile
            self.randomization_seed = self.fixed_scenario.seed
        else:
            if randomization_profile not in PROFILE_NAMES:
                raise ValueError(
                    "randomization_profile must be one of "
                    f"{PROFILE_NAMES}, got {randomization_profile!r}"
                )
            self.randomization_profile = randomization_profile
        self.current_scenario = self._scenario_for_episode()
        self.initial_snapshot = (
            None
            if snapshot_file is None
            else CarryBoxSnapshot.load(config.resolve_path(snapshot_file))
        )
        self.open_loop_action_trace_path = (
            None
            if action_trace_file is None
            else config.resolve_path(action_trace_file)
        )
        self.open_loop_commands = self._load_open_loop_commands(
            self.open_loop_action_trace_path
        )
        print(
            "MuJoCo physics profile: "
            + json.dumps(self.physics_fingerprint, sort_keys=True)
        )
        self.recorder.write(
            {
                "kind": "run_metadata",
                "component": "mujoco_server",
                "protocol_version": 2,
                "policy_frame": self.robot_cfg["policy_frame"],
                "physics": self.physics_fingerprint,
                "camera": self.camera_metadata,
                "randomization": {
                    "profile": self.randomization_profile,
                    "seed": self.randomization_seed,
                    "scenario_file": (
                        None
                        if scenario_file is None
                        else str(config.resolve_path(scenario_file))
                    ),
                    "scenario_fingerprint": self.current_scenario.fingerprint,
                    "approximation": self.current_scenario.approximation,
                },
                "collision_profile": self.collision_profile.name,
                "snapshot_file": (
                    None
                    if self.initial_snapshot is None
                    else str(self.initial_snapshot.path)
                ),
                "action_trace_file": (
                    None
                    if self.open_loop_action_trace_path is None
                    else str(self.open_loop_action_trace_path)
                ),
                "control_mode": (
                    "closed_loop_policy"
                    if not self.open_loop_commands
                    else "open_loop_action_replay"
                ),
                "goal_profile": self.goal_profile,
                "source_platform_profile": self.source_platform_profile,
                "box_size_scale": (
                    None
                    if self.box_size_scale is None
                    else self.box_size_scale.tolist()
                ),
            }
        )
        self.reset()

    @staticmethod
    def _address(value) -> tuple[str, int]:
        return str(value[0]), int(value[1])

    def _configure_transport(self) -> None:
        self.task_publisher = UdpPublisher(
            self._address(self.network_cfg["task_state_udp"])
        )
        if self.transport_name == "udp":
            self.robot_publisher = UdpPublisher(
                self._address(self.network_cfg["robot_state_udp"])
            )
            self.command_receiver = UdpLatestReceiver(
                self._address(self.network_cfg["robot_command_udp"]),
                unpack_policy_command,
            )
            self.unitree_bridge = None
        elif self.transport_name == "unitree_dds":
            from .unitree_bridge import UnitreeSimulatorBridge

            self.unitree_bridge = UnitreeSimulatorBridge(
                domain_id=int(self.network_cfg["domain_id"]),
                interface=str(self.network_cfg["interface"]),
                policy_to_motor=self.robot_cfg["policy_to_motor"],
            )
            self.robot_publisher = None
            self.command_receiver = None
        else:
            raise ValueError(f"unsupported transport {self.transport_name!r}")

    def _configure_scene(self) -> None:
        self.box_size = np.asarray(self.sim_cfg["box_size"], dtype=np.float64)
        self.box_initial_position = np.asarray(
            self.sim_cfg["box_initial_position"], dtype=np.float64
        )
        self.goal_position = np.asarray(
            self.sim_cfg["goal_position"], dtype=np.float64
        )
        self.box_body_id = int(self.model.body("carry_box").id)
        self.box_geom_id = int(self.model.geom("carry_box_geom").id)
        self.source_platform_body_id = int(
            self.model.body("source_platform").id
        )
        self.source_platform_geom_id = int(
            self.model.geom("source_platform_geom").id
        )
        self.goal_site_id = int(self.model.site("goal_site").id)
        self.head_body_id = int(self.model.body("mid360_link").id)
        self.left_palm_body_id = int(self.model.body("left_palm_link").id)
        self.right_palm_body_id = int(self.model.body("right_palm_link").id)
        self.left_hand_body_id = int(
            self.model.body("left_rubber_hand").id
        )
        self.right_hand_body_id = int(
            self.model.body("right_rubber_hand").id
        )
        self.left_hand_geom_ids = self._collidable_geoms_for_body(
            self.left_hand_body_id
        )
        self.right_hand_geom_ids = self._collidable_geoms_for_body(
            self.right_hand_body_id
        )
        if not self.left_hand_geom_ids or not self.right_hand_geom_ids:
            raise RuntimeError(
                "generated MJCF must contain collidable left/right rubber-hand "
                "geometries"
            )
        self.hip_yaw_body_ids = np.asarray(
            [
                int(self.model.body("left_hip_yaw_link").id),
                int(self.model.body("right_hip_yaw_link").id),
            ],
            dtype=np.int32,
        )
        self.model.geom_size[self.box_geom_id, :3] = self.box_size / 2.0
        density = float(self.sim_cfg["box_density"])
        mass = density * float(np.prod(self.box_size))
        x, y, z = self.box_size
        self.model.body_mass[self.box_body_id] = mass
        self.model.body_inertia[self.box_body_id] = (
            mass
            / 12.0
            * np.asarray((y * y + z * z, x * x + z * z, x * x + y * y))
        )
        source_platform_cfg = self.sim_cfg["source_platform"]
        configured_platform_enabled = bool(source_platform_cfg["enabled"])
        self.source_platform_enabled = (
            configured_platform_enabled
            if self.source_platform_profile == "configured"
            else self.source_platform_profile == "enabled"
        )
        self.source_platform_size = np.asarray(
            source_platform_cfg["size"], dtype=np.float64
        )
        self.source_platform_box_gap = float(
            source_platform_cfg["box_gap"]
        )
        self.source_platform_friction = np.asarray(
            source_platform_cfg["friction"], dtype=np.float64
        )
        self.source_platform_position = source_platform_center(
            self.box_initial_position,
            self.box_size,
            self.source_platform_size,
            self.source_platform_box_gap,
        )
        self.model.geom_size[
            self.source_platform_geom_id, :3
        ] = self.source_platform_size / 2.0
        self.model.geom_friction[
            self.source_platform_geom_id, :3
        ] = self.source_platform_friction
        if self.source_platform_enabled:
            self.model.body_pos[
                self.source_platform_body_id
            ] = self.source_platform_position
            self.model.geom_contype[self.source_platform_geom_id] = 1
            self.model.geom_conaffinity[self.source_platform_geom_id] = 1
        else:
            self.model.body_pos[self.source_platform_body_id] = np.asarray(
                (
                    self.source_platform_position[0],
                    self.source_platform_position[1],
                    -5.0,
                ),
                dtype=np.float64,
            )
            self.model.geom_contype[self.source_platform_geom_id] = 0
            self.model.geom_conaffinity[self.source_platform_geom_id] = 0
        self.model.site_pos[self.goal_site_id] = self.goal_position

    def _collidable_geoms_for_body(self, body_id: int) -> tuple[int, ...]:
        return tuple(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == int(body_id)
            and (
                int(self.model.geom_contype[geom_id]) != 0
                or int(self.model.geom_conaffinity[geom_id]) != 0
            )
        )

    def _configure_collision_filter(self) -> None:
        external_body_ids = frozenset(
            (0, self.box_body_id, self.source_platform_body_id)
        )
        self.robot_body_ids = frozenset(
            body_id
            for body_id in range(1, self.model.nbody)
            if body_id not in external_body_ids
        )
        excluded: set[frozenset[int]] = set()
        for body_a, body_b in self.collision_profile.excluded_body_pairs:
            try:
                excluded.add(
                    frozenset(
                        (
                            int(self.model.body(body_a).id),
                            int(self.model.body(body_b).id),
                        )
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"collision profile {self.collision_profile.name!r} "
                    f"references an unknown body: {(body_a, body_b)}"
                ) from exc
        self.excluded_body_id_pairs = frozenset(excluded)
        self._collision_filter_callback = None
        if (
            self.collision_profile.disable_robot_self
            or self.excluded_body_id_pairs
        ):
            if not hasattr(self.mujoco, "set_mjcb_contactfilter"):
                raise RuntimeError(
                    "selected collision profile requires a MuJoCo build with "
                    "set_mjcb_contactfilter"
                )

            def contact_filter(model, data, geom1, geom2):
                del data
                body1 = int(model.geom_bodyid[int(geom1)])
                body2 = int(model.geom_bodyid[int(geom2)])
                return int(
                    should_filter_collision(
                        body1,
                        body2,
                        robot_body_ids=self.robot_body_ids,
                        disable_robot_self=(
                            self.collision_profile.disable_robot_self
                        ),
                        excluded_body_id_pairs=(
                            self.excluded_body_id_pairs
                        ),
                    )
                )

            self._collision_filter_callback = contact_filter
            self.mujoco.set_mjcb_contactfilter(contact_filter)

    def _configure_camera(self) -> None:
        camera_name = str(self.camera_cfg["name"])
        camera_body_name = str(self.camera_cfg["body"])
        try:
            self.camera_id = int(self.model.camera(camera_name).id)
            self.camera_body_id = int(
                self.model.body(camera_body_name).id
            )
        except KeyError as exc:
            raise RuntimeError(
                "generated MJCF does not contain the configured D455 "
                "camera; run `python -m deploy.tools.build_mjcf`"
            ) from exc
        actual_body_id = int(self.model.cam_bodyid[self.camera_id])
        if actual_body_id != self.camera_body_id:
            actual_body = self.model.body(actual_body_id).name
            raise RuntimeError(
                f"camera {camera_name!r} belongs to {actual_body!r}, "
                f"expected {camera_body_name!r}"
            )
        self.camera_intrinsics = CameraIntrinsics.from_config(
            self.camera_cfg
        )
        self.camera_metadata = {
            "name": camera_name,
            "body": camera_body_name,
            "position": [
                float(value) for value in self.camera_cfg["position"]
            ],
            "quaternion_wxyz": [
                float(value)
                for value in self.camera_cfg["quaternion_wxyz"]
            ],
            **self.camera_intrinsics.to_dict(),
        }

    def _validate_camera_axes(self) -> None:
        camera_rotation = self.data.cam_xmat[
            self.camera_id
        ].reshape(3, 3)
        optical_rotation = self.data.xmat[
            self.camera_body_id
        ].reshape(3, 3)
        camera_right = camera_rotation[:, 0]
        camera_up = camera_rotation[:, 1]
        camera_forward = -camera_rotation[:, 2]
        expected_right = optical_rotation[:, 0]
        expected_up = -optical_rotation[:, 1]
        expected_forward = optical_rotation[:, 2]
        if not (
            np.allclose(camera_right, expected_right, atol=1e-6)
            and np.allclose(camera_up, expected_up, atol=1e-6)
            and np.allclose(
                camera_forward, expected_forward, atol=1e-6
            )
        ):
            raise RuntimeError(
                "D455 camera axes do not match RealSense optical "
                "frame (+Z forward, +X right, +Y down)"
            )

    def box_camera_projection(self) -> dict[str, object]:
        return project_box_to_camera(
            camera_position_world=self.data.cam_xpos[
                self.camera_id
            ],
            camera_rotation_to_world=self.data.cam_xmat[
                self.camera_id
            ].reshape(3, 3),
            box_position_world=self.data.xpos[self.box_body_id],
            box_rotation_to_world=self.data.xmat[
                self.box_body_id
            ].reshape(3, 3),
            box_half_size=self.box_size / 2.0,
            intrinsics=self.camera_intrinsics,
        )

    def _configure_physics_profile(self) -> dict:
        contact_margin = float(self.sim_cfg.get("contact_margin", 0.0))
        collision_mask = (self.model.geom_contype != 0) | (
            self.model.geom_conaffinity != 0
        )
        external_body_ids = {
            0,
            self.box_body_id,
            self.source_platform_body_id,
        }
        robot_geom_mask = np.asarray(
            [
                bool(collision_mask[geom_id])
                and int(self.model.geom_bodyid[geom_id])
                not in external_body_ids
                for geom_id in range(self.model.ngeom)
            ],
            dtype=bool,
        )
        external_geom_mask = collision_mask & ~robot_geom_mask
        self.model.geom_margin[robot_geom_mask] = (
            self.collision_profile.robot_margin
        )
        self.model.geom_margin[external_geom_mask] = (
            self.collision_profile.external_margin
        )

        solref = np.asarray(
            self.sim_cfg.get("joint_limit_solref", (0.005, 1.0)),
            dtype=np.float64,
        )
        limited = self.model.jnt_limited.astype(bool)
        self.model.jnt_solref[limited, :2] = solref
        solimp = np.asarray(
            self.sim_cfg.get(
                "joint_limit_solimp",
                (0.9, 0.95, 0.001, 0.5, 2.0),
            ),
            dtype=np.float64,
        )
        self.model.jnt_solimp[limited, :5] = solimp
        disable_refsafe = bool(
            self.sim_cfg.get("disable_refsafe_for_joint_limits", False)
        )
        if disable_refsafe:
            self.model.opt.disableflags |= int(
                self.mujoco.mjtDisableBit.mjDSBL_REFSAFE
            )

        base_joint_id = int(self.model.joint("floating_base_joint").id)
        base_dof_adr = int(self.model.jnt_dofadr[base_joint_id])
        linear_damping = float(
            self.sim_cfg.get("body_linear_damping", 0.0)
        )
        angular_damping = float(
            self.sim_cfg.get("body_angular_damping", 0.0)
        )
        self.model.dof_damping[base_dof_adr : base_dof_adr + 3] = (
            linear_damping
        )
        self.model.dof_damping[base_dof_adr + 3 : base_dof_adr + 6] = (
            angular_damping
        )
        return {
            "timestep": float(self.model.opt.timestep),
            "integrator": int(self.model.opt.integrator),
            "solver": int(self.model.opt.solver),
            "iterations": int(self.model.opt.iterations),
            "contact_margin": contact_margin,
            "collision_profile": self.collision_profile.name,
            "robot_collision_margin": (
                self.collision_profile.robot_margin
            ),
            "external_collision_margin": (
                self.collision_profile.external_margin
            ),
            "robot_self_collision_disabled": (
                self.collision_profile.disable_robot_self
            ),
            "excluded_body_pairs": [
                list(pair)
                for pair in self.collision_profile.excluded_body_pairs
            ],
            "collision_geom_count": int(np.count_nonzero(collision_mask)),
            "joint_limit_solref": solref.tolist(),
            "joint_limit_solimp": solimp.tolist(),
            "refsafe_disabled": disable_refsafe,
            "limited_joint_count": int(np.count_nonzero(limited)),
            "body_linear_damping": linear_damping,
            "body_angular_damping": angular_damping,
            "ground_friction": self.model.geom_friction[
                int(self.model.geom("floor").id)
            ].tolist(),
            "box_mass": float(self.model.body_mass[self.box_body_id]),
            "box_inertia": self.model.body_inertia[
                self.box_body_id
            ].tolist(),
            "source_platform": {
                "enabled": self.source_platform_enabled,
                "size": self.source_platform_size.tolist(),
                "position": self.source_platform_position.tolist(),
                "box_gap": self.source_platform_box_gap,
                "friction": self.source_platform_friction.tolist(),
            },
        }

    def _capture_nominal_model(self) -> None:
        self._nominal_model = {
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "body_ipos": self.model.body_ipos.copy(),
            "body_pos": self.model.body_pos.copy(),
            "geom_size": self.model.geom_size.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "site_pos": self.model.site_pos.copy(),
        }

    def _restore_nominal_model(self) -> None:
        for name, value in self._nominal_model.items():
            getattr(self.model, name)[:] = value

    def _scenario_for_episode(self) -> ScenarioSample:
        if self.fixed_scenario is not None:
            return self.fixed_scenario
        return self.scenario_sampler.sample(
            self.randomization_profile,
            self.randomization_seed,
            self.episode_index,
        )

    def _apply_scenario_to_model(self) -> dict:
        scenario = self.current_scenario
        self._restore_nominal_model()
        self.box_size = np.asarray(scenario.box_size, dtype=np.float64)
        self.box_initial_position = np.asarray(
            scenario.box_position, dtype=np.float64
        )
        self.goal_position = np.asarray(scenario.goal_position, dtype=np.float64)
        if self.goal_profile == "isaac_sector":
            goal_rng = np.random.default_rng(
                np.random.SeedSequence(
                    (int(scenario.seed), int(self.episode_index), 0xCABB0)
                )
            )
            signed_bearing = float(goal_rng.uniform(10.0, 80.0))
            signed_bearing *= float(goal_rng.choice((-1.0, 1.0)))
            self.goal_position = isaac_sector_goal(
                robot_position=np.asarray(
                    scenario.robot_position, dtype=np.float64
                ),
                box_position=self.box_initial_position,
                box_size=self.box_size,
                distance=float(goal_rng.uniform(0.6, 4.0)),
                bearing_degrees=signed_bearing,
                sampled_height=float(goal_rng.uniform(0.0, 0.4)),
                platform_height=float(self.source_platform_size[2]),
            )
        self.model.geom_size[self.box_geom_id, :3] = self.box_size / 2.0
        box_mass = float(scenario.box_density) * float(np.prod(self.box_size))
        x, y, z = self.box_size
        self.model.body_mass[self.box_body_id] = box_mass
        self.model.body_inertia[self.box_body_id] = (
            box_mass
            / 12.0
            * np.asarray((y * y + z * z, x * x + z * z, x * x + y * y))
        )

        excluded_bodies = {
            0,
            self.box_body_id,
            self.source_platform_body_id,
        }
        mass_factors = scenario.link_mass_factors(self.model.nbody)
        for body_id in range(1, self.model.nbody):
            if body_id in excluded_bodies:
                continue
            self.model.body_mass[body_id] *= mass_factors[body_id]
            self.model.body_inertia[body_id] *= mass_factors[body_id]
        torso_id = self.name_map.torso_body_id
        torso_mass_before_payload = float(self.model.body_mass[torso_id])
        torso_mass = max(
            0.05,
            torso_mass_before_payload + float(scenario.torso_payload_mass),
        )
        self.model.body_mass[torso_id] = torso_mass
        self.model.body_inertia[torso_id] *= (
            torso_mass / torso_mass_before_payload
        )
        self.model.body_ipos[torso_id] += np.asarray(
            scenario.torso_com_displacement, dtype=np.float64
        )

        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id in excluded_bodies or body_id == 0:
                continue
            if (
                int(self.model.geom_contype[geom_id]) != 0
                or int(self.model.geom_conaffinity[geom_id]) != 0
            ):
                self.model.geom_friction[geom_id, 0] = (
                    scenario.robot_friction
                )

        self.source_platform_position = source_platform_center(
            self.box_initial_position,
            self.box_size,
            self.source_platform_size,
            self.source_platform_box_gap,
        )
        self.model.geom_size[
            self.source_platform_geom_id, :3
        ] = self.source_platform_size / 2.0
        if self.source_platform_enabled:
            self.model.body_pos[
                self.source_platform_body_id
            ] = self.source_platform_position
        else:
            self.model.body_pos[self.source_platform_body_id] = np.asarray(
                (
                    self.source_platform_position[0],
                    self.source_platform_position[1],
                    -5.0,
                )
            )
        self.model.site_pos[self.goal_site_id] = self.goal_position
        self.mujoco.mj_setConst(self.model, self.data)
        return {
            "box_mass": box_mass,
            "box_inertia": self.model.body_inertia[
                self.box_body_id
            ].copy(),
            "torso_mass": torso_mass,
            "torso_payload_effective": (
                torso_mass - torso_mass_before_payload
            ),
            "robot_friction": float(scenario.robot_friction),
            "link_mass_min": float(np.min(mass_factors[1:])),
            "link_mass_max": float(np.max(mass_factors[1:])),
        }

    def _apply_snapshot_to_model(self) -> dict[str, object] | None:
        snapshot = self.initial_snapshot
        if snapshot is None:
            return None
        self.box_size = snapshot.box_size.copy()
        self.box_initial_position = snapshot.box_position_world.copy()
        self.goal_position = snapshot.goal_position_world.copy()
        self.model.geom_size[self.box_geom_id, :3] = self.box_size / 2.0
        mass = (
            float(snapshot.box_mass)
            if snapshot.box_mass is not None
            else float(self.current_scenario.box_density)
            * float(np.prod(self.box_size))
        )
        x, y, z = self.box_size
        self.model.body_mass[self.box_body_id] = mass
        self.model.body_inertia[self.box_body_id] = (
            mass
            / 12.0
            * np.asarray((y * y + z * z, x * x + z * z, x * x + y * y))
        )
        self.source_platform_position = (
            snapshot.platform_position_world.copy()
            if snapshot.platform_position_world is not None
            else source_platform_center(
                self.box_initial_position,
                self.box_size,
                self.source_platform_size,
                self.source_platform_box_gap,
            )
        )
        if self.source_platform_enabled:
            self.model.body_pos[
                self.source_platform_body_id
            ] = self.source_platform_position
        self.model.site_pos[self.goal_site_id] = self.goal_position
        self.mujoco.mj_setConst(self.model, self.data)
        return {
            "path": str(snapshot.path),
            "phase": snapshot.phase,
            "box_mass": mass,
            "box_density": float(mass / np.prod(self.box_size)),
            "platform_position_world": (
                self.source_platform_position.copy()
            ),
        }

    def _apply_box_size_ablation(self) -> dict[str, object] | None:
        if self.box_size_scale is None:
            return None
        original_size = self.box_size.copy()
        density = float(
            self.model.body_mass[self.box_body_id] / np.prod(original_size)
        )
        self.box_size = original_size * self.box_size_scale
        self.model.geom_size[self.box_geom_id, :3] = self.box_size / 2.0
        mass = density * float(np.prod(self.box_size))
        x, y, z = self.box_size
        self.model.body_mass[self.box_body_id] = mass
        self.model.body_inertia[self.box_body_id] = (
            mass
            / 12.0
            * np.asarray((y * y + z * z, x * x + z * z, x * x + y * y))
        )
        self.mujoco.mj_setConst(self.model, self.data)
        return {
            "scale": self.box_size_scale.copy(),
            "original_size": original_size,
            "physical_and_observed_size": self.box_size.copy(),
            "density": density,
            "mass": mass,
        }

    def _free_joint_address(self, name: str) -> int:
        return int(self.model.joint(name).qposadr)

    def _load_open_loop_commands(
        self, path
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        if path is None:
            return ()
        commands = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("kind") != "isaac_carrybox_step":
                    continue
                action = np.asarray(
                    record.get("policy_action"), dtype=np.float64
                ).reshape(-1)
                if action.shape != (29,) or not np.isfinite(action).all():
                    raise ValueError(
                        f"{path}:{line_number} has an invalid policy_action"
                    )
                q_target_value = record.get("q_target")
                if q_target_value is None:
                    q_target = np.asarray(DEFAULT_DOF_POS) + float(
                        self.control_cfg["action_scale"]
                    ) * action
                else:
                    q_target = np.asarray(
                        q_target_value, dtype=np.float64
                    ).reshape(-1)
                    if q_target.shape != (29,) or not np.isfinite(
                        q_target
                    ).all():
                        raise ValueError(
                            f"{path}:{line_number} has an invalid q_target"
                        )
                commands.append((action, q_target))
        if not commands:
            raise ValueError(
                f"action trace contains no isaac_carrybox_step records: {path}"
            )
        return tuple(commands)

    def reset(self, *, advance_episode: bool = False) -> None:
        if advance_episode and self.fixed_scenario is None:
            self.episode_index += 1
            self.current_scenario = self._scenario_for_episode()
        scenario = self.current_scenario
        randomized_physics = self._apply_scenario_to_model()
        snapshot_metadata = self._apply_snapshot_to_model()
        box_size_ablation = self._apply_box_size_ablation()
        self.mujoco.mj_resetData(self.model, self.data)
        base_adr = self._free_joint_address("floating_base_joint")
        snapshot = self.initial_snapshot
        self.data.qpos[base_adr : base_adr + 3] = np.asarray(
            (
                scenario.robot_position
                if snapshot is None
                else snapshot.root_position_world
            ),
            dtype=np.float64,
        )
        self.data.qpos[base_adr + 3 : base_adr + 7] = np.asarray(
            (
                scenario.robot_quaternion_wxyz
                if snapshot is None
                else snapshot.root_quaternion_wxyz
            ),
            dtype=np.float64,
        )
        joint_position = np.clip(
            np.asarray(
                (
                    scenario.joint_position
                    if snapshot is None
                    else snapshot.joint_pos
                ),
                dtype=np.float64,
            ),
            self.robot.lower_limits,
            self.robot.upper_limits,
        )
        self.data.qpos[self.name_map.joint_qpos_adr] = joint_position
        base_joint_id = int(self.model.joint("floating_base_joint").id)
        base_dof_adr = int(self.model.jnt_dofadr[base_joint_id])
        self.data.qvel[base_dof_adr : base_dof_adr + 3] = np.asarray(
            (
                scenario.root_linear_velocity
                if snapshot is None
                else snapshot.root_linear_velocity_world
            ),
            dtype=np.float64,
        )
        self.data.qvel[base_dof_adr + 3 : base_dof_adr + 6] = np.asarray(
            (
                scenario.root_angular_velocity
                if snapshot is None
                else snapshot.root_angular_velocity_world
            ),
            dtype=np.float64,
        )
        if snapshot is not None:
            self.data.qvel[self.name_map.joint_dof_adr] = snapshot.joint_vel
        box_adr = self._free_joint_address("box_free_joint")
        self.data.qpos[box_adr : box_adr + 3] = self.box_initial_position
        self.data.qpos[box_adr + 3 : box_adr + 7] = np.asarray(
            (
                scenario.box_quaternion_wxyz
                if snapshot is None
                else snapshot.box_quaternion_wxyz
            ),
            dtype=np.float64,
        )
        if snapshot is not None:
            box_joint_id = int(self.model.joint("box_free_joint").id)
            box_dof_adr = int(self.model.jnt_dofadr[box_joint_id])
            self.data.qvel[box_dof_adr : box_dof_adr + 3] = (
                snapshot.box_linear_velocity_world
            )
            self.data.qvel[box_dof_adr + 3 : box_dof_adr + 6] = (
                snapshot.box_angular_velocity_world
            )
        self.mujoco.mj_forward(self.model, self.data)
        self._validate_camera_axes()
        self.sequence = 0
        if self.unitree_bridge is not None:
            self.unitree_bridge.clear_command()
        elif self.command_receiver is not None:
            self.command_receiver.poll_latest()
        self.last_command = PolicyCommand(
            sequence=0,
            timestamp_ns=time.monotonic_ns(),
            raw_action=np.zeros(29),
            # Training pre-fills the delay buffer from the reset joint state,
            # so delayed startup holds the randomized pose instead of pulling
            # immediately toward the nominal pose.
            q_target=joint_position.copy(),
            kp=np.asarray(KP),
            kd=np.asarray(KD),
            armed=False,
            reason="simulator default-pose hold",
        )
        self.has_received_command = bool(self.open_loop_commands)
        self.physics_started = bool(self.open_loop_commands)
        self.pending_command: PolicyCommand | None = None
        self._delayed_commands: deque[PolicyCommand] = deque()
        self.control_substep = 0
        self.control_decimation = int(self.control_cfg["decimation"])
        self.last_boundary_wait_ms = 0.0
        self._interval_max_gravity_xy = 0.0
        self._interval_max_joint_violation = 0.0
        self._interval_max_contact_force = 0.0
        self._interval_first_torque = np.zeros(29, dtype=np.float64)
        self._interval_ground_contact_bodies: set[str] = set()
        self.episode_failed = False
        self.episode_failure_reason = ""
        self.episode_failure_sequence: int | None = None
        self.episode_failure_sim_time: float | None = None
        self.policy_step_index = 0
        self.observation_step_index = 0
        self.current_disturbance = np.zeros(3, dtype=np.float64)
        self.grasp_tracker = GraspTracker()
        self.data.xfrc_applied[self.name_map.torso_body_id, :3] = 0.0
        self.recorder.write(
            {
                "kind": "episode_reset",
                "episode_index": self.episode_index,
                "scenario": scenario.to_dict(),
                "randomized_physics": randomized_physics,
                "snapshot": snapshot_metadata,
                "joint_position_clipped": joint_position,
                "goal_profile": self.goal_profile,
                "goal_position_world": self.goal_position.copy(),
                "box_size_ablation": box_size_ablation,
            }
        )
        self.reset_requested = False

    def _poll_command(self) -> None:
        if self.open_loop_commands:
            if self.control_substep != 0 or self.pending_command is not None:
                return
            command_index = self.policy_step_index
            if command_index >= len(self.open_loop_commands):
                self.stop_requested = True
                self.physics_started = False
                return
            action, q_target = self.open_loop_commands[command_index]
            self.pending_command = PolicyCommand(
                sequence=self.sequence,
                timestamp_ns=time.monotonic_ns(),
                raw_action=action,
                q_target=q_target,
                kp=np.asarray(KP),
                kd=np.asarray(KD),
                armed=True,
                reason=f"open-loop Isaac trace step {command_index}",
            )
            return
        command = (
            self.command_receiver.poll_latest()
            if self.command_receiver is not None
            else self.unitree_bridge.poll_command()
        )
        if command is not None and command.is_finite():
            self.has_received_command = True
            if command.armed and self.episode_failed:
                pass
            elif command.armed:
                if not self.physics_started:
                    self.pending_command = command
                    self.control_substep = 0
                    # Paused-state publications may advance while the first
                    # action is in flight. Re-anchor the active physics
                    # sequence to the exact state used by that action so the
                    # next policy boundary is source_sequence + 4.
                    self.sequence = int(command.sequence)
                    self.physics_started = True
                else:
                    self.pending_command = command
            else:
                self.last_command = command
                self.pending_command = None
                self.control_substep = 0
                self.physics_started = False
        newest_command = (
            self.pending_command
            if self.pending_command is not None
            else self.last_command
        )
        age_ns = time.monotonic_ns() - newest_command.timestamp_ns
        if self.has_received_command and age_ns > int(0.2e9):
            joint_pos, _ = self.name_map.joint_state(self.data)
            self.last_command = PolicyCommand(
                sequence=self.last_command.sequence,
                timestamp_ns=time.monotonic_ns(),
                raw_action=np.zeros(29),
                q_target=joint_pos,
                kp=np.zeros(29),
                kd=np.asarray(KD),
                armed=False,
                reason="command timeout damping hold",
            )
            self.pending_command = None
            self.control_substep = 0
            self.physics_started = False

    @staticmethod
    def _roll_pitch_from_wxyz(quaternion: np.ndarray) -> tuple[float, float]:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = np.arcsin(
            np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        )
        return float(roll), float(pitch)

    def _training_termination_reason(
        self,
        robot_state,
        projected_gravity: np.ndarray,
    ) -> str | None:
        gravity_xy = float(np.linalg.norm(projected_gravity[:2]))
        if gravity_xy > float(
            self.cfg.section("safety")["max_projected_gravity_xy"]
        ):
            return (
                "training projected-gravity termination: "
                f"xy={gravity_xy:.6f}"
            )
        head_height = float(self.data.xpos[self.head_body_id, 2])
        if head_height < float(
            self.sim_cfg.get("termination_head_height", 0.6)
        ):
            return (
                "training head-height termination: "
                f"z={head_height:.6f}"
            )
        root_height = float(
            self.data.xpos[self.name_map.pelvis_body_id, 2]
        )
        if root_height < float(
            self.sim_cfg.get("termination_root_height", 0.2)
        ):
            return (
                "training root-height termination: "
                f"z={root_height:.6f}"
            )
        hip_heights = self.data.xpos[self.hip_yaw_body_ids, 2]
        hip_min_index = int(np.argmin(hip_heights))
        hip_min = float(hip_heights[hip_min_index])
        if hip_min < float(
            self.sim_cfg.get("termination_hip_yaw_height", 0.15)
        ):
            body_name = self.model.body(
                int(self.hip_yaw_body_ids[hip_min_index])
            ).name
            return (
                "training hip-height termination: "
                f"body={body_name} z={hip_min:.6f}"
            )
        roll, pitch = self._roll_pitch_from_wxyz(
            robot_state.policy_frame_quat_wxyz
        )
        if abs(roll) > float(
            self.sim_cfg.get("termination_roll_abs", 0.5)
        ):
            return f"training roll termination: roll={roll:.6f}"
        if abs(pitch) > float(
            self.sim_cfg.get("termination_pitch_abs", 1.1)
        ):
            return f"training pitch termination: pitch={pitch:.6f}"
        return None

    def _latch_episode_failure(self, reason: str) -> None:
        if self.episode_failed:
            return
        self.episode_failed = True
        self.episode_failure_reason = reason
        self.episode_failure_sequence = self.sequence
        self.episode_failure_sim_time = float(self.data.time)
        joint_pos, _ = self.name_map.joint_state(self.data)
        self.last_command = PolicyCommand(
            sequence=self.sequence,
            timestamp_ns=time.monotonic_ns(),
            raw_action=np.zeros(29),
            q_target=joint_pos,
            kp=np.zeros(29),
            kd=np.asarray(KD),
            armed=False,
            reason=f"episode failed: {reason}",
        )
        self.pending_command = None
        self._delayed_commands.clear()
        self.control_substep = 0
        self.physics_started = False

    def _apply_command(self) -> np.ndarray:
        joint_pos, joint_vel = self.name_map.joint_state(self.data)
        command = self.last_command
        scenario = self.current_scenario
        pd_torque = (
            command.tau_ff
            + command.kp
            * np.asarray(scenario.kp_factors)
            * (command.q_target - joint_pos)
            - command.kd * np.asarray(scenario.kd_factors) * joint_vel
        )
        torque = (
            pd_torque * np.asarray(scenario.motor_strength)
            + np.asarray(scenario.torque_bias_fraction)
            * self.robot.effort_limits
        )
        torque = np.clip(
            torque, -self.robot.effort_limits, self.robot.effort_limits
        )
        self.data.ctrl[self.name_map.actuator_ids] = torque
        return torque

    def _activate_pending_command(self) -> None:
        if self.pending_command is None:
            return
        self._delayed_commands.append(self.pending_command)
        self.pending_command = None
        delay = int(self.current_scenario.action_delay_steps)
        if len(self._delayed_commands) > delay:
            self.last_command = self._delayed_commands.popleft()

    def _apply_scheduled_disturbance(self) -> None:
        if self.control_substep == 0:
            self.policy_step_index += 1
            self.current_disturbance = self.current_scenario.disturbance_at(
                self.policy_step_index
            )
        self.data.xfrc_applied[self.name_map.torso_body_id, :3] = (
            self.current_disturbance
        )

    def _wait_for_boundary_command(self) -> None:
        """Synchronize the next 4-step block with its boundary observation."""

        if not self.physics_started or self.control_substep != 0:
            self.last_boundary_wait_ms = 0.0
            return
        wait_start = time.perf_counter()
        required_state_sequence = self.sequence
        if (
            self.pending_command is None
            and self.last_command.sequence >= required_state_sequence
        ):
            self.last_boundary_wait_ms = 0.0
            return
        timeout_s = float(
            self.sim_cfg.get("policy_boundary_timeout_ms", 40.0)
        ) / 1000.0
        deadline = time.perf_counter() + timeout_s
        next_republish = time.perf_counter()
        while self.physics_started:
            if (
                self.pending_command is not None
                and self.pending_command.sequence >= required_state_sequence
            ):
                self.last_boundary_wait_ms = (
                    time.perf_counter() - wait_start
                ) * 1000.0
                return
            if time.perf_counter() >= deadline:
                joint_pos, _ = self.name_map.joint_state(self.data)
                self.last_command = PolicyCommand(
                    sequence=required_state_sequence,
                    timestamp_ns=time.monotonic_ns(),
                    raw_action=np.zeros(29),
                    q_target=joint_pos,
                    kp=np.zeros(29),
                    kd=np.asarray(KD),
                    armed=False,
                    reason=(
                        "policy boundary timeout waiting for state "
                        f"{required_state_sequence}"
                    ),
                )
                self.pending_command = None
                self._delayed_commands.clear()
                self.physics_started = False
                self.last_boundary_wait_ms = (
                    time.perf_counter() - wait_start
                ) * 1000.0
                return
            if time.perf_counter() >= next_republish:
                # UDP is intentionally best-effort. Re-publish the frozen
                # boundary with a fresh monotonic timestamp so one dropped
                # state/task packet cannot deadlock the 4-step handshake.
                self._publish(advance_sequence=False)
                next_republish = time.perf_counter() + 0.002
            time.sleep(0.0005)
            self._poll_command()

    def _publish(self, *, advance_sequence: bool = True):
        if advance_sequence:
            self.sequence += 1
        truth_robot_state = self.name_map.robot_state(
            self.model, self.data, sequence=self.sequence
        )
        truth_task_state = task_state_from_mujoco(
            self.model,
            self.data,
            self.name_map,
            sequence=self.sequence,
            box_body="carry_box",
            box_size=self.box_size,
            goal_position_world=self.goal_position,
            success_position_threshold=float(
                self.sim_cfg["success_position_threshold"]
            ),
            success_tilt_threshold=float(self.sim_cfg["success_tilt_threshold"]),
        )
        observed_robot_state, observed_task_state = (
            self.current_scenario.observed_states(
                truth_robot_state,
                truth_task_state,
                sample_index=self.observation_step_index,
            )
        )
        if self.robot_publisher is not None:
            self.robot_publisher.send(pack_robot_state(observed_robot_state))
        else:
            self.unitree_bridge.publish_state(observed_robot_state, self.data.time)
        self.task_publisher.send(pack_task_state(observed_task_state))
        return (
            truth_robot_state,
            truth_task_state,
            observed_robot_state,
            observed_task_state,
        )

    def _minimum_geom_distance(
        self, geom_ids: tuple[int, ...], other_geom_id: int
    ) -> float | None:
        if not hasattr(self.mujoco, "mj_geomDistance"):
            return None
        distances = []
        for geom_id in geom_ids:
            from_to = np.zeros(6, dtype=np.float64)
            distance = self.mujoco.mj_geomDistance(
                self.model,
                self.data,
                int(geom_id),
                int(other_geom_id),
                10.0,
                from_to,
            )
            if np.isfinite(distance):
                distances.append(float(distance))
        return min(distances) if distances else None

    def _grasp_geometry(
        self, *, include_geom_distance: bool
    ) -> dict[str, object]:
        box_position = self.data.xpos[self.box_body_id].copy()
        box_quaternion = self.data.xquat[self.box_body_id].copy()
        left_palm = self.data.xpos[self.left_palm_body_id].copy()
        right_palm = self.data.xpos[self.right_palm_body_id].copy()
        platform_top = (
            float(
                self.data.xpos[self.source_platform_body_id, 2]
                + 0.5 * self.source_platform_size[2]
            )
            if self.source_platform_enabled
            else 0.0
        )
        box_clearance = (
            obb_bottom_height(
                box_position, box_quaternion, self.box_size
            )
            - platform_top
        )
        return {
            "left_palm_position_world": left_palm,
            "right_palm_position_world": right_palm,
            "left_palm_box_signed_distance": (
                point_to_obb_signed_distance(
                    left_palm,
                    box_position,
                    box_quaternion,
                    self.box_size,
                )
            ),
            "right_palm_box_signed_distance": (
                point_to_obb_signed_distance(
                    right_palm,
                    box_position,
                    box_quaternion,
                    self.box_size,
                )
            ),
            "left_hand_box_geom_distance": (
                self._minimum_geom_distance(
                    self.left_hand_geom_ids, self.box_geom_id
                )
                if include_geom_distance
                else None
            ),
            "right_hand_box_geom_distance": (
                self._minimum_geom_distance(
                    self.right_hand_geom_ids, self.box_geom_id
                )
                if include_geom_distance
                else None
            ),
            "box_bottom_height": obb_bottom_height(
                box_position, box_quaternion, self.box_size
            ),
            "support_height": platform_top,
            "box_clearance": box_clearance,
        }

    def step(self) -> None:
        self._poll_command()
        self._wait_for_boundary_command()
        if self.physics_started:
            if self.control_substep == 0 and self.pending_command is not None:
                self._activate_pending_command()
            self._apply_scheduled_disturbance()
            torque = self._apply_command()
            if self.control_substep == 0:
                self._interval_first_torque = torque.copy()
            self.mujoco.mj_step(self.model, self.data)
            self.observation_step_index += 1
            self.control_substep = (
                self.control_substep + 1
            ) % self.control_decimation
        else:
            torque = np.zeros(29, dtype=np.float64)
            self.data.ctrl[:] = 0.0
            self.current_disturbance[:] = 0.0
            self.data.xfrc_applied[self.name_map.torso_body_id, :3] = 0.0
            self.data.time += float(self.model.opt.timestep)
            self.mujoco.mj_forward(self.model, self.data)
        (
            robot_state,
            task_state,
            observed_robot_state,
            observed_task_state,
        ) = self._publish()
        projected_gravity = quat_rotate_inverse_wxyz(
            robot_state.policy_frame_quat_wxyz,
            np.array([0.0, 0.0, -1.0]),
        )
        failure_reason = self._training_termination_reason(
            robot_state, projected_gravity
        )
        if failure_reason is not None:
            self._latch_episode_failure(failure_reason)
        if not self.recorder.enabled:
            return
        contacts = []
        ground_contact_bodies: set[str] = set()
        max_contact_force = 0.0
        detailed_log = (
            self.control_substep == 0
            if self.physics_started
            else self.sequence % self.control_decimation == 0
        )
        joint_violation = np.maximum(
            np.maximum(
                self.robot.lower_limits - robot_state.joint_pos,
                robot_state.joint_pos - self.robot.upper_limits,
            ),
            0.0,
        )
        self._interval_max_gravity_xy = max(
            self._interval_max_gravity_xy,
            float(np.linalg.norm(projected_gravity[:2])),
        )
        self._interval_max_joint_violation = max(
            self._interval_max_joint_violation,
            float(np.max(joint_violation)),
        )
        if not detailed_log:
            return
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(
                self.model, self.data, contact_index, force
            )
            max_contact_force = max(
                max_contact_force, float(np.linalg.norm(force[:3]))
            )
            geom1 = (
                self.mujoco.mj_id2name(
                    self.model,
                    self.mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom1),
                )
                or int(contact.geom1)
            )
            geom2 = (
                self.mujoco.mj_id2name(
                    self.model,
                    self.mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom2),
                )
                or int(contact.geom2)
            )
            body1 = self.model.body(
                int(self.model.geom_bodyid[int(contact.geom1)])
            ).name
            body2 = self.model.body(
                int(self.model.geom_bodyid[int(contact.geom2)])
            ).name
            if str(geom1).lower() == "floor":
                ground_contact_bodies.add(body2)
            if str(geom2).lower() == "floor":
                ground_contact_bodies.add(body1)
            contacts.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "body1": body1,
                    "body2": body2,
                    "force_contact_frame": force,
                }
            )
        grasp_geometry = self._grasp_geometry(
            include_geom_distance=detailed_log
        )
        grasp_state = self.grasp_tracker.update(
            sim_time=float(self.data.time),
            contacts=contacts,
            box_clearance=float(grasp_geometry["box_clearance"]),
            left_palm_signed_distance=float(
                grasp_geometry["left_palm_box_signed_distance"]
            ),
            right_palm_signed_distance=float(
                grasp_geometry["right_palm_box_signed_distance"]
            ),
        )
        grasp_summary = self.grasp_tracker.summary()
        self._interval_max_contact_force = max(
            self._interval_max_contact_force, max_contact_force
        )
        self._interval_ground_contact_bodies.update(
            ground_contact_bodies
        )
        camera_projection = self.box_camera_projection()
        record = {
            "kind": "mujoco_step",
            "wall_timestamp_ns": time.monotonic_ns(),
            "sim_time": self.data.time,
            "sequence": robot_state.sequence,
            "physics_step_stride": self.control_decimation,
            "projected_gravity": projected_gravity,
            "max_projected_gravity_xy_interval": (
                self._interval_max_gravity_xy
            ),
            "pelvis_position_world": self.data.xpos[
                self.name_map.pelvis_body_id
            ].copy(),
            "command_armed": self.last_command.armed,
            "command_reason": self.last_command.reason,
            "active_command_source_sequence": self.last_command.sequence,
            "pending_command_source_sequence": (
                None
                if self.pending_command is None
                else self.pending_command.sequence
            ),
            "control_substep": self.control_substep,
            "boundary_wait_ms": self.last_boundary_wait_ms,
            "physics_started": self.physics_started,
            "episode_failed": self.episode_failed,
            "episode_failure_reason": self.episode_failure_reason,
            "episode_failure_sequence": self.episode_failure_sequence,
            "episode_failure_sim_time": self.episode_failure_sim_time,
            "joint_limit_violation_max": (
                self._interval_max_joint_violation
            ),
            "box_position_world": self.data.xpos[self.box_body_id].copy(),
            "success": task_state.success,
            "contact_count": int(self.data.ncon),
            "max_contact_force": self._interval_max_contact_force,
            "ground_contact_bodies": sorted(
                self._interval_ground_contact_bodies
            ),
            "detail": detailed_log,
            "camera_view": self.camera_view,
            "episode_index": self.episode_index,
            "scenario_fingerprint": self.current_scenario.fingerprint,
            "randomization_profile": self.current_scenario.profile,
            "action_delay_steps": self.current_scenario.action_delay_steps,
            "policy_step_index": self.policy_step_index,
            "disturbance_force_local": self.current_disturbance.copy(),
            "collision_profile": self.collision_profile.name,
            **grasp_geometry,
            **grasp_state,
            **grasp_summary,
            "grasp_summary": grasp_summary,
            **camera_projection,
        }
        if detailed_log:
            record.update(
                {
                    "joint_pos": robot_state.joint_pos,
                    "joint_vel": robot_state.joint_vel,
                    "policy_frame_quat_wxyz": (
                        robot_state.policy_frame_quat_wxyz
                    ),
                    "policy_frame_ang_vel": (
                        robot_state.policy_frame_ang_vel
                    ),
                    "raw_action": self.last_command.raw_action,
                    "q_target": self.last_command.q_target,
                    "torque": torque,
                    "first_substep_torque": self._interval_first_torque,
                    "joint_limit_violation": joint_violation,
                    "box_pos_policy_frame": (
                        task_state.box_pos_policy_frame
                    ),
                    "box_quat_policy_frame_wxyz": (
                        task_state.box_quat_policy_frame_wxyz
                    ),
                    "goal_pos_policy_frame": (
                        task_state.goal_pos_policy_frame
                    ),
                    "observed_joint_pos": observed_robot_state.joint_pos,
                    "observed_joint_vel": observed_robot_state.joint_vel,
                    "observed_policy_frame_quat_wxyz": (
                        observed_robot_state.policy_frame_quat_wxyz
                    ),
                    "observed_policy_frame_ang_vel": (
                        observed_robot_state.policy_frame_ang_vel
                    ),
                    "observed_box_pos_policy_frame": (
                        observed_task_state.box_pos_policy_frame
                    ),
                    "observed_box_quat_policy_frame_wxyz": (
                        observed_task_state.box_quat_policy_frame_wxyz
                    ),
                    "observed_goal_pos_policy_frame": (
                        observed_task_state.goal_pos_policy_frame
                    ),
                    "contacts": contacts,
                }
            )
        self.recorder.write(record)
        self._interval_max_gravity_xy = 0.0
        self._interval_max_joint_violation = 0.0
        self._interval_max_contact_force = 0.0
        self._interval_first_torque[:] = 0.0
        self._interval_ground_contact_bodies.clear()

    def _key_callback(self, keycode: int) -> None:
        if keycode == 259:  # GLFW_BACKSPACE
            self.reset_requested = True
        elif keycode == 67:  # GLFW_C
            self.camera_view = (
                "free" if self.camera_view == "d455" else "d455"
            )
            self._viewer_camera_dirty = True
            print(f"MuJoCo viewer camera: {self.camera_view}")
        elif keycode in (81, 256):  # Q or ESC
            self.stop_requested = True

    def _apply_viewer_camera(self, viewer_context) -> None:
        if not self._viewer_camera_dirty:
            return
        with viewer_context.lock():
            if self.camera_view == "d455":
                viewer_context.cam.type = (
                    self.mujoco.mjtCamera.mjCAMERA_FIXED
                )
                viewer_context.cam.fixedcamid = self.camera_id
            else:
                viewer_context.cam.type = (
                    self.mujoco.mjtCamera.mjCAMERA_FREE
                )
                viewer_context.cam.fixedcamid = -1
        self._viewer_camera_dirty = False

    @staticmethod
    def _wait_until(deadline: float) -> None:
        """Wait accurately enough for 200 Hz on Linux and Windows hosts."""

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return
            if remaining > 0.0015:
                time.sleep(remaining - 0.001)
            else:
                # A short yield/spin tail avoids accumulating millisecond
                # scheduler oversleep at every 5 ms physics tick.
                time.sleep(0)

    def run(self, *, duration: float | None = None) -> None:
        period = float(self.sim_cfg["physics_dt"])
        next_tick = time.perf_counter()
        start_time = next_tick
        if self.viewer_enabled:
            viewer_context = self.mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
            self._apply_viewer_camera(viewer_context)
        else:
            viewer_context = None
        try:
            while not self.stop_requested:
                if duration is not None and time.perf_counter() - start_time >= duration:
                    break
                if viewer_context is not None and not viewer_context.is_running():
                    break
                if bool(self.sim_cfg.get("realtime", True)):
                    self._wait_until(next_tick)
                if self.reset_requested:
                    self.reset(advance_episode=True)
                self.step()
                if viewer_context is not None:
                    self._apply_viewer_camera(viewer_context)
                    viewer_context.sync()
                if bool(self.sim_cfg.get("realtime", True)):
                    next_tick += period
                    if next_tick < time.perf_counter() - 1.0:
                        next_tick = time.perf_counter()
        finally:
            if viewer_context is not None:
                viewer_context.close()
            self.close()

    def close(self) -> None:
        if (
            self._collision_filter_callback is not None
            and hasattr(self.mujoco, "set_mjcb_contactfilter")
        ):
            self.mujoco.set_mjcb_contactfilter(None)
            self._collision_filter_callback = None
        self.recorder.close()
        self.task_publisher.close()
        if self.robot_publisher is not None:
            self.robot_publisher.close()
        if self.command_receiver is not None:
            self.command_receiver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--transport", choices=("unitree_dds", "udp"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--camera-view",
        choices=("free", "d455"),
        default="free",
        help="initial MuJoCo viewer camera; press C to toggle",
    )
    parser.add_argument("--log", help="optional per-physics-step JSONL log")
    parser.add_argument("--duration", type=float, help="optional wall-clock duration")
    parser.add_argument(
        "--randomization-profile",
        choices=PROFILE_NAMES,
        default="nominal",
        help="seeded Sim2Sim randomization profile",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario-file",
        help="serialized ScenarioSample; overrides profile and seed",
    )
    parser.add_argument(
        "--snapshot-file",
        help="Isaac CarryBox .npz snapshot used as the exact reset state",
    )
    parser.add_argument(
        "--collision-profile",
        choices=COLLISION_PROFILE_NAMES,
        help="collision differential-diagnosis profile",
    )
    parser.add_argument(
        "--action-trace",
        help="Isaac JSONL trace whose actions/q_target are replayed open-loop",
    )
    parser.add_argument(
        "--goal-profile",
        choices=GOAL_PROFILE_NAMES,
        default="configured",
        help="configured goal or a seeded sample from Isaac default reset sector",
    )
    parser.add_argument(
        "--source-platform-profile",
        choices=SOURCE_PLATFORM_PROFILE_NAMES,
        default="configured",
        help="single-variable source-platform presence ablation",
    )
    parser.add_argument(
        "--box-size-scale",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        help="single-variable scale applied to physical and observed box size",
    )
    args = parser.parse_args()
    cfg = load_deploy_config(args.config)
    server = MujocoServer(
        cfg,
        transport=args.transport or cfg.section("simulation")["transport"],
        viewer=bool(cfg.section("simulation")["viewer"]) and not args.headless,
        camera_view=args.camera_view,
        log_path=args.log,
        randomization_profile=args.randomization_profile,
        seed=args.seed,
        scenario_file=args.scenario_file,
        snapshot_file=args.snapshot_file,
        collision_profile=args.collision_profile,
        action_trace_file=args.action_trace,
        goal_profile=args.goal_profile,
        source_platform_profile=args.source_platform_profile,
        box_size_scale=(
            None
            if args.box_size_scale is None
            else tuple(args.box_size_scale)
        ),
    )
    server.run(duration=args.duration)


if __name__ == "__main__":
    main()
