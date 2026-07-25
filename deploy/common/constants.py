"""Dimensions and ordered names that define the actor contract."""

from __future__ import annotations

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

END_EFFECTOR_NAMES = (
    "left_palm_link",
    "right_palm_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "mid360_link",
)

ACTION_DIM = 29
PROPRIO_OBS_DIM = 108
TASK_OBS_DIM = 15
FRAME_OBS_DIM = 123
HISTORY_LENGTH = 6
ACTOR_OBS_DIM = FRAME_OBS_DIM * HISTORY_LENGTH

OBSERVATION_SLICES = {
    "base_ang_vel": (0, 3),
    "projected_gravity": (3, 6),
    "dof_pos": (6, 35),
    "dof_vel": (35, 64),
    "end_effector_pos": (64, 79),
    "previous_action": (79, 108),
    "task": (108, 123),
}

DEFAULT_DOF_POS = (
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
)

KP = (
    150.0,
    150.0,
    150.0,
    300.0,
    40.0,
    40.0,
    150.0,
    150.0,
    150.0,
    300.0,
    40.0,
    40.0,
    300.0,
    300.0,
    300.0,
    200.0,
    200.0,
    200.0,
    100.0,
    20.0,
    20.0,
    20.0,
    200.0,
    200.0,
    200.0,
    100.0,
    20.0,
    20.0,
    20.0,
)

KD = (
    2.0,
    2.0,
    2.0,
    4.0,
    1.0,
    1.0,
    2.0,
    2.0,
    2.0,
    4.0,
    1.0,
    1.0,
    4.0,
    4.0,
    4.0,
    3.0,
    3.0,
    3.0,
    1.0,
    0.5,
    0.5,
    0.5,
    3.0,
    3.0,
    3.0,
    1.0,
    0.5,
    0.5,
    0.5,
)

POLICY_TO_UNITREE_MOTOR = tuple(range(ACTION_DIM))

assert len(JOINT_NAMES) == ACTION_DIM
assert len(DEFAULT_DOF_POS) == ACTION_DIM
assert len(KP) == ACTION_DIM
assert len(KD) == ACTION_DIM
