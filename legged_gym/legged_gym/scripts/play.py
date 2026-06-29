import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_jit_to_onnx, load_onnx_policy, task_registry, Logger

import numpy as np
import torch
import keyboard
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


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


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.env.test = True
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = True

    # carrybox
    if args.task == 'carrybox':
        env_cfg.asset.box.random_props = False
        env_cfg.asset.box.reset_mode = 'default'
        env_cfg.env.episode_length_s = 10
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
    policy = ppo_runner.get_inference_policy(device=env.device)
    
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
        result = env.gym.fetch_results(env.sim, True)
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
