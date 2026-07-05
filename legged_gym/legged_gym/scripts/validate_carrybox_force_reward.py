"""Runtime smoke test for carrybox_boxperturb_force_reward_resume.

Run inside the Isaac Gym Python environment. This intentionally does not load a checkpoint.
Set FORCE_DEBUG_C1=1 to enable the deterministic C1 debug event and viewer vectors.
"""
import os

import isaacgym  # noqa: F401 - must precede torch in Isaac Gym environments
import torch

from legged_gym.envs import *  # noqa: F401,F403 - performs task registration
from legged_gym.utils import get_args, task_registry


TASK = "carrybox_boxperturb_force_reward_resume"


def main():
    args = get_args()
    args.task = TASK
    cfg, _ = task_registry.get_cfgs(TASK)
    cfg.env.num_envs = min(int(cfg.env.num_envs), 4)
    cfg.force_reward.debug_force_c1 = os.environ.get("FORCE_DEBUG_C1", "0") == "1"
    env, _ = task_registry.make_env(TASK, args=args, env_cfg=cfg)

    assert env.obs_buf.shape == (env.num_envs, 738), env.obs_buf.shape
    assert env.privileged_obs_buf.shape == (env.num_envs, 143), env.privileged_obs_buf.shape
    assert env.num_interaction_priv_obs == 17
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, critic, reward, _, _, _, _, _ = env.step(actions)
    assert obs.shape[-1] == 738 and critic.shape[-1] == 143
    tensors = (
        env.force_raw_world_buf, env.force_fn_ema_buf, env.force_ft_ema_buf,
        env.force_fn_over_mg_buf, env.force_rho_buf, env.force_normal_reward_buf,
        env.force_shear_penalty_buf, env.force_overcompression_penalty_buf, reward,
    )
    assert all(torch.isfinite(value).all() for value in tensors)

    ids = torch.arange(env.num_envs, device=env.device)
    env.force_fn_ema_buf.fill_(1.0)
    env.force_face_locked_buf.fill_(True)
    env.reset_idx(ids)
    assert torch.count_nonzero(env.force_fn_ema_buf) == 0
    assert not torch.any(env.force_face_locked_buf)
    print("task_registry_resolves=True")
    print("actor_obs_dim=738 critic_obs_dim=143 interaction_privileged_tail_dim=17")
    print("force_tensors_finite=True force_rewards_finite=True reset_clears_force_buffers=True")
    print("checkpoint_test=skipped_by_request")


if __name__ == "__main__":
    main()
