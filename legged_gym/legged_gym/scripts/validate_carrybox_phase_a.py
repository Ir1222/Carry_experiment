import argparse
from types import SimpleNamespace

import isaacgym  # noqa: F401
from isaacgym import gymapi
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry


def _make_args(num_envs, headless, rl_device):
    return SimpleNamespace(
        task="carrybox",
        resume=False,
        resume_path=None,
        experiment_name=None,
        run_name="carrybox_phase_a_validation",
        load_run=None,
        checkpoint=None,
        exptid=None,
        resumeid=None,
        headless=headless,
        horovod=False,
        rl_device=rl_device,
        sim_device=rl_device,
        device=rl_device,
        num_envs=num_envs,
        seed=1,
        max_iterations=None,
        play_dataset=False,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=rl_device.startswith("cuda"),
        use_gpu_pipeline=rl_device.startswith("cuda"),
        subscenes=0,
        num_threads=10,
    )


def _disable_amp_reward_and_loss(runner):
    device = runner.device

    def zero_predict_reward(agent_obs, normalizer=None):
        return torch.zeros(agent_obs.shape[0], 1, device=device)

    def zero_compute_loss(agent_obs, expert_obs):
        zero = torch.zeros((), device=device)
        return zero, zero, zero

    runner.alg.amp.predict_reward = zero_predict_reward
    runner.alg.amp.compute_loss = zero_compute_loss
    if runner.alg.amp_normalizer is not None:
        runner.alg.amp_normalizer.update = lambda x: None


def validate(num_envs, smoke_iterations, rollout_steps, rl_device):
    args = _make_args(num_envs=num_envs, headless=True, rl_device=rl_device)
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = num_envs
    env_cfg.noise.add_noise = False
    train_cfg.runner.resume = False
    train_cfg.runner.num_steps_per_env = rollout_steps
    train_cfg.runner.max_iterations = smoke_iterations
    train_cfg.runner.run_name = "carrybox_phase_a_validation"
    train_cfg.amp.amp_coef = 0.0

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env=env, args=args, train_cfg=train_cfg, log_root=None)
    _disable_amp_reward_and_loss(runner)

    zero_actions = torch.zeros(num_envs, env.num_actions, device=env.device)
    obs, critic_obs, _, _, _, _, _, _ = env.step(zero_actions)
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    assert list(obs.shape) == [num_envs, 738], obs.shape
    assert list(critic_obs.shape) == [num_envs, 758], critic_obs.shape
    assert torch.isfinite(obs).all()
    assert torch.isfinite(critic_obs).all()

    interaction_proxy = env._compute_interaction_privileged_proxy(log_stats=False)
    assert list(interaction_proxy.shape) == [num_envs, 17], interaction_proxy.shape
    assert torch.allclose(critic_obs[:, 741:758], interaction_proxy)

    obs_before_terminal = obs.clone()
    term_ids = torch.arange(min(4, num_envs), device=env.device)
    termination_critic_obs = env.compute_termination_observations(term_ids)
    assert termination_critic_obs.shape[-1] == 758, termination_critic_obs.shape
    assert torch.isfinite(termination_critic_obs).all()
    assert torch.equal(env.obs_buf, obs_before_terminal)

    actor_first = runner.alg.actor_critic.actor[0]
    critic_first = runner.alg.actor_critic.critic[0]
    assert actor_first.in_features == 738, actor_first.in_features
    assert critic_first.in_features == 758, critic_first.in_features
    with torch.inference_mode():
        actions = runner.alg.actor_critic.act_inference(obs.to(runner.device))
    assert list(actions.shape) == [num_envs, env.num_actions], actions.shape

    runner.learn(num_learning_iterations=smoke_iterations, init_at_random_ep_len=True)
    obs = env.get_observations()
    critic_obs = env.get_privileged_observations()
    assert list(obs.shape) == [num_envs, 738], obs.shape
    assert list(critic_obs.shape) == [num_envs, 758], critic_obs.shape
    assert torch.isfinite(obs).all()
    assert torch.isfinite(critic_obs).all()

    print("carrybox_phase_a_validation_passed")
    print(f"actor_obs_shape={tuple(obs.shape)}")
    print(f"critic_obs_shape={tuple(critic_obs.shape)}")
    print(f"termination_critic_obs_shape={tuple(termination_critic_obs.shape)}")
    print(f"actor_first_layer_in_features={actor_first.in_features}")
    print(f"critic_first_layer_in_features={critic_first.in_features}")
    print(f"smoke_iterations={smoke_iterations}")
    print(f"rollout_steps_per_env={rollout_steps}")
    print("amp_reward_and_loss_disabled_for_validation=True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--smoke-iterations", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--rl-device", type=str, default="cuda:0")
    parsed = parser.parse_args()
    validate(parsed.num_envs, parsed.smoke_iterations, parsed.rollout_steps, parsed.rl_device)
