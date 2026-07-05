import argparse
from types import SimpleNamespace

import isaacgym  # noqa: F401
from isaacgym import gymapi
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry


def _make_args(num_envs, rl_device):
    return SimpleNamespace(
        task="carrybox_boxperturb_resume",
        resume=False,
        resume_path=None,
        experiment_name=None,
        run_name="carrybox_boxperturb_smoke",
        load_run=None,
        checkpoint=None,
        exptid=None,
        resumeid=None,
        headless=True,
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


def validate(num_envs, rl_device):
    args = _make_args(num_envs, rl_device)
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = num_envs
    env_cfg.noise.add_noise = False
    env_cfg.box_perturbation.debug_force_event = True
    env_cfg.box_perturbation.manual_stage_override = "C1"
    train_cfg.runner.resume = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg, log_root=None
    )

    assert env.cfg.domain_rand.disturbance is False
    assert env.cfg.box_perturbation.enabled is True
    assert env.box_cfg.random_size is False
    assert env.box_cfg.random_density is False
    assert env.box_cfg.density_default == 50.0
    assert env._pulse_physics_steps() == 20
    assert env._recovery_policy_steps() == 50
    assert env.box_perturb_force_tensor.shape == env.contact_forces.shape
    assert int(env.box_net_contact_force_index) == 2

    # Isolate the scheduler test from contact availability: debug mode deterministically
    # schedules C1 once the confirmed-carry streak reaches its configured threshold.
    env.confirmed_carry_buf[:] = True
    threshold = env.cfg.box_perturbation.stable_confirmed_carry_policy_steps
    for _ in range(threshold):
        env._update_box_perturbation_state()
    assert torch.all(env.box_perturb_event_count_buf == 1)
    assert torch.all(env.box_perturb_remaining_physics_steps == 20)
    assert torch.all(env.box_perturb_beta >= 0.05)
    assert torch.all(env.box_perturb_beta <= 0.10)
    assert torch.allclose(
        env.box_perturb_peak_force_N,
        torch.clamp(env.box_perturb_beta * env.box_masses * 9.81, max=10.0),
    )

    zero_actions = torch.zeros(num_envs, env.num_actions, device=env.device)
    obs, critic_obs, _, _, infos, _, _, _ = env.step(zero_actions)
    assert tuple(obs.shape) == (num_envs, 738)
    assert tuple(critic_obs.shape) == (num_envs, 143)
    interaction_proxy = env._compute_interaction_privileged_proxy(log_stats=False)
    assert tuple(interaction_proxy.shape) == (num_envs, 17)
    assert torch.isfinite(obs).all()
    assert torch.isfinite(critic_obs).all()
    assert torch.all(
        env.box_perturb_force_tensor[:, : int(env.box_net_contact_force_index), :] == 0
    )
    assert torch.all(
        env.box_perturb_force_tensor[:, int(env.box_net_contact_force_index) + 1 :, :] == 0
    )

    actor_first = runner.alg.actor_critic.actor[0]
    critic_first = runner.alg.actor_critic.critic[0]
    assert actor_first.in_features == 738
    assert critic_first.in_features == 143
    assert "perturb" in infos
    for key in (
        "stage_id",
        "enabled",
        "event_trigger_rate",
        "events_per_episode",
        "force_peak_N_mean",
        "beta_mean",
        "mass_kg_mean",
        "force_cap_usage_rate",
        "recovery_success_rate",
    ):
        assert key in infos["perturb"], key

    print("carrybox_boxperturb_smoke_passed")
    print(f"num_envs={num_envs}")
    print(f"actor_obs_shape={tuple(obs.shape)}")
    print(f"critic_obs_shape={tuple(critic_obs.shape)}")
    print(f"interaction_priv_shape={tuple(interaction_proxy.shape)}")
    print(f"pulse_physics_steps={env._pulse_physics_steps()}")
    print(f"box_mass_kg_mean={float(env.box_masses.mean()):.6f}")
    print(f"force_peak_N_mean={float(env.box_perturb_peak_force_N.mean()):.6f}")
    print("debug_force_event=True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rl-device", type=str, default="cuda:0")
    parsed = parser.parse_args()
    validate(parsed.num_envs, parsed.rl_device)
