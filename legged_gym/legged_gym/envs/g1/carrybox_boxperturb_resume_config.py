from .carrybox_resume_config import G1Cfg as CarryBoxResumeCfg
from .carrybox_resume_config import G1CfgPPO as CarryBoxResumeCfgPPO


class G1Cfg(CarryBoxResumeCfg):
    """Resume-compatible box-COM perturbation curriculum configuration."""

    class env(CarryBoxResumeCfg.env):
        # Explicit compatibility contract for existing carry-box checkpoints.
        num_actor_obs = 738
        num_privileged_obs = 143
        num_interaction_priv_obs = 17

    class asset(CarryBoxResumeCfg.asset):
        class box(CarryBoxResumeCfg.asset.box):
            random_size = False
            random_density = False
            density_default = 50.0

    class domain_rand(CarryBoxResumeCfg.domain_rand):
        # Keep all inherited robot DR; disable only the legacy torso force.
        disturbance = False

    class box_perturbation:
        # Master experiment gate. False preserves nominal carry dynamics.
        enabled = True
        stable_confirmed_carry_policy_steps = 20
        max_events_per_episode = 1
        pulse_duration_s = 0.10
        force_peak_cap_N = 10.0

        schedule_mode = "staged_policy_steps"
        manual_stage_override = None
        stage_start_policy_steps = {
            "C1": 0,
            "C2": 250_000,
            "C3": 750_000,
            "C4": 1_500_000,
        }
        stages = {
            "C1": {
                "directions": ("+box_x", "-box_x"),
                "beta": {"x": (0.05, 0.10)},
                "probability": 0.25,
            },
            "C2": {
                "directions": ("+box_x", "-box_x"),
                "beta": {"x": (0.10, 0.20)},
                "probability": 0.40,
            },
            "C3": {
                "directions": ("+box_x", "-box_x", "-z_world"),
                "beta": {"x": (0.10, 0.25), "z": (0.05, 0.15)},
                "probability": 0.50,
            },
            "C4": {
                "directions": (
                    "+box_x",
                    "-box_x",
                    "+box_y",
                    "-box_y",
                    "-z_world",
                ),
                "beta": {
                    "x": (0.10, 0.30),
                    "y": (0.05, 0.15),
                    "z": (0.05, 0.20),
                },
                "probability": 0.60,
            },
        }

        recovery_window_s = 1.0
        recovery_confirmed_carry_steps = 5
        debug_force_event = True
        debug_draw_force = False
        # Viewer arrow length is |F| times this scale.
        debug_force_draw_scale_m_per_N = 0.08
        debug_force_arrow_head_length_m = 0.08
        debug_force_draw_max_envs = 10
        debug_force_log_interval_policy_steps = 1

        # Evaluation-only deterministic sweep. For every beta level, test all
        # five directions once before increasing the force level.
        debug_sweep_enabled = False
        debug_sweep_beta_values = (0.25, 0.50, 0.75, 1.00)
        debug_sweep_directions = (
            "+box_x",
            "-box_x",
            "+box_y",
            "-box_y",
            "-z_world",
        )
        debug_sweep_inter_event_policy_steps = 75


class G1CfgPPO(CarryBoxResumeCfgPPO):
    class runner(CarryBoxResumeCfgPPO.runner):
        run_name = "carrybox_boxperturb_resume"

    amp = G1Cfg.amp
