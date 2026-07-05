import numpy as np

from .carrybox_resume_config import G1Cfg as CarryBoxResumeCfg
from .carrybox_resume_config import G1CfgPPO as CarryBoxResumeCfgPPO


class G1Cfg(CarryBoxResumeCfg):
    """Isolated carry-box force-quality experiment; policy interfaces are unchanged."""

    class force_reward:
        enabled = False
        calibration_only = True
        force_filter_tau_s = 0.04
        eps = 1.0e-6

        # Dimensionless calibration placeholders. Replace from calibration logs.
        f_n_min = 0.50
        f_n_sat = 1.50
        f_n_max = 3.00
        rho_safe = 0.60
        rho_max = 5.00
        huber_delta = 0.50

        sign_verification_samples = 12
        min_positive_normal_fraction = 0.70
        warning_check_samples = 32
        debug_log_interval = 100
        debug_force_c1 = False
        debug_force_n = 25.0
        debug_event_steps = 10
        debug_direction_id = 0
        debug_draw_scale = 0.005

    class rewards(CarryBoxResumeCfg.rewards):
        class scales(CarryBoxResumeCfg.rewards.scales):
            normal_force_margin = 0.06
            tangential_compression_risk = -0.08
            normal_force_overcompression = -0.02


class G1CfgPPO(CarryBoxResumeCfgPPO):
    class runner(CarryBoxResumeCfgPPO.runner):
        run_name = "carrybox_boxperturb_force_reward_resume"
        # Calibration can start without a checkpoint; --resume --resume_path remains supported.
        resume = False
        resume_path = None

    amp = G1Cfg.amp
