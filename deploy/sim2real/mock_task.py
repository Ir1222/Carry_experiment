"""Publish a stationary torso-relative TaskState for hardware dry-runs."""

from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.common.config import load_deploy_config
from deploy.common.transport import UdpPublisher, pack_task_state
from deploy.sim2real.task_provider import MockTaskStateProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--box-pos", nargs=3, type=float, default=(1.0, 0.0, -0.65))
    parser.add_argument("--goal-pos", nargs=3, type=float, default=(2.5, 0.75, -0.65))
    args = parser.parse_args()
    cfg = load_deploy_config(args.config)
    address_value = cfg.section("network")["task_state_udp"]
    publisher = UdpPublisher((str(address_value[0]), int(address_value[1])))
    box_size = np.asarray(cfg.section("simulation")["box_size"], dtype=np.float64)
    provider = MockTaskStateProvider(
        box_pos_torso=args.box_pos,
        box_size=box_size,
        goal_pos_torso=args.goal_pos,
    )
    period = 1.0 / args.rate
    try:
        while True:
            publisher.send(pack_task_state(provider.poll()))
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()
        publisher.close()


if __name__ == "__main__":
    main()
