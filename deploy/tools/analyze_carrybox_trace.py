"""Summarize grasp contacts and compare Isaac/MuJoCo open-loop trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _read(path: str | Path, kind: str) -> list[dict]:
    records = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            if record.get("kind") == kind:
                records.append(record)
    return records


def _pair(contact: dict) -> tuple[str, str]:
    return tuple(
        sorted((str(contact.get("body1", "")), str(contact.get("body2", ""))))
    )


def _first_box_contact(records: list[dict]) -> int | None:
    for index, record in enumerate(records):
        if any("carry_box" in _pair(contact) for contact in record.get("contacts", ())):
            return index
    return None


def _contact_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(
        " <-> ".join(_pair(contact))
        for record in records
        for contact in record.get("contacts", ())
        if float(
            contact.get(
                "force_norm",
                np.linalg.norm(contact.get("force_contact_frame", (0, 0, 0))[:3]),
            )
        )
        > 1.0
    )
    return dict(counts.most_common())


def analyze(
    isaac_trace: str | Path,
    mujoco_trace: str | Path | None = None,
) -> dict[str, object]:
    isaac = _read(isaac_trace, "isaac_carrybox_step")
    if not isaac:
        raise ValueError("Isaac trace contains no isaac_carrybox_step records")
    report: dict[str, object] = {
        "isaac": {
            "steps": len(isaac),
            "first_box_contact_step": _first_box_contact(isaac),
            "contact_pair_steps": _contact_counts(isaac),
        }
    }
    if mujoco_trace is None:
        return report
    mujoco = _read(mujoco_trace, "mujoco_step")
    if not mujoco:
        raise ValueError("MuJoCo trace contains no mujoco_step records")
    first_isaac = _first_box_contact(isaac)
    first_mujoco = _first_box_contact(mujoco)
    limit_candidates = [max(0, len(isaac) - 1), len(mujoco)]
    if first_isaac is not None:
        limit_candidates.append(first_isaac)
    if first_mujoco is not None:
        limit_candidates.append(first_mujoco)
    precontact_steps = min(limit_candidates)

    def trajectory_error(
        key_isaac: str, key_mujoco: str, *, isaac_offset: int = 0
    ):
        errors = []
        for left, right in zip(
            isaac[
                isaac_offset : isaac_offset + precontact_steps
            ],
            mujoco[:precontact_steps],
        ):
            if key_isaac not in left or key_mujoco not in right:
                continue
            errors.append(
                np.abs(
                    np.asarray(left[key_isaac], dtype=np.float64)
                    - np.asarray(right[key_mujoco], dtype=np.float64)
                )
            )
        if not errors:
            return None
        stacked = np.stack(errors)
        return {
            "max_abs": float(np.max(stacked)),
            "mean_abs": float(np.mean(stacked)),
            "last_step_max_abs": float(np.max(stacked[-1])),
        }

    final_grasp = mujoco[-1].get("grasp_summary", {})
    report["mujoco"] = {
        "steps": len(mujoco),
        "collision_profile": mujoco[-1].get("collision_profile"),
        "first_box_contact_step": first_mujoco,
        "contact_pair_steps": _contact_counts(mujoco),
        "grasp_summary": final_grasp,
        "min_left_hand_box_geom_distance": min(
            (
                float(item["left_hand_box_geom_distance"])
                for item in mujoco
                if item.get("left_hand_box_geom_distance") is not None
            ),
            default=None,
        ),
        "min_right_hand_box_geom_distance": min(
            (
                float(item["right_hand_box_geom_distance"])
                for item in mujoco
                if item.get("right_hand_box_geom_distance") is not None
            ),
            default=None,
        ),
    }
    report["open_loop_pre_box_contact"] = {
        "compared_policy_steps": precontact_steps,
        "alignment": (
            "Isaac boundary state i+1 versus MuJoCo state after action i; "
            "commands compare at i"
        ),
        "joint_pos": trajectory_error(
            "joint_pos", "joint_pos", isaac_offset=1
        ),
        "joint_vel": trajectory_error(
            "joint_vel", "joint_vel", isaac_offset=1
        ),
        "q_target": trajectory_error("q_target", "q_target"),
        "action": trajectory_error("policy_action", "raw_action"),
        "first_substep_torque": trajectory_error(
            "torque", "first_substep_torque"
        ),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isaac_trace")
    parser.add_argument("--mujoco-trace")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = analyze(args.isaac_trace, args.mujoco_trace)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(
            text + "\n", encoding="utf-8"
        )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
