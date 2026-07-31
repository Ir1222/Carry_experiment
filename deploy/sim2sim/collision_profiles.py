"""Selectable MuJoCo collision filters for CarryBox differential diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


COLLISION_PROFILE_NAMES = ("current", "no_robot_self", "isaac_parity")


def normalized_pair(body_a: str, body_b: str) -> frozenset[str]:
    return frozenset((str(body_a), str(body_b)))


def should_filter_collision(
    body1: int,
    body2: int,
    *,
    robot_body_ids: frozenset[int],
    disable_robot_self: bool,
    excluded_body_id_pairs: frozenset[frozenset[int]],
) -> bool:
    pair = frozenset((int(body1), int(body2)))
    if pair in excluded_body_id_pairs:
        return True
    return bool(
        disable_robot_self
        and int(body1) in robot_body_ids
        and int(body2) in robot_body_ids
    )


@dataclass(frozen=True, slots=True)
class CollisionProfile:
    name: str
    disable_robot_self: bool
    robot_margin: float
    external_margin: float
    excluded_body_pairs: tuple[tuple[str, str], ...]

    @classmethod
    def from_simulation_config(
        cls,
        simulation: Mapping[str, object],
        name: str | None,
    ) -> "CollisionProfile":
        collision = simulation.get("collision")
        if not isinstance(collision, Mapping):
            collision = {}
        profiles = collision.get("profiles")
        if not isinstance(profiles, Mapping):
            profiles = {}
        selected = str(
            name
            or collision.get("default_profile", "current")
        )
        if selected not in COLLISION_PROFILE_NAMES:
            raise ValueError(
                f"collision profile must be one of {COLLISION_PROFILE_NAMES}, "
                f"got {selected!r}"
            )
        configured = profiles.get(selected)
        if not isinstance(configured, Mapping):
            default_margin = float(simulation.get("contact_margin", 0.0))
            configured = {
                "disable_robot_self": selected == "no_robot_self",
                "robot_margin": (
                    0.0 if selected == "isaac_parity" else default_margin
                ),
                "external_margin": default_margin,
                "exclude_body_pairs": (),
            }
        excluded = configured.get("exclude_body_pairs", ())
        pairs: list[tuple[str, str]] = []
        if not isinstance(excluded, Iterable) or isinstance(
            excluded, (str, bytes)
        ):
            raise ValueError("exclude_body_pairs must be a sequence")
        for item in excluded:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not all(str(value) for value in item)
            ):
                raise ValueError(
                    "each collision exclusion must contain two body names"
                )
            pairs.append((str(item[0]), str(item[1])))
        robot_margin = float(configured.get("robot_margin", 0.0))
        external_margin = float(configured.get("external_margin", 0.0))
        if robot_margin < 0.0 or external_margin < 0.0:
            raise ValueError("collision margins must be non-negative")
        return cls(
            name=selected,
            disable_robot_self=bool(
                configured.get("disable_robot_self", False)
            ),
            robot_margin=robot_margin,
            external_margin=external_margin,
            excluded_body_pairs=tuple(pairs),
        )
