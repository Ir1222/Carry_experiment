"""Create a temporary, safety-locked G1 sim2real dry-run configuration."""

from __future__ import annotations

import argparse
import socket
import tempfile
from pathlib import Path
from typing import Sequence

import yaml

from deploy.common.config import load_deploy_config


def _validate_interface(interface: str, *, allow_loopback: bool) -> None:
    names = {name for _, name in socket.if_nameindex()}
    if interface not in names:
        available = ", ".join(sorted(names)) or "<none>"
        raise ValueError(f"interface {interface!r} does not exist; available: {available}")
    if not allow_loopback and interface in {"lo", "lo0"}:
        raise ValueError("use the dedicated G1 Ethernet interface, not loopback")


def create_dryrun_config(
    source: str | Path,
    output: str | Path,
    interface: str,
    *,
    validate_interface: bool = True,
    allow_loopback: bool = False,
) -> Path:
    """Write a validated copy with dry-run and zero proportional gain locked on."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("refusing to overwrite the project deployment configuration")
    if validate_interface:
        _validate_interface(interface, allow_loopback=allow_loopback)

    source_cfg = load_deploy_config(source_path)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("deployment YAML root must be a mapping")

    raw["project_root"] = str(source_cfg.project_root)
    raw.setdefault("network", {})["interface"] = interface
    raw.setdefault("network", {})["domain_id"] = int(
        source_cfg.section("network")["domain_id"]
    )
    raw.setdefault("simulation", {})["transport"] = "unitree_dds"
    raw.setdefault("safety", {})["dry_run"] = True
    raw.setdefault("control", {})["hardware_kp_scale"] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(output_path)

    locked = load_deploy_config(output_path)
    if (
        not bool(locked.section("safety")["dry_run"])
        or float(locked.section("control")["hardware_kp_scale"]) != 0.0
    ):
        raise RuntimeError("failed to apply the sim2real safety lock")
    if str(locked.section("network")["interface"]) != interface:
        raise RuntimeError("failed to apply the selected network interface")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--interface", required=True, help="dedicated Ethernet NIC connected to G1")
    parser.add_argument(
        "--output",
        default=str(Path(tempfile.gettempdir()) / "g1_carrybox_dryrun.yaml"),
    )
    parser.add_argument(
        "--skip-interface-check",
        action="store_true",
        help="only for offline configuration tests; do not use when connected to G1",
    )
    parser.add_argument(
        "--allow-loopback",
        action="store_true",
        help="only for offline DDS tests; real G1 dry-run must use its Ethernet NIC",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = create_dryrun_config(
        args.config,
        args.output,
        args.interface,
        validate_interface=not args.skip_interface_check,
        allow_loopback=args.allow_loopback,
    )
    cfg = load_deploy_config(path)
    print(f"Wrote safety-locked dry-run config: {path}")
    print(f"project_root: {cfg.project_root}")
    print(f"interface: {cfg.section('network')['interface']}")
    print(f"dry_run: {str(bool(cfg.section('safety')['dry_run'])).lower()}")
    print(f"hardware_kp_scale: {float(cfg.section('control')['hardware_kp_scale'])}")
    print("LowCmd hardware writes remain disabled unless code and configuration are changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
