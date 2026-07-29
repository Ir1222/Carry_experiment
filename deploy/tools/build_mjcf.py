"""Generate a MuJoCo robot model from the PhysHSI URDF.

MuJoCo performs the URDF conversion. This tool then adds name-matched torque
actuators and a torso IMU site. The scene wrapper in ``deploy/assets`` adds the
ground, free box, and goal marker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from deploy.common.config import load_deploy_config
from deploy.common.constants import JOINT_NAMES
from deploy.common.mapping import RobotDescription


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required to build the MJCF; install deploy/requirements.txt"
        ) from exc
    return mujoco


def _normalise_mesh_paths(root: ET.Element, urdf_path: Path) -> Path | None:
    """Resolve meshes independently of the rewritten URDF's temp directory."""
    mesh_dirs: set[Path] = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf_path.parent / mesh_path).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"URDF mesh not found: {mesh_path}")
        mesh_dirs.add(mesh_path.parent)
        mesh.attrib["filename"] = mesh_path.name

    if not mesh_dirs:
        return None
    if len(mesh_dirs) != 1:
        raise ValueError(
            "Expected all PhysHSI G1 meshes in one directory, got: "
            + ", ".join(str(path) for path in sorted(mesh_dirs))
        )
    return next(iter(mesh_dirs))


def _enable_floating_base(root: ET.Element) -> None:
    if any(j.attrib.get("name") == "floating_base_joint" for j in root.findall("joint")):
        return
    world = ET.Element("link", {"name": "world"})
    floating = ET.Element(
        "joint", {"name": "floating_base_joint", "type": "floating"}
    )
    ET.SubElement(floating, "parent", {"link": "world"})
    ET.SubElement(floating, "child", {"link": "pelvis"})
    root.insert(0, world)
    root.insert(1, floating)


def _add_actuators_and_imu(
    mjcf_path: Path,
    robot: RobotDescription,
    *,
    joint_armature: float,
    camera_config: Mapping[str, Any],
) -> None:
    root = ET.parse(mjcf_path).getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.attrib.get("meshdir"):
        mesh_dir = Path(compiler.attrib.pop("meshdir")).resolve()
        for mesh in root.findall("./asset/mesh"):
            filename = mesh.attrib.get("file")
            if filename:
                mesh.attrib["file"] = (mesh_dir / filename).resolve().as_posix()
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    else:
        actuator.clear()
    for index, name in enumerate(robot.joint_names):
        joint = next(
            (
                element
                for element in root.findall(".//joint")
                if element.attrib.get("name") == name
            ),
            None,
        )
        if joint is None:
            raise ValueError(f"converted MJCF has no joint {name}")
        joint.attrib["armature"] = f"{float(joint_armature):.9g}"
        effort = float(robot.effort_limits[index])
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{-effort:.9g} {effort:.9g}",
            },
        )

    torso = next(
        (body for body in root.findall(".//body") if body.attrib.get("name") == "torso_link"),
        None,
    )
    if torso is None:
        raise ValueError("converted MJCF has no torso_link body")
    if not any(site.attrib.get("name") == "torso_imu_site" for site in torso.findall("site")):
        ET.SubElement(
            torso,
            "site",
            {
                "name": "torso_imu_site",
                "pos": "0 0 0",
                "size": "0.005",
                "rgba": "0 0 0 0",
            },
        )
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(
        sensor,
        "framequat",
        {
            "name": "torso_quat",
            "objtype": "site",
            "objname": "torso_imu_site",
        },
    )
    ET.SubElement(
        sensor, "gyro", {"name": "torso_gyro", "site": "torso_imu_site"}
    )

    camera_name = str(camera_config["name"])
    camera_body_name = str(camera_config["body"])
    camera_body = next(
        (
            body
            for body in root.findall(".//body")
            if body.attrib.get("name") == camera_body_name
        ),
        None,
    )
    if camera_body is None:
        raise ValueError(
            f"converted MJCF has no camera body {camera_body_name!r}"
        )
    existing_cameras = [
        camera
        for camera in root.findall(".//camera")
        if camera.attrib.get("name") == camera_name
    ]
    if existing_cameras:
        raise ValueError(
            f"converted MJCF already contains camera {camera_name!r}"
        )
    position = " ".join(
        f"{float(value):.9g}"
        for value in camera_config["position"]
    )
    quaternion = " ".join(
        f"{float(value):.9g}"
        for value in camera_config["quaternion_wxyz"]
    )
    resolution = (
        f"{int(camera_config['width'])} "
        f"{int(camera_config['height'])}"
    )
    ET.SubElement(
        camera_body,
        "camera",
        {
            "name": camera_name,
            "mode": "fixed",
            "pos": position,
            "quat": quaternion,
            "fovy": f"{float(camera_config['vertical_fov_deg']):.9g}",
            "resolution": resolution,
        },
    )
    ET.indent(root)
    ET.ElementTree(root).write(mjcf_path, encoding="unicode", xml_declaration=False)


def build_robot_mjcf(
    urdf_path: str | Path,
    output_path: str | Path,
    *,
    joint_armature: float = 0.01,
    camera_config: Mapping[str, Any] | None = None,
) -> Path:
    mujoco = _require_mujoco()
    urdf_path = Path(urdf_path).resolve()
    output_path = Path(output_path).resolve()
    robot = RobotDescription.from_urdf(urdf_path)
    if camera_config is None:
        raise ValueError(
            "camera_config is required so the generated MJCF preserves "
            "the trained D455 optical frame"
        )
    urdf_root = ET.parse(urdf_path).getroot()
    mesh_dir = _normalise_mesh_paths(urdf_root, urdf_path)
    _enable_floating_base(urdf_root)
    compiler = urdf_root.find("mujoco/compiler")
    if compiler is None:
        mujoco_tag = urdf_root.find("mujoco")
        if mujoco_tag is None:
            mujoco_tag = ET.SubElement(urdf_root, "mujoco")
        compiler = ET.SubElement(mujoco_tag, "compiler")
    if mesh_dir is not None:
        compiler.attrib["meshdir"] = mesh_dir.as_posix()
    compiler.attrib["fusestatic"] = "false"
    compiler.attrib["discardvisual"] = "false"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="physhsi_urdf_") as temp_dir:
        temp_urdf = Path(temp_dir) / "g1_29dof_floating.urdf"
        ET.ElementTree(urdf_root).write(
            temp_urdf, encoding="unicode", xml_declaration=False
        )
        model = mujoco.MjModel.from_xml_path(str(temp_urdf))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mujoco.mj_saveLastXML(str(output_path), model)

    _add_actuators_and_imu(
        output_path,
        robot,
        joint_armature=float(joint_armature),
        camera_config=camera_config,
    )

    validation_model = mujoco.MjModel.from_xml_path(str(output_path))
    validation_names = {
        validation_model.joint(index).name for index in range(validation_model.njnt)
    }
    missing = [name for name in JOINT_NAMES if name not in validation_names]
    if missing:
        raise RuntimeError(f"generated MJCF is missing policy joints: {missing}")
    if validation_model.nu != len(JOINT_NAMES):
        raise RuntimeError(
            f"generated MJCF has {validation_model.nu} actuators, expected 29"
        )
    camera_name = str(camera_config["name"])
    camera_body_name = str(camera_config["body"])
    try:
        camera_id = int(validation_model.camera(camera_name).id)
        camera_body_id = int(
            validation_model.body(camera_body_name).id
        )
    except KeyError as exc:
        raise RuntimeError(
            "generated MJCF is missing the configured D455 camera/body"
        ) from exc
    if int(validation_model.cam_bodyid[camera_id]) != camera_body_id:
        actual_body = validation_model.body(
            int(validation_model.cam_bodyid[camera_id])
        ).name
        raise RuntimeError(
            f"camera {camera_name!r} belongs to {actual_body!r}, "
            f"expected {camera_body_name!r}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="deploy/config/g1_carrybox.yaml")
    parser.add_argument("--urdf")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_deploy_config(args.config)
    urdf = Path(args.urdf).resolve() if args.urdf else cfg.urdf_path
    output = (
        Path(args.output).resolve()
        if args.output
        else cfg.resolve_path(cfg.section("simulation")["generated_robot_mjcf"])
    )
    result = build_robot_mjcf(
        urdf,
        output,
        joint_armature=float(cfg.section("simulation").get("joint_armature", 0.01)),
        camera_config=cfg.section("camera"),
    )
    print(f"Generated PhysHSI G1 MJCF: {result}")


if __name__ == "__main__":
    main()
