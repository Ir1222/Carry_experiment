#!/usr/bin/env python3
"""Convert the active G1 visual STL meshes to geometry-only OBJ surfaces.

The generated OBJ files are deliberately not added to the URDF.  They are
geometry assets for downstream sensor/taxel layout.  OBJ face N corresponds
to STL triangle N (both one-based), so a downstream face mask can always be
mapped back to the source STL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence


Vec3 = tuple[float, float, float]
Triangle = tuple[Vec3, Vec3, Vec3]

SENSOR_SURFACE_LINKS = frozenset(
    {
        "head_link",
        "torso_link",
        "waist_yaw_link",
        "pelvis",
        "pelvis_contour_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_roll_link",
    }
)

GENERATED_OBJ_MARKER = (
    "# Geometry-only sensor surface candidate; not a collision mesh."
)


def _read_binary_stl(data: bytes) -> list[Triangle] | None:
    if len(data) < 84:
        return None
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * triangle_count != len(data):
        return None

    triangles: list[Triangle] = []
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        triangles.append(
            (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )
        )
        offset += 50
    return triangles


def _read_ascii_stl(data: bytes) -> list[Triangle]:
    vertices: list[Vec3] = []
    for raw_line in data.decode("utf-8", errors="strict").splitlines():
        fields = raw_line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in fields[1:]))  # type: ignore[arg-type]
    if not vertices or len(vertices) % 3:
        raise ValueError("ASCII STL does not contain a complete triangle list")
    return [
        (vertices[index], vertices[index + 1], vertices[index + 2])
        for index in range(0, len(vertices), 3)
    ]


def read_stl(path: Path) -> list[Triangle]:
    data = path.read_bytes()
    triangles = _read_binary_stl(data)
    if triangles is not None:
        return triangles
    return _read_ascii_stl(data)


def triangle_area(triangle: Triangle) -> float:
    a, b, c = triangle
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def indexed_mesh(
    triangles: Sequence[Triangle],
) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    vertices: list[Vec3] = []
    vertex_indices: dict[Vec3, int] = {}
    faces: list[tuple[int, int, int]] = []
    for triangle in triangles:
        face: list[int] = []
        for vertex in triangle:
            index = vertex_indices.get(vertex)
            if index is None:
                index = len(vertices) + 1
                vertex_indices[vertex] = index
                vertices.append(vertex)
            face.append(index)
        faces.append((face[0], face[1], face[2]))
    return vertices, faces


def topology_counts(
    faces: Iterable[tuple[int, int, int]],
) -> tuple[int, int]:
    edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        edges.update(
            (
                tuple(sorted((a, b))),
                tuple(sorted((b, c))),
                tuple(sorted((c, a))),
            )
        )
    boundary_edges = sum(count == 1 for count in edges.values())
    nonmanifold_edges = sum(count > 2 for count in edges.values())
    return boundary_edges, nonmanifold_edges


def write_obj(
    path: Path,
    source_stl: Path,
    links: Sequence[str],
    vertices: Sequence[Vec3],
    faces: Sequence[tuple[int, int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{GENERATED_OBJ_MARKER}\n")
        stream.write(f"# source_stl: {source_stl.as_posix()}\n")
        stream.write(f"# urdf_links: {', '.join(links)}\n")
        stream.write("# units: meter (inherited from the source G1 asset)\n")
        stream.write("# mapping: OBJ face N == source STL triangle N (one-based)\n")
        stream.write(f"o {path.stem}\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        stream.write(f"g {path.stem}\n")
        for a, b, c in faces:
            stream.write(f"f {a} {b} {c}\n")


def parse_vector(value: str | None, default: str) -> list[float]:
    return [float(item) for item in (value or default).split()]


def active_visual_meshes(urdf_path: Path) -> dict[Path, list[dict[str, object]]]:
    robot = ET.parse(urdf_path).getroot()
    meshes: dict[Path, list[dict[str, object]]] = {}
    for link in robot.findall("link"):
        link_name = link.attrib["name"]
        if link_name not in SENSOR_SURFACE_LINKS:
            continue
        for visual_index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            source = (urdf_path.parent / mesh.attrib["filename"]).resolve()
            origin = visual.find("origin")
            meshes.setdefault(source, []).append(
                {
                    "urdf_link": link_name,
                    "visual_index": visual_index,
                    "origin_xyz": parse_vector(
                        None if origin is None else origin.attrib.get("xyz"), "0 0 0"
                    ),
                    "origin_rpy": parse_vector(
                        None if origin is None else origin.attrib.get("rpy"), "0 0 0"
                    ),
                    "mesh_scale": parse_vector(mesh.attrib.get("scale"), "1 1 1"),
                }
            )
    return meshes


def relative_posix(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def convert(robot_root: Path, urdf_path: Path, output_dir: Path) -> None:
    source_meshes = active_visual_meshes(urdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_obj_names = {f"{path.stem}.obj" for path in source_meshes}
    for obj_path in output_dir.glob("*.obj"):
        if obj_path.name in expected_obj_names:
            continue
        with obj_path.open("r", encoding="utf-8", errors="replace") as stream:
            generated_by_this_script = (
                stream.readline().rstrip("\r\n") == GENERATED_OBJ_MARKER
            )
        if generated_by_this_script:
            obj_path.unlink()

    records: list[dict[str, object]] = []
    for source_path in sorted(source_meshes, key=lambda item: item.name.lower()):
        if not source_path.is_file():
            raise FileNotFoundError(f"URDF visual mesh does not exist: {source_path}")

        references = source_meshes[source_path]
        links = sorted({str(reference["urdf_link"]) for reference in references})
        triangles = read_stl(source_path)
        vertices, faces = indexed_mesh(triangles)
        areas = [triangle_area(triangle) for triangle in triangles]
        nondegenerate_areas = [area for area in areas if area > 1.0e-14]
        boundary_edges, nonmanifold_edges = topology_counts(faces)

        obj_path = output_dir / f"{source_path.stem}.obj"
        write_obj(
            obj_path,
            Path(relative_posix(source_path, robot_root)),
            links,
            vertices,
            faces,
        )

        coordinates = list(zip(*vertices))
        bbox_min = [min(axis) for axis in coordinates]
        bbox_max = [max(axis) for axis in coordinates]
        area_m2 = sum(nondegenerate_areas)
        records.append(
            {
                "source_stl": relative_posix(source_path, robot_root),
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "sensor_surface_obj": relative_posix(obj_path, robot_root),
                "urdf_references": references,
                "triangle_count": len(triangles),
                "obj_vertex_count": len(vertices),
                "surface_area_m2": area_m2,
                "surface_area_cm2": area_m2 * 10_000.0,
                "bbox_min_m": bbox_min,
                "bbox_max_m": bbox_max,
                "degenerate_triangle_count": len(areas) - len(nondegenerate_areas),
                "boundary_edge_count_exact_weld": boundary_edges,
                "nonmanifold_edge_count_exact_weld": nonmanifold_edges,
                "surface_scope": "all triangles in active URDF visual mesh",
                "manual_exposure_review_required": True,
                "face_mapping": "OBJ face N equals STL triangle N (one-based)",
            }
        )

    manifest = {
        "schema_version": 1,
        "purpose": "geometry-only surface meshes for downstream tactile/taxel layout",
        "source_urdf": relative_posix(urdf_path, robot_root),
        "urdf_modified": False,
        "collision_enabled": False,
        "coordinate_transform": "identity",
        "units": "meter (assumed from the source G1 URDF asset)",
        "selected_urdf_links": sorted(SENSOR_SURFACE_LINKS),
        "selection_policy": (
            "Every triangle of the active visual STL for each manually selected "
            "sensor-surface link is exported as a conservative candidate. "
            "Joint-internal and permanently occluded faces must be removed during "
            "manual exposure review."
        ),
        "mesh_count": len(records),
        "meshes": records,
    }
    manifest_path = robot_root / "sensor_surface_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report_path = robot_root / "sensor_surface_area_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "urdf_links",
            "source_stl",
            "sensor_surface_obj",
            "triangle_count",
            "obj_vertex_count",
            "surface_area_m2",
            "surface_area_cm2",
            "degenerate_triangle_count",
            "boundary_edge_count_exact_weld",
            "nonmanifold_edge_count_exact_weld",
            "manual_exposure_review_required",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "urdf_links": ";".join(
                        sorted(
                            {
                                str(reference["urdf_link"])
                                for reference in record["urdf_references"]  # type: ignore[union-attr]
                            }
                        )
                    ),
                    **{name: record[name] for name in fieldnames if name != "urdf_links"},
                }
            )

    print(f"Converted {len(records)} selected sensor-surface STL meshes")
    print(f"OBJ directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Area report: {report_path}")


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, default=default_root)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Defaults to <robot-root>/urdf/g1_29dof.urdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <robot-root>/meshes/sensor_surface_obj",
    )
    args = parser.parse_args()

    robot_root = args.robot_root.resolve()
    urdf_path = (
        args.urdf.resolve()
        if args.urdf is not None
        else robot_root / "urdf" / "g1_29dof.urdf"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else robot_root / "meshes" / "sensor_surface_obj"
    )
    convert(robot_root, urdf_path, output_dir)


if __name__ == "__main__":
    main()
