from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import trimesh

from config import WedgeCarvingConfig
from directions import DIRECTION, face_from_normal


@dataclass(frozen=True)
class WedgePlacement3D:
    wedge_id: int
    face: DIRECTION
    sign_index: int
    wedge_index: int
    wedge_type: str
    size_scale: float
    position: np.ndarray
    normal: np.ndarray
    direction: np.ndarray
    depth: float
    angle: float
    tilt_angle: float = 0.0
    sign_code: Optional[str] = None
    line_index: Optional[int] = None
    char_index: Optional[int] = None


def create_wedge(size: float = 5.0, height: Optional[float] = None) -> trimesh.Trimesh:
    if height is None:
        height = size

    base = np.array(
        [[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]],
        dtype=float,
    )
    apex = np.array([[size / 2.0, size / 2.0, height]], dtype=float)
    vertices = np.vstack([base, apex])
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=int,
    )

    wedge = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    wedge.update_faces(wedge.unique_faces())
    wedge.remove_unreferenced_vertices()
    wedge.fix_normals()
    return wedge.convex_hull


def get_wedge_trafo(
    vertex: np.ndarray,
    normal: np.ndarray,
    depth: float,
    direction: np.ndarray,
    angle: float,
    mesh_size: float = 5.0,
    beta: float = -45.0,
    collinear_threshold: float = 0.9,
) -> np.ndarray:
    s = float(mesh_size)
    if abs(s) < 1e-12:
        raise ValueError("mesh_size must be non-zero")

    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)

    d = np.asarray(direction, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        d = np.array([1.0, 0.0, 0.0], dtype=float)
        d_norm = 1.0
    d /= d_norm

    t = d - np.dot(d, n) * n
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-8:
        a = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(a, n)) > collinear_threshold:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
        t = a - np.dot(a, n) * n
        t_norm = np.linalg.norm(t)
    t /= t_norm

    a_rad = np.deg2rad(angle)
    u = t * np.cos(a_rad) + n * np.sin(a_rad)
    w = -t * np.sin(a_rad) + n * np.cos(a_rad)
    u /= np.linalg.norm(u)
    w /= np.linalg.norm(w)

    o = np.cross(u, w)
    o /= np.linalg.norm(o)

    def rodrigues(x, axis, angle_rad):
        axis = axis / np.linalg.norm(axis)
        return (
            x * np.cos(angle_rad)
            + np.cross(axis, x) * np.sin(angle_rad)
            + axis * np.dot(axis, x) * (1 - np.cos(angle_rad))
        )

    b_rad = np.deg2rad(beta)
    w = rodrigues(w, u, b_rad)
    o = rodrigues(o, u, b_rad)

    v = np.asarray(vertex, dtype=float)
    q1 = v - n * depth
    q2 = q1 + u * s
    q3 = q1 + w * (np.sqrt(2.0) * s)
    q4 = 0.5 * (q1 + q3) - o * (s / np.sqrt(2.0))

    t0 = q1
    a0 = (q2 - q1) / s
    a2 = (q4 - q1) / s
    a1 = (q3 - t0 - a2 * s) / s

    A = np.column_stack([a0, a1, a2])
    T = np.eye(4, dtype=float)
    T[:3, :3] = A
    T[:3, 3] = t0
    return T


def build_wedge_meshes(
    wedges: Iterable[WedgePlacement3D],
    config: WedgeCarvingConfig,
) -> List[trimesh.Trimesh]:
    """Build a cutter trimesh for each wedge; set .wedge_id on each mesh."""
    meshes: List[trimesh.Trimesh] = []
    for wedge in wedges:
        size = config.base_wedge_size * wedge.size_scale
        mesh = create_wedge(size=size)
        trafo = get_wedge_trafo(
            wedge.position,
            wedge.normal,
            wedge.depth,
            wedge.direction,
            wedge.angle,
            mesh_size=size,
            beta=wedge.tilt_angle,
            collinear_threshold=config.tangent_collinear_threshold,
        )
        mesh.apply_transform(trafo)
        mesh.wedge_id = wedge.wedge_id  # used by sdf_boolean for labelling
        meshes.append(mesh)
    return meshes


def map_placed_wedges3d_to_surface(
    placed_wedges3d,
    *,
    config: WedgeCarvingConfig,
    flip_tail_direction_x: bool = False,
    rng: np.random.Generator,
) -> List[WedgePlacement3D]:
    """Sample random carving parameters and convert PlacedWedge3D → WedgePlacement3D."""
    wedges_3d: List[WedgePlacement3D] = []
    next_id = 1

    for pw in placed_wedges3d:
        depth = float(rng.uniform(config.min_depth, config.max_depth))
        angle = float(rng.uniform(config.min_angle, config.max_angle))
        tilt_angle = float(rng.uniform(config.min_tilt_angle, config.max_tilt_angle))

        face = getattr(pw, "face", None)
        if face is None:
            face = face_from_normal(pw.normal3)

        direction = np.array(pw.tail_dir3, dtype=float)
        if flip_tail_direction_x:
            direction[0] *= -1.0

        wedges_3d.append(
            WedgePlacement3D(
                wedge_id=next_id,
                face=face,
                sign_index=int(pw.sign_index),
                wedge_index=int(pw.wedge_index),
                wedge_type=str(pw.wedge_type),
                size_scale=float(pw.size_scale),
                position=np.array(pw.pos3, dtype=float),
                normal=np.array(pw.normal3, dtype=float),
                direction=direction,
                depth=depth,
                angle=angle,
                tilt_angle=tilt_angle,
                sign_code=str(getattr(pw, "sign_code", "")) or None,
                line_index=int(pw.line_index) if getattr(pw, "line_index", None) is not None else None,
                char_index=int(pw.char_index) if getattr(pw, "char_index", None) is not None else None,
            )
        )
        next_id += 1

    return wedges_3d
