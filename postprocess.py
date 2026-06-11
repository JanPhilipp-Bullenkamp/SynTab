"""Mesh post-processing pipeline."""
from __future__ import annotations

import math

import numpy as np
import trimesh
import pymeshlab as ml

from config import PostprocessConfig
from sdf_boolean import transfer_vertex_labels_nearest


def rescale_mesh(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh:
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    mesh.apply_transform(matrix)
    return mesh


def remesh_and_smooth(mesh: trimesh.Trimesh, config: PostprocessConfig) -> trimesh.Trimesh:
    """Isotropic remesh then a light Taubin smoothing pass.

    Reduces SDF voxelisation artifacts without heavy mesh shrinkage.
    """
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(mesh.vertices, mesh.faces))
    ms.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        iterations=config.remesh_iterations,
    )
    remeshed = ms.current_mesh()
    out = trimesh.Trimesh(
        vertices=remeshed.vertex_matrix(),
        faces=remeshed.face_matrix(),
    )
    try:
        trimesh.smoothing.filter_taubin(
            out,
            lamb=config.smooth_lambda,
            nu=config.smooth_mu,
            iterations=config.smooth_iterations,
        )
    except Exception:
        pass
    return out


def add_noise(
    mesh: trimesh.Trimesh,
    magnitude: float,
    rng: np.random.Generator,
) -> trimesh.Trimesh:
    """Add per-vertex random displacement bounded by magnitude."""
    raw_noise = rng.standard_normal(size=mesh.vertices.shape)
    scale = rng.uniform(0, magnitude, size=(raw_noise.shape[0], 1))
    norms = np.linalg.norm(raw_noise, axis=1, keepdims=True)
    mesh.vertices += raw_noise / norms * scale
    return mesh


def subdivide_to_target(mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
    """Loop-subdivide until the mesh has at least target_vertices vertices."""
    current = len(mesh.vertices)
    if current >= target_vertices:
        return mesh
    steps = math.ceil(math.log(target_vertices / current, 4))
    for _ in range(steps):
        mesh = mesh.subdivide_loop()
        if len(mesh.vertices) >= target_vertices:
            break
    return mesh


def postprocess_mesh(
    mesh_raw: trimesh.Trimesh,
    labels_raw: np.ndarray,
    *,
    target_vertices: int,
    config: PostprocessConfig,
    rng: np.random.Generator,
    scale: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Run the full post-processing chain and return (mesh_final, labels_final).

    Steps: remesh → smooth → subdivide → label transfer → noise → rescale.
    The label transfer is performed once, from the raw mesh vertices to the
    remeshed/subdivided vertices.  This is the only KD-tree build needed at
    this stage (a separate one happens inside sdf_difference_with_labels for the
    initial raw mesh).
    """
    mesh = remesh_and_smooth(mesh_raw, config)
    mesh = subdivide_to_target(mesh, target_vertices)
    labels = transfer_vertex_labels_nearest(mesh_raw, labels_raw, mesh)
    mesh = add_noise(mesh, magnitude=config.noise_magnitude, rng=rng)
    mesh = rescale_mesh(mesh, scale=scale)
    return mesh, labels
