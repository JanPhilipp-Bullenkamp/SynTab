"""SDF-based boolean difference with per-vertex wedge labelling.

Key difference from the original sdf_imprint.py:
  - Label tracking uses a single `label_grid` array (int32, same shape as the
    voxel grid) instead of accumulating per-cutter index lists.  This bounds
    peak memory to O(grid_cells × 4 bytes) regardless of how many wedges
    overlap a given voxel, versus O(total_rasterised_voxels × 16 bytes) in the
    old approach.
  - `_dedup_voxel_labels` is no longer needed.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from scipy import ndimage

from config import SdfConfig


def _voxelgrid_origin(vg: trimesh.voxel.VoxelGrid) -> np.ndarray:
    if hasattr(vg, "origin"):
        return np.asarray(vg.origin, dtype=np.float64)
    if hasattr(vg, "translation"):
        return np.asarray(vg.translation, dtype=np.float64)
    if hasattr(vg, "transform"):
        return np.asarray(vg.transform[:3, 3], dtype=np.float64)
    if hasattr(vg, "bounds"):
        return np.asarray(vg.bounds[0], dtype=np.float64)
    raise AttributeError("VoxelGrid origin/translation/transform not available")


def _occupancy_superquadric(
    *,
    dims: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    a: float,
    b: float,
    c: float,
    epsilon: float,
    eta: float,
) -> np.ndarray:
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    x = origin[0] + np.arange(nx, dtype=np.float64) * pitch
    y = origin[1] + np.arange(ny, dtype=np.float64) * pitch
    z = origin[2] + np.arange(nz, dtype=np.float64) * pitch

    ax = max(a, 1e-9)
    by = max(b, 1e-9)
    cz = max(c, 1e-9)

    exp_xy = 2.0 / max(eta, 1e-9)
    exp_z = 2.0 / max(epsilon, 1e-9)
    power = max(eta, 1e-9) / max(epsilon, 1e-9)

    X = (np.abs(x) / ax) ** exp_xy
    Y = (np.abs(y) / by) ** exp_xy
    Z = (np.abs(z) / cz) ** exp_z

    XY = (X[:, None] + Y[None, :]) ** power
    occ = np.empty((nx, ny, nz), dtype=bool)

    # Chunk over z to avoid large temporary broadcast arrays.
    base = max(1, nx * ny)
    z_chunk = max(1, min(nz, int(32_000_000 // base)))
    for z0 in range(0, nz, z_chunk):
        z1 = min(nz, z0 + z_chunk)
        occ[:, :, z0:z1] = (XY[:, :, None] + Z[None, None, z0:z1]) <= 1.0
    return occ


def _trilinear_sample(
    volume: np.ndarray, pts_world: np.ndarray, origin: np.ndarray, pitch: float
) -> np.ndarray:
    pts = (pts_world - origin[None, :]) / pitch
    coords = np.vstack([pts[:, 0], pts[:, 1], pts[:, 2]])
    return ndimage.map_coordinates(volume, coords, order=1, mode="nearest")


def sdf_from_occupancy(
    occ: np.ndarray, pitch: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    dist_out = np.empty(occ.shape, dtype=np.float64)
    ndimage.distance_transform_edt(~occ, sampling=pitch, distances=dist_out)
    dist_in = np.empty(occ.shape, dtype=np.float64)
    ndimage.distance_transform_edt(occ, sampling=pitch, distances=dist_in)
    dist_out[occ] = -dist_in[occ]
    if dtype == np.float64:
        return dist_out
    return dist_out.astype(dtype, copy=False)


def _rasterize_cutter_contains(
    cutter: trimesh.Trimesh,
    occB: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    return_indices: bool = False,
) -> np.ndarray | None:
    bounds = cutter.bounds
    imin = np.floor((bounds[0] - origin) / pitch).astype(int) - 1
    imax = np.ceil((bounds[1] - origin) / pitch).astype(int) + 1
    imin = np.maximum(imin, 0)
    imax = np.minimum(imax, np.array(occB.shape) - 1)

    if np.any(imax < imin):
        return None

    xs = np.arange(imin[0], imax[0] + 1)
    ys = np.arange(imin[1], imax[1] + 1)
    zs = np.arange(imin[2], imax[2] + 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)
    pts_world = origin[None, :] + pitch * pts

    inside = cutter.contains(pts_world)
    if not np.any(inside):
        return None

    inside_idx = pts[inside].astype(int)
    ix, iy, iz = inside_idx[:, 0], inside_idx[:, 1], inside_idx[:, 2]
    occB[ix, iy, iz] = True

    if return_indices:
        return inside_idx
    return None


def _rasterize_cutter_voxelized(
    cutter: trimesh.Trimesh,
    occB: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    return_indices: bool = False,
) -> np.ndarray | None:
    vg = cutter.voxelized(pitch)
    vg_f = vg.fill()

    mat = vg_f.matrix.astype(bool)
    if not np.any(mat):
        return None

    vg_origin = _voxelgrid_origin(vg_f)
    off = np.round((vg_origin - origin) / pitch).astype(int)

    x0, y0, z0 = off
    x1, y1, z1 = off + np.array(mat.shape)

    dims = occB.shape
    gx0, gy0, gz0 = max(x0, 0), max(y0, 0), max(z0, 0)
    gx1, gy1, gz1 = min(x1, dims[0]), min(y1, dims[1]), min(z1, dims[2])

    if (gx1 <= gx0) or (gy1 <= gy0) or (gz1 <= gz0):
        return None

    mx0, my0, mz0 = gx0 - x0, gy0 - y0, gz0 - z0
    mx1, my1, mz1 = mx0 + (gx1 - gx0), my0 + (gy1 - gy0), mz0 + (gz1 - gz0)

    sub = mat[mx0:mx1, my0:my1, mz0:mz1]
    if not np.any(sub):
        return None

    occB[gx0:gx1, gy0:gy1, gz0:gz1] |= sub

    if not return_indices:
        return None

    inside_local = np.argwhere(sub)
    if inside_local.size == 0:
        return None

    return (inside_local + np.array([gx0, gy0, gz0], dtype=np.int64)).astype(np.int32)


def rasterize_cutter_into_grid(
    cutter: trimesh.Trimesh,
    occB: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    *,
    method: str = "voxelize",
    return_indices: bool = False,
) -> np.ndarray | None:
    if method == "voxelize":
        try:
            return _rasterize_cutter_voxelized(
                cutter=cutter, occB=occB, origin=origin, pitch=pitch,
                return_indices=return_indices,
            )
        except Exception:
            return _rasterize_cutter_contains(
                cutter=cutter, occB=occB, origin=origin, pitch=pitch,
                return_indices=return_indices,
            )
    if method == "contains":
        return _rasterize_cutter_contains(
            cutter=cutter, occB=occB, origin=origin, pitch=pitch,
            return_indices=return_indices,
        )
    raise ValueError("method must be 'voxelize' or 'contains'")


def mesh_from_sdf(sdf: np.ndarray, origin: np.ndarray, pitch: float) -> trimesh.Trimesh:
    try:
        from skimage.measure import marching_cubes

        sdf_t = np.transpose(sdf, (2, 1, 0))
        verts_zyx, faces, _, _ = marching_cubes(sdf_t, level=0.0, spacing=(pitch, pitch, pitch))
        verts_xyz = np.stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]], axis=1)
        verts_xyz = verts_xyz + origin[None, :]
        return trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)
    except Exception:
        occ = sdf <= 0.0
        m = trimesh.voxel.ops.matrix_to_marching_cubes(occ, pitch=pitch)
        m.apply_translation(origin)
        return m


def sdf_difference_with_labels(
    cutter_meshes: List[trimesh.Trimesh],
    *,
    config: SdfConfig,
    pitch: float,
    base_implicit: Dict[str, float],
    order_bias: bool = True,
    return_face_labels: bool = False,
    sdf_dtype: np.dtype = np.float32,
    cutter_raster: str = "voxelize",
    tight_bounds: bool = True,
    clip_bounds_to_base: bool = True,
    max_voxels: int | None = None,
    debug: bool = False,
    debug_every: int = 200,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    def _dbg(msg: str) -> None:
        if debug:
            print(msg, flush=True)

    eps_active = config.eps_active_fraction * pitch
    base_label = config.base_label
    cutter_label_offset = config.cutter_label_offset
    padding = config.padding
    max_grid = config.max_grid
    base_bounds_margin = config.base_bounds_margin

    a = float(base_implicit["a"])
    b = float(base_implicit["b"])
    c = float(base_implicit["c"])
    base_bmin = np.array([-a, -b, -c], dtype=np.float64)
    base_bmax = np.array([a, b, c], dtype=np.float64)

    if tight_bounds and cutter_meshes:
        cutter_mins = np.vstack([cm.bounds[0] for cm in cutter_meshes])
        cutter_maxs = np.vstack([cm.bounds[1] for cm in cutter_meshes])
        bmin = np.minimum(base_bmin, cutter_mins.min(axis=0))
        bmax = np.maximum(base_bmax, cutter_maxs.max(axis=0))
        if debug:
            base_span = base_bmax - base_bmin
            tight_span = bmax - bmin
            span_ratio = tight_span / np.maximum(base_span, 1e-12)
            low_overshoot = np.maximum(0.0, base_bmin[None, :] - cutter_mins)
            high_overshoot = np.maximum(0.0, cutter_maxs - base_bmax[None, :])
            overshoot = np.maximum(low_overshoot, high_overshoot)
            overshoot_mag = overshoot.max(axis=1) if overshoot.size > 0 else np.zeros((0,))
            n_outliers = int(np.count_nonzero(overshoot_mag > 1e-9)) if overshoot_mag.size else 0
            max_out = float(overshoot_mag.max()) if overshoot_mag.size else 0.0
            p95_out = float(np.percentile(overshoot_mag, 95)) if overshoot_mag.size else 0.0
            _dbg(
                f"[sdf] tight bounds span_ratio={np.round(span_ratio, 3).tolist()} "
                f"outliers={n_outliers}/{len(cutter_meshes)} "
                f"max_outlier={max_out:.4f} p95={p95_out:.4f}"
            )
    else:
        bmin, bmax = base_bmin, base_bmax

    if clip_bounds_to_base:
        margin = max(0.0, float(base_bounds_margin))
        bmin = np.maximum(bmin, base_bmin - margin)
        bmax = np.minimum(bmax, base_bmax + margin)

    bmin_raw = bmin.astype(np.float64, copy=True)
    bmax_raw = bmax.astype(np.float64, copy=True)
    pitch_in = float(pitch)

    for _ in range(4):
        pad = padding * pitch
        bmin = bmin_raw - pad
        bmax = bmax_raw + pad
        dims = np.ceil((bmax - bmin) / pitch).astype(int) + 1

        changed = False
        if max_grid is not None and int(max_grid) > 1:
            mg = int(max_grid)
            max_dim = int(dims.max())
            if max_dim > mg:
                pitch *= (max_dim - 1) / float(mg - 1)
                changed = True

        nvox_try = int(np.prod(dims, dtype=np.int64))
        if max_voxels is not None and int(max_voxels) > 0 and nvox_try > int(max_voxels):
            pitch *= (nvox_try / float(int(max_voxels))) ** (1.0 / 3.0)
            changed = True

        if not changed:
            break

    pad = padding * pitch
    bmin = bmin_raw - pad
    bmax = bmax_raw + pad
    dims = np.ceil((bmax - bmin) / pitch).astype(int) + 1
    origin = bmin.astype(np.float64)
    nvox = int(np.prod(dims, dtype=np.int64))

    if abs(pitch - pitch_in) > 1e-15:
        _dbg(
            f"[sdf] pitch adjusted {pitch_in:.7f} → {pitch:.7f} "
            f"(factor={pitch / pitch_in:.3f}x, max_grid={max_grid})"
        )
    _dbg(
        f"[sdf] grid dims={tuple(int(x) for x in dims.tolist())} nvox={nvox:,} "
        f"pitch={pitch:.7f} bmin={np.round(bmin, 4).tolist()}"
    )

    occA = _occupancy_superquadric(
        dims=dims, origin=origin, pitch=pitch,
        a=a, b=b, c=c,
        epsilon=float(base_implicit["epsilon"]),
        eta=float(base_implicit["eta"]),
    )

    occB = np.zeros(tuple(dims), dtype=bool)
    # label_grid stores the wedge id for each voxel; 0 = unlabelled.
    # Using a dense grid eliminates the old List[np.ndarray] accumulation and
    # _dedup_voxel_labels, bounding peak memory to grid_cells × 4 bytes.
    label_grid = np.zeros(tuple(dims), dtype=np.int32)

    skipped = 0
    raster_t0 = time.perf_counter()

    for idx, cutter in enumerate(cutter_meshes):
        cbmin, cbmax = cutter.bounds
        if np.any(cbmax < bmin) or np.any(cbmin > bmax):
            skipped += 1
            continue

        cid = int(getattr(cutter, "wedge_id", idx + 1))

        inside_idx = rasterize_cutter_into_grid(
            cutter=cutter, occB=occB, origin=origin, pitch=pitch,
            method=cutter_raster, return_indices=True,
        )

        if inside_idx is not None and inside_idx.size > 0:
            ix = inside_idx[:, 0]
            iy = inside_idx[:, 1]
            iz = inside_idx[:, 2]
            if order_bias:
                # last writer wins — later wedge id overwrites earlier
                label_grid[ix, iy, iz] = cid
            else:
                # first writer wins — only write to unlabelled voxels
                mask = label_grid[ix, iy, iz] == 0
                label_grid[ix[mask], iy[mask], iz[mask]] = cid

        if debug and ((idx + 1) % max(1, int(debug_every)) == 0):
            _dbg(
                f"[sdf] rasterized {idx + 1}/{len(cutter_meshes)} "
                f"elapsed={time.perf_counter() - raster_t0:.1f}s"
            )

    _dbg(
        f"[sdf] raster done: cutters={len(cutter_meshes)} skipped={skipped} "
        f"labelled_voxels={int(np.count_nonzero(label_grid)):,} "
        f"elapsed={time.perf_counter() - raster_t0:.2f}s"
    )

    t0 = time.perf_counter()
    sdfA = sdf_from_occupancy(occA, pitch=pitch, dtype=sdf_dtype)
    _dbg(f"[sdf] sdfA done {time.perf_counter() - t0:.2f}s")
    t0 = time.perf_counter()
    sdfB = sdf_from_occupancy(occB, pitch=pitch, dtype=sdf_dtype)
    _dbg(f"[sdf] sdfB done {time.perf_counter() - t0:.2f}s")
    del occA
    del occB

    sdfDiff = np.maximum(sdfA, -sdfB)
    out_mesh = mesh_from_sdf(sdfDiff, origin=origin, pitch=pitch)
    out_mesh.remove_unreferenced_vertices()
    out_mesh.fix_normals()

    verts = out_mesh.vertices
    a_v = _trilinear_sample(sdfA, verts, origin, pitch)
    b_v = _trilinear_sample(-sdfB, verts, origin, pitch)
    carved_v = b_v > (a_v + eps_active)
    _dbg(f"[sdf] carved vertices: {int(np.count_nonzero(carved_v))}/{len(verts)}")

    v_labels = np.full(len(verts), base_label, dtype=np.int32)

    if np.any(label_grid > 0) and np.any(carved_v):
        label_mask = label_grid > 0
        coords = origin[None, :] + pitch * np.argwhere(label_mask).astype(np.float64)
        ids_flat = label_grid[label_mask]
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(coords)
            _, nn = tree.query(verts[carved_v], k=1)
            ids_v = ids_flat[nn]
            carved_labels = np.where(
                ids_v > 0, ids_v + cutter_label_offset, base_label
            ).astype(np.int32)
            v_labels[carved_v] = carved_labels
        except Exception:
            pass

    if return_face_labels:
        faces = out_mesh.faces
        if len(faces) == 0:
            face_labels = np.zeros((0,), dtype=np.int32)
        else:
            vf = v_labels[faces]
            l0, l1, l2 = vf[:, 0], vf[:, 1], vf[:, 2]
            eq01 = l0 == l1
            eq02 = l0 == l2
            eq12 = l1 == l2
            face_labels = np.where(
                eq01 | eq02, l0, np.where(eq12, l1, l0)
            ).astype(np.int32, copy=False)
            no_majority = ~(eq01 | eq02 | eq12)
            if np.any(no_majority):
                i_nm = np.where(no_majority)[0]
                pick = np.where(
                    l0[i_nm] != base_label, l0[i_nm],
                    np.where(l1[i_nm] != base_label, l1[i_nm],
                             np.where(l2[i_nm] != base_label, l2[i_nm], l0[i_nm]))
                )
                face_labels[i_nm] = pick
        del sdfA, sdfB
        return out_mesh, face_labels

    del sdfA, sdfB
    return out_mesh, v_labels


def transfer_vertex_labels_nearest(
    src_mesh: trimesh.Trimesh,
    src_labels: np.ndarray,
    dst_mesh: trimesh.Trimesh,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(src_mesh.vertices)
        _, idx = tree.query(dst_mesh.vertices, k=1)
        return src_labels[idx].astype(np.int32)
    except Exception:
        pass

    kdtree = getattr(src_mesh, "kdtree", None)
    if kdtree is not None:
        _, idx = kdtree.query(dst_mesh.vertices)
        return src_labels[idx].astype(np.int32)

    labels = np.zeros((len(dst_mesh.vertices),), dtype=np.int32)
    for i, v in enumerate(dst_mesh.vertices):
        d = np.linalg.norm(src_mesh.vertices - v[None, :], axis=1)
        labels[i] = src_labels[int(np.argmin(d))]
    return labels
