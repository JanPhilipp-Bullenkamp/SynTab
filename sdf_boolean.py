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
from concurrent.futures import ThreadPoolExecutor
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


def _rasterize_to_indices(
    cutter: trimesh.Trimesh,
    origin: np.ndarray,
    pitch: float,
    dims: tuple,
    method: str,
) -> np.ndarray | None:
    """Pure function: return voxel indices (K, 3) inside cutter, or None.

    Does not write to any shared array — safe to call from threads.

    Uses grid-clipped column-ray casting: shoots rays along the grid axis with
    the most voxels (fewest column rays), restricted to the grid-clipped
    bounding box of the cutter.  This is much faster than `cutter.voxelized()`
    when cutter bounding boxes extend far outside the grid (common for pyramid
    cutters that span well beyond the tablet surface).

    Column processing is fully vectorised — no Python loops over columns.

    Falls back to `cutter.voxelized()` or `cutter.contains()` on ray failure.
    """
    bounds = cutter.bounds
    dims_arr = np.array(dims, dtype=np.int32)
    imin = np.maximum(np.floor((bounds[0] - origin) / pitch).astype(np.int32) - 1, 0)
    imax = np.minimum(np.ceil( (bounds[1] - origin) / pitch).astype(np.int32) + 1, dims_arr - 1)
    if np.any(imax < imin):
        return None

    # Choose ray axis = grid axis with most voxels → fewest (col_a × col_b) columns.
    # e.g. grid (173, 200, 64): ray_ax=1 gives 173*64=11 072 cols vs 34 600 for z-rays.
    ray_ax = int(np.argmax(dims_arr))
    ca0, ca1 = [i for i in range(3) if i != ray_ax]

    ia0 = np.arange(int(imin[ca0]), int(imax[ca0]) + 1, dtype=np.int32)
    ia1 = np.arange(int(imin[ca1]), int(imax[ca1]) + 1, dtype=np.int32)
    IA0, IA1 = np.meshgrid(ia0, ia1, indexing="ij")   # (n0, n1)
    flat_a0 = IA0.ravel()
    flat_a1 = IA1.ravel()
    n_rays = len(flat_a0)

    ray_origins = np.empty((n_rays, 3), dtype=np.float64)
    ray_origins[:, ca0] = origin[ca0] + flat_a0 * pitch
    ray_origins[:, ca1] = origin[ca1] + flat_a1 * pitch
    ray_start = min(float(bounds[0][ray_ax]) - pitch,
                    float(origin[ray_ax]) + int(imin[ray_ax]) * pitch - pitch)
    ray_origins[:, ray_ax] = ray_start

    ray_dirs = np.zeros((n_rays, 3), dtype=np.float64)
    ray_dirs[:, ray_ax] = 1.0

    try:
        hit_locs, hit_ray_idx, _ = cutter.ray.intersects_location(
            ray_origins, ray_dirs, multiple_hits=True
        )

        if len(hit_locs) > 0:
            # Sort hits by (column, ray-axis coord).
            sort_order = np.lexsort((hit_locs[:, ray_ax], hit_ray_idx))
            s_col = hit_ray_idx[sort_order]
            s_r   = hit_locs[sort_order, ray_ax]

            # Find which columns have exactly 2 hits (enter + exit for closed convex mesh).
            u_col, first_pos, counts = np.unique(s_col, return_index=True, return_counts=True)
            two_hit = counts == 2
            if not np.any(two_hit):
                hit_locs = np.empty((0, 3))   # trigger fallback
            else:
                fp = first_pos[two_hit]
                r_in  = s_r[fp]
                r_out = s_r[fp + 1]

                o_r    = float(origin[ray_ax])
                ir_min = int(imin[ray_ax])
                ir_max = int(imax[ray_ax])

                ir_in  = np.maximum(ir_min, np.ceil( (r_in  - o_r) / pitch).astype(np.int32))
                ir_out = np.minimum(ir_max, np.floor((r_out - o_r) / pitch).astype(np.int32))

                valid = ir_in <= ir_out
                if not np.any(valid):
                    return None

                col_v  = u_col[two_hit][valid]
                ir_in_v  = ir_in[valid]
                ir_out_v = ir_out[valid]
                ia0_v = flat_a0[col_v]
                ia1_v = flat_a1[col_v]

                # Fully vectorised range generation — no Python loop over columns.
                n_vox = (ir_out_v - ir_in_v + 1).astype(np.int64)
                total_vox = int(n_vox.sum())
                if total_vox == 0:
                    return None

                g_starts = np.empty(len(n_vox), dtype=np.int64)
                g_starts[0] = 0
                if len(n_vox) > 1:
                    g_starts[1:] = np.cumsum(n_vox[:-1])

                all_idx = np.arange(total_vox, dtype=np.int64)
                within  = all_idx - np.repeat(g_starts, n_vox)

                ir_flat  = (np.repeat(ir_in_v, n_vox) + within).astype(np.int32)
                ia0_flat = np.repeat(ia0_v, n_vox)
                ia1_flat = np.repeat(ia1_v, n_vox)

                result = np.empty((total_vox, 3), dtype=np.int32)
                result[:, ca0]   = ia0_flat
                result[:, ca1]   = ia1_flat
                result[:, ray_ax] = ir_flat
                return result

    except Exception:
        pass

    # Fallback 1: cutter.voxelized (may be slow for large bounding boxes)
    if method == "voxelize":
        try:
            vg = cutter.voxelized(pitch).fill()
            mat = vg.matrix.astype(bool)
            if not np.any(mat):
                return None
            vg_origin = _voxelgrid_origin(vg)
            off = np.round((vg_origin - origin) / pitch).astype(int)
            local_idx = np.argwhere(mat)
            global_idx = local_idx + off
            mask = np.all((global_idx >= 0) & (global_idx < dims_arr), axis=1)
            return global_idx[mask].astype(np.int32) if np.any(mask) else None
        except Exception:
            pass

    # Fallback 2: contains — tests all grid points in the clipped bounding box
    ia0_all = np.arange(int(imin[ca0]), int(imax[ca0]) + 1)
    ia1_all = np.arange(int(imin[ca1]), int(imax[ca1]) + 1)
    ir_all  = np.arange(int(imin[ray_ax]), int(imax[ray_ax]) + 1)
    G0, G1, GR = np.meshgrid(ia0_all, ia1_all, ir_all, indexing="ij")
    pts_i = np.empty((G0.size, 3), dtype=np.int32)
    pts_i[:, ca0]    = G0.ravel()
    pts_i[:, ca1]    = G1.ravel()
    pts_i[:, ray_ax] = GR.ravel()
    pts_world = origin[None, :] + pitch * pts_i.astype(np.float64)
    inside = cutter.contains(pts_world)
    return pts_i[inside].astype(np.int32) if np.any(inside) else None


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
    raise ValueError(f"method must be 'voxelize' or 'contains', got '{method}'")


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

    raster_threads = int(config.raster_threads)
    raster_t0 = time.perf_counter()
    dims_tuple = tuple(int(d) for d in dims)

    def _rasterize_one(args):
        idx, cutter = args
        cbmin, cbmax = cutter.bounds
        if np.any(cbmax < bmin) or np.any(cbmin > bmax):
            return idx, None, None
        cid = int(getattr(cutter, "wedge_id", idx + 1))
        return idx, cid, _rasterize_to_indices(cutter, origin, pitch, dims_tuple, cutter_raster)

    if raster_threads > 1:
        with ThreadPoolExecutor(max_workers=raster_threads) as ex:
            results = list(ex.map(_rasterize_one, enumerate(cutter_meshes)))
    else:
        results = [_rasterize_one(item) for item in enumerate(cutter_meshes)]

    skipped = 0
    for idx, cid, inside_idx in results:
        if cid is None:
            skipped += 1
            continue
        if inside_idx is None or inside_idx.size == 0:
            continue
        ix, iy, iz = inside_idx[:, 0], inside_idx[:, 1], inside_idx[:, 2]
        occB[ix, iy, iz] = True
        if order_bias:
            label_grid[ix, iy, iz] = cid
        else:
            mask = label_grid[ix, iy, iz] == 0
            label_grid[ix[mask], iy[mask], iz[mask]] = cid

        if debug and ((idx + 1) % max(1, int(debug_every)) == 0):
            _dbg(
                f"[sdf] merged {idx + 1}/{len(cutter_meshes)} "
                f"elapsed={time.perf_counter() - raster_t0:.1f}s"
            )

    _dbg(
        f"[sdf] raster done: cutters={len(cutter_meshes)} skipped={skipped} "
        f"labelled_voxels={int(np.count_nonzero(label_grid)):,} "
        f"elapsed={time.perf_counter() - raster_t0:.2f}s"
    )

    # sdfA and sdfB are independent — compute in parallel threads.
    # scipy.ndimage.distance_transform_edt releases the GIL, so two threads
    # run truly concurrently on multi-core hardware.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futA = ex.submit(sdf_from_occupancy, occA, pitch=pitch, dtype=sdf_dtype)
        futB = ex.submit(sdf_from_occupancy, occB, pitch=pitch, dtype=sdf_dtype)
        sdfA = futA.result()
        sdfB = futB.result()
    _dbg(f"[sdf] sdfA+sdfB done (parallel) {time.perf_counter() - t0:.2f}s")
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
