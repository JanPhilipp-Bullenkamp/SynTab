"""Synthetic cuneiform tablet dataset generator.

Entry point:

    python generate_tablets.py

Or from code:

    from generate_tablets import generate_dataset
    from config import TabletConfig, SdfConfig

    generate_dataset(
        cfg=TabletConfig(sdf=SdfConfig(max_grid=400)),
        paleocodes_path="paleocodes.json",
        output_dirs={"mesh": "./out/plys/", "labels": "./out/labels/",
                     "annotations": "./out/annos/", "configs": "./out/configs/"},
        num_tablets=5,
    )
"""
from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional

import numpy as np

from annotation_export import export_webannotation_3d
from config import TabletConfig
from directions import DIRECTION
from imprint import build_wedge_meshes, map_placed_wedges3d_to_surface
from layout3d import layout_signs_on_superquadric
from paleocodage import parse_signs
from postprocess import postprocess_mesh
from signs import drop_signs_without_wedges, load_sign_reference_metadata, resolve_sign_codes
from sdf_boolean import sdf_difference_with_labels
from util import export_labels


def _compute_pitch(height: float, width: float, depth: float, cfg: TabletConfig) -> float:
    base_extent = 2.0 * max(height, width, depth)
    denom = max(1.0, (cfg.sdf.max_grid - 1) - 2.0 * cfg.sdf.padding)
    return max(cfg.sdf.target_pitch, base_extent / denom)


def _generate_single_tablet(
    tablet_index: int,
    seed: int,
    cfg: TabletConfig,
    mesh_path: str,
    labels_path: str,
    annotations_path: Optional[str],
    config_path: Optional[str],
    sign_source: str,
    sign_list: Optional[List[str]],
    paleocodes_path: str,
    faces: List[DIRECTION],
    sign_reference: Dict[str, Dict[str, str]],
) -> None:
    def _log(msg: str) -> None:
        print(msg, flush=True)

    tag = f"[tablet {tablet_index}]"

    if os.path.exists(mesh_path) and os.path.exists(labels_path):
        _log(f"{tag} already exists, skipping")
        return

    rng = np.random.default_rng(seed)
    tablet_t0 = time.perf_counter()
    _log(f"{tag} start")

    imprinted_mesh_raw = None
    labels_raw = None
    wedges_3d = None
    height = width = depth = epsilon = eta = scale = None

    max_attempts = cfg.generation.max_attempts
    for attempt in range(1, max_attempts + 1):
        atag = f"{tag}[attempt {attempt}/{max_attempts}]"
        t0 = time.perf_counter()
        try:
            height  = float(rng.uniform(*cfg.geometry.height_range))
            width   = float(rng.uniform(*cfg.geometry.width_range))
            depth   = float(rng.uniform(*cfg.geometry.depth_range))
            epsilon = float(rng.uniform(*cfg.geometry.epsilon_range))
            eta     = float(rng.uniform(*cfg.geometry.eta_range))
            scale   = float(rng.uniform(*cfg.geometry.scale_range))
            _log(
                f"{atag} h={height:.4f} w={width:.4f} d={depth:.4f} "
                f"eps={epsilon:.4f} eta={eta:.4f} scale={scale:.2f}"
            )

            sign_codes = resolve_sign_codes(
                sign_source, sign_list, paleocodes_path, rng, cfg.generation
            )
            parsed_raw = parse_signs(sign_codes, config=cfg.paleocodage)
            parsed_signs, n_dropped = drop_signs_without_wedges(parsed_raw)
            _log(f"{atag} signs={len(parsed_signs)} (dropped_empty={n_dropped})")
            if not parsed_signs:
                raise ValueError("No valid signs with wedge strokes after filtering")

            placed = layout_signs_on_superquadric(
                parsed_signs,
                a=height, b=width, c=depth,
                epsilon=epsilon, eta=eta,
                config=cfg.layout,
                rng=rng,
                allowed_faces=set(faces),
            )
            _log(f"{atag} layout done wedges={len(placed)}")

            wedges_3d = map_placed_wedges3d_to_surface(
                placed,
                config=cfg.carving,
                flip_tail_direction_x=cfg.generation.flip_tail_direction_x,
                rng=rng,
            )
            if not wedges_3d:
                raise ValueError("No wedge placements mapped to surface")

            wedge_meshes = build_wedge_meshes(wedges_3d, config=cfg.carving)
            _log(f"{atag} wedge meshes built count={len(wedge_meshes)}")

            pitch = _compute_pitch(height, width, depth, cfg)
            _log(f"{atag} sdf start pitch={pitch:.7f}")
            imprinted_mesh_raw, labels_raw = sdf_difference_with_labels(
                wedge_meshes,
                config=cfg.sdf,
                pitch=pitch,
                base_implicit={
                    "a": height, "b": width, "c": depth,
                    "epsilon": epsilon, "eta": eta,
                },
                debug=cfg.generation.debug,
            )
            _log(
                f"{atag} sdf done verts={len(imprinted_mesh_raw.vertices)} "
                f"dt={time.perf_counter() - t0:.2f}s"
            )
            break

        except Exception as e:
            _log(
                f"{atag} ERROR {type(e).__name__}: {e} "
                + ("Retrying..." if attempt < max_attempts else "Giving up.")
            )
            traceback.print_exc()
    else:
        _log(f"{tag} failed after {max_attempts} attempts, skipping")
        return

    target_vertices = int(rng.integers(*cfg.postprocess.target_vertices_range))
    _log(f"{tag} postprocess start target_vertices={target_vertices}")

    mesh, labels = postprocess_mesh(
        imprinted_mesh_raw,
        labels_raw,
        target_vertices=target_vertices,
        config=cfg.postprocess,
        rng=rng,
        scale=scale,
    )

    mesh.export(mesh_path)
    export_labels(labels, labels_path)

    if annotations_path:
        export_webannotation_3d(
            wedges_3d,
            annotations_path,
            scale=scale,
            source=mesh_path,
            config=cfg.sdf,
            label_mesh=mesh,
            vertex_labels=labels,
            sign_reference=sign_reference,
        )

    if config_path:
        cfg.to_json(config_path)

    _log(f"{tag} done total_dt={time.perf_counter() - tablet_t0:.2f}s")


def generate_dataset(
    cfg: Optional[TabletConfig] = None,
    *,
    paleocodes_path: str = "./paleocodes.json",
    output_dirs: Optional[Dict[str, str]] = None,
    filename_prefix: str = "tablet",
    num_tablets: Optional[int] = None,
    sign_source: str = "paleocode",
    sign_list: Optional[List[str]] = None,
    faces: Iterable[DIRECTION] = tuple(DIRECTION),
    exclude_faces: Optional[Iterable[DIRECTION]] = None,
) -> None:
    """Generate a dataset of synthetic cuneiform tablets.

    Args:
        cfg: Master config (defaults to TabletConfig() if None).
        paleocodes_path: Path to paleocodes JSON database.
        output_dirs: Dict with keys "mesh", "labels", and optionally
            "annotations" and "configs".
        filename_prefix: Prefix for output filenames.
        num_tablets: Number of tablets to generate.
        sign_source: "paleocode" or "transliteration".
        sign_list: Fixed sign list; if None, samples randomly from database.
        faces: Faces on which signs may be placed.
        exclude_faces: Faces to exclude (e.g. TOP, BOTTOM).
    """
    cfg = cfg or TabletConfig()
    n = num_tablets if num_tablets is not None else 10

    dirs = output_dirs or {}
    mesh_dir = dirs.get("mesh",  "./out/plys/")
    label_dir = dirs.get("labels", "./out/labels/")
    anno_dir  = dirs.get("annotations")
    cfg_dir   = dirs.get("configs")

    for d in filter(None, [mesh_dir, label_dir, anno_dir, cfg_dir]):
        os.makedirs(d, exist_ok=True)

    excluded = set(exclude_faces) if exclude_faces else set()
    active_faces = [f for f in faces if f not in excluded]
    if not active_faces:
        raise ValueError("No faces remain for sign placement after applying exclude_faces")

    sign_reference = load_sign_reference_metadata(paleocodes_path)

    rng = np.random.default_rng(cfg.generation.base_seed)
    seeds = rng.integers(0, 2**31, size=n).tolist()

    def _build_args(i: int) -> dict:
        return dict(
            tablet_index=i,
            seed=int(seeds[i]),
            cfg=cfg,
            mesh_path=os.path.join(mesh_dir, f"{filename_prefix}{i}.ply"),
            labels_path=os.path.join(label_dir, f"{filename_prefix}{i}.txt"),
            annotations_path=os.path.join(anno_dir, f"{filename_prefix}{i}.json") if anno_dir else None,
            config_path=os.path.join(cfg_dir, f"{filename_prefix}{i}_config.json") if cfg_dir else None,
            sign_source=sign_source,
            sign_list=sign_list,
            paleocodes_path=paleocodes_path,
            faces=active_faces,
            sign_reference=sign_reference,
        )

    workers = cfg.generation.workers
    if workers == 1:
        for i in range(n):
            _generate_single_tablet(**_build_args(i))
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
            futures = {pool.submit(_generate_single_tablet, **_build_args(i)): i for i in range(n)}
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    generate_dataset(
        cfg=TabletConfig(),
        paleocodes_path="./paleocodes.json",
        output_dirs={
            "mesh":        "./out/plys/",
            "labels":      "./out/labels/",
            "annotations": "./out/annotations/",
            "configs":     "./out/configs/",
        },
        filename_prefix="tablet",
        num_tablets=10,
        sign_source="paleocode",
        exclude_faces=[DIRECTION.BOTTOM, DIRECTION.TOP],
    )
