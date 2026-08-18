"""Generate one tablet with exactly one cuneiform sign — for debugging/visualization.

Usage:
    python generate_single_sign.py              # uses first sign in paleocodes.json
    python generate_single_sign.py "d:c:d:c"   # uses a specific paleocode

Output: ./out_single/<code>.ply  +  ./out_single/<code>.txt  +  ./out_single/<code>.json
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from annotation_export import export_labels, export_webannotation_3d
from config import (
    DEBUG_TABLET,
    PostprocessConfig,
    TabletConfig,
    WedgeCarvingConfig,
    WedgeSizeConfig,
)
from imprint import build_wedge_meshes, map_placed_wedges3d_to_surface
from layout3d import _surface_from_param, place_sign_at_front_center
from paleocodage import parse_signs
from postprocess import postprocess_mesh
from sdf_boolean import compute_pitch, sdf_difference_with_labels
from signs import drop_signs_without_wedges, load_paleocodes_json, load_sign_reference_metadata

# ---------------------------------------------------------------------------
# Wedge carving — ranges; each wedge draws a random value from [min, max].
# Narrow ranges keep this debug render tidy; the same block is applied to every
# size class so a single sign shows the notation's own size differences rather
# than the sampling spread.
# ---------------------------------------------------------------------------
DEBUG_WEDGE_RANGES = WedgeSizeConfig(
    min_depth=0.050, max_depth=0.070,
    min_angle=7.0, max_angle=12.0,
    min_tilt_angle=-3.0, max_tilt_angle=3.0,
)

PALEOCODES_PATH = "./paleocodes.json"
OUT_DIR = "./out_single"


def main() -> None:
    code = sys.argv[1] if len(sys.argv) > 1 else None
    if not code:
        pool = load_paleocodes_json(PALEOCODES_PATH)
        if not pool:
            print("paleocodes.json is empty")
            sys.exit(1)
        code = pool[0]
    print(f"Sign: {code!r}")

    parsed_raw = parse_signs([code])
    signs, _ = drop_signs_without_wedges(parsed_raw)
    if not signs:
        print("No wedges found for this sign — try a different paleocode.")
        sys.exit(1)

    sign = signs[0]
    shape = DEBUG_TABLET
    surface_height = 2.0 * shape.b
    target_sign_height = shape.sign_height_frac * surface_height

    # Compute front-face arc-length (u: π/4 → 3π/4 at equator) to constrain sign width
    u_front = np.linspace(np.pi / 4, 3 * np.pi / 4, 128)
    front_pts = _surface_from_param(
        u_front, np.zeros(128),
        shape.a, shape.b, shape.c, shape.epsilon, shape.eta,
        band_axis="y", project_iters=8,
    )
    front_face_arc = float(np.sum(np.linalg.norm(np.diff(front_pts, axis=0), axis=1)))

    scale_h = target_sign_height / max(1e-6, sign.height)
    scale_w = (front_face_arc * 0.85) / max(1e-6, sign.width)
    scale = min(scale_h, scale_w)
    print(f"  h={sign.height:.1f}  w={sign.width:.1f}  wedges={len(sign.normalized_wedges())}")
    print(f"  front_face_arc={front_face_arc:.3f}")
    print(f"  scaled: h={sign.height*scale:.3f}  w={sign.width*scale:.3f}  (limited by {'height' if scale_h < scale_w else 'width'})")

    cfg = TabletConfig(
        carving=WedgeCarvingConfig(
            normal=DEBUG_WEDGE_RANGES,
            small=DEBUG_WEDGE_RANGES,
            large=DEBUG_WEDGE_RANGES,
        ),
        postprocess=PostprocessConfig(
            remesh_iterations=3,
            target_vertices_range=(50_000, 80_000),
        ),
    )

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    placed = place_sign_at_front_center(
        sign, scale, shape.a, shape.b, shape.c, shape.epsilon, shape.eta,
    )
    print(f"[layout]      {len(placed)} wedges  dt={time.perf_counter()-t0:.3f}s")

    ts = time.perf_counter()
    wedges_3d = map_placed_wedges3d_to_surface(
        placed, config=cfg.carving,
        flip_tail_direction_x=cfg.generation.flip_tail_direction_x,
        rng=rng,
    )
    wedge_meshes = build_wedge_meshes(wedges_3d, config=cfg.carving)
    print(f"[imprint]     {len(wedge_meshes)} meshes  dt={time.perf_counter()-ts:.3f}s")

    pitch = compute_pitch(shape.a, shape.b, shape.c, cfg.sdf)
    print(f"[sdf]         pitch={pitch:.7f}")

    ts = time.perf_counter()
    mesh_raw, labels_raw = sdf_difference_with_labels(
        wedge_meshes,
        config=cfg.sdf, pitch=pitch,
        base_implicit={
            "a": shape.a, "b": shape.b, "c": shape.c,
            "epsilon": shape.epsilon, "eta": shape.eta,
        },
    )
    print(f"[sdf]         verts={len(mesh_raw.vertices):,}  dt={time.perf_counter()-ts:.3f}s")

    ts = time.perf_counter()
    mesh, labels = postprocess_mesh(
        mesh_raw, labels_raw,
        target_vertices=60_000,
        config=cfg.postprocess,
        rng=rng,
        scale=shape.scale_mm,
    )
    print(f"[postprocess] verts={len(mesh.vertices):,}  dt={time.perf_counter()-ts:.3f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    safe = code.replace("/", "_").replace(":", "-")[:40]
    mesh_path  = os.path.join(OUT_DIR, f"{safe}.ply")
    label_path = os.path.join(OUT_DIR, f"{safe}.txt")
    anno_path  = os.path.join(OUT_DIR, f"{safe}.json")

    mesh.export(mesh_path)
    export_labels(labels, label_path)

    sign_reference = load_sign_reference_metadata(PALEOCODES_PATH)
    export_webannotation_3d(
        wedges_3d, anno_path,
        scale=shape.scale_mm, source=mesh_path,
        config=cfg.sdf, label_mesh=mesh, vertex_labels=labels,
        sign_reference=sign_reference,
    )

    print(f"\nSaved → {mesh_path}")
    print(f"Total: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
