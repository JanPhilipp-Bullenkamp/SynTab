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

from annotation_export import export_webannotation_3d
from config import (
    LayoutConfig,
    PostprocessConfig,
    TabletConfig,
    WedgeCarvingConfig,
    WedgeSizeConfig,
)
from directions import face_from_normal
from imprint import build_wedge_meshes, map_placed_wedges3d_to_surface
from layout3d import (
    PlacedWedge3D,
    _surface_from_param,
    project_to_superquadric,
    superquadric_F_and_grad,
    tangent_frame_from_param,
    writing_frame,
)
from paleocodage import parse_signs
from postprocess import postprocess_mesh
from sdf_boolean import sdf_difference_with_labels
from signs import drop_signs_without_wedges, load_paleocodes_json, load_sign_reference_metadata
from util import export_labels

# ---------------------------------------------------------------------------
# Tablet shape (normalised units)
# ---------------------------------------------------------------------------
TABLET_H = 0.70   # superquadric a (x half-axis)
TABLET_W = 0.80   # superquadric b (y — height axis when band_axis="y")
TABLET_D = 0.25   # superquadric c (z — depth)
EPSILON  = 0.85   # z-axis roundness
ETA      = 0.82   # xy-plane roundness
SCALE_MM = 10.0   # final mesh scale in mm

# Sign placement
SIGN_HEIGHT_FRAC = 0.15   # sign height as fraction of tablet height (2 * TABLET_W)

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


def _place_sign_at_front_center(
    sign, scale: float, a, b, c, epsilon, eta,
    layout_config: LayoutConfig | None = None,
) -> list[PlacedWedge3D]:
    """Place one sign centred on the front face (u=π/2, v=0) of the superquadric."""
    u_center = np.pi / 2   # front face
    v_center = 0.35        # slightly above equator (range: -π/2 bottom … +π/2 top)

    center3 = _surface_from_param(
        np.array([u_center]), np.array([v_center]),
        a, b, c, epsilon, eta, band_axis="y", project_iters=8,
    )[0]

    t_u, t_v, _, _ = tangent_frame_from_param(
        u_center, v_center, a, b, c, epsilon, eta,
        band_axis="y", project_iters=8, delta=1e-4, center_hint=center3,
    )
    # Shared with the production layout so the two cannot drift apart; both
    # wedge positions and wedge directions must use this one frame.
    e_x, e_y = writing_frame(t_u, t_v, layout_config or LayoutConfig())

    _, g = superquadric_F_and_grad(center3, a, b, c, epsilon, eta)
    n3 = g / max(1e-12, np.linalg.norm(g))
    face = face_from_normal(n3)

    sign_w = float(sign.width * scale)
    sign_h = float(sign.height * scale)
    x0 = -0.5 * sign_w
    y0 = -0.5 * sign_h

    wedges = sign.normalized_wedges()
    placed: list[PlacedWedge3D] = []
    for wedge_idx, wedge in enumerate(wedges):
        wx = x0 + float(wedge.pos[0]) * scale
        wy = y0 + float(wedge.pos[1]) * scale

        p3 = center3 + wx * e_x + wy * e_y
        p3 = project_to_superquadric(p3, a, b, c, epsilon, eta, iters=8)

        _, g_w = superquadric_F_and_grad(p3, a, b, c, epsilon, eta)
        n3_w = g_w / max(1e-12, np.linalg.norm(g_w))

        d2 = np.array(wedge.direction, dtype=float).reshape(2,)
        d3 = d2[0] * e_x + d2[1] * e_y
        d3 = d3 - np.dot(d3, n3_w) * n3_w
        d3 = d3 / max(1e-12, np.linalg.norm(d3))

        placed.append(PlacedWedge3D(
            sign_index=0,
            wedge_index=wedge_idx,
            wedge_type=wedge.wedge_type,
            size_scale=wedge.size_scale,
            size_class=wedge.size_class,
            pos3=p3,
            tail_dir3=d3,
            normal3=n3_w,
            face=face,
            sign_code=str(sign.code),
            line_index=1,
            char_index=1,
        ))

    return placed


def main() -> None:
    code = sys.argv[1] if len(sys.argv) > 1 else None
    if not code:
        pool = load_paleocodes_json(PALEOCODES_PATH)
        if not pool:
            print("paleocodes.json is empty")
            sys.exit(1)
        code = pool[0]
    print(f"Sign: {code!r}")
    code= str("b:::b--d:c_a::a") #str("b:::b--a--a") #

    parsed_raw = parse_signs([code])
    signs, _ = drop_signs_without_wedges(parsed_raw)
    if not signs:
        print("No wedges found for this sign — try a different paleocode.")
        sys.exit(1)

    sign = signs[0]
    surface_height = 2.0 * TABLET_W
    target_sign_height = SIGN_HEIGHT_FRAC * surface_height

    # Compute front-face arc-length (u: π/4 → 3π/4 at equator) to constrain sign width
    u_front = np.linspace(np.pi / 4, 3 * np.pi / 4, 128)
    front_pts = _surface_from_param(
        u_front, np.zeros(128),
        TABLET_H, TABLET_W, TABLET_D, EPSILON, ETA, band_axis="y", project_iters=8,
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

    placed = _place_sign_at_front_center(
        sign, scale, TABLET_H, TABLET_W, TABLET_D, EPSILON, ETA,
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

    h, w, d = TABLET_H, TABLET_W, TABLET_D
    base_extent = 2.0 * max(h, w, d)
    denom = max(1.0, (cfg.sdf.max_grid - 1) - 2.0 * cfg.sdf.padding)
    pitch = max(cfg.sdf.target_pitch, base_extent / denom)
    print(f"[sdf]         pitch={pitch:.7f}")

    ts = time.perf_counter()
    mesh_raw, labels_raw = sdf_difference_with_labels(
        wedge_meshes,
        config=cfg.sdf, pitch=pitch,
        base_implicit={"a": h, "b": w, "c": d, "epsilon": EPSILON, "eta": ETA},
    )
    print(f"[sdf]         verts={len(mesh_raw.vertices):,}  dt={time.perf_counter()-ts:.3f}s")

    ts = time.perf_counter()
    mesh, labels = postprocess_mesh(
        mesh_raw, labels_raw,
        target_vertices=60_000,
        config=cfg.postprocess,
        rng=rng,
        scale=SCALE_MM,
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
        scale=SCALE_MM, source=mesh_path,
        config=cfg.sdf, label_mesh=mesh, vertex_labels=labels,
        sign_reference=sign_reference,
    )

    print(f"\nSaved → {mesh_path}")
    print(f"Total: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
