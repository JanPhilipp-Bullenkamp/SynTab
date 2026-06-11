from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from config import LayoutConfig
from directions import DIRECTION, face_from_normal


@dataclass(frozen=True)
class PlacedWedge3D:
    sign_index: int
    wedge_index: int
    wedge_type: str
    size_scale: float
    pos3: np.ndarray
    tail_dir3: np.ndarray
    normal3: np.ndarray
    face: DIRECTION
    sign_code: Optional[str] = None
    line_index: Optional[int] = None
    char_index: Optional[int] = None


def _spow(t: np.ndarray, e: float) -> np.ndarray:
    return np.sign(t) * (np.abs(t) ** e)


def superquadric_param(u, v, a, b, c, epsilon, eta) -> np.ndarray:
    eps = max(float(epsilon), 1e-9)
    et = max(float(eta), 1e-9)

    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)

    cu_e = _spow(cu, et)
    su_e = _spow(su, et)
    cv_e = _spow(cv, eps)
    sv_e = _spow(sv, eps)

    x = a * cv_e * cu_e
    y = b * cv_e * su_e
    z = c * sv_e
    return np.stack([x, y, z], axis=-1)


def _superquadric_param_y_seed(u, v, a, b, c, epsilon, eta) -> np.ndarray:
    eps = max(float(epsilon), 1e-9)
    et = max(float(eta), 1e-9)

    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)

    cu_e = _spow(cu, et)
    su_e = _spow(su, et)
    cv_e = _spow(cv, eps)
    sv_e = _spow(sv, eps)

    x = a * cv_e * cu_e
    y = b * sv_e
    z = c * cv_e * su_e
    return np.stack([x, y, z], axis=-1)


def _surface_from_param(
    u,
    v,
    a,
    b,
    c,
    epsilon,
    eta,
    *,
    band_axis: str,
    project_iters: int,
) -> np.ndarray:
    axis = str(band_axis).lower()
    if axis == "z":
        return superquadric_param(u, v, a, b, c, epsilon, eta)
    if axis != "y":
        raise ValueError(f"Unsupported band_axis '{band_axis}', expected 'y' or 'z'")

    seeds = _superquadric_param_y_seed(u, v, a, b, c, epsilon, eta)
    flat = seeds.reshape((-1, 3))
    projected = np.array(
        [
            project_to_superquadric(p, a, b, c, epsilon, eta, iters=project_iters)
            for p in flat
        ],
        dtype=float,
    )
    return projected.reshape(seeds.shape)


def superquadric_F_and_grad(p, a, b, c, epsilon, eta) -> Tuple[float, np.ndarray]:
    ax = max(float(a), 1e-9)
    by = max(float(b), 1e-9)
    cz = max(float(c), 1e-9)
    eps = max(float(epsilon), 1e-9)
    et = max(float(eta), 1e-9)

    x, y, z = float(p[0]), float(p[1]), float(p[2])

    exp_xy = 2.0 / et
    exp_z = 2.0 / eps
    power = et / eps

    X = (abs(x) / ax) ** exp_xy
    Y = (abs(y) / by) ** exp_xy
    Z = (abs(z) / cz) ** exp_z

    S = X + Y
    S_pow = S ** power if S > 0.0 else 0.0
    F = (S_pow + Z) - 1.0

    dSfac = power * (S ** (power - 1.0)) if S > 0.0 else 0.0

    dX_dx = exp_xy * (abs(x) / ax) ** (exp_xy - 1.0) * (np.sign(x) / ax) if abs(x) > 0.0 else 0.0
    dY_dy = exp_xy * (abs(y) / by) ** (exp_xy - 1.0) * (np.sign(y) / by) if abs(y) > 0.0 else 0.0
    dZ_dz = exp_z * (abs(z) / cz) ** (exp_z - 1.0) * (np.sign(z) / cz) if abs(z) > 0.0 else 0.0

    grad = np.array([dSfac * dX_dx, dSfac * dY_dy, dZ_dz], dtype=float)
    return float(F), grad


def project_to_superquadric(p, a, b, c, epsilon, eta, iters=8, tol=1e-7) -> np.ndarray:
    q = np.array(p, dtype=float)
    for _ in range(int(iters)):
        F, g = superquadric_F_and_grad(q, a, b, c, epsilon, eta)
        gg = float(np.dot(g, g))
        if gg < 1e-18:
            break
        q = q - (F / gg) * g
        if abs(F) < tol:
            break
    return q


def tangent_frame_from_param(
    u,
    v,
    a,
    b,
    c,
    epsilon,
    eta,
    *,
    band_axis: str = "z",
    project_iters: int = 8,
    delta: float = 1e-4,
    center_hint: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_u, t_v, normal, center).

    If center_hint is provided it is used as the surface point at (u, v),
    avoiding one Newton-projection call (~33% cheaper in the placement loop).
    """
    if center_hint is not None:
        p = center_hint
    else:
        p = _surface_from_param(
            np.array([u]), np.array([v]), a, b, c, epsilon, eta,
            band_axis=band_axis, project_iters=project_iters,
        )[0]

    pu = _surface_from_param(
        np.array([u + delta]), np.array([v]), a, b, c, epsilon, eta,
        band_axis=band_axis, project_iters=project_iters,
    )[0]
    pv = _surface_from_param(
        np.array([u]), np.array([v + delta]), a, b, c, epsilon, eta,
        band_axis=band_axis, project_iters=project_iters,
    )[0]

    xu = pu - p
    xv = pv - p

    t_u = xu / max(1e-12, np.linalg.norm(xu))
    xv_orth = xv - np.dot(xv, t_u) * t_u
    t_v = xv_orth / max(1e-12, np.linalg.norm(xv_orth))

    n = np.cross(t_u, t_v)
    n = n / max(1e-12, np.linalg.norm(n))
    return t_u, t_v, n, p


def _band_arclength_u(
    v, a, b, c, epsilon, eta,
    *, band_axis: str, project_iters: int, n_samples: int,
):
    u = np.linspace(-np.pi, np.pi, int(n_samples), dtype=float)
    v_arr = np.full_like(u, float(v))
    pts = _surface_from_param(
        u, v_arr, a, b, c, epsilon, eta,
        band_axis=band_axis, project_iters=project_iters,
    )
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return u, s


def _pick_v_bands_equal_meridian(
    n_bands, a, b, c, epsilon, eta,
    *, band_axis: str, project_iters: int, n_samples: int,
):
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(n_samples), dtype=float)
    u0 = np.zeros_like(v)
    pts = _surface_from_param(
        u0, v, a, b, c, epsilon, eta,
        band_axis=band_axis, project_iters=project_iters,
    )
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if n_bands <= 1 or total <= 1e-12:
        return np.array([0.0], dtype=float)
    step = total / float(n_bands)
    targets = (np.arange(int(n_bands), dtype=float) + 0.5) * step
    return np.interp(targets, s, v)


def layout_signs_on_superquadric(
    signs,
    *,
    a: float,
    b: float,
    c: float,
    epsilon: float,
    eta: float,
    config: LayoutConfig = LayoutConfig(),
    rng: Optional[np.random.Generator] = None,
    allowed_faces: Optional[Set[DIRECTION]] = None,
) -> List[PlacedWedge3D]:
    if rng is None:
        rng = np.random.default_rng()
    if not signs:
        return []

    sign_entries: List[Tuple] = []
    for sign in signs:
        wedges = sign.normalized_wedges()
        if not wedges:
            continue
        sign_entries.append((sign, wedges))

    if not sign_entries:
        return []

    sign_heights = np.array([max(1e-6, s.height) for s, _ in sign_entries], dtype=float)
    sign_widths = np.array([max(1e-6, s.width) for s, _ in sign_entries], dtype=float)
    sign_aspects = sign_widths / sign_heights

    band_axis = str(config.band_axis).lower()
    surface_height = 2.0 * max(1e-9, float(b if band_axis == "y" else c))
    target_sign_height = config.target_sign_height_frac * surface_height
    target_line_height = target_sign_height / max(1e-9, config.sign_height_ratio)

    n_bands = max(1, int(np.floor(surface_height / max(1e-9, target_line_height))))
    v_bands = _pick_v_bands_equal_meridian(
        n_bands, a, b, c, epsilon, eta,
        band_axis=band_axis,
        project_iters=config.project_iters,
        n_samples=config.arclength_v_samples,
    )

    placed: List[PlacedWedge3D] = []
    placed_centers: List[np.ndarray] = []
    placed_radii: List[float] = []
    sign_counter = 0

    for line_index, v in enumerate(v_bands, start=1):
        u_samp, s_cum = _band_arclength_u(
            v, a, b, c, epsilon, eta,
            band_axis=band_axis,
            project_iters=config.project_iters,
            n_samples=config.arclength_u_samples,
        )
        band_length = float(s_cum[-1])
        if band_length <= 1e-9:
            continue

        line_height = target_line_height
        spacing = line_height * config.sign_spacing_ratio

        target_height = line_height * config.sign_height_ratio
        pad_abs = float(config.sign_padding)
        pad_frac = float(config.sign_padding_frac or 0.0)
        pad = pad_abs + pad_frac * target_height

        widths_at_target = sign_aspects * target_height
        widths_effective = widths_at_target + 2.0 * pad

        s_cursor = 0.0
        u_phase = float(rng.uniform(-np.pi, np.pi))
        char_in_line = 0

        def s_to_u(s: float) -> float:
            u_base = float(np.interp(np.clip(s, 0.0, band_length), s_cum, u_samp))
            return ((u_base + u_phase + np.pi) % (2.0 * np.pi)) - np.pi

        while True:
            remaining = band_length - s_cursor
            if remaining <= 1e-9:
                break

            candidates = np.where(widths_effective <= remaining)[0]
            if candidates.size == 0:
                break

            candidate_pool = candidates.copy()
            placed_sign = False

            while candidate_pool.size > 0:
                widths_fit = widths_effective[candidate_pool]

                if config.small_sign_bias > 0.0 and widths_fit.size > 1:
                    threshold = float(np.median(widths_fit))
                    if remaining < threshold:
                        weights = 1.0 / np.maximum(widths_fit, 1e-9)
                        weights = np.power(weights, config.small_sign_bias)
                        weights = weights / weights.sum()
                        pick = int(rng.choice(candidate_pool, p=weights))
                    else:
                        pick = int(rng.choice(candidate_pool))
                else:
                    pick = int(rng.choice(candidate_pool))

                sign, wedges = sign_entries[pick]
                scale = target_height / max(1e-6, sign.height)
                sign_w = float(sign.width * scale)
                sign_h = float(sign.height * scale)

                eff_w = sign_w + 2.0 * pad
                if eff_w > remaining + 1e-6:
                    candidate_pool = candidate_pool[candidate_pool != pick]
                    continue

                s_center = s_cursor + pad + 0.5 * sign_w
                u_center = s_to_u(s_center)

                # Compute center3 first; pass as hint to avoid a redundant Newton projection.
                center3 = _surface_from_param(
                    np.array([u_center]), np.array([v]), a, b, c, epsilon, eta,
                    band_axis=band_axis, project_iters=config.project_iters,
                )[0]
                t_u, t_v, _, _ = tangent_frame_from_param(
                    u_center, float(v), a, b, c, epsilon, eta,
                    band_axis=band_axis,
                    project_iters=config.project_iters,
                    delta=config.tangent_delta,
                    center_hint=center3,
                )

                if config.flip_vertical_axis:
                    t_v = -t_v

                _, g_center = superquadric_F_and_grad(center3, a, b, c, epsilon, eta)
                n_center = g_center / max(1e-12, np.linalg.norm(g_center))
                face = face_from_normal(n_center)
                if allowed_faces is not None and face not in allowed_faces:
                    candidate_pool = candidate_pool[candidate_pool != pick]
                    continue

                footprint_radius = 0.5 * np.sqrt(sign_w * sign_w + sign_h * sign_h) + pad

                if config.global_center_separation > 0.0 and placed_centers:
                    ok = True
                    for c0, r0 in zip(placed_centers, placed_radii):
                        d = float(np.linalg.norm(center3 - c0))
                        if d < config.global_center_separation * (footprint_radius + r0):
                            ok = False
                            break
                    if not ok:
                        candidate_pool = candidate_pool[candidate_pool != pick]
                        continue

                x0 = -0.5 * sign_w
                y0 = -0.5 * sign_h
                next_char_index = char_in_line + 1

                for wedge_idx, wedge in enumerate(wedges):
                    wx = x0 + float(wedge.pos[0]) * scale
                    wy = y0 + float(wedge.pos[1]) * scale

                    p3 = center3 + wx * t_u + wy * t_v
                    p3 = project_to_superquadric(
                        p3, a, b, c, epsilon, eta, iters=config.project_iters
                    )

                    _, g = superquadric_F_and_grad(p3, a, b, c, epsilon, eta)
                    n3 = g / max(1e-12, np.linalg.norm(g))

                    d2 = np.array(wedge.direction, dtype=float).reshape(2,)
                    d3 = d2[0] * t_u + d2[1] * t_v
                    d3 = d3 - np.dot(d3, n3) * n3
                    d3 = d3 / max(1e-12, np.linalg.norm(d3))

                    placed.append(
                        PlacedWedge3D(
                            sign_index=sign_counter,
                            wedge_index=wedge_idx,
                            wedge_type=wedge.wedge_type,
                            size_scale=wedge.size_scale,
                            pos3=p3,
                            tail_dir3=d3,
                            normal3=n3,
                            face=face,
                            sign_code=str(sign.code),
                            line_index=line_index,
                            char_index=next_char_index,
                        )
                    )

                placed_centers.append(center3)
                placed_radii.append(footprint_radius)

                s_cursor += sign_w + 2.0 * pad + spacing
                sign_counter += 1
                char_in_line = next_char_index
                placed_sign = True
                break

            if not placed_sign:
                break

    return placed
