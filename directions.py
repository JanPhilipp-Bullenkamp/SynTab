from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np


class DIRECTION(Enum):
    FRONT = [0, 0, 1]
    BACK = [0, 0, -1]
    LEFT = [-1, 0, 0]
    RIGHT = [1, 0, 0]
    TOP = [0, 1, 0]
    BOTTOM = [0, -1, 0]


WRITING_DIRECTIONS = {
    DIRECTION.FRONT: ([1, 0, 0], [0, -1, 0]),
    DIRECTION.BACK: ([1, 0, 0], [0, 1, 0]),
    DIRECTION.LEFT: ([0, 0, -1], [0, -1, 0]),
    DIRECTION.RIGHT: ([0, 0, 1], [0, -1, 0]),
    DIRECTION.TOP: ([1, 0, 0], [0, 0, 1]),
    DIRECTION.BOTTOM: ([1, 0, 0], [0, 0, -1]),
}


FACE_AXES = {
    DIRECTION.FRONT: (0, 1),
    DIRECTION.BACK: (0, 1),
    DIRECTION.LEFT: (2, 1),
    DIRECTION.RIGHT: (2, 1),
    DIRECTION.TOP: (0, 2),
    DIRECTION.BOTTOM: (0, 2),
}


def face_basis(face: DIRECTION) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.array(face.value, dtype=float)
    normal /= np.linalg.norm(normal)
    horizontal, vertical = WRITING_DIRECTIONS[face]
    horizontal = np.array(horizontal, dtype=float)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.array(vertical, dtype=float)
    vertical /= np.linalg.norm(vertical)
    return normal, horizontal, vertical


def face_from_normal(n: np.ndarray) -> DIRECTION:
    """Return the face whose outward normal best matches n."""
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ax >= ay and ax >= az:
        return DIRECTION.RIGHT if nx > 0 else DIRECTION.LEFT
    if ay >= ax and ay >= az:
        return DIRECTION.TOP if ny > 0 else DIRECTION.BOTTOM
    return DIRECTION.FRONT if nz > 0 else DIRECTION.BACK


def face_extents(bounds: np.ndarray, face: DIRECTION) -> Tuple[float, float]:
    axis_h, axis_v = FACE_AXES[face]
    half_extents = (bounds[1] - bounds[0]) / 2.0
    return float(half_extents[axis_h]), float(half_extents[axis_v])
