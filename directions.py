from __future__ import annotations

from enum import Enum

import numpy as np


class DIRECTION(Enum):
    FRONT = [0, 0, 1]
    BACK = [0, 0, -1]
    LEFT = [-1, 0, 0]
    RIGHT = [1, 0, 0]
    TOP = [0, 1, 0]
    BOTTOM = [0, -1, 0]


def face_from_normal(n: np.ndarray) -> DIRECTION:
    """Return the face whose outward normal best matches n."""
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ax >= ay and ax >= az:
        return DIRECTION.RIGHT if nx > 0 else DIRECTION.LEFT
    if ay >= ax and ay >= az:
        return DIRECTION.TOP if ny > 0 else DIRECTION.BOTTOM
    return DIRECTION.FRONT if nz > 0 else DIRECTION.BACK
