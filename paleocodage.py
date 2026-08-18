from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Dict, Iterable, Iterator, List

import numpy as np

from config import PaleocodageConfig

# Mappings adapted from PaleoCodage-master/js/paleocodage.js
OPERATOR_TO_LOCAL_ROT = {
    "a": 0, "A": 0,
    "b": 90, "B": 90,
    "c": 45, "C": 45,
    "d": 135, "D": 135,
    "e": 225, "E": 225,
    "f": 315, "F": 315,
    "w": 270, "W": 270,
    "x": 0, "X": 0,
    "y": -90, "Y": -90,
}

OPERATOR_TO_POSITIONING = {
    "a": [0, 0], "A": [0, 0],
    "b": [0.3, -0.35], "B": [0.4, -0.4],
    "c": [0.2, -0.65], "C": [0.2, -0.5],
    "d": [0.2, -0.05], "D": [0.2, -0.3],
    "e": [0.8, -0.05], "E": [0.4, -0.3],
    "f": [0.8, -0.65], "F": [0.4, -0.5],
    "w": [-0.6, -0.4], "W": [-0.55, -0.38],
    "x": [0.5, 0.5], "X": [0.5, 0.5],
    "y": [0.5, 0.5], "Y": [0.5, 0.5],
}

OPERATOR_TO_SCALING = {
    "a": 1, "A": 1,
    "b": 1, "B": 1,
    "c": 1, "C": 1,
    "d": 1, "D": 1,
    "e": 1, "E": 1,
    "f": 1, "F": 1,
    "w": 2, "W": 2,
    "x": 0, "X": 0,
    "y": 0, "Y": 0,
}

WEDGE_TOKENS = {
    token
    for token in OPERATOR_TO_LOCAL_ROT.keys()
    if token.lower() in {"a", "b", "c", "d", "e", "f", "w"}
}


class WedgeSize:
    """The three wedge sizes the notation distinguishes.

    Deliberately separate from `size_scale`: that number also carries
    `OPERATOR_TO_SCALING`, which doubles the Winkelhaken as a *drawing*
    convention rather than because the impression is bigger.  A plain `w`
    therefore has size_scale 2.0 but is a NORMAL wedge, and `sw` has size_scale
    1.0 but is SMALL — so the class has to come from the modifiers, not from the
    scale factor.
    """
    SMALL = "small"
    NORMAL = "normal"
    LARGE = "large"

    ALL = ("small", "normal", "large")


@dataclass(frozen=True)
class Wedge2D:
    pos: np.ndarray
    direction: np.ndarray
    wedge_type: str
    size_scale: float
    # Which of WedgeSize the notation asked for: 's' prefix → SMALL, an
    # uppercase token → LARGE, otherwise NORMAL.  Drives the carving depth and
    # impression angle in `imprint.map_placed_wedges3d_to_surface`.
    size_class: str = WedgeSize.NORMAL


@dataclass(frozen=True)
class ParsedSign:
    code: str
    wedges: List[Wedge2D]
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    @property
    def width(self) -> float:
        return float(self.bbox_max[0] - self.bbox_min[0])

    @property
    def height(self) -> float:
        return float(self.bbox_max[1] - self.bbox_min[1])

    def normalized_wedges(self) -> List[Wedge2D]:
        shift = self.bbox_min
        return [
            Wedge2D(
                pos=w.pos - shift,
                direction=w.direction,
                wedge_type=w.wedge_type,
                size_scale=w.size_scale,
                size_class=w.size_class,
            )
            for w in self.wedges
        ]


def _angle_to_dir(angle_degrees: float) -> np.ndarray:
    # PaleoCodage uses screen-like coordinates: +y is downwards.
    # angle 0 -> down, 90 -> right.
    theta = np.deg2rad(angle_degrees)
    return np.array([np.sin(theta), np.cos(theta)], dtype=float)


@dataclass(frozen=True)
class ParseStep:
    """Decoder state immediately after consuming one character of a paleocode.

    Yielded by `iter_paleocode_steps` so callers can follow the decode — the
    figure that explains the notation walks a code character by character and
    needs the cursor position at each one.
    """
    index: int              # position of the consumed character in the code
    char: str
    curx: float             # cursor, in paleocodage units
    cury: float
    emitted: Wedge2D | None  # wedge placed by this character, if any
    smaller: bool           # pending 's' (half-size) modifier
    mirror: bool            # pending '!' (mirror) modifier
    rot: float              # pending '<' / '>' rotation, degrees

    # Live decoder list plus how many entries existed at this step, so taking a
    # snapshot stays O(1) during parsing and only costs when actually read.
    _wedges: List[Wedge2D] = field(repr=False, default_factory=list)
    _count: int = field(repr=False, default=0)

    @property
    def wedges(self) -> tuple[Wedge2D, ...]:
        """Wedges emitted up to and including this step."""
        return tuple(self._wedges[: self._count])


def iter_paleocode_steps(
    code: str, config: PaleocodageConfig | None = None
) -> Iterator[ParseStep]:
    """Decode `code`, yielding a ParseStep for every character consumed.

    This is the single implementation of the PaleoCodage grammar;
    `parse_paleocode` is a thin consumer of it.
    """
    if config is None:
        config = PaleocodageConfig()

    curx = 0.0
    cury = 0.0
    starty = 0.0
    smaller = False
    mirror = False
    rot = 0.0

    wedges: List[Wedge2D] = []

    for i, ch in enumerate(code):
        emitted: Wedge2D | None = None

        if ch == "s":
            smaller = True

        elif ch in WEDGE_TOKENS:
            # A pending 's' wins over the token's case: `sA` is a small wedge
            # drawn from the big variant, and reads as small on the tablet.
            if smaller:
                size_class = WedgeSize.SMALL
            elif ch.isupper():
                size_class = WedgeSize.LARGE
            else:
                size_class = WedgeSize.NORMAL

            base_scale = config.big_scale if ch.isupper() else 1.0
            if smaller:
                base_scale *= config.small_scale
                smaller = False

            size_scale = base_scale * OPERATOR_TO_SCALING.get(ch, 1.0)
            local_offset = OPERATOR_TO_POSITIONING.get(ch, [0.0, 0.0])
            pos = np.array(
                [
                    curx + local_offset[0] * config.stroke_length,
                    cury + local_offset[1] * config.stroke_length,
                ],
                dtype=float,
            )

            angle = OPERATOR_TO_LOCAL_ROT.get(ch, 0.0) + rot
            if mirror:
                angle += 180.0

            emitted = Wedge2D(
                pos=pos,
                direction=_angle_to_dir(angle),
                wedge_type=ch.lower(),
                size_scale=size_scale,
                size_class=size_class,
            )
            wedges.append(emitted)

            mirror = False
            rot = 0.0

        elif ch == "-":
            curx += config.step_x
            cury = starty
        elif ch == "_":
            curx += config.stroke_length
            cury = starty
        elif ch == ":":
            cury += config.step_y
        elif ch == ";":
            cury += config.stroke_length
        elif ch == "/":
            cury += config.step_y / 2.0
        elif ch == ".":
            cury += config.step_y
            curx += config.step_y
        elif ch == ",":
            cury -= config.step_y
            curx -= config.step_y
        elif ch == "'":
            cury = starty
        elif ch == "\"":
            cury = 0.0
        elif ch == "~":
            curx -= config.step_x
            cury = starty
        elif ch == " ":
            curx += 1.5 * config.stroke_length
            cury = starty
        elif ch == "!":
            mirror = True
        elif ch == "<":
            rot -= config.rotation_constant
        elif ch == ">":
            rot += config.rotation_constant

        yield ParseStep(
            index=i, char=ch, curx=curx, cury=cury, emitted=emitted,
            smaller=smaller, mirror=mirror, rot=rot,
            _wedges=wedges, _count=len(wedges),
        )


def parse_paleocode(code: str, config: PaleocodageConfig | None = None) -> ParsedSign:
    if config is None:
        config = PaleocodageConfig()

    wedges: List[Wedge2D] = []
    for step in iter_paleocode_steps(code, config=config):
        if step.emitted is not None:
            wedges.append(step.emitted)

    if wedges:
        positions = np.stack([w.pos for w in wedges], axis=0)
        bbox_min = positions.min(axis=0) - config.bbox_padding
        bbox_max = positions.max(axis=0) + config.bbox_padding
    else:
        bbox_min = np.array([0.0, 0.0], dtype=float)
        bbox_max = np.array([0.0, 0.0], dtype=float)

    return ParsedSign(code=code, wedges=wedges, bbox_min=bbox_min, bbox_max=bbox_max)


def load_paleocodes(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = re.compile(
        r"\"Transliteration\"\s*:\s*\"(.*?)\".*?\"PaleoCode\"\s*:\s*\"(.*?)\"",
        re.DOTALL,
    )

    mapping: Dict[str, str] = {}
    for translit, code in pattern.findall(text):
        if translit and code:
            mapping[translit] = code

    return mapping


def load_paleocodes_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [str(value) for value in data.values() if value]

    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], str):
            return [str(item) for item in data if item]
        if isinstance(data[0], dict):
            codes = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                code = item.get("PaleoCode") or item.get("paleocode") or item.get("code")
                if code:
                    codes.append(str(code))
            return codes

    return []


def parse_signs(
    codes: Iterable[str],
    config: PaleocodageConfig | None = None,
) -> List[ParsedSign]:
    return [parse_paleocode(code, config=config) for code in codes]
