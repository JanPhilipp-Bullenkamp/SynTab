from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Dict, Iterable, List

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


@dataclass(frozen=True)
class Wedge2D:
    pos: np.ndarray
    direction: np.ndarray
    wedge_type: str
    size_scale: float


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
            )
            for w in self.wedges
        ]


def _angle_to_dir(angle_degrees: float) -> np.ndarray:
    # PaleoCodage uses screen-like coordinates: +y is downwards.
    # angle 0 -> down, 90 -> right.
    theta = np.deg2rad(angle_degrees)
    return np.array([np.sin(theta), np.cos(theta)], dtype=float)


def parse_paleocode(code: str, config: PaleocodageConfig | None = None) -> ParsedSign:
    if config is None:
        config = PaleocodageConfig()

    curx = 0.0
    cury = 0.0
    starty = 0.0
    smaller = False
    mirror = False
    rot = 0.0

    wedges: List[Wedge2D] = []

    i = 0
    while i < len(code):
        ch = code[i]

        if ch == "s":
            smaller = True
            i += 1
            continue

        if ch in WEDGE_TOKENS:
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

            direction = _angle_to_dir(angle)
            wedges.append(
                Wedge2D(
                    pos=pos,
                    direction=direction,
                    wedge_type=ch.lower(),
                    size_scale=size_scale,
                )
            )

            mirror = False
            rot = 0.0
            i += 1
            continue

        if ch == "-":
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

        i += 1

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
