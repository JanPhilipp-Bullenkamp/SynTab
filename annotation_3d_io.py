"""Read and write W3C Web Annotation JSON for 2D and 3D selectors.

Supports the Web Annotation Data Model and extends it for 3D annotations as
described in the DANES paper (WKTSelector, MeshIndexSelector,
MeshVertexSelector) and 2D annotations using SvgSelector / FragmentSelector.

Both the top-level id-map format used in HS1174_front.png.json and
bbox_anno.json, and the standard AnnotationCollection list format, are
supported for reading and writing.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

Annotation = Dict[str, Any]
AnnotationMap = Dict[str, Annotation]

DEFAULT_CONTEXT = "http://www.w3.org/ns/anno.jsonld"
DEFAULT_CUNEIFORM_TAG = {
    "type": "SpecificResource",
    "purpose": "tagging",
    "source": {
        "id": "http://purl.org/cuneiform/Wedge",
        "label": "Wedge",
    },
}

# Source URIs for 3D body items (from bbox_anno.json)
_SOURCE_TRANSLITERATION = "http://purl.org/cuneiform/Transliteration"
_SOURCE_CHARACTER       = "http://purl.org/cuneiform/Character"
_SOURCE_LINE            = "http://purl.org/cuneiform/Line"
_SOURCE_CHARINDEX       = "http://purl.org/cuneiform/CharacterIndex"
_SOURCE_WORDINDEX       = "http://purl.org/cuneiform/WordIndex"
_SOURCE_RELCHARINDEX    = "http://purl.org/cuneiform/RelativeCharacterIndex"

_SELECTOR_TYPES_3D = {"WKTSelector", "MeshIndexSelector", "MeshVertexSelector"}
_SELECTOR_TYPES_2D = {"SvgSelector", "FragmentSelector"}


# ---------------------------------------------------------------------------
# I/O (unchanged)
# ---------------------------------------------------------------------------

def read(file_path: str) -> List[Annotation]:
    """Read a W3C Web Annotation JSON file and return a list of annotations."""
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        if _looks_like_annotation_map(data):
            return [_merge_id_into_annotation(annotation_id, annotation)
                    for annotation_id, annotation in data.items()]

        if data.get("type") == "AnnotationCollection":
            items = data.get("items", [])
            return [_merge_id_into_annotation(item.get("id"), item) for item in items]

        if data.get("type") == "Annotation":
            return [_merge_id_into_annotation(data.get("id"), data)]

    if isinstance(data, list):
        return [_merge_id_into_annotation(annotation.get("id"), annotation)
                for annotation in data]

    raise ValueError(f"Unsupported annotation JSON shape in {file_path}")


def write(
    file_path: str,
    annotations: Union[Sequence[Annotation], AnnotationMap],
    context: Optional[str] = None,
    indent: int = 2,
    as_map: bool = True,
) -> None:
    """Write annotation data to a JSON file.

    If as_map is True, the output uses a top-level mapping of ids to Annotation
    objects, matching HS1174_front.png.json style.
    """
    context = context or DEFAULT_CONTEXT
    if isinstance(annotations, dict) and _looks_like_annotation_map(annotations):
        output_data = annotations
    else:
        output_data = _annotations_to_map(list(annotations), context=context) if as_map else {
            "type": "AnnotationCollection",
            "@context": context,
            "items": [_ensure_annotation(annotation, context=context) for annotation in annotations],
        }

    os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# WKT utilities
# ---------------------------------------------------------------------------

def to_wkt_polygon_z(points: Sequence[Sequence[float]]) -> str:
    """Serialise a list of 3D points to a WKT POLYGON Z string.

    Format: spaces within a point, commas between points, matching bbox_anno.json.

    Args:
        points: Non-empty sequence of [x, y, z] coordinate triples.

    Returns:
        ``"POLYGON Z((x y z,x y z,...))"``

    Raises:
        ValueError: If points is empty or any element is not 3-element.
    """
    if not points:
        raise ValueError("points must be non-empty")
    coord_strs = []
    for pt in points:
        if len(pt) != 3:
            raise ValueError(f"Each point must have exactly 3 coordinates, got {len(pt)}")
        coord_strs.append(f"{float(pt[0])} {float(pt[1])} {float(pt[2])}")
    return "POLYGON Z((" + ",".join(coord_strs) + "))"


def parse_wkt_polygon_z(wkt: str) -> List[List[float]]:
    """Parse a WKT POLYGON Z string into a list of [x, y, z] coordinate triples.

    Args:
        wkt: A string of the form ``"POLYGON Z((x y z,x y z,...))"``

    Returns:
        List of [x, y, z] float lists.

    Raises:
        ValueError: If wkt does not match the POLYGON Z format.
    """
    m = re.match(r'POLYGON\s+Z\s*\(\((.*)\)\)\s*$', wkt.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse WKT POLYGON Z: {wkt!r}")
    inner = m.group(1)
    result = []
    for pt_str in inner.split(","):
        parts = pt_str.split()
        if len(parts) != 3:
            raise ValueError(
                f"Expected 3 coordinates per point, got {len(parts)} in {pt_str!r}"
            )
        result.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return result


def _bbox_to_wkt_corners(
    min_xyz: Sequence[float],
    max_xyz: Sequence[float],
) -> List[List[float]]:
    """Return the 8 corners of an axis-aligned 3D bounding box.

    Ordering verified against bbox_anno.json: fix x (max→min), iterate
    y (max→min) and z (max→min) for each x.
    """
    mn = [float(v) for v in min_xyz]
    mx = [float(v) for v in max_xyz]
    return [
        [mx[0], mx[1], mx[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mx[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mn[0], mn[1], mn[2]],
    ]


# ---------------------------------------------------------------------------
# Selector creators — 3D
# ---------------------------------------------------------------------------

def create_3d_bbox_selector(
    min_xyz: Sequence[float],
    max_xyz: Sequence[float],
    urs: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a WKTSelector for an axis-aligned 3D bounding box.

    The 8 corners of the box are enumerated and serialised as a
    ``POLYGON Z((…))`` string, matching bbox_anno.json.

    Args:
        min_xyz: [x, y, z] coordinates of the minimum corner.
        max_xyz: [x, y, z] coordinates of the maximum corner.
        urs:     Optional URI of the coordinate reference system.

    Returns:
        ``{"type": "WKTSelector", "value": "POLYGON Z((8 corners))", ["urs": …]}``

    Raises:
        ValueError: If min_xyz or max_xyz are not 3-element sequences.
    """
    if len(min_xyz) != 3 or len(max_xyz) != 3:
        raise ValueError("min_xyz and max_xyz must be 3-element sequences")
    corners = _bbox_to_wkt_corners(min_xyz, max_xyz)
    selector: Dict[str, Any] = {
        "type": "WKTSelector",
        "value": to_wkt_polygon_z(corners),
    }
    if urs is not None:
        selector["urs"] = urs
    return selector


def create_3d_polygon_selector(
    points: Sequence[Sequence[float]],
    urs: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a WKTSelector for an arbitrary 3D polygon.

    Args:
        points: Non-empty sequence of [x, y, z] coordinate triples.
        urs:    Optional URI of the coordinate reference system.

    Returns:
        ``{"type": "WKTSelector", "value": "POLYGON Z((…))", ["urs": …]}``

    Raises:
        ValueError: If points is empty or any element is not 3-element.
    """
    if not points or any(len(pt) != 3 for pt in points):
        raise ValueError("points must be a non-empty sequence of 3-element coordinates")
    selector: Dict[str, Any] = {
        "type": "WKTSelector",
        "value": to_wkt_polygon_z([[float(x), float(y), float(z)] for x, y, z in points]),
    }
    if urs is not None:
        selector["urs"] = urs
    return selector


def create_wkt_selector(
    points: Sequence[Sequence[float]],
    urs: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a WKTSelector from a list of 3D points.

    Thin alias for :func:`create_3d_polygon_selector`.

    Args:
        points: Non-empty sequence of [x, y, z] coordinate triples.
        urs:    Optional coordinate reference system URI.

    Returns:
        ``{"type": "WKTSelector", "value": "POLYGON Z((…))", ["urs": …]}``
    """
    return create_3d_polygon_selector(points, urs=urs)


def create_mesh_index_selector(
    indices: Sequence[int],
    urs: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a MeshIndexSelector referencing mesh vertices by index.

    Per the DANES paper, the value is a list of integer vertex indices.

    Args:
        indices: Non-empty list of integer vertex indices.
        urs:     Optional coordinate reference system URI.

    Returns:
        ``{"type": "MeshIndexSelector", "value": [5788, 5804, …], ["urs": …]}``

    Raises:
        ValueError: If indices is empty.
    """
    if not indices:
        raise ValueError("indices must be non-empty")
    selector: Dict[str, Any] = {
        "type": "MeshIndexSelector",
        "value": [int(i) for i in indices],
    }
    if urs is not None:
        selector["urs"] = urs
    return selector


def create_mesh_vertex_selector(
    vertices: Sequence[Sequence[float]],
    urs: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a MeshVertexSelector from explicit 3D vertex coordinates.

    Per the DANES paper, the value is a list of [x, y, z] coordinate lists.

    Args:
        vertices: Non-empty sequence of [x, y, z] float triples.
        urs:      Optional coordinate reference system URI.

    Returns:
        ``{"type": "MeshVertexSelector", "value": [[x,y,z], …], ["urs": …]}``

    Raises:
        ValueError: If vertices is empty or any element is not 3-element.
    """
    if not vertices or any(len(v) != 3 for v in vertices):
        raise ValueError("vertices must be a non-empty sequence of 3-element coordinates")
    selector: Dict[str, Any] = {
        "type": "MeshVertexSelector",
        "value": [[float(x), float(y), float(z)] for x, y, z in vertices],
    }
    if urs is not None:
        selector["urs"] = urs
    return selector


# ---------------------------------------------------------------------------
# Selector creators — 2D
# ---------------------------------------------------------------------------

def create_svg_selector(points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Create an SvgSelector for a 2D polygon.

    Produces the format used in HS1174_front.png.json:
    ``<svg><polygon points="x1,y1 x2,y2 …"></polygon></svg>``

    Args:
        points: Non-empty sequence of [x, y] 2D coordinate pairs.

    Returns:
        ``{"type": "SvgSelector", "value": "<svg><polygon points=\"…\"></polygon></svg>"}``

    Raises:
        ValueError: If points is empty or any element is not 2-element.
    """
    if not points or any(len(pt) != 2 for pt in points):
        raise ValueError("points must be a non-empty sequence of 2-element [x, y] pairs")
    pts_str = " ".join(f"{float(x)},{float(y)}" for x, y in points)
    return {
        "type": "SvgSelector",
        "value": f'<svg><polygon points="{pts_str}"></polygon></svg>',
    }


def create_fragment_selector(
    x: float,
    y: float,
    w: float,
    h: float,
) -> Dict[str, Any]:
    """Create a FragmentSelector using the XYWH pixel format.

    Produces the format parsed by annotationExport.py in cuneur_template.

    Args:
        x: Left edge of the bounding box in pixels.
        y: Top edge of the bounding box in pixels.
        w: Width of the bounding box in pixels.
        h: Height of the bounding box in pixels.

    Returns:
        ``{"type": "FragmentSelector", "value": "xywh=pixel:x,y,w,h"}``
    """
    return {
        "type": "FragmentSelector",
        "value": f"xywh=pixel:{float(x)},{float(y)},{float(w)},{float(h)}",
    }


# ---------------------------------------------------------------------------
# Selector parsing
# ---------------------------------------------------------------------------

def _parse_svg_selector_coords(svg_value: str) -> Optional[List[List[float]]]:
    """Extract 2D [x, y] points from an SvgSelector value string."""
    start = svg_value.find('points="')
    if start == -1:
        return None
    start += len('points="')
    end = svg_value.find('"', start)
    if end == -1:
        return None
    pts_str = svg_value[start:end]
    result = []
    for token in pts_str.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected 'x,y' token in SVG points, got {token!r}")
        result.append([float(parts[0]), float(parts[1])])
    return result


def _parse_fragment_selector_coords(value: str) -> Dict[str, float]:
    """Parse a FragmentSelector 'xywh=pixel:x,y,w,h' value string."""
    cleaned = value.replace("xywh=pixel:", "").replace("xywh=", "").strip()
    parts = cleaned.split(",")
    if len(parts) != 4:
        raise ValueError(f"Expected 4 values in FragmentSelector, got {value!r}")
    keys = ("x", "y", "w", "h")
    return {k: float(v) for k, v in zip(keys, parts)}


def parse_selector(selector: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a selector dict into a normalised form with a ``"coords"`` key.

    Coord format by selector type:

    - ``WKTSelector``: ``List[List[float]]`` — parsed POLYGON Z points
    - ``MeshIndexSelector``: ``List[int]`` — vertex indices
    - ``MeshVertexSelector``: ``List[List[float]]`` — vertex coordinates
    - ``SvgSelector``: ``List[List[float]]`` — 2D [x, y] pairs
    - ``FragmentSelector``: ``{"x":…, "y":…, "w":…, "h":…}``
    - Unknown type: ``None``

    The optional ``"urs"`` field is passed through if present.

    Args:
        selector: A raw selector dict from an annotation target.

    Returns:
        Dict with at minimum ``"type"`` and ``"coords"`` keys.

    Raises:
        ValueError: If the selector value is present but malformed.
    """
    sel_type = selector.get("type", "")
    result: Dict[str, Any] = {"type": sel_type}
    if "urs" in selector:
        result["urs"] = selector["urs"]

    value = selector.get("value")

    if sel_type == "WKTSelector":
        result["coords"] = parse_wkt_polygon_z(value) if value is not None else None
    elif sel_type == "MeshIndexSelector":
        result["coords"] = [int(i) for i in value] if value is not None else None
    elif sel_type == "MeshVertexSelector":
        result["coords"] = (
            [[float(c) for c in pt] for pt in value] if value is not None else None
        )
    elif sel_type == "SvgSelector":
        result["coords"] = _parse_svg_selector_coords(value) if value is not None else None
    elif sel_type == "FragmentSelector":
        result["coords"] = _parse_fragment_selector_coords(value) if value is not None else None
    else:
        result["coords"] = None

    return result


def get_selector_coords(annotation: Annotation) -> Optional[Any]:
    """Return the parsed coordinates from the selector of an annotation.

    Convenience wrapper around :func:`parse_selector`.

    Args:
        annotation: A full annotation dict.

    Returns:
        The ``"coords"`` value from :func:`parse_selector`, or ``None`` if the
        annotation has no target/selector or the type is unknown.
    """
    target = annotation.get("target")
    if not isinstance(target, dict):
        return None
    selector = target.get("selector")
    if not isinstance(selector, dict):
        return None
    return parse_selector(selector).get("coords")


# ---------------------------------------------------------------------------
# Body construction
# ---------------------------------------------------------------------------

def create_cuneiform_body(
    sign_label: str = "Wedge",
    line: Optional[str] = None,
    charindex: Optional[str] = None,
    wedgeindex: Optional[str] = None,
    wedgetype: Optional[str] = None,
    extra_fields: Optional[Dict[str, str]] = None,
    *,
    mode: str = "2d",
    transliteration: Optional[str] = None,
    wordindex: Optional[str] = None,
    relcharindex: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create a cuneiform annotation body.

    Two modes are supported:

    ``"2d"`` (default, backward-compatible)
        Produces a wedge-style body matching HS1174_front.png.json:
        a SpecificResource wedge tag followed by TextualBody items for
        Line, Charindex, Wedgeindex, and Wedgetype.

    ``"3d"``
        Produces a character-level body matching bbox_anno.json:
        TextualBody items for Transliteration, Character tagging, Line,
        and Charindex, each with a ``"source"`` URI.  Optionally includes
        Wordindex and RelCharindex.

    Args:
        sign_label:     (2D only) Label for the wedge SpecificResource.
        line:           Line number as string.
        charindex:      Character index as string.
        wedgeindex:     (2D only) Wedge index within character.
        wedgetype:      (2D only) Wedge type code (e.g. ``"w"``).
        extra_fields:   Additional ``{purpose: value}`` TextualBody items.
        mode:           ``"2d"`` or ``"3d"``.  Keyword-only.
        transliteration: (3D only) Transliteration string (e.g. ``"9(disz)"``).
        wordindex:      (3D only) Word index as string.
        relcharindex:   (3D only) Relative character index as string.

    Returns:
        List of body dicts.

    Raises:
        ValueError: If mode is not ``"2d"`` or ``"3d"``.
    """
    if mode == "2d":
        body: List[Dict[str, Any]] = [DEFAULT_CUNEIFORM_TAG.copy()]
        if line is not None:
            body.append({"type": "TextualBody", "purpose": "Line", "value": str(line)})
        if charindex is not None:
            body.append({"type": "TextualBody", "purpose": "Charindex", "value": str(charindex)})
        if wedgeindex is not None:
            body.append({"type": "TextualBody", "purpose": "Wedgeindex", "value": str(wedgeindex)})
        if wedgetype is not None:
            body.append({"type": "TextualBody", "purpose": "Wedgetype", "value": str(wedgetype)})
        if extra_fields:
            for purpose, value in extra_fields.items():
                body.append({"type": "TextualBody", "purpose": purpose, "value": str(value)})
        return body

    elif mode == "3d":
        body = []
        if transliteration is not None:
            body.append({
                "type": "TextualBody",
                "purpose": "Transliteration",
                "value": str(transliteration),
                "source": _SOURCE_TRANSLITERATION,
            })
        body.append({
            "type": "TextualBody",
            "value": "Character",
            "purpose": "tagging",
            "source": _SOURCE_CHARACTER,
        })
        if line is not None:
            body.append({
                "type": "TextualBody",
                "purpose": "Line",
                "value": str(line),
                "source": _SOURCE_LINE,
            })
        if charindex is not None:
            body.append({
                "type": "TextualBody",
                "purpose": "Charindex",
                "value": str(charindex),
                "source": _SOURCE_CHARINDEX,
            })
        if wordindex is not None:
            body.append({
                "type": "TextualBody",
                "purpose": "Wordindex",
                "value": str(wordindex),
                "source": _SOURCE_WORDINDEX,
            })
        if relcharindex is not None:
            body.append({
                "type": "TextualBody",
                "purpose": "RelCharindex",
                "value": str(relcharindex),
                "source": _SOURCE_RELCHARINDEX,
            })
        if extra_fields:
            for purpose, value in extra_fields.items():
                body.append({"type": "TextualBody", "purpose": purpose, "value": str(value)})
        return body

    else:
        raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")


# ---------------------------------------------------------------------------
# Annotation construction
# ---------------------------------------------------------------------------

def create_annotation(
    source: str,
    selector: Dict[str, Any],
    body: Optional[List[Dict[str, Any]]] = None,
    annotation_id: Optional[str] = None,
    context: Optional[str] = None,
    *,
    rights: Optional[str] = None,
    generator: Optional[Dict[str, Any]] = None,
    target_rights: Optional[str] = None,
    target_creator: Optional[Dict[str, Any]] = None,
    target_dimensions: Optional[Dict[str, Any]] = None,
) -> Annotation:
    """Create a full Annotation object suitable for the Web Annotation model.

    Args:
        source:            URL of the annotated resource.
        selector:          Selector dict (e.g. from :func:`create_wkt_selector`).
        body:              List of body dicts.  Defaults to a single wedge tag.
        annotation_id:     Explicit annotation ID.  Auto-generated if omitted.
        context:           JSON-LD ``@context`` URL.  Defaults to DEFAULT_CONTEXT.
        rights:            (keyword-only) Top-level ``"rights"`` field.
        generator:         (keyword-only) Top-level ``"generator"`` dict.
        target_rights:     (keyword-only) ``"rights"`` inside the target dict.
        target_creator:    (keyword-only) ``"creator"`` inside the target dict.
        target_dimensions: (keyword-only) ``"dimensions"`` inside the target dict.

    Returns:
        A fully-formed annotation dict.
    """
    annotation_id = annotation_id or _generate_annotation_id()
    context = context or DEFAULT_CONTEXT

    target: Dict[str, Any] = {"source": source, "selector": selector}
    if target_rights is not None:
        target["rights"] = target_rights
    if target_dimensions is not None:
        target["dimensions"] = target_dimensions
    if target_creator is not None:
        target["creator"] = target_creator

    annotation: Annotation = {
        "type": "Annotation",
        "id": annotation_id,
        "@context": context,
        "body": body if body is not None else [DEFAULT_CUNEIFORM_TAG.copy()],
        "target": target,
    }
    if rights is not None:
        annotation["rights"] = rights
    if generator is not None:
        annotation["generator"] = generator

    return annotation


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def get_body_value(annotation: Annotation, purpose: str) -> Optional[str]:
    """Return the value of the first body item matching purpose, or None.

    Args:
        annotation: A full annotation dict.
        purpose:    The purpose to search for, e.g. ``"Transliteration"``.

    Returns:
        The matching value as a string, or ``None``.
    """
    for item in annotation.get("body", []):
        if item.get("purpose") == purpose and "value" in item:
            return str(item["value"])
    return None


def get_all_body_values(annotation: Annotation) -> Dict[str, str]:
    """Return all {purpose: value} pairs from an annotation's body.

    Only items that have both ``"purpose"`` and ``"value"`` are included.
    If multiple items share the same purpose, the last one wins.

    Args:
        annotation: A full annotation dict.

    Returns:
        Dict mapping purpose strings to value strings.
    """
    result: Dict[str, str] = {}
    for item in annotation.get("body", []):
        if "purpose" in item and "value" in item:
            result[item["purpose"]] = str(item["value"])
    return result


def get_selector_type(annotation: Annotation) -> Optional[str]:
    """Return the selector type string from an annotation's target, or None.

    Args:
        annotation: A full annotation dict.
    """
    target = annotation.get("target")
    if not isinstance(target, dict):
        return None
    selector = target.get("selector")
    if not isinstance(selector, dict):
        return None
    return selector.get("type")


def is_3d_annotation(annotation: Annotation) -> bool:
    """Return True if the annotation uses a recognised 3D selector type.

    Recognised types: ``WKTSelector``, ``MeshIndexSelector``, ``MeshVertexSelector``.
    """
    return get_selector_type(annotation) in _SELECTOR_TYPES_3D


def is_2d_annotation(annotation: Annotation) -> bool:
    """Return True if the annotation uses a recognised 2D selector type.

    Recognised types: ``SvgSelector``, ``FragmentSelector``.
    """
    return get_selector_type(annotation) in _SELECTOR_TYPES_2D


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_id_into_annotation(annotation_id: Optional[str], annotation: Annotation) -> Annotation:
    if annotation_id is None:
        return dict(annotation)
    if annotation.get("id") == annotation_id:
        return dict(annotation)
    merged = dict(annotation)
    merged["id"] = annotation_id
    return merged


def _generate_annotation_id() -> str:
    return f"#{uuid.uuid4()}"


def _looks_like_annotation_map(data: Dict[str, Any]) -> bool:
    if not data:
        return False
    return all(isinstance(key, str) and isinstance(value, dict) and value.get("type") == "Annotation"
               for key, value in data.items())


def _ensure_annotation(annotation: Annotation, context: Optional[str] = None) -> Annotation:
    annotation = dict(annotation)
    annotation.setdefault("type", "Annotation")
    annotation.setdefault("id", _generate_annotation_id())
    if context is not None:
        annotation.setdefault("@context", context)
    annotation.setdefault("body", [DEFAULT_CUNEIFORM_TAG.copy()])
    target = annotation.get("target")
    if not target or not isinstance(target, dict):
        raise ValueError("Annotation target must be a dictionary containing source and selector")
    if "source" not in target or "selector" not in target:
        raise ValueError("Annotation target requires both 'source' and 'selector' fields")
    return annotation


def _annotations_to_map(annotations: List[Annotation], context: Optional[str] = None) -> AnnotationMap:
    result: AnnotationMap = {}
    for annotation in annotations:
        safe_annotation = _ensure_annotation(annotation, context=context)
        annotation_id = safe_annotation["id"]
        if annotation_id in result:
            raise ValueError(f"Duplicate annotation id: {annotation_id}")
        result[annotation_id] = safe_annotation
    return result
