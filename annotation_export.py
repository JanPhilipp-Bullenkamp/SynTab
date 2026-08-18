"""Export the labelled outputs of a generated tablet: per-vertex labels and
W3C WebAnnotation JSON."""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh

from config import SdfConfig


def export_labels(labels, filepath) -> None:
    """Write per-vertex labels as a commented `index label` text file."""
    with open(filepath, "w") as f:
        f.write("# +-----------------------------------------------------+\n")
        f.write("# | txt file with labels per index                      |\n")
        f.write("# +-----------------------------------------------------+\n")
        f.write("# | Format: index label                                 |\n")
        f.write("# +-----------------------------------------------------+\n")
        for index, label in enumerate(labels):
            if isinstance(label, list):
                label = " ".join(str(l) for l in label)
            f.write(f"{index} {label}\n")


def _textual_body(purpose: str, value: Any) -> Dict[str, object]:
    return {"type": "TextualBody", "purpose": purpose, "value": str(value)}


def _tag_body(resource_id: str, label: str) -> Dict[str, object]:
    return {
        "type": "SpecificResource",
        "purpose": "tagging",
        "source": {"id": resource_id, "label": label},
    }


def _bbox3d_selector_from_bounds(bounds: np.ndarray) -> Dict[str, object]:
    bmin = np.asarray(bounds[0], dtype=float)
    bmax = np.asarray(bounds[1], dtype=float)
    size = bmax - bmin
    center = 0.5 * (bmin + bmax)
    x0, y0, z0 = bmin.tolist()
    x1, y1, z1 = bmax.tolist()
    corners = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    return {
        "type": "Box3DSelector",
        "value": {
            "format": "AABB-XYZXYZ",
            "min": bmin.tolist(),
            "max": bmax.tolist(),
            "size": size.tolist(),
            "center": center.tolist(),
            "corners": corners,
        },
    }


def _new_annotation_id() -> str:
    return f"#{uuid.uuid4()}"


def export_webannotation_3d(
    wedges_3d,
    filepath: str,
    *,
    scale: float,
    source: str,
    config: SdfConfig,
    label_mesh: trimesh.Trimesh,
    vertex_labels: np.ndarray,
    sign_reference: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """Write per-wedge and per-sign W3C WebAnnotation JSON alongside the mesh.

    Each wedge gets one annotation with a 3-D bounding box selector derived from
    the vertices that carry its label.  Signs are aggregated from their wedge
    records and written as a second set of annotations.
    """
    base_label = config.base_label
    cutter_label_offset = config.cutter_label_offset

    annotations: Dict[str, Dict[str, object]] = {}
    signs_agg: Dict[int, Dict[str, object]] = {}
    skipped_wedge_ids: List[int] = []

    labels_arr = np.asarray(vertex_labels, dtype=np.int32).reshape(-1)
    if len(labels_arr) != len(label_mesh.vertices):
        raise ValueError(
            f"vertex_labels length ({len(labels_arr)}) does not match "
            f"mesh vertices ({len(label_mesh.vertices)})"
        )

    # Pre-compute per-label bounding boxes from vertex positions.
    label_bounds_by_id: Dict[int, np.ndarray] = {}
    for label_id in np.unique(labels_arr):
        lid = int(label_id)
        if lid <= int(base_label):
            continue
        pts = np.asarray(label_mesh.vertices[labels_arr == lid], dtype=float)
        if pts.size == 0:
            continue
        label_bounds_by_id[lid] = np.stack([pts.min(axis=0), pts.max(axis=0)], axis=0)

    for wedge in wedges_3d:
        label_id = int(wedge.wedge_id) + int(cutter_label_offset)
        bounds_scaled = label_bounds_by_id.get(label_id)
        if bounds_scaled is None:
            skipped_wedge_ids.append(int(wedge.wedge_id))
            continue

        selector = _bbox3d_selector_from_bounds(bounds_scaled)

        sign_index0 = int(wedge.sign_index)
        sign_index = sign_index0 + 1
        wedge_index = int(wedge.wedge_index) + 1
        line_index = int(wedge.line_index) if wedge.line_index is not None else 1
        char_index = int(wedge.char_index) if wedge.char_index is not None else sign_index
        sign_code = str(wedge.sign_code) if wedge.sign_code else ""

        ref = (sign_reference or {}).get(sign_code, {})
        transliteration = ref.get("transliteration")
        sign_char = ref.get("sign")

        body: List[Dict[str, object]] = [
            _tag_body("http://purl.org/cuneiform/Wedge", "Wedge"),
            _textual_body("Line", line_index),
            _textual_body("Charindex", char_index),
            _textual_body("Signindex", sign_index),
            _textual_body("Wedgeindex", wedge_index),
            _textual_body("Wedgetype", wedge.wedge_type),
            _textual_body("Face", wedge.face.name),
            _textual_body("Depth", float(wedge.depth) * float(scale)),
            _textual_body("Angle", float(wedge.angle)),
            _textual_body("SizeScale", float(wedge.size_scale)),
            _textual_body("Wedgesize", getattr(wedge, "size_class", "normal")),
        ]
        if sign_code:
            body.append(_textual_body("Paleocode", sign_code))
        if transliteration:
            body.append(_textual_body("Transliteration", transliteration))
        if sign_char:
            body.append(_textual_body("Sign", sign_char))

        ann_id = _new_annotation_id()
        annotations[ann_id] = {
            "type": "Annotation",
            "body": body,
            "target": {"source": source, "selector": selector},
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": ann_id,
        }

        # Aggregate per-sign record
        sign_record = signs_agg.get(sign_index0)
        if sign_record is None:
            signs_agg[sign_index0] = {
                "bbox_min": bounds_scaled[0].copy(),
                "bbox_max": bounds_scaled[1].copy(),
                "line_index": line_index,
                "char_index": char_index,
                "sign_index": sign_index,
                "sign_code": sign_code,
                "transliteration": transliteration,
                "sign_char": sign_char,
                "wedge_types": [str(wedge.wedge_type)],
                "faces": [str(wedge.face.name)],
                "wedge_ids": [int(wedge.wedge_id)],
            }
        else:
            sign_record["bbox_min"] = np.minimum(sign_record["bbox_min"], bounds_scaled[0])
            sign_record["bbox_max"] = np.maximum(sign_record["bbox_max"], bounds_scaled[1])
            if not sign_record.get("transliteration") and transliteration:
                sign_record["transliteration"] = transliteration
            if not sign_record.get("sign_char") and sign_char:
                sign_record["sign_char"] = sign_char
            if not sign_record.get("sign_code") and sign_code:
                sign_record["sign_code"] = sign_code
            sign_record["wedge_types"].append(str(wedge.wedge_type))
            sign_record["faces"].append(str(wedge.face.name))
            sign_record["wedge_ids"].append(int(wedge.wedge_id))

    for sign_key in sorted(signs_agg.keys()):
        sr = signs_agg[sign_key]
        bounds = np.stack([sr["bbox_min"], sr["bbox_max"]], axis=0)
        selector = _bbox3d_selector_from_bounds(bounds)

        unique_wedge_types = sorted(set(sr["wedge_types"]))
        unique_faces = sorted(set(sr["faces"]))

        body = [
            _tag_body("http://purl.org/cuneiform/Character", "Character"),
            _textual_body("Line", sr["line_index"]),
            _textual_body("Charindex", sr["char_index"]),
            _textual_body("Signindex", sr["sign_index"]),
            _textual_body("Wedgecount", len(sr["wedge_ids"])),
            _textual_body("WedgeTypes", ",".join(unique_wedge_types)),
            _textual_body("Faces", ",".join(unique_faces)),
            _textual_body("Wedgeids", ",".join(str(x) for x in sr["wedge_ids"])),
        ]

        transliteration = sr.get("transliteration")
        sign_code = sr.get("sign_code")
        sign_char = sr.get("sign_char")

        if transliteration:
            body.insert(1, _textual_body("Transliteration", transliteration))
        elif sign_code:
            body.insert(1, _textual_body("Transliteration", sign_code))

        if sign_code:
            body.append(_textual_body("Paleocode", sign_code))
        if sign_char:
            body.append(_textual_body("Sign", sign_char))

        ann_id = _new_annotation_id()
        annotations[ann_id] = {
            "type": "Annotation",
            "body": body,
            "target": {"source": source, "selector": selector},
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "id": ann_id,
        }

    if skipped_wedge_ids:
        print(
            f"Warning: skipped {len(skipped_wedge_ids)} wedges without label-imprint region "
            f"in annotation export (sample ids: {skipped_wedge_ids[:10]})"
        )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
