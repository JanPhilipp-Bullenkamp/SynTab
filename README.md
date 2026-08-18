# SynTab

Procedural generation of labeled 3D cuneiform tablet meshes.

SynTab creates synthetic training data for machine learning, 3D reconstruction, and computational paleography research. Given a database of cuneiform signs in PaleoCodage notation, it generates realistic 3D tablet meshes with wedge imprints carved via SDF-based boolean operations, together with per-vertex labels and W3C WebAnnotation JSON metadata.

## Features

- Superquadric tablet geometry
- PaleoCodage string parsing into 2D wedge type/position/rotation parameters
- Automatic sign layout on blank tablet
- SDF-based boolean difference to carve wedge impressions into the mesh
- Multi-step mesh post-processing: noise, remesh, smooth, subdivide
- Different per-vertex wedge ID labels and W3C Web Annotation JSON with 3D bounding boxes

## Installation

Requires Python 3.10+.

```bash
pip install numpy trimesh scipy pymeshlab
```

## Quick Start

```bash
# Generate 10 labeled tablets (output in out/)
python generate_tablets.py

# Single sign on blank tablet using paleocode for debugging/visualization
python generate_single_sign.py "d:c:d:c"
```

Output is written to `out/plys/`, `out/labels/`, `out/annotations/`, and `out/configs/`.

## Configuration

All parameters are controlled in [config.py](config.py):

| Class | Controls |
|---|---|
| `GeometryConfig` | Tablet dimensions and superquadric shape |
| `WedgeCarvingConfig` | Wedge depth, angle, and tilt ranges |
| `SdfConfig` | Voxel grid resolution and padding |
| `PostprocessConfig` | Remesh iterations, smoothing, target vertex counts |
| `LayoutConfig` | Sign spacing, margins, multi-column layout |
| `PaleocodageConfig` | Stroke lengths and scaling factors |
| `GenerationConfig` | Number of tablets, parallel workers, RNG seed |
| `DebugTabletConfig` | Fixed tablet shape used by `generate_single_sign.py` |

## Output Format

Four files are generated per tablet:

| File | Description |
|---|---|
| `out/plys/tablet_<id>.ply` | Binary PLY mesh |
| `out/labels/tablet_<id>.txt` | Per-vertex wedge IDs (`vertex_index label_id`) |
| `out/annotations/tablet_<id>.json` | W3C WebAnnotation with 3D bounding boxes |
| `out/configs/tablet_<id>.json` | Generation parameters for reproducibility |

Annotation entries follow the W3C Web Annotation Data Model and include wedge type, PaleoCodage string, transliteration, tablet face, and an axis-aligned 3D bounding box (`Box3DSelector`).

## License

GNU GPL v3 — see [LICENSE](LICENSE).
