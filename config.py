"""
Central configuration for synthetic cuneiform tablet generation.

All tunable parameters live here. Import TabletConfig and construct it with
defaults or keyword overrides — no other file defines numeric knobs.

    cfg = TabletConfig()                     # all defaults
    cfg = TabletConfig(sdf=SdfConfig(max_grid=400))  # override one sub-config
    cfg.to_json("run_config.json")           # save alongside outputs
    cfg2 = TabletConfig.from_json("run_config.json")  # reload
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryConfig:
    """Superquadric shape parameters — sampled uniformly per tablet."""
    height_range: tuple = (0.6, 1.0)
    width_range:  tuple = (0.7, 1.0)
    depth_range:  tuple = (0.2, 0.4)
    epsilon_range: tuple = (0.7, 0.9)   # z-axis roundness
    eta_range:    tuple = (0.7, 0.9)    # xy-plane roundness
    scale_range:  tuple = (8.0, 25.0)   # final mesh scale in mm

    def __post_init__(self):
        for name in ("height_range", "width_range", "depth_range",
                     "epsilon_range", "eta_range", "scale_range"):
            lo, hi = getattr(self, name)
            if lo >= hi:
                raise ValueError(f"GeometryConfig.{name}: lo ({lo}) must be < hi ({hi})")
            if lo <= 0:
                raise ValueError(f"GeometryConfig.{name}: lo ({lo}) must be > 0")


@dataclass(frozen=True)
class WedgeCarvingConfig:
    """Randomisation ranges for individual wedge carving."""
    min_depth: float = 0.025
    max_depth: float = 0.055
    min_angle: float = 11.0
    max_angle: float = 29.0
    min_tilt_angle: float = -12.0
    max_tilt_angle: float = 28.0
    base_wedge_size: float = 0.65
    # Previously hardcoded constants promoted to config
    wedge_beta: float = -45.0               # rotation angle in get_wedge_trafo
    tangent_collinear_threshold: float = 0.9  # near-parallel check in imprint

    def __post_init__(self):
        if self.min_depth >= self.max_depth:
            raise ValueError(f"WedgeCarvingConfig: min_depth ({self.min_depth}) must be < max_depth ({self.max_depth})")
        if self.min_angle >= self.max_angle:
            raise ValueError(f"WedgeCarvingConfig: min_angle ({self.min_angle}) must be < max_angle ({self.max_angle})")
        if self.min_tilt_angle >= self.max_tilt_angle:
            raise ValueError(f"WedgeCarvingConfig: min_tilt_angle ({self.min_tilt_angle}) must be < max_tilt_angle ({self.max_tilt_angle})")
        if self.base_wedge_size <= 0:
            raise ValueError(f"WedgeCarvingConfig: base_wedge_size must be > 0")


@dataclass(frozen=True)
class SdfConfig:
    """Voxelisation and SDF boolean-difference parameters."""
    target_pitch: float = 1.0 / 120.0  # desired voxel size (normalised units)
    padding: float = 1.5                # padding around bounds in voxels
    max_grid: int = 500                 # max grid dimension; caps memory use
    base_bounds_margin: float = 0.02   # margin around base superquadric
    eps_active_fraction: float = 0.5   # eps_active = fraction * pitch
    base_label: int = 1                # label assigned to unlabelled tablet surface
    cutter_label_offset: int = 1       # offset added to wedge_id for final label
    raster_threads: int = 4            # parallel threads for cutter rasterization (1 = serial)

    def __post_init__(self):
        if self.target_pitch <= 0:
            raise ValueError("SdfConfig: target_pitch must be > 0")
        if self.max_grid <= 0:
            raise ValueError("SdfConfig: max_grid must be > 0")
        if self.raster_threads < 1:
            raise ValueError("SdfConfig: raster_threads must be >= 1")


@dataclass(frozen=True)
class PostprocessConfig:
    """Mesh post-processing pipeline parameters."""
    target_vertices_range: tuple = (70_000, 200_000)
    noise_magnitude: float = 0.0018
    remesh_iterations: int = 3
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5   # Taubin λ — previously hardcoded in util
    smooth_mu: float = -0.53     # Taubin μ — previously hardcoded in util

    def __post_init__(self):
        lo, hi = self.target_vertices_range
        if lo >= hi:
            raise ValueError(f"PostprocessConfig: target_vertices_range lo ({lo}) must be < hi ({hi})")
        if self.remesh_iterations < 0:
            raise ValueError("PostprocessConfig: remesh_iterations must be >= 0")
        if self.smooth_iterations < 0:
            raise ValueError("PostprocessConfig: smooth_iterations must be >= 0")


@dataclass(frozen=True)
class LayoutConfig:
    """Parameters for placing cuneiform signs on the superquadric surface."""
    target_sign_height_frac: float = 0.04
    sign_height_ratio: float = 0.5
    sign_spacing_ratio: float = 0.04
    sign_padding: float = 0.05
    sign_padding_frac: float = 0.02
    small_sign_bias: float = 0.7
    project_iters: int = 8              # Newton iterations for surface projection
    band_axis: str = "y"
    flip_vertical_axis: bool = True
    global_center_separation: float = 0.25
    tangent_delta: float = 1e-4         # finite-difference step for tangent frames
    arclength_u_samples: int = 1024     # samples for u-direction arc-length
    arclength_v_samples: int = 2048     # samples for v-band arc-length
    # Margins
    margin_top_frac: float = 0.04       # fraction of meridian arc to skip at top
    margin_bottom_frac: float = 0.04    # fraction of meridian arc to skip at bottom
    margin_side_frac: float = 0.05      # fraction of band-length to skip per side
    # Row alignment
    row_phase: float = 0.0              # fixed u-phase for all rows (0 = front-center start)
    # Justified filling
    justify_rows: bool = True           # spread signs evenly across each row
    justify_threshold: float = 0.55     # min fill ratio to justify; below → center block
    # Column-block layout
    n_columns: int = 1                  # 1 = full-width rows; >1 = column-block layout
    column_gap_frac: float = 0.03       # gap between columns as fraction of band-length

    def __post_init__(self):
        if self.band_axis not in ("y", "z"):
            raise ValueError(f"LayoutConfig: band_axis must be 'y' or 'z', got '{self.band_axis}'")
        if self.project_iters < 1:
            raise ValueError("LayoutConfig: project_iters must be >= 1")
        if self.n_columns < 1:
            raise ValueError("LayoutConfig: n_columns must be >= 1")
        if not (0.0 <= self.justify_threshold <= 1.0):
            raise ValueError("LayoutConfig: justify_threshold must be in [0, 1]")


@dataclass(frozen=True)
class PaleocodageConfig:
    """Controls how PaleoCodage strings are decoded into 2-D wedge positions."""
    stroke_length: float = 30.0
    step_x: float = 10.0
    step_y: float = 7.0
    big_scale: float = 1.5      # scale for uppercase / 'B'-prefixed wedges
    small_scale: float = 0.5    # scale for 's'-prefixed wedges
    rotation_constant: float = 15.0  # angle delta for '<' / '>' operators
    bbox_padding: float = 2.0


@dataclass(frozen=True)
class GenerationConfig:
    """Dataset-level generation settings."""
    num_signs_range: tuple = (180, 320)   # signs sampled per tablet
    max_attempts: int = 10                # retries if generation fails
    workers: int = 1                      # parallel workers (1 = serial)
    base_seed: Optional[int] = None       # RNG seed (None = random)
    max_paleocode_length: Optional[int] = None  # filter by code length
    flip_tail_direction_x: bool = False   # mirror wedge tails along x
    debug: bool = False

    def __post_init__(self):
        lo, hi = self.num_signs_range
        if lo >= hi:
            raise ValueError(f"GenerationConfig: num_signs_range lo ({lo}) must be < hi ({hi})")
        if self.max_attempts < 1:
            raise ValueError("GenerationConfig: max_attempts must be >= 1")
        if self.workers < 1:
            raise ValueError("GenerationConfig: workers must be >= 1")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TabletConfig:
    """
    Master configuration object.  Construct with defaults and override any
    sub-config you need:

        cfg = TabletConfig(sdf=SdfConfig(max_grid=400))
    """
    geometry:    GeometryConfig     = field(default_factory=GeometryConfig)
    carving:     WedgeCarvingConfig = field(default_factory=WedgeCarvingConfig)
    sdf:         SdfConfig          = field(default_factory=SdfConfig)
    postprocess: PostprocessConfig  = field(default_factory=PostprocessConfig)
    layout:      LayoutConfig       = field(default_factory=LayoutConfig)
    paleocodage: PaleocodageConfig  = field(default_factory=PaleocodageConfig)
    generation:  GenerationConfig   = field(default_factory=GenerationConfig)

    def __post_init__(self):
        # Sub-configs validate themselves; this just ensures they are the right types.
        for fname, ftype in (
            ("geometry", GeometryConfig),
            ("carving", WedgeCarvingConfig),
            ("sdf", SdfConfig),
            ("postprocess", PostprocessConfig),
            ("layout", LayoutConfig),
            ("paleocodage", PaleocodageConfig),
            ("generation", GenerationConfig),
        ):
            val = getattr(self, fname)
            if not isinstance(val, ftype):
                raise TypeError(f"TabletConfig.{fname} must be a {ftype.__name__}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TabletConfig":
        def _coerce(sub_cls, sub_dict: dict) -> object:
            if not sub_dict:
                return sub_cls()
            fmap = {f.name: f for f in dataclasses.fields(sub_cls)}
            kwargs: dict[str, Any] = {}
            for k, v in sub_dict.items():
                if k not in fmap:
                    continue
                # JSON deserialises tuples as lists; restore them.
                if isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v
            return sub_cls(**kwargs)

        return cls(
            geometry=_coerce(GeometryConfig,     d.get("geometry", {})),
            carving=_coerce(WedgeCarvingConfig,  d.get("carving", {})),
            sdf=_coerce(SdfConfig,               d.get("sdf", {})),
            postprocess=_coerce(PostprocessConfig, d.get("postprocess", {})),
            layout=_coerce(LayoutConfig,         d.get("layout", {})),
            paleocodage=_coerce(PaleocodageConfig, d.get("paleocodage", {})),
            generation=_coerce(GenerationConfig, d.get("generation", {})),
        )

    @classmethod
    def from_json(cls, path: str) -> "TabletConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
