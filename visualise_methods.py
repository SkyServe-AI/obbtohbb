#!/usr/bin/env python3
"""
visualise_methods.py  -  Save individual 64x64 px chip PNGs for each HBB method.

Produces 9 chips (3 per angle bin: 0-15°, 15-30°, 30-45°).
Each chip is saved as a separate PNG with all method overlays drawn on it:
  - OBB outline (white dashed)
  - GT_HBB      (green)
  - Outer_HBB   (red)
  - Area_Equiv_HBB (orange)
  - GBB_Marginalized (purple)
  - Novel_ShapeAware (cyan)

Output files: <output_dir>/chip_<bin>_<n>_<scene>_<angle>.png

Usage
-----
  python visualise_methods.py \
      --tif_roots  Data/mock_raw_tif_files_and_shp_files/MOCK/aditi_08_04_mock \
                   Data/mock_raw_tif_files_and_shp_files/MOCK/aki_mock_images \
      --obb_gt_root Data/OBB_GT \
      --hbb_gt_root Data/HBB_GT \
      --output_dir  chips \
      --seed        42
"""
from __future__ import annotations
import argparse
import math
import random
import sys
from pathlib import Path

import cairocffi as cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent))
import obb2hbb
from obbhbbstats import (
    outer_hbb, area_equiv_hbb, gbb_marginalized_hbb, novel_shape_aware,
)

ANGLE_BINS      = ["0-15", "15-30", "30-45"]
CHIP_HALF       = 16          # pixels each side of centroid → 32×32 chip
SCALE           = 1           # upscale factor for drawing clarity → 512×512 saved
SAMPLES_PER_BIN = 3

EXTRA_METHODS   = ["Outer_HBB", "Area_Equiv_HBB", "GBB_Marginalized"]
DEFAULT_METHODS = ["OBB", "GT_HBB", "Novel_ShapeAware"]

# BGR-style tuples as PIL RGB
METHOD_STYLES = {
    "OBB":              {"color": (255,  60,  60), "width": 0.5, "dash": False},
    "GT_HBB":           {"color": (0,   220,   0), "width": 0.5, "dash": False},
    "Outer_HBB":        {"color": (255, 255, 255), "width": 0.5, "dash": False},
    "Area_Equiv_HBB":   {"color": (255, 160,   0), "width": 0.5, "dash": False},
    "GBB_Marginalized": {"color": (180,   0, 255), "width": 0.5, "dash": False},
    "Novel_ShapeAware": {"color": (255, 255,   0), "width": 0.5, "dash": False},
}


def _angle_bin(poly: Polygon) -> str | None:
    try:
        _, _, _, _, theta = obb2hbb._mrr_params(poly)
        deg = abs(theta) * 180 / math.pi % 90
        if deg > 45:
            deg = 90 - deg
        if deg < 15:   return "0-15"
        elif deg < 30: return "15-30"
        else:          return "30-45"
    except Exception:
        return None


def _find_tif(scene_name: str, tif_roots: list[Path]) -> Path | None:
    """Find a matching .tif file by stem across roots."""
    for root in tif_roots:
        c = root / f"{scene_name}.tif"
        if c.exists():
            return c
    return None


def _find_image(scene_name: str, img_roots: list[Path]) -> Path | None:
    """Find a matching PNG/JPG file by stem across roots."""
    for root in img_roots:
        for ext in (".png", ".jpg", ".jpeg"):
            c = root / f"{scene_name}{ext}"
            if c.exists():
                return c
    return None


def _pixel_identity_transform():
    """Affine identity so pixel-space coords pass through ~transform unchanged."""
    return rasterio.transform.Affine(1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0)


def _load_png_image(img_path: Path):
    """Load a PNG/JPG with PIL — already uint8, no percentile stretch needed.
    CLAHE is applied per channel for consistency with the TIF path.
    Returns (rgb_uint8_HxWx3, pixel_identity_transform, width, height).
    """
    img = Image.open(str(img_path)).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    for c in range(3):
        rgb[:, :, c] = _clahe_channel(rgb[:, :, c])
    transform = _pixel_identity_transform()
    height, width = rgb.shape[:2]
    return rgb, transform, width, height


def _best_gt_hbb(ref_hbb: Polygon, gt_geoms: list) -> Polygon | None:
    best_iou, best_gt = -1.0, None
    for g in gt_geoms:
        if g is None or g.is_empty:
            continue
        try:
            inter = ref_hbb.intersection(g).area
            union = ref_hbb.union(g).area
            iou   = inter / union if union > 0 else 0.0
        except Exception:
            continue
        if iou > best_iou:
            best_iou, best_gt = iou, g
    return best_gt


def _clahe_channel(ch: np.ndarray, clip_limit: float = 2.0, tile: int = 4) -> np.ndarray:
    """Apply CLAHE to a single uint8 channel via cv2."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(ch)


def _tif_meta(tif_path: Path) -> tuple:
    """Read transform + dimensions. Dispatches on extension:
    - .tif: rasterio (georeferenced transform)
    - .png/.jpg/.jpeg: PIL + pixel-identity transform
    """
    if tif_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        img = Image.open(str(tif_path))
        w, h = img.size
        return _pixel_identity_transform(), w, h
    with rasterio.open(str(tif_path)) as src:
        return src.transform, src.width, src.height


def _load_normalised_image(tif_path: Path):
    """Load and normalise an image. Dispatches on extension:
    - .tif: rasterio + 2-98 percentile stretch + CLAHE
    - .png/.jpg/.jpeg: PIL + CLAHE only (already uint8)
    """
    if tif_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return _load_png_image(tif_path)
    with rasterio.open(str(tif_path)) as src:
        data      = src.read([1, 2, 3])
        transform = src.transform
        width     = src.width
        height    = src.height
    rgb = np.moveaxis(data, 0, -1).astype(np.float32)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        span = max(float(hi - lo), 1.0)
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / span * 255.0, 0, 255)
    rgb = rgb.astype(np.uint8)
    for c in range(3):
        rgb[:, :, c] = _clahe_channel(rgb[:, :, c])
    return rgb, transform, width, height


def _chip_from_image(rgb_full: np.ndarray, transform, width: int, height: int,
                     cx_geo: float, cy_geo: float):
    """Slice a chip from an already-normalised full image array.
    Returns (chip_rgb_HxWx3, chip_transform) or None if out of bounds.
    """
    inv      = ~transform
    col_f, row_f = inv * (cx_geo, cy_geo)
    col_off  = int(col_f) - CHIP_HALF
    row_off  = int(row_f) - CHIP_HALF
    w = h    = 2 * CHIP_HALF
    if col_off < 0 or row_off < 0 or col_off + w > width or row_off + h > height:
        return None
    chip           = rgb_full[row_off:row_off + h, col_off:col_off + w].copy()
    chip_transform = transform * rasterio.transform.Affine.translation(col_off, row_off)
    return chip, chip_transform


def _geo_to_px(coords, chip_transform, scale: int = 1):
    """Convert geo coords → integer pixel coords, optionally scaled."""
    inv = ~chip_transform
    return [tuple(int(round(v * scale)) for v in inv * (x, y)) for x, y in coords]


def _geo_to_px_float(coords, chip_transform, scale: float = 1.0):
    """Convert geo coords → float pixel coords (for Cairo anti-aliased drawing)."""
    inv = ~chip_transform
    return [tuple(v * scale for v in inv * (x, y)) for x, y in coords]


def _draw_poly(draw: ImageDraw.Draw, poly: Polygon, chip_transform,
               color, width: int, dash: bool, scale: int):
    """PIL-based polygon drawing (integer widths >= 1)."""
    coords = list(poly.exterior.coords)
    px     = _geo_to_px(coords, chip_transform, scale)
    pts = px + [px[0]]
    if dash:
        seg_len = max(4 * scale, 1)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]; x1, y1 = pts[i + 1]
            dx, dy = x1 - x0, y1 - y0
            dist   = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            steps = max(1, int(dist / seg_len))
            for s in range(steps):
                if s % 2 == 0:
                    ax = x0 + dx * s / steps
                    ay = y0 + dy * s / steps
                    bx = x0 + dx * (s + 1) / steps
                    by = y0 + dy * (s + 1) / steps
                    draw.line([(ax, ay), (bx, by)], fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width)


def _draw_poly_cairo(ctx, poly: Polygon, chip_transform,
                     color, width: float, dash: bool, scale: float):
    """Cairo-based polygon drawing — supports fractional line widths with anti-aliasing.
    Strokes twice to reinforce thin anti-aliased lines so they appear brighter.
    """
    coords = list(poly.exterior.coords)
    px     = _geo_to_px_float(coords, chip_transform, scale)
    pts    = px + [px[0]]
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    ctx.set_source_rgb(r, g, b)
    ctx.set_line_width(width * scale)
    if dash:
        ctx.set_dash([4.0 * scale, 4.0 * scale])
    else:
        ctx.set_dash([])
    # Double-stroke for brighter sub-pixel lines
    for _ in range(3):
        ctx.move_to(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            ctx.line_to(x, y)
        ctx.close_path()
        ctx.stroke()



def collect_pool(tif_roots, obb_gt_root, hbb_gt_root):
    hbb_scene_names = {d.name for d in hbb_gt_root.iterdir() if d.is_dir()}
    obb_scene_names = {d.name for d in obb_gt_root.iterdir() if d.is_dir()}
    matched = sorted([h for h in hbb_scene_names if (h + "_RGB") in obb_scene_names])

    pool = {b: [] for b in ANGLE_BINS}

    for hbb_name in matched:
        obb_name = hbb_name + "_RGB"
        obb_shp  = obb_gt_root / obb_name / "detections.shp"
        hbb_shp  = hbb_gt_root / hbb_name / "detections.shp"
        tif_path = _find_tif(obb_name, tif_roots)

        if not obb_shp.exists() or not hbb_shp.exists() or tif_path is None:
            continue

        gdf_obb    = gpd.read_file(str(obb_shp))
        gdf_hbb_gt = gpd.read_file(str(hbb_shp))
        if gdf_obb.empty or gdf_hbb_gt.empty:
            continue
        if gdf_hbb_gt.crs != gdf_obb.crs:
            gdf_hbb_gt = gdf_hbb_gt.to_crs(gdf_obb.crs)

        gt_geoms = list(gdf_hbb_gt.geometry)

        for geom in gdf_obb.geometry:
            if geom is None or geom.is_empty:
                continue
            poly     = geom if isinstance(geom, Polygon) else geom.convex_hull
            bin_name = _angle_bin(poly)
            if bin_name is None:
                continue

            novel   = novel_shape_aware(poly)
            outer   = outer_hbb(poly)
            area_eq = area_equiv_hbb(poly)
            gbb     = gbb_marginalized_hbb(poly, sigma_scale=1.0)
            gt_hbb  = _best_gt_hbb(novel, gt_geoms)
            if gt_hbb is None:
                continue

            _, _, _, _, theta = obb2hbb._mrr_params(poly)
            angle_deg = abs(theta) * 180 / math.pi % 90
            if angle_deg > 45:
                angle_deg = 90 - angle_deg

            pool[bin_name].append(dict(
                obb=poly, gt_hbb=gt_hbb,
                outer=outer, area_equiv=area_eq, gbb=gbb, novel=novel,
                cx_geo=poly.centroid.x, cy_geo=poly.centroid.y,
                tif_path=tif_path,
                scene=hbb_name,
                angle_deg=round(angle_deg, 1),
            ))

    return pool


def collect_pool_hbb(tif_roots: list, hbb_label_root: Path) -> dict:
    """
    Like collect_pool() but sources GT HBBs from YOLO HBB label .txt files
    (cx cy w h, normalized) instead of shapefiles.

    Each .txt file in hbb_label_root is matched by stem to a .tif found in
    any of tif_roots.  The HBB geo-polygon is used as both 'gt_hbb' and 'obb'
    in the candidate dict so that save_chips() works unchanged.  Because HBBs
    are axis-aligned, every candidate falls in the '0-15' angle bin.
    """
    from obbhbbstats import parse_hbb_yolo_line

    pool = {b: [] for b in ANGLE_BINS}

    for label_path in sorted(hbb_label_root.glob("*.txt")):
        tif_path = _find_tif(label_path.stem, tif_roots)
        if tif_path is None:
            continue

        with open(label_path) as f:
            lines = f.readlines()

        # Read TIF transform to convert normalized coords → geo coords
        with rasterio.open(str(tif_path)) as src:
            transform = src.transform
            img_w, img_h = src.width, src.height

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm  = float(parts[3])
                h_norm  = float(parts[4])
            except ValueError:
                continue
            if w_norm < 1e-8 or h_norm < 1e-8:
                continue

            # Denormalize to pixel space
            cx_px = cx_norm * img_w
            cy_px = cy_norm * img_h
            w_px  = w_norm  * img_w
            h_px  = h_norm  * img_h

            # Centroid in geo coords (for chip extraction)
            cx_geo, cy_geo = transform * (cx_px, cy_px)

            # Convert pixel-space box corners to geo coords for drawing
            corners_px = [
                (cx_px - w_px / 2, cy_px - h_px / 2),
                (cx_px + w_px / 2, cy_px - h_px / 2),
                (cx_px + w_px / 2, cy_px + h_px / 2),
                (cx_px - w_px / 2, cy_px + h_px / 2),
            ]
            corners_geo = [transform * pt for pt in corners_px]
            hbb_geo = Polygon(corners_geo)
            if not hbb_geo.is_valid or hbb_geo.area < 1e-8:
                continue

            novel   = novel_shape_aware(hbb_geo)
            outer   = outer_hbb(hbb_geo)
            area_eq = area_equiv_hbb(hbb_geo)
            gbb     = gbb_marginalized_hbb(hbb_geo, sigma_scale=1.0)

            pool["0-15"].append(dict(
                obb=hbb_geo, gt_hbb=hbb_geo,
                outer=outer, area_equiv=area_eq, gbb=gbb, novel=novel,
                cx_geo=cx_geo, cy_geo=cy_geo,
                tif_path=tif_path,
                scene=label_path.stem,
                angle_deg=0.0,
            ))

    return pool


def collect_pool_txt(tif_roots: list, obb_label_root: Path, hbb_label_root: Path) -> dict:
    """
    Like collect_pool() but reads both OBB and HBB labels from .txt files
    instead of shapefiles.

    OBB labels: 4-corner normalized (class x1 y1 x2 y2 x3 y3 x4 y4)
    HBB labels: YOLO format normalized (class cx cy w h)

    Files are paired by stem. The TIF geotransform is used to convert all
    normalized coords → geo coords so that save_chips() works unchanged.
    """
    pool = {b: [] for b in ANGLE_BINS}

    obb_files = {p.stem: p for p in sorted(obb_label_root.glob("*.txt"))}
    hbb_files = {p.stem: p for p in sorted(hbb_label_root.glob("*.txt"))}
    paired_stems = sorted(set(obb_files) & set(hbb_files))

    for stem in paired_stems:
        tif_path = _find_tif(stem, tif_roots)
        if tif_path is None:
            continue

        with rasterio.open(str(tif_path)) as src:
            transform = src.transform
            img_w, img_h = src.width, src.height

        # Load GT HBBs in geo coords from YOLO HBB label file
        gt_hbbs_geo = []
        with open(hbb_files[stem]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cx_px = float(parts[1]) * img_w
                    cy_px = float(parts[2]) * img_h
                    w_px  = float(parts[3]) * img_w
                    h_px  = float(parts[4]) * img_h
                except ValueError:
                    continue
                if w_px < 1e-8 or h_px < 1e-8:
                    continue
                corners_px = [
                    (cx_px - w_px / 2, cy_px - h_px / 2),
                    (cx_px + w_px / 2, cy_px - h_px / 2),
                    (cx_px + w_px / 2, cy_px + h_px / 2),
                    (cx_px - w_px / 2, cy_px + h_px / 2),
                ]
                corners_geo = [transform * pt for pt in corners_px]
                hbb_geo = Polygon(corners_geo)
                if hbb_geo.is_valid and hbb_geo.area >= 1e-8:
                    gt_hbbs_geo.append(hbb_geo)

        if not gt_hbbs_geo:
            continue

        # Load OBBs in geo coords from 4-corner normalized label file
        with open(obb_files[stem]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(x) for x in parts[1:9]]
                except ValueError:
                    continue

                # Denormalize corners to pixel space then to geo coords
                points_px  = [(coords[i] * img_w, coords[i+1] * img_h)
                              for i in range(0, 8, 2)]
                points_geo = [transform * pt for pt in points_px]

                try:
                    obb_geo = Polygon(points_geo)
                    if not obb_geo.is_valid:
                        obb_geo = obb_geo.buffer(0)
                    if obb_geo.area < 1e-8:
                        continue
                except Exception:
                    continue

                bin_name = _angle_bin(obb_geo)
                if bin_name is None:
                    continue

                novel   = novel_shape_aware(obb_geo)
                outer   = outer_hbb(obb_geo)
                area_eq = area_equiv_hbb(obb_geo)
                gbb     = gbb_marginalized_hbb(obb_geo, sigma_scale=1.0)
                gt_hbb  = _best_gt_hbb(novel, gt_hbbs_geo)
                if gt_hbb is None:
                    continue

                _, _, _, _, theta = obb2hbb._mrr_params(obb_geo)
                angle_deg = abs(theta) * 180 / math.pi % 90
                if angle_deg > 45:
                    angle_deg = 90 - angle_deg

                pool[bin_name].append(dict(
                    obb=obb_geo, gt_hbb=gt_hbb,
                    outer=outer, area_equiv=area_eq, gbb=gbb, novel=novel,
                    cx_geo=obb_geo.centroid.x, cy_geo=obb_geo.centroid.y,
                    tif_path=tif_path,
                    scene=stem,
                    angle_deg=round(angle_deg, 1),
                ))

    return pool


def collect_pool_txt_png(img_roots: list, obb_label_root: Path,
                         hbb_label_root: Path) -> dict:
    """
    Like collect_pool_txt() but for plain PNG/JPG images with no geospatial
    context. Labels are in normalized 0-1 coords and are converted to pixel
    space using the image dimensions. All polygons live in pixel space with
    a pixel-identity transform, so save_chips() works unchanged.

    OBB labels: 4-corner normalized (class x1 y1 x2 y2 x3 y3 x4 y4)
    HBB labels: YOLO format normalized (class cx cy w h)

    Files are paired by stem across obb_label_root and hbb_label_root.
    """
    pool = {b: [] for b in ANGLE_BINS}

    obb_files = {p.stem: p for p in sorted(obb_label_root.glob("*.txt"))}
    hbb_files = {p.stem: p for p in sorted(hbb_label_root.glob("*.txt"))}
    paired_stems = sorted(set(obb_files) & set(hbb_files))

    for stem in paired_stems:
        img_path = _find_image(stem, img_roots)
        if img_path is None:
            continue

        # Get image dimensions via PIL — no rasterio needed
        img_pil  = Image.open(str(img_path))
        img_w, img_h = img_pil.size
        img_pil.close()

        transform = _pixel_identity_transform()

        # Load GT HBBs in pixel space from YOLO HBB label file
        gt_hbbs_px = []
        with open(hbb_files[stem]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cx_px = float(parts[1]) * img_w
                    cy_px = float(parts[2]) * img_h
                    w_px  = float(parts[3]) * img_w
                    h_px  = float(parts[4]) * img_h
                except ValueError:
                    continue
                if w_px < 1e-8 or h_px < 1e-8:
                    continue
                corners = [
                    (cx_px - w_px / 2, cy_px - h_px / 2),
                    (cx_px + w_px / 2, cy_px - h_px / 2),
                    (cx_px + w_px / 2, cy_px + h_px / 2),
                    (cx_px - w_px / 2, cy_px + h_px / 2),
                ]
                hbb_px = Polygon(corners)
                if hbb_px.is_valid and hbb_px.area >= 1e-8:
                    gt_hbbs_px.append(hbb_px)

        if not gt_hbbs_px:
            continue

        # Load OBBs in pixel space from 4-corner normalized label file
        with open(obb_files[stem]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(x) for x in parts[1:9]]
                except ValueError:
                    continue

                points_px = [(coords[i] * img_w, coords[i+1] * img_h)
                             for i in range(0, 8, 2)]

                try:
                    obb_px = Polygon(points_px)
                    if not obb_px.is_valid:
                        obb_px = obb_px.buffer(0)
                    if obb_px.area < 1e-8:
                        continue
                except Exception:
                    continue

                bin_name = _angle_bin(obb_px)
                if bin_name is None:
                    continue

                novel   = novel_shape_aware(obb_px)
                outer   = outer_hbb(obb_px)
                area_eq = area_equiv_hbb(obb_px)
                gbb     = gbb_marginalized_hbb(obb_px, sigma_scale=1.0)
                gt_hbb  = _best_gt_hbb(novel, gt_hbbs_px)
                if gt_hbb is None:
                    continue

                _, _, _, _, theta = obb2hbb._mrr_params(obb_px)
                angle_deg = abs(theta) * 180 / math.pi % 90
                if angle_deg > 45:
                    angle_deg = 90 - angle_deg

                pool[bin_name].append(dict(
                    obb=obb_px, gt_hbb=gt_hbb,
                    outer=outer, area_equiv=area_eq, gbb=gbb, novel=novel,
                    cx_geo=obb_px.centroid.x, cy_geo=obb_px.centroid.y,
                    tif_path=img_path,      # save_chips uses this key for any image
                    scene=stem,
                    angle_deg=round(angle_deg, 1),
                ))

    return pool

def _tif_meta(tif_path: Path) -> tuple:
    """Read only transform + dimensions from a TIF — no pixel data."""
    with rasterio.open(str(tif_path)) as src:
        return src.transform, src.width, src.height


def _in_bounds(transform, width: int, height: int, cx_geo: float, cy_geo: float) -> bool:
    inv = ~transform
    col_f, row_f = inv * (cx_geo, cy_geo)
    col_off = int(col_f) - CHIP_HALF
    row_off = int(row_f) - CHIP_HALF
    w = h = 2 * CHIP_HALF
    return col_off >= 0 and row_off >= 0 and col_off + w <= width and row_off + h <= height


def save_chips(pool, output_dir: Path, all_methods: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    active_methods = list(METHOD_STYLES.keys()) if all_methods else DEFAULT_METHODS
    use_cairo = any(METHOD_STYLES[m]["width"] < 1 for m in active_methods)

    # ── Step 1: read only TIF metadata (no pixels) to check chip bounds ──────
    from collections import defaultdict
    tif_meta_cache: dict = {}
    for bin_name in ANGLE_BINS:
        for cand in pool[bin_name]:
            tp = cand["tif_path"]
            if tp not in tif_meta_cache:
                tif_meta_cache[tp] = _tif_meta(tp)

    # ── Step 2: select all in-bounds candidates ─────────────────────────────
    print("  Selecting chips (bounds check only, no pixel reads)...")
    selected: list[tuple[str, int, dict]] = []   # (bin_name, idx_within_bin, cand)
    needed_tifs: set = set()
    for bin_name in ANGLE_BINS:
        idx = 0
        for cand in pool[bin_name]:
            transform, w_img, h_img = tif_meta_cache[cand["tif_path"]]
            if not _in_bounds(transform, w_img, h_img, cand["cx_geo"], cand["cy_geo"]):
                continue
            selected.append((bin_name, idx, cand))
            needed_tifs.add(cand["tif_path"])
            idx += 1
        print(f"  Bin {bin_name}°: {idx} chips selected")

    # ── Step 3: load + normalise only the required TIFs ──────────────────────
    print(f"  Loading and normalising {len(needed_tifs)} required TIF(s)...")
    tif_images: dict = {}
    for tif_path in needed_tifs:
        print(f"    {tif_path.name}")
        tif_images[tif_path] = _load_normalised_image(tif_path)

    # ── Step 4: slice chips and save ─────────────────────────────────────────
    for bin_name, pick_idx, cand in selected:
        rgb_full, transform, w_img, h_img = tif_images[cand["tif_path"]]
        rgb, chip_transform = _chip_from_image(
            rgb_full, transform, w_img, h_img, cand["cx_geo"], cand["cy_geo"]
        )

        S    = SCALE
        H, W = rgb.shape[:2]

        poly_map = {
            "OBB":              cand["obb"],
            "GT_HBB":           cand["gt_hbb"],
            "Outer_HBB":        cand["outer"],
            "Area_Equiv_HBB":   cand["area_equiv"],
            "GBB_Marginalized": cand["gbb"],
            "Novel_ShapeAware": cand["novel"],
        }

        if use_cairo:
            # Cairo path: render onto ARGB surface, composite over the chip
            out_w, out_h = W * S, H * S
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, out_w, out_h)
            ctx     = cairo.Context(surface)
            # Paint the chip as background
            chip_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, out_w, out_h)
            if S == 1:
                scaled_rgb = rgb
            else:
                scaled_rgb = np.array(
                    Image.fromarray(rgb, "RGB").resize((out_w, out_h), Image.NEAREST)
                )
            # Convert RGB → BGRA for Cairo
            bgra = np.zeros((out_h, out_w, 4), dtype=np.uint8)
            bgra[:, :, 0] = scaled_rgb[:, :, 2]  # B
            bgra[:, :, 1] = scaled_rgb[:, :, 1]  # G
            bgra[:, :, 2] = scaled_rgb[:, :, 0]  # R
            bgra[:, :, 3] = 255                   # A
            chip_surface = cairo.ImageSurface.create_for_data(
                bytearray(bgra.tobytes()), cairo.FORMAT_ARGB32, out_w, out_h, out_w * 4
            )
            ctx.set_source_surface(chip_surface, 0, 0)
            ctx.paint()

            for name in active_methods:
                style = METHOD_STYLES[name]
                _draw_poly_cairo(ctx, poly_map[name], chip_transform,
                                 color=style["color"],
                                 width=style["width"],
                                 dash=style["dash"],
                                 scale=float(S))

            # Extract result as PIL Image
            buf  = surface.get_data()
            arr  = np.frombuffer(buf, dtype=np.uint8).reshape((out_h, out_w, 4)).copy()
            # BGRA → RGB
            base = Image.fromarray(
                np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1), "RGB"
            )
        else:
            # PIL path: integer widths >= 1
            base = Image.fromarray(rgb, "RGB").resize((W * S, H * S), Image.NEAREST)
            draw = ImageDraw.Draw(base)
            for name in active_methods:
                style = METHOD_STYLES[name]
                _draw_poly(draw, poly_map[name], chip_transform,
                           color=style["color"],
                           width=max(1, int(style["width"] * S)),
                           dash=style["dash"],
                           scale=S)

        scene_short = cand["scene"].split("_")[0] + "_" + cand["scene"][-8:-5]
        fname    = f"chip_{bin_name.replace('-','_')}_{pick_idx+1}_{scene_short}_{cand['angle_deg']}deg.png"
        out_path = output_dir / fname
        base.save(str(out_path))
        print(f"  Saved {out_path}  (bin={bin_name}  angle={cand['angle_deg']}°  scene={cand['scene'][:40]})")
        saved.append(out_path)

    return saved


def main():
    ap = argparse.ArgumentParser(description="Save HBB method chip PNGs")
    ap.add_argument("--tif_roots",      nargs="+",
                    help="Folders of .tif raster files (shapefile / txt-label TIF modes)")
    ap.add_argument("--img_roots",      nargs="+",
                    help="Folders of .png/.jpg image files (txt-label PNG/JPG mode)")
    ap.add_argument("--obb_gt_root",    default="Data/OBB_GT",
                    help="Root of shapefile-based OBB GT dirs (shapefile mode)")
    ap.add_argument("--hbb_gt_root",    default="Data/HBB_GT",
                    help="Root of shapefile-based HBB GT dirs (shapefile mode)")
    ap.add_argument("--obb_label_root",
                    help="Folder of OBB .txt label files (4-corner normalized).")
    ap.add_argument("--hbb_label_root",
                    help="Folder of YOLO HBB .txt label files (cx cy w h, normalized).")
    ap.add_argument("--output_dir",     default="chips")
    ap.add_argument("--all_methods",    action="store_true",
                    help="Draw all 6 overlays; default draws only OBB, GT_HBB, Novel_ShapeAware")
    args = ap.parse_args()

    print("Collecting candidates...")

    if args.img_roots and args.obb_label_root and args.hbb_label_root:
        # PNG/JPG mode — pixel-space coords, no rasterio
        img_roots = [Path(r) for r in args.img_roots]
        pool = collect_pool_txt_png(img_roots,
                                    Path(args.obb_label_root),
                                    Path(args.hbb_label_root))
    elif args.tif_roots and args.obb_label_root and args.hbb_label_root:
        # TIF + txt label mode — geo coords via rasterio transform
        tif_roots = [Path(r) for r in args.tif_roots]
        pool = collect_pool_txt(tif_roots,
                                Path(args.obb_label_root),
                                Path(args.hbb_label_root))
    elif args.tif_roots and args.hbb_label_root:
        # HBB-only txt label mode
        tif_roots = [Path(r) for r in args.tif_roots]
        pool = collect_pool_hbb(tif_roots, Path(args.hbb_label_root))
    elif args.tif_roots:
        # Original shapefile mode
        tif_roots = [Path(r) for r in args.tif_roots]
        pool = collect_pool(tif_roots, Path(args.obb_gt_root), Path(args.hbb_gt_root))
    else:
        ap.error("Provide --tif_roots (TIF/shapefile modes) or --img_roots (PNG/JPG mode)")

    for b in ANGLE_BINS:
        print(f"  Bin {b}°: {len(pool[b])} candidates")

    methods_label = "all" if args.all_methods else "default (OBB, GT_HBB, Novel_ShapeAware)"
    print(f"\nSaving chips to {args.output_dir}/  [overlays: {methods_label}]")
    saved = save_chips(pool, Path(args.output_dir), all_methods=args.all_methods)
    print(f"\nDone — {len(saved)} chips written.")


if __name__ == "__main__":
    main()