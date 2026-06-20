import sys
import math
import glob
import argparse
from pathlib import Path
from shapely.geometry import Polygon, box
import importlib.util
import json
import warnings

# Import the user's obb2hbb script dynamically
sys.path.append("/run/media/brr/PrimeStore/SkyServe/OBB_HBB")
import obb2hbb

def outer_hbb(geom: Polygon) -> Polygon:
    """Standard Min-Max Bounding Box"""
    return box(*geom.bounds)

def area_equiv_hbb(geom: Polygon) -> Polygon:
    """HBB centered on centroid with dimensions equal to L and W of MRR, but aligned to axes"""
    cx, cy, L, W, _ = obb2hbb._mrr_params(geom)
    return box(cx - L/2, cy - W/2, cx + L/2, cy + W/2)

def gbb_marginalized_hbb(geom: Polygon, sigma_scale: float = 2.0) -> Polygon:
    """
    Gaussian Bounding Box marginalized.
    Treats OBB as uniform distrib. W = L, H = W.
    Var_x = (L^2 cos^2(theta) + W^2 sin^2(theta)) / 12
    Var_y = (L^2 sin^2(theta) + W^2 cos^2(theta)) / 12
    sigma_x = sqrt(Var_x), sigma_y = sqrt(Var_y)
    Width_HBB = sigma_scale * sqrt(12) * sigma_x
    """
    cx, cy, L, W, theta = obb2hbb._mrr_params(geom)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    var_x = (L**2 * cos_t**2 + W**2 * sin_t**2) / 12.0
    var_y = (L**2 * sin_t**2 + W**2 * cos_t**2) / 12.0

    # 2*sqrt(3) * sigma would be full width if angle is 0.
    # sigma_scale=1.0 gives a box bounded by the variances exactly.
    hw = sigma_scale * math.sqrt(3 * var_x)
    hh = sigma_scale * math.sqrt(3 * var_y)
    return box(cx - hw, cy - hh, cx + hw, cy + hh)

def novel_shape_aware(geom: Polygon, shape_q=1.5, shape_fullness=0.8, shrink=0.95) -> Polygon:
    """User's Novel Method"""
    hbb, _ = obb2hbb.obb_to_hbb_shapeaware(geom, shape_q=shape_q, shape_fullness=shape_fullness, shrink=shrink)
    return hbb


def novel_shape_aware_gt(
    hbb_gt_root: str,
    obb_gt_root: str,
    shape_q: float = 2.2,
    shape_fullness: float = 0.9,
    shrink: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Benchmark all 4 HBB methods (Outer_HBB, Area_Equiv_HBB, GBB_Marginalized,
    Novel_ShapeAware) against ground-truth HBB shapefiles for all matched scenes.

    Folder matching rule:
        HBB_GT/<scene_name>.SAFE  <-->  OBB_GT/<scene_name>.SAFE_RGB

    For each matched pair and each method:
      - Reads OBB polygons from OBB_GT/<scene>/detections.shp
      - Converts each OBB -> candidate HBB via the method
      - Reads GT HBBs from HBB_GT/<scene>/detections.shp
      - Spatially matches each candidate HBB to best-IoU GT HBB
      - Accumulates IoU, overshoot, undershoot per method per scene and overall

    Returns a dict keyed by method name, each with 'overall' and 'per_scene'.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required: pip install geopandas")

    METHOD_NAMES = ["Outer_HBB", "Area_Equiv_HBB", "GBB_Marginalized", "Novel_ShapeAware"]

    ANGLE_BINS = ["0-15", "15-30", "30-45"]

    def _make_acc():
        return {
            "iou_sum": 0.0, "over_sum": 0.0, "under_sum": 0.0, "count": 0,
            "angle_bins": {b: {"iou_sum": 0.0, "count": 0} for b in ANGLE_BINS},
        }

    def _acc_stats(acc):
        c = acc["count"]
        angle_miou = {}
        for b in ANGLE_BINS:
            bc = acc["angle_bins"][b]["count"]
            angle_miou[b] = round(acc["angle_bins"][b]["iou_sum"] / bc, 4) if bc > 0 else None
        return {
            "Mean_IoU":          round(acc["iou_sum"]   / c, 4) if c > 0 else None,
            "Overshoot":         round(acc["over_sum"]  / c, 4) if c > 0 else None,
            "Undershoot":        round(acc["under_sum"] / c, 4) if c > 0 else None,
            "Angle_mIoU_0_15":   angle_miou["0-15"],
            "Angle_mIoU_15_30":  angle_miou["15-30"],
            "Angle_mIoU_30_45":  angle_miou["30-45"],
            "Samples":           c,
        }

    def _best_iou_match(pred_hbb, gt_geoms):
        best_iou, best_gt = -1.0, None
        for gt_geom in gt_geoms:
            if gt_geom is None or gt_geom.is_empty:
                continue
            try:
                inter = pred_hbb.intersection(gt_geom).area
                union = pred_hbb.union(gt_geom).area
                iou   = inter / union if union > 0 else 0.0
            except Exception:
                continue
            if iou > best_iou:
                best_iou, best_gt = iou, gt_geom
        return best_iou, best_gt

    hbb_root = Path(hbb_gt_root)
    obb_root = Path(obb_gt_root)

    hbb_scene_names = {d.name for d in hbb_root.iterdir() if d.is_dir()}
    obb_scene_names = {d.name for d in obb_root.iterdir() if d.is_dir()}

    matched_pairs = []
    for hbb_name in sorted(hbb_scene_names):
        obb_candidate = hbb_name + "_RGB"
        if obb_candidate in obb_scene_names:
            matched_pairs.append((hbb_name, obb_candidate))

    if not matched_pairs:
        warnings.warn("No matched scene folders found between HBB_GT and OBB_GT.")
        return {}

    if verbose:
        print(f"Found {len(matched_pairs)} matched scene pair(s).")

    aggregates  = {m: _make_acc() for m in METHOD_NAMES}
    per_scene   = {m: {}          for m in METHOD_NAMES}

    for hbb_name, obb_name in matched_pairs:
        hbb_shp = hbb_root / hbb_name / "detections.shp"
        obb_shp = obb_root / obb_name / "detections.shp"

        if not hbb_shp.exists() or not obb_shp.exists():
            warnings.warn(f"Missing shapefile for scene {hbb_name}, skipping.")
            continue

        gdf_hbb_gt = gpd.read_file(str(hbb_shp))
        gdf_obb    = gpd.read_file(str(obb_shp))

        if gdf_hbb_gt.empty or gdf_obb.empty:
            warnings.warn(f"Empty shapefile for scene {hbb_name}, skipping.")
            continue

        # Reproject GT HBBs to OBB CRS (projected, metric) for area comparisons
        target_crs = gdf_obb.crs
        if gdf_hbb_gt.crs != target_crs:
            gdf_hbb_gt = gdf_hbb_gt.to_crs(target_crs)

        gt_geoms = list(gdf_hbb_gt.geometry)

        # Build per-method candidate HBBs for every OBB in this scene
        scene_polys = []
        for geom in gdf_obb.geometry:
            if geom is None or geom.is_empty:
                scene_polys.append(None)
                continue
            poly = geom if isinstance(geom, Polygon) else geom.convex_hull
            scene_polys.append(poly)

        scene_accs = {m: _make_acc() for m in METHOD_NAMES}

        for poly in scene_polys:
            if poly is None:
                continue

            # Compute angle bin from OBB MRR (same logic as process_file)
            try:
                _, _, _, _, theta = obb2hbb._mrr_params(poly)
                angle_deg = abs(theta) * 180 / math.pi % 90
                if angle_deg > 45:
                    angle_deg = 90 - angle_deg
                if angle_deg < 15:
                    bin_name = "0-15"
                elif angle_deg < 30:
                    bin_name = "15-30"
                else:
                    bin_name = "30-45"
            except Exception:
                bin_name = None

            candidates = {
                "Outer_HBB":       outer_hbb(poly),
                "Area_Equiv_HBB":  area_equiv_hbb(poly),
                "GBB_Marginalized": gbb_marginalized_hbb(poly, sigma_scale=1.0),
                "Novel_ShapeAware": novel_shape_aware(poly, shape_q, shape_fullness, shrink),
            }

            for method_name, pred_hbb in candidates.items():
                if pred_hbb is None or pred_hbb.is_empty:
                    continue

                best_iou, best_gt = _best_iou_match(pred_hbb, gt_geoms)
                if best_gt is None:
                    continue

                try:
                    inter      = pred_hbb.intersection(best_gt).area
                    pred_area  = pred_hbb.area
                    gt_area    = best_gt.area
                    overshoot  = (pred_area - inter) / pred_area if pred_area > 0 else 0.0
                    undershoot = (gt_area   - inter) / gt_area   if gt_area   > 0 else 0.0
                except Exception:
                    continue

                for acc in (scene_accs[method_name], aggregates[method_name]):
                    acc["iou_sum"]   += best_iou
                    acc["over_sum"]  += overshoot
                    acc["under_sum"] += undershoot
                    acc["count"]     += 1
                    if bin_name is not None:
                        acc["angle_bins"][bin_name]["iou_sum"] += best_iou
                        acc["angle_bins"][bin_name]["count"]   += 1

        if verbose:
            row = "  [HBB: {} <-> OBB: {}]".format(hbb_name, obb_name)
            for m in METHOD_NAMES:
                s = _acc_stats(scene_accs[m])
                row += f"  {m}: IoU={s['Mean_IoU']} over={s['Overshoot']} under={s['Undershoot']} n={s['Samples']}"
            print(row)

        for m in METHOD_NAMES:
            per_scene[m][hbb_name] = _acc_stats(scene_accs[m])

    output = {}
    if verbose:
        print(f"\n--- GT Shapefile Benchmark Overall ({len(matched_pairs)} scenes) ---")
    for m in METHOD_NAMES:
        overall = _acc_stats(aggregates[m])
        output[m] = {"overall": overall, "per_scene": per_scene[m]}
        if verbose:
            print(f"  {m}: Mean_IoU={overall['Mean_IoU']}  "
                  f"Overshoot={overall['Overshoot']}  "
                  f"Undershoot={overall['Undershoot']}  "
                  f"Angle_mIoU_0_15={overall['Angle_mIoU_0_15']}  "
                  f"Angle_mIoU_15_30={overall['Angle_mIoU_15_30']}  "
                  f"Angle_mIoU_30_45={overall['Angle_mIoU_30_45']}  "
                  f"Samples={overall['Samples']}")

    return output


def calculate_metrics(obb_poly: Polygon, hbb_poly: Polygon):
    if not hbb_poly.is_valid:
        hbb_poly = hbb_poly.buffer(0)

    inter = obb_poly.intersection(hbb_poly).area
    union = obb_poly.union(hbb_poly).area
    iou = inter / union if union > 0 else 0

    obb_area = obb_poly.area
    hbb_area = hbb_poly.area

    overshoot = (hbb_area - inter) / hbb_area if hbb_area > 0 else 0
    undershoot = (obb_area - inter) / obb_area if obb_area > 0 else 0

    return iou, overshoot, undershoot

def process_file(filepath: str, results: dict):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue

        # Parse points (x1, y1, x2, y2, x3, y3, x4, y4)
        c_id = int(parts[0])
        coords = [float(x) for x in parts[1:9]]
        points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]

        try:
            obb = Polygon(points)
            if not obb.is_valid:
                obb = obb.buffer(0)
            if obb.area < 1e-8:
                continue
        except Exception:
            continue

        # Extract orientation for stratification
        _, _, L, W, theta = obb2hbb._mrr_params(obb)
        angle_deg = abs(theta) * 180 / math.pi
        # Normalize angle to 0-90 (due to symmetry)
        angle_deg = angle_deg % 90
        if angle_deg > 45:
            angle_deg = 90 - angle_deg

        aspect_ratio = L / W if W > 0 else 1.0

        algorithms = {
            "Outer_HBB": outer_hbb(obb),
            "Area_Equiv_HBB": area_equiv_hbb(obb),
            "GBB_Marginalized": gbb_marginalized_hbb(obb, sigma_scale=1.0),
            "Novel_ShapeAware": novel_shape_aware(obb)
        }

        for name, hbb in algorithms.items():
            iou, over, under = calculate_metrics(obb, hbb)
            results[name]["count"] += 1
            results[name]["iou_sum"] += iou
            results[name]["over_sum"] += over
            results[name]["under_sum"] += under

            # Stratify by angle (0-15, 15-30, 30-45)
            if angle_deg < 15:
                bin_name = "0-15"
            elif angle_deg < 30:
                bin_name = "15-30"
            else:
                bin_name = "30-45"

            results[name]["angle_bins"][bin_name]["iou_sum"] += iou
            results[name]["angle_bins"][bin_name]["count"] += 1

def main():
    ap = argparse.ArgumentParser(description="OBB→HBB statistics benchmark")
    ap.add_argument(
        "--mode",
        choices=["novel_shape_aware", "novel_shape_aware_gt"],
        default="novel_shape_aware",
        help=(
            "novel_shape_aware : benchmark all methods against OBB labels (txt files)  "
            "novel_shape_aware_gt : compare shape-aware HBBs against GT HBB shapefiles"
        ),
    )
    ap.add_argument("--dataset_path",
                    help="Glob pattern for label .txt files (novel_shape_aware mode)")
    ap.add_argument("--hbb_gt_root",
                    help="Root folder of HBB ground-truth shapefiles (novel_shape_aware_gt mode)")
    ap.add_argument("--obb_gt_root",
                    help="Root folder of OBB ground-truth shapefiles (novel_shape_aware_gt mode)")
    ap.add_argument("--output",
                    help="Path to write JSON results")
    ap.add_argument("--shape_q",        type=float, default=2.2)
    ap.add_argument("--shape_fullness", type=float, default=0.9)
    ap.add_argument("--shrink",         type=float, default=1.0)
    args = ap.parse_args()

    if args.mode == "novel_shape_aware_gt":
        results = novel_shape_aware_gt(
            hbb_gt_root=args.hbb_gt_root,
            obb_gt_root=args.obb_gt_root,
            shape_q=args.shape_q,
            shape_fullness=args.shape_fullness,
            shrink=args.shrink,
            verbose=True,
        )
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults written to {args.output}")

    else:
        files = glob.glob(args.dataset_path)

        results = {
            name: {
                "count": 0, "iou_sum": 0, "over_sum": 0, "under_sum": 0,
                "angle_bins": {
                    "0-15": {"iou_sum": 0, "count": 0},
                    "15-30": {"iou_sum": 0, "count": 0},
                    "30-45": {"iou_sum": 0, "count": 0}
                }
            }
            for name in ["Outer_HBB", "Area_Equiv_HBB", "GBB_Marginalized", "Novel_ShapeAware"]
        }

        print(f"Processing {len(files)} files...")

        for i, file in enumerate(files):
            if i % 500 == 0:
                print(f"  Processed {i}/{len(files)} files")
            process_file(file, results)

        print("\n--- Benchmark Results ---")

        final_stats = {}
        for name, data in results.items():
            c = data["count"]
            if c == 0: continue
            miou = data["iou_sum"] / c
            mover = data["over_sum"] / c
            munder = data["under_sum"] / c

            angle_miou = {}
            for b_name, b_data in data["angle_bins"].items():
                if b_data["count"] > 0:
                    angle_miou[b_name] = b_data["iou_sum"] / b_data["count"]
                else:
                    angle_miou[b_name] = 0

            final_stats[name] = {
                "Mean_IoU": round(miou, 4),
                "Overshoot": round(mover, 4),
                "Undershoot": round(munder, 4),
                "Angle_mIoU_0_15": round(angle_miou["0-15"], 4),
                "Angle_mIoU_15_30": round(angle_miou["15-30"], 4),
                "Angle_mIoU_30_45": round(angle_miou["30-45"], 4),
                "Samples": c
            }

        for name, stats in final_stats.items():
            print(f"\nMethod: {name}")
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        with open(args.output, "w") as f:
            json.dump(final_stats, f, indent=4)
        print(f"\nResults written to {args.output}")

if __name__ == "__main__":
    main()
