#!/usr/bin/env python3
"""
obb2hbb.py  –  Convert OBB polygons (shapefile / GeoJSON / CSV) to HBB
using the shape-aware method with best-found defaults.

Best defaults (from tuning on Sentinel-2 ship dataset):
  q=1.5  fullness=0.8  shrink=0.95  (no physical size cap)

Usage examples
--------------
  # shapefile input  →  shapefile output
  python3 obb2hbb.py --input detections.shp --output hbb_out.shp

  # GeoJSON input  →  GeoJSON output
  python3 obb2hbb.py --input detections.geojson --output hbb_out.geojson

  # CSV with wkt column  →  CSV output
  python3 obb2hbb.py --input detections.csv --wkt_col geometry --output hbb_out.csv

  # override params
  python3 obb2hbb.py --input detections.shp --output out.shp \
      --shape_q 1.4 --shape_fullness 0.9 --shrink 0.95
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
from typing import Optional, Tuple

# ── shape-aware core ──────────────────────────────────────────────────────────
def _mrr_params(geom) -> Tuple[float, float, float, float, float]:
    from shapely.geometry import box as shapely_box
    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]
    if len(coords) != 4:
        b = geom.bounds
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        return cx, cy, b[2]-b[0], b[3]-b[1], 0.0
    edges = []
    for i in range(4):
        x1,y1 = coords[i]; x2,y2 = coords[(i+1)%4]
        dx,dy = x2-x1, y2-y1
        edges.append((math.hypot(dx,dy), dx, dy))
    edges.sort(key=lambda t: t[0], reverse=True)
    length = edges[0][0]; width = edges[2][0] if len(edges)>2 else edges[1][0]
    angle  = math.atan2(edges[0][1], edges[0][2])
    return mrr.centroid.x, mrr.centroid.y, max(length,width), min(length,width), angle


def obb_to_hbb_shapeaware(
    geom,
    shape_q: float       = 1.5,
    shape_fullness: float = 0.8,
    shrink: float         = 0.95,
    max_len_m: float      = 1e9,
    max_wid_m: float      = 1e9,
    meters_per_unit: float= 1.0,
):
    """Convert a Shapely geometry to a shape-aware HBB (axis-aligned box)."""
    from shapely.geometry import box as shapely_box
    if geom is None or geom.is_empty:
        return geom, {}
    cx, cy, L, W, theta = _mrr_params(geom)
    if L <= 0 or W <= 0:
        return shapely_box(cx,cy,cx,cy), {}
    a = L/2.0
    b = (W/2.0) * shape_fullness
    q = max(float(shape_q), 1.000001)
    t = abs(theta)
    s, c = abs(math.sin(t)), abs(math.cos(t))
    if t < 1e-12:
        tx, ty = 0.0, 1.0
    elif abs(math.pi/2 - t) < 1e-12:
        tx, ty = 1.0, 0.0
    else:
        ks = max(q*b/a, 1e-12)
        tx = min(1.0, max(0.0, ((s/max(c,1e-12))/ks)**(1.0/(q-1.0))))
        ty = min(1.0, max(0.0, ((c/max(s,1e-12))/ks)**(1.0/(q-1.0))))
    xh = (a*tx*s + b*(1.0-tx**q)*c) * shrink
    yh = (a*ty*c + b*(1.0-ty**q)*s) * shrink
    max_l = max_len_m/meters_per_unit/2.0
    max_w = max_wid_m/meters_per_unit/2.0
    cap = False
    if xh >= yh:
        if xh>max_l: xh=max_l; cap=True
        if yh>max_w: yh=max_w; cap=True
    else:
        if xh>max_w: xh=max_w; cap=True
        if yh>max_l: yh=max_l; cap=True
    meta = dict(L=L, W=W, angle_deg=abs(theta)*180/math.pi,
                tx=tx, ty=ty, x_half=xh, y_half=yh, cap_hit=cap)
    return shapely_box(cx-xh, cy-yh, cx+xh, cy+yh), meta

# ── I/O helpers ───────────────────────────────────────────────────────────────
def _meters_per_unit(crs) -> float:
    try:
        from pyproj import CRS as PCRS
        c = PCRS.from_user_input(crs)
        if c.is_geographic: return 111320.0
        f = c.axis_info[0].unit_conversion_factor
        return float(f) if f else 1.0
    except Exception:
        return 1.0

def run_shp(src_path: Path, dst_path: Path, args):
    import fiona, json
    from fiona.crs import from_epsg
    from shapely.geometry import shape, mapping
    with fiona.open(str(src_path)) as src:
        mpu = _meters_per_unit(src.crs)
        schema = dict(src.schema)
        schema['geometry'] = 'Polygon'
        extra_fields = ['sa_L','sa_W','sa_angle','sa_tx','sa_ty','sa_xhalf','sa_yhalf','sa_cap']
        for ef in extra_fields:
            schema['properties'][ef] = 'float'
        with fiona.open(str(dst_path), 'w', driver='ESRI Shapefile',
                        crs=src.crs, schema=schema) as dst:
            for feat in src:
                geom_raw = feat.get('geometry')
                if not geom_raw:
                    continue
                geom = shape(geom_raw)
                hbb, meta = obb_to_hbb_shapeaware(
                    geom, args.shape_q, args.shape_fullness, args.shrink,
                    args.max_ship_length_m, args.max_ship_width_m, mpu)
                props = dict(feat['properties'])
                props['sa_L']     = round(meta.get('L',0.0),6)
                props['sa_W']     = round(meta.get('W',0.0),6)
                props['sa_angle'] = round(meta.get('angle_deg',0.0),6)
                props['sa_tx']    = round(meta.get('tx',0.0),6)
                props['sa_ty']    = round(meta.get('ty',0.0),6)
                props['sa_xhalf'] = round(meta.get('x_half',0.0),6)
                props['sa_yhalf'] = round(meta.get('y_half',0.0),6)
                props['sa_cap']   = float(meta.get('cap_hit',False))
                dst.write({'geometry': mapping(hbb), 'properties': props})
    print(f"Wrote {dst_path}")

def run_geojson(src_path: Path, dst_path: Path, args):
    from shapely.geometry import shape, mapping
    with open(src_path) as f:
        fc = json.load(f)
    out_features = []
    for feat in fc.get('features',[]):
        geom = shape(feat['geometry'])
        hbb, meta = obb_to_hbb_shapeaware(
            geom, args.shape_q, args.shape_fullness, args.shrink,
            args.max_ship_length_m, args.max_ship_width_m)
        props = dict(feat.get('properties') or {})
        props.update({k: round(v,6) if isinstance(v,float) else v for k,v in meta.items()})
        out_features.append({'type':'Feature','geometry':mapping(hbb),'properties':props})
    with open(dst_path,'w') as f:
        json.dump({'type':'FeatureCollection','features':out_features}, f, indent=2)
    print(f"Wrote {dst_path}")

def run_csv(src_path: Path, dst_path: Path, wkt_col: str, args):
    from shapely.wkt import loads, dumps
    with open(src_path, newline='') as f:
        reader = csv.DictReader(f)
        rows_out = []
        for row in reader:
            wkt = row.get(wkt_col,'')
            try:
                geom = loads(wkt)
                hbb, meta = obb_to_hbb_shapeaware(
                    geom, args.shape_q, args.shape_fullness, args.shrink,
                    args.max_ship_length_m, args.max_ship_width_m)
                row[wkt_col] = dumps(hbb)
                for k,v in meta.items():
                    row[f'sa_{k}'] = round(v,6) if isinstance(v,float) else v
            except Exception as e:
                row['sa_error'] = str(e)
            rows_out.append(row)
    with open(dst_path,'w',newline='') as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
    print(f"Wrote {dst_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='OBB → HBB (shape-aware, best defaults)')
    ap.add_argument('--input',  required=True,  help='.shp / .geojson / .csv')
    ap.add_argument('--output', required=True,  help='Output path (same type)')
    ap.add_argument('--wkt_col', default='geometry', help='WKT column name (CSV only)')
    ap.add_argument('--shape_q',        type=float, default=1.5)
    ap.add_argument('--shape_fullness', type=float, default=0.8)
    ap.add_argument('--shrink',         type=float, default=0.95)
    ap.add_argument('--max_ship_length_m', type=float, default=1e9)
    ap.add_argument('--max_ship_width_m',  type=float, default=1e9)
    args = ap.parse_args()
    src = Path(args.input); dst = Path(args.output)
    ext = src.suffix.lower()
    if ext == '.shp':       run_shp(src, dst, args)
    elif ext == '.geojson': run_geojson(src, dst, args)
    elif ext == '.csv':     run_csv(src, dst, args.wkt_col, args)
    else:
        print(f"Unsupported format: {ext}. Use .shp / .geojson / .csv"); sys.exit(1)

if __name__ == '__main__':
    main()
