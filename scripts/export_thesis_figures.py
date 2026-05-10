#!/usr/bin/env python3
"""Coordinate-driven thesis figure exporter (tile/16:9/1x4/2x2).

Primary mode: scenario YAML.
Fallback mode: single coordinate from CLI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

try:
    from scripts.label_tiles import CNNPredictor, _apply_sld, _compute_heatmap
except ModuleNotFoundError:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from scripts.label_tiles import CNNPredictor, _apply_sld, _compute_heatmap

try:
    from src.cdw_detect.wms_utils import build_wms_layer_name, fetch_wms_for_bbox
except ModuleNotFoundError:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    _src = _repo_root / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from cdw_detect.wms_utils import build_wms_layer_name, fetch_wms_for_bbox

ORTHO_ATTRIBUTION = "Ortofoto Maa- ja Ruumiamet"


@dataclass
class RenderView:
    key: str
    view_type: str  # chm|orthophoto|intgrad|probability|prediction|hotspot
    title: str
    raster_path: Path | None = None


@dataclass
class RenderScene:
    scene_id: str
    x: float
    y: float
    crs: str
    tile_size: int
    output_dir: Path
    views: list[RenderView]
    layouts: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --scenario mode (pip/conda install pyyaml)") from exc
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Scenario YAML root must be a mapping/object.")
    return data


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_default_raster(root: Path) -> Path:
    if root.is_file():
        return root
    tifs = sorted(root.rglob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in {root}")
    return tifs[0]


def _context_window_for_point(src: rasterio.io.DatasetReader, x: float, y: float, tile_size: int) -> tuple[Window, Window, int, int]:
    row, col = src.index(x, y)
    # Snap to tile origin that contains point
    tile_row = (row // tile_size) * tile_size
    tile_col = (col // tile_size) * tile_size
    tile_win = Window(tile_col, tile_row, tile_size, tile_size)
    # 0.5 tile margin each side => 2T x 2T context
    ctx_row = tile_row - tile_size // 2
    ctx_col = tile_col - tile_size // 2
    ctx_win = Window(ctx_col, ctx_row, tile_size * 2, tile_size * 2)
    return tile_win, ctx_win, tile_row, tile_col


def _read_context(src: rasterio.io.DatasetReader, ctx_win: Window) -> tuple[np.ndarray, rasterio.Affine]:
    arr = src.read(
        1,
        window=ctx_win,
        boundless=True,
        fill_value=src.nodata if src.nodata is not None else 0,
        out_shape=(int(ctx_win.height), int(ctx_win.width)),
        resampling=Resampling.bilinear,
    ).astype(np.float32)
    if src.nodata is not None:
        arr[arr == src.nodata] = 0.0
    tr = window_transform(ctx_win, src.transform)
    return arr, tr


def _draw_l_marker(img: np.ndarray, tile_origin_px: tuple[int, int], tile_size: int, color: tuple[int, int, int] = (255, 68, 68)) -> np.ndarray:
    out = img.copy()
    r0, c0 = tile_origin_px
    r1, c1 = r0 + tile_size, c0 + tile_size
    l = max(10, min(int(tile_size * 0.2), int(tile_size * 0.35)))
    t = 2
    # top-left
    cv2.line(out, (c0, r0), (c0 + l, r0), color, t)
    cv2.line(out, (c0, r0), (c0, r0 + l), color, t)
    # top-right
    cv2.line(out, (c1 - l, r0), (c1, r0), color, t)
    cv2.line(out, (c1, r0), (c1, r0 + l), color, t)
    # bottom-left
    cv2.line(out, (c0, r1), (c0 + l, r1), color, t)
    cv2.line(out, (c0, r1 - l), (c0, r1), color, t)
    # bottom-right
    cv2.line(out, (c1 - l, r1), (c1, r1), color, t)
    cv2.line(out, (c1, r1 - l), (c1, r1), color, t)
    return out


def _draw_scale_bar(img: np.ndarray, meters_per_px: float) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    base_x = 32
    base_y = h - 28
    max_m = 10.0
    pix_10m = max(1, int(round(max_m / meters_per_px)))
    pix_5m = max(1, int(round(5.0 / meters_per_px)))
    color = (255, 255, 255)
    border = (0, 0, 0)
    # bar
    cv2.line(out, (base_x, base_y), (base_x + pix_10m, base_y), border, 4)
    cv2.line(out, (base_x, base_y), (base_x + pix_10m, base_y), color, 2)
    for x in (base_x, base_x + pix_5m, base_x + pix_10m):
        cv2.line(out, (x, base_y - 8), (x, base_y + 8), border, 3)
        cv2.line(out, (x, base_y - 8), (x, base_y + 8), color, 1)
    cv2.putText(out, "0", (base_x - 6, base_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 2, cv2.LINE_AA)
    cv2.putText(out, "0", (base_x - 6, base_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(out, "5", (base_x + pix_5m - 5, base_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 2, cv2.LINE_AA)
    cv2.putText(out, "5", (base_x + pix_5m - 5, base_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(out, "10", (base_x + pix_10m - 10, base_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 2, cv2.LINE_AA)
    cv2.putText(out, "10", (base_x + pix_10m - 10, base_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(out, "m", (base_x + pix_10m + 10, base_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 2, cv2.LINE_AA)
    cv2.putText(out, "m", (base_x + pix_10m + 10, base_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _draw_ortho_attribution(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    text = ORTHO_ATTRIBUTION
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = max(8, w - tw - 12)
    y = max(20, h - 16)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _pad_to_16_9(img: np.ndarray, bg: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    h, w = img.shape[:2]
    target_ratio = 16.0 / 9.0
    cur_ratio = w / h if h else target_ratio
    if abs(cur_ratio - target_ratio) < 1e-6:
        return img
    if cur_ratio > target_ratio:
        new_h = int(round(w / target_ratio))
        pad = max(0, new_h - h)
        top = pad // 2
        bot = pad - top
        return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=bg)
    new_w = int(round(h * target_ratio))
    pad = max(0, new_w - w)
    left = pad // 2
    right = pad - left
    return cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=bg)


def _compose_grid(images: list[np.ndarray], rows: int, cols: int, pad: int = 12, bg: tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    assert len(images) == rows * cols
    hh = max(img.shape[0] for img in images)
    ww = max(img.shape[1] for img in images)
    canvas_h = rows * hh + (rows + 1) * pad
    canvas_w = cols * ww + (cols + 1) * pad
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = np.array(bg, dtype=np.uint8)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y = pad + r * (hh + pad)
        x = pad + c * (ww + pad)
        patch = img
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        if patch.shape[:2] != (hh, ww):
            patch = cv2.resize(patch, (ww, hh), interpolation=cv2.INTER_AREA)
        canvas[y : y + hh, x : x + ww] = patch
    return canvas


def _write_geotiff(path: Path, data: np.ndarray, transform: rasterio.Affine, crs: Any, nodata: float | None = None) -> None:
    _ensure_dir(path.parent)
    if data.ndim == 2:
        count = 1
        out = data[np.newaxis, ...]
    else:
        count = data.shape[2]
        out = np.transpose(data, (2, 0, 1))
    dtype = out.dtype
    profile = {
        "driver": "GTiff",
        "height": out.shape[1],
        "width": out.shape[2],
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out)


def _render_view(
    view: RenderView,
    base_raster: Path,
    x: float,
    y: float,
    tile_size: int,
    predictor: CNNPredictor,
) -> dict[str, Any]:
    raster_path = view.raster_path or base_raster
    with rasterio.open(raster_path) as src:
        tile_win, ctx_win, tile_row, tile_col = _context_window_for_point(src, x, y, tile_size)
        ctx, tr = _read_context(src, ctx_win)
        mpp = abs(src.transform.a)
        rgb = _apply_sld(ctx)

        # Focus tile region inside 2T context is always [T/2 : T/2+T]
        origin_rc = (tile_size // 2, tile_size // 2)
        focus_tile = src.read(
            1,
            window=tile_win,
            boundless=True,
            fill_value=src.nodata if src.nodata is not None else 0,
            out_shape=(tile_size, tile_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        if src.nodata is not None:
            focus_tile[focus_tile == src.nodata] = 0.0

        prob = predictor.predict_proba_cdw(focus_tile) if predictor._trained else None
        pred = None if prob is None else ("cdw" if prob >= predictor._thresh else "no_cdw")
        hm = _compute_heatmap("IntGrad", predictor, focus_tile, tile_row, tile_col, {})

        if view.view_type == "orthophoto":
            layer = build_wms_layer_name(raster_path.name)
            if layer is not None:
                bbox = window_bounds(ctx_win, src.transform)
                ortho = fetch_wms_for_bbox(
                    layer=layer,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    width=int(ctx_win.width),
                    height=int(ctx_win.height),
                )
                if ortho is not None:
                    rgb = ortho
            rgb = _draw_ortho_attribution(rgb)
        elif view.view_type in {"intgrad", "hotspot"}:
            hm_big = np.zeros((tile_size * 2, tile_size * 2), dtype=np.uint8)
            r0, c0 = origin_rc
            hm_big[r0 : r0 + tile_size, c0 : c0 + tile_size] = hm
            hm_color = cv2.applyColorMap(hm_big, cv2.COLORMAP_INFERNO)[:, :, ::-1]
            rgb = np.clip(0.35 * rgb.astype(np.float32) + 0.65 * hm_color.astype(np.float32), 0, 255).astype(
                np.uint8
            )
        elif view.view_type == "probability":
            txt = "P(CDW)=n/a" if prob is None else f"P(CDW)={prob:.3f}"
            cv2.putText(rgb, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(rgb, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        elif view.view_type == "prediction":
            txt = "Prediction: n/a" if pred is None else f"Prediction: {pred.upper()}"
            cv2.putText(rgb, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(rgb, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        rgb = _draw_l_marker(rgb, origin_rc, tile_size)
        rgb = _draw_scale_bar(rgb, meters_per_px=mpp)

        return {
            "rgb": rgb,
            "ctx": ctx,
            "transform": tr,
            "crs": src.crs,
            "focus_tile": focus_tile,
            "prob": prob,
            "pred": pred,
            "intgrad": hm,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "tile_origin_rc": origin_rc,
            "tile_size": tile_size,
            "raster_path": raster_path,
            "view_key": view.key,
            "view_type": view.view_type,
        }


def _export_single_view_artifacts(scene_dir: Path, rendered: dict[str, Any]) -> None:
    key = rendered["view_key"]
    _ensure_dir(scene_dir / "layers")
    rgb = rendered["rgb"]
    ctx = rendered["ctx"]
    tr = rendered["transform"]
    crs = rendered["crs"]
    tsize = rendered["tile_size"]
    r0, c0 = rendered["tile_origin_rc"]

    # GeoTIFFs
    _write_geotiff(scene_dir / "layers" / f"{key}_chm_context.tif", ctx.astype(np.float32), tr, crs, nodata=-9999.0)
    _write_geotiff(scene_dir / "layers" / f"{key}_visual_context.tif", rgb.astype(np.uint8), tr, crs)

    intgrad_ctx = np.full((tsize * 2, tsize * 2), np.nan, dtype=np.float32)
    intgrad_ctx[r0 : r0 + tsize, c0 : c0 + tsize] = rendered["intgrad"].astype(np.float32) / 255.0
    _write_geotiff(scene_dir / "layers" / f"{key}_intgrad_context.tif", intgrad_ctx, tr, crs, nodata=np.nan)

    prob_map = np.full((tsize * 2, tsize * 2), np.nan, dtype=np.float32)
    if rendered["prob"] is not None:
        prob_map[r0 : r0 + tsize, c0 : c0 + tsize] = float(rendered["prob"])
    _write_geotiff(scene_dir / "layers" / f"{key}_probability_context.tif", prob_map, tr, crs, nodata=np.nan)

    pred_map = np.full((tsize * 2, tsize * 2), 255, dtype=np.uint8)
    if rendered["pred"] is not None:
        pred_val = 1 if rendered["pred"] == "cdw" else 0
        pred_map[r0 : r0 + tsize, c0 : c0 + tsize] = pred_val
    _write_geotiff(scene_dir / "layers" / f"{key}_prediction_context.tif", pred_map, tr, crs, nodata=255)

    cv2.imwrite(str(scene_dir / "layers" / f"{key}_context.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_layouts(scene_dir: Path, rendered_views: list[dict[str, Any]], layouts: list[str]) -> None:
    _ensure_dir(scene_dir / "figures")
    panels = [rv["rgb"] for rv in rendered_views]
    if not panels:
        return
    # deterministic: first 4 panels for grouped layouts
    while len(panels) < 4:
        panels.append(panels[-1].copy())

    if "tile" in layouts:
        cv2.imwrite(
            str(scene_dir / "figures" / "tile.png"),
            cv2.cvtColor(rendered_views[0]["rgb"], cv2.COLOR_RGB2BGR),
        )
    if "16:9" in layouts or "16x9" in layouts:
        out = _pad_to_16_9(rendered_views[0]["rgb"])
        cv2.imwrite(str(scene_dir / "figures" / "single_16x9.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    if "1x4" in layouts:
        strip = _compose_grid(panels[:4], rows=1, cols=4)
        cv2.imwrite(str(scene_dir / "figures" / "grid_1x4.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    if "2x2" in layouts:
        grid = _compose_grid(panels[:4], rows=2, cols=2)
        cv2.imwrite(str(scene_dir / "figures" / "grid_2x2.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def export_scene(scene: RenderScene, base_raster: Path, predictor: CNNPredictor) -> dict[str, Any]:
    scene_dir = scene.output_dir / scene.scene_id
    _ensure_dir(scene_dir)
    rendered = []
    for view in scene.views:
        rv = _render_view(
            view=view,
            base_raster=base_raster,
            x=scene.x,
            y=scene.y,
            tile_size=scene.tile_size,
            predictor=predictor,
        )
        _export_single_view_artifacts(scene_dir, rv)
        rendered.append(rv)
    _save_layouts(scene_dir, rendered, scene.layouts)
    meta = {
        "scene_id": scene.scene_id,
        "x": scene.x,
        "y": scene.y,
        "crs": scene.crs,
        "tile_size": scene.tile_size,
        "views": [{"key": v.key, "type": v.view_type, "title": v.title} for v in scene.views],
        "layouts": scene.layouts,
    }
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def quick_export_current_view(
    *,
    chm_path: Path,
    x: float,
    y: float,
    output_dir: Path,
    tile_size: int = 128,
    scene_id: str = "quick_export",
) -> Path:
    predictor = CNNPredictor()
    model_path = output_dir / "ensemble_model.pt"
    ensemble_meta = model_path.parent / "ensemble_meta.json"
    if ensemble_meta.exists():
        predictor.load_ensemble_meta(ensemble_meta)
    elif model_path.exists():
        predictor.load_from_disk(model_path)
    scene = RenderScene(
        scene_id=scene_id,
        x=x,
        y=y,
        crs="EPSG:3301",
        tile_size=tile_size,
        output_dir=output_dir / "thesis_exports",
        views=[
            RenderView("chm", "chm", "CHM"),
            RenderView("orthophoto", "orthophoto", "Ortofoto"),
            RenderView("probability", "probability", "Probability"),
            RenderView("intgrad", "intgrad", "IntGrad"),
        ],
        layouts=["tile", "16:9", "1x4", "2x2"],
    )
    export_scene(scene, chm_path, predictor)
    predictor.shutdown()
    return scene.output_dir / scene.scene_id


def _scene_from_yaml_obj(obj: dict[str, Any], default_output: Path) -> RenderScene:
    scene_id = str(obj.get("id", "scene"))
    x = float(obj["x"])
    y = float(obj["y"])
    crs = str(obj.get("crs", "EPSG:3301"))
    tile_size = int(obj.get("tile_size", 128))
    layouts = [str(v) for v in obj.get("layouts", ["tile", "16:9", "1x4", "2x2"])]
    views_data = obj.get("views", [])
    if not views_data:
        views_data = [
            {"key": "chm", "type": "chm", "title": "CHM"},
            {"key": "orthophoto", "type": "orthophoto", "title": "Ortofoto"},
            {"key": "probability", "type": "probability", "title": "Probability"},
            {"key": "intgrad", "type": "intgrad", "title": "IntGrad"},
        ]
    views = []
    for idx, vv in enumerate(views_data):
        if not isinstance(vv, dict):
            raise ValueError(f"Invalid views[{idx}] entry")
        views.append(
            RenderView(
                key=str(vv.get("key", f"view{idx+1}")),
                view_type=str(vv.get("type", "chm")),
                title=str(vv.get("title", vv.get("type", "view"))),
                raster_path=Path(vv["raster_path"]) if vv.get("raster_path") else None,
            )
        )
    out = Path(obj.get("output_dir", default_output))
    return RenderScene(
        scene_id=scene_id,
        x=x,
        y=y,
        crs=crs,
        tile_size=tile_size,
        output_dir=out,
        views=views,
        layouts=layouts,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export coordinate-based thesis figures.")
    p.add_argument("--scenario", type=str, default="", help="Path to scenario YAML.")
    p.add_argument("--chm", type=str, default="", help="Fallback single-scene CHM path.")
    p.add_argument("--x", type=float, default=None, help="Fallback X/Easting in EPSG:3301.")
    p.add_argument("--y", type=float, default=None, help="Fallback Y/Northing in EPSG:3301.")
    p.add_argument("--tile-size", type=int, default=128, help="Tile size in pixels.")
    p.add_argument("--output", type=str, default="output/thesis_exports", help="Output root directory.")
    p.add_argument("--scene-id", type=str, default="single_scene", help="Scene id for fallback mode.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_root = Path(args.output)
    _ensure_dir(out_root)

    predictor = CNNPredictor()
    model_path = Path("output/tile_labels/ensemble_model.pt")
    meta_path = model_path.parent / "ensemble_meta.json"
    if meta_path.exists():
        predictor.load_ensemble_meta(meta_path)
    elif model_path.exists():
        predictor.load_from_disk(model_path)
    else:
        print("[export] No ensemble model found; probability/heatmap outputs may be empty.")

    exported: list[dict[str, Any]] = []

    if args.scenario:
        data = _load_yaml(Path(args.scenario))
        base_raster = _find_default_raster(Path(data.get("default_raster", args.chm or ".")))
        scenes_data = data.get("scenes", [])
        if not scenes_data:
            raise ValueError("Scenario must contain non-empty 'scenes'.")
        for s_obj in scenes_data:
            if not isinstance(s_obj, dict):
                raise ValueError("Each scene entry must be a mapping/object.")
            scene = _scene_from_yaml_obj(s_obj, out_root)
            exported.append(export_scene(scene, base_raster=base_raster, predictor=predictor))
    else:
        if args.x is None or args.y is None or not args.chm:
            raise ValueError("Fallback mode requires --chm --x --y.")
        scene = RenderScene(
            scene_id=args.scene_id,
            x=float(args.x),
            y=float(args.y),
            crs="EPSG:3301",
            tile_size=int(args.tile_size),
            output_dir=out_root,
            views=[
                RenderView("chm", "chm", "CHM"),
                RenderView("orthophoto", "orthophoto", "Ortofoto"),
                RenderView("prediction", "prediction", "Prediction"),
                RenderView("probability", "probability", "Probability"),
                RenderView("intgrad", "intgrad", "IntGrad"),
            ],
            layouts=["tile", "16:9", "1x4", "2x2"],
        )
        exported.append(export_scene(scene, base_raster=Path(args.chm), predictor=predictor))

    report = {"exported_scenes": exported}
    (out_root / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictor.shutdown()
    print(f"[export] Done. Scene count: {len(exported)}  Output: {out_root}")


if __name__ == "__main__":
    main()
