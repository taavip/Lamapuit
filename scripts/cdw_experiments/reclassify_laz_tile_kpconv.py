#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import pickle

import laspy
import numpy as np
from plyfile import PlyData, PlyElement


ROOT = Path(__file__).resolve().parents[2]


PRETRAINED_IDS: dict[str, str] = {
    "light": "14sz0hdObzsf_exxInXdOIbnUTe0foOOz",
    "heavy": "1ySQq3SRBgk2Vt5Bvj-0N7jDPi0QTPZiZ",
    "deform": "1ObGr2Srfj0f7Bd3bBbuQzxtjf0ULbpSA",
    "deform_light": "1gZfv6q6lUT9STFh7Fk4qVa5IVTgwmWIr",
}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _write_ply_xyzrgbc(path: Path, xyz: np.ndarray, rgb: np.ndarray, cls: np.ndarray) -> None:
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[0] != cls.shape[0]:
        raise ValueError("xyz/rgb/class arrays must have matching lengths")

    vertices = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "i4"),
        ],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    vertices["class"] = cls
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def _write_pred_ply(path: Path, xyz: np.ndarray, preds: np.ndarray) -> None:
    if xyz.shape[0] != preds.shape[0]:
        raise ValueError("xyz/preds arrays must have matching lengths")
    vertices = np.empty(
        xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "i4")],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["preds"] = preds.astype(np.int32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def _extract_rgb_u8(las: laspy.LasData) -> np.ndarray:
    n = len(las.x)
    if all(hasattr(las, c) for c in ("red", "green", "blue")):
        r = np.asarray(las.red, dtype=np.float32)
        g = np.asarray(las.green, dtype=np.float32)
        b = np.asarray(las.blue, dtype=np.float32)
        maxv = float(max(np.nanmax(r), np.nanmax(g), np.nanmax(b), 1.0))
        if maxv > 255.0:
            scale = 255.0 / maxv
            r = r * scale
            g = g * scale
            b = b * scale
        rgb = np.stack([r, g, b], axis=1)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    if hasattr(las, "intensity"):
        inten = np.asarray(las.intensity, dtype=np.float32)
        if inten.size == 0:
            return np.zeros((n, 3), dtype=np.uint8)
        vmax = float(np.nanmax(inten))
        if vmax <= 0:
            return np.zeros((n, 3), dtype=np.uint8)
        u8 = np.clip(255.0 * (inten / vmax), 0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=1)

    return np.zeros((n, 3), dtype=np.uint8)


def _ensure_pretrained_log(kpconv_root: Path, variant: str) -> Path:
    logs_root = kpconv_root / "pretrained_logs"
    variant_root = logs_root / variant
    zip_path = logs_root / f"{variant}.zip"
    logs_root.mkdir(parents=True, exist_ok=True)

    if not variant_root.exists():
        variant_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        p.parent
        for p in variant_root.rglob("parameters.txt")
        if (p.parent / "checkpoints").exists()
    ]
    if candidates:
        return sorted(candidates)[0]

    if not zip_path.exists():
        model_id = PRETRAINED_IDS[variant]
        _run([sys.executable, "-m", "gdown", "--id", model_id, "-O", str(zip_path)])

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(variant_root)

    candidates = [
        p.parent
        for p in variant_root.rglob("parameters.txt")
        if (p.parent / "checkpoints").exists()
    ]
    if not candidates:
        raise RuntimeError(f"No KPConv log with checkpoints found after extracting {zip_path}")
    return sorted(candidates)[0]


def _ensure_kpconv_wrappers(kpconv_root: Path) -> None:
    _run([sys.executable, "-m", "pip", "install", "--quiet", "setuptools<60"])
    _run(["bash", "-lc", "sh compile_wrappers.sh"], cwd=kpconv_root / "cpp_wrappers")


def _prepare_s3dis_layout(base_s3dis: Path) -> None:
    ply_dir = base_s3dis / "original_ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    placeholder_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    placeholder_rgb = np.array([[0, 0, 0]], dtype=np.uint8)
    placeholder_cls = np.array([0], dtype=np.int32)

    for idx in range(1, 7):
        area_ply = ply_dir / f"Area_{idx}.ply"
        if not area_ply.exists():
            _write_ply_xyzrgbc(area_ply, placeholder_xyz, placeholder_rgb, placeholder_cls)


def _clear_area5_cache(base_s3dis: Path, first_subsampling_dl: float) -> None:
    tree_dir = base_s3dis / f"input_{first_subsampling_dl:.3f}"
    for name in [
        "Area_5_KDTree.pkl",
        "Area_5.ply",
        "Area_5_proj.pkl",
        "Area_5_coarse_KDTree.pkl",
    ]:
        p = tree_dir / name
        if p.exists():
            p.unlink()


def _seed_calibration_cache(base_s3dis: Path, config: object) -> None:
    batch_lim_file = base_s3dis / "batch_limits.pkl"
    neighb_lim_file = base_s3dis / "neighbors_limits.pkl"

    if batch_lim_file.exists():
        with open(batch_lim_file, "rb") as f:
            batch_lim_dict = pickle.load(f)
    else:
        batch_lim_dict = {}

    if neighb_lim_file.exists():
        with open(neighb_lim_file, "rb") as f:
            neighb_lim_dict = pickle.load(f)
    else:
        neighb_lim_dict = {}

    key = "potentials_{:.3f}_{:.3f}_{:d}".format(
        float(config.in_radius),
        float(config.first_subsampling_dl),
        int(config.batch_num),
    )
    if key not in batch_lim_dict:
        batch_lim_dict[key] = 3000.0

    num_layers = int(config.num_layers)
    for layer_ind in range(num_layers):
        dl = float(config.first_subsampling_dl) * (2 ** layer_ind)
        if config.deform_layers[layer_ind]:
            r = dl * float(config.deform_radius)
        else:
            r = dl * float(config.conv_radius)
        neighb_key = "{:.3f}_{:.3f}".format(dl, r)
        if neighb_key not in neighb_lim_dict:
            neighb_lim_dict[neighb_key] = 64

    with open(batch_lim_file, "wb") as f:
        pickle.dump(batch_lim_dict, f)
    with open(neighb_lim_file, "wb") as f:
        pickle.dump(neighb_lim_dict, f)


def _run_single_inference(
    kpconv_root: Path,
    base_s3dis: Path,
    chosen_log: Path,
    save_tag: str,
    num_votes: int,
    validation_size: int,
    subsampling_dl: float | None,
    in_radius: float | None,
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from datasets.S3DIS import S3DISCollate, S3DISDataset, S3DISSampler
    from models.architectures import KPFCNN
    from utils.config import Config
    from utils.tester import ModelTester

    config = Config()
    config.load(str(chosen_log))
    config.input_threads = 0
    config.validation_size = int(validation_size)
    if subsampling_dl is not None:
        config.first_subsampling_dl = float(subsampling_dl)
    if in_radius is not None:
        config.in_radius = float(in_radius)
    config.saving = True
    config.saving_path = save_tag

    _seed_calibration_cache(base_s3dis, config)

    chkp_dir = chosen_log / "checkpoints"
    chkps = sorted([p for p in chkp_dir.iterdir() if p.name.startswith("chkp")])
    if not chkps:
        raise RuntimeError(f"No checkpoints found in {chkp_dir}")
    chosen_chkp = chkps[-1]

    test_dataset = S3DISDataset(config, set="validation", use_potentials=True)
    test_sampler = S3DISSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=S3DISCollate,
        num_workers=config.input_threads,
        pin_memory=True,
    )
    test_sampler.calibration(test_loader, verbose=False)

    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Some pretrained logs were saved on CUDA devices; force CPU remap when CUDA is unavailable.
    original_torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
    try:
        tester = ModelTester(net, chkp_path=str(chosen_chkp))
    finally:
        torch.load = original_torch_load

    original_cuda_sync = torch.cuda.synchronize
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *args, **kwargs: None
    try:
        tester.cloud_segmentation_test(net, test_loader, config, num_votes=num_votes, debug=False)
    finally:
        torch.cuda.synchronize = original_cuda_sync

    pred_path = kpconv_root / "test" / save_tag / "predictions" / "Area_5.ply"
    if not pred_path.exists() and getattr(tester, "test_probs", None) is not None:
        # Fallback for early-exit runs where tester does not hit its internal save trigger.
        probs = tester.test_probs[0][test_dataset.test_proj[0], :]
        for l_ind, label_value in enumerate(test_dataset.label_values):
            if label_value in test_dataset.ignored_labels:
                probs = np.insert(probs, l_ind, 0, axis=1)
        preds = test_dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
        points = test_dataset.load_evaluation_points(test_dataset.files[0]).astype(np.float32)
        _write_pred_ply(pred_path, points, preds)

    if not pred_path.exists():
        raise RuntimeError(f"Expected prediction file not found: {pred_path}")
    return pred_path


def _preds_to_las_classes(preds: np.ndarray, original_cls: np.ndarray) -> np.ndarray:
    out = original_cls.astype(np.uint8, copy=True)

    # S3DIS classes: floor(1) is mapped to ground; wall/structure classes map to LAS building.
    ground_mask = preds == 1
    building_mask = np.isin(preds, [2, 3, 4, 5, 6])
    clutter_mask = np.isin(preds, [0, 7, 8, 9, 10, 11, 12])

    out[ground_mask] = 2
    out[building_mask] = 6
    out[clutter_mask] = 1
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reclassify tile LAZ files with KPConv (S3DIS pretrained)")
    p.add_argument("--input-dir", type=Path, default=Path("data/lamapuit/laz"))
    p.add_argument("--tile-id", default="436646")
    p.add_argument("--years", default="2018,2020,2022,2024")
    p.add_argument("--kpconv-root", type=Path, default=Path("kpconv_tmp/KPConv-PyTorch"))
    p.add_argument(
        "--pretrained-variant",
        choices=sorted(PRETRAINED_IDS.keys()),
        default="light",
        help="KPConv pretrained S3DIS model variant",
    )
    p.add_argument("--chosen-log", type=Path, default=None, help="Use an existing extracted KPConv log")
    p.add_argument("--num-votes", type=int, default=0, help="Votes for KPConv tester (0 is fastest)")
    p.add_argument("--validation-size", type=int, default=80)
    p.add_argument("--subsampling-dl", type=float, default=None, help="Override KPConv first_subsampling_dl")
    p.add_argument("--in-radius", type=float, default=None, help="Override KPConv in_radius")
    p.add_argument(
        "--max-kpconv-points",
        type=int,
        default=35000000,
        help="If input exceeds this size, run KPConv on a sampled subset and remap classes on full cloud",
    )
    p.add_argument("--out-dir", type=Path, default=Path("output/laz_reclassified_kpconv"))
    p.add_argument("--summary-json", type=Path, default=Path("output/laz_reclassified_kpconv/summary_436646_kpconv.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    if not years:
        raise ValueError("No years provided")

    kpconv_root = (ROOT / args.kpconv_root).resolve()
    if not kpconv_root.exists():
        raise FileNotFoundError(f"KPConv root not found: {kpconv_root}")

    chosen_log = (ROOT / args.chosen_log).resolve() if args.chosen_log is not None else None
    if chosen_log is None:
        chosen_log = _ensure_pretrained_log(kpconv_root, args.pretrained_variant)
    if not chosen_log.exists():
        raise FileNotFoundError(f"Chosen log not found: {chosen_log}")

    os.chdir(kpconv_root)
    if str(kpconv_root) not in sys.path:
        sys.path.insert(0, str(kpconv_root))

    _ensure_kpconv_wrappers(kpconv_root)

    # S3DISDataset uses a hardcoded relative path '../../Data/S3DIS' from kpconv_root.
    base_s3dis = kpconv_root.parents[1] / "Data" / "S3DIS"
    _prepare_s3dis_layout(base_s3dis)

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, object]] = []
    area5_ply = base_s3dis / "original_ply" / "Area_5.ply"

    for year in years:
        laz_path = (ROOT / args.input_dir / f"{args.tile_id}_{year}_madal.laz").resolve()
        if not laz_path.exists():
            raise FileNotFoundError(f"Missing LAZ: {laz_path}")

        print(f"\n[KPConv] Processing {laz_path.name}")
        las = laspy.read(str(laz_path))
        xyz = np.stack([
            np.asarray(las.x, dtype=np.float32),
            np.asarray(las.y, dtype=np.float32),
            np.asarray(las.z, dtype=np.float32),
        ], axis=1)
        rgb = _extract_rgb_u8(las)
        original_cls = np.asarray(las.classification, dtype=np.uint8)

        sampled_for_inference = False
        class_map: dict[int, int] = {}
        if xyz.shape[0] > int(args.max_kpconv_points):
            sampled_for_inference = True
            rng = np.random.default_rng(42 + int(year))
            infer_indices = np.sort(rng.choice(xyz.shape[0], size=int(args.max_kpconv_points), replace=False))
        else:
            infer_indices = np.arange(xyz.shape[0], dtype=np.int64)

        xyz_infer = xyz[infer_indices]
        rgb_infer = rgb[infer_indices]
        original_cls_infer = original_cls[infer_indices]

        labels = np.zeros(xyz_infer.shape[0], dtype=np.int32)
        _write_ply_xyzrgbc(area5_ply, xyz_infer, rgb_infer, labels)

        # Ensure fresh KDTree/reprojection for the newly written Area_5 cloud.
        # first_subsampling_dl is read from pretrained parameters and used in S3DISDataset.
        from utils.config import Config

        cfg = Config()
        cfg.load(str(chosen_log))
        dl_for_cache = float(args.subsampling_dl) if args.subsampling_dl is not None else float(cfg.first_subsampling_dl)
        _clear_area5_cache(base_s3dis, dl_for_cache)

        save_tag = f"kpconv_{args.tile_id}_{year}"
        pred_ply = _run_single_inference(
            kpconv_root=kpconv_root,
            base_s3dis=base_s3dis,
            chosen_log=chosen_log,
            save_tag=save_tag,
            num_votes=args.num_votes,
            validation_size=args.validation_size,
            subsampling_dl=args.subsampling_dl,
            in_radius=args.in_radius,
        )

        pred_data = PlyData.read(str(pred_ply))["vertex"].data
        if "preds" not in pred_data.dtype.names:
            raise RuntimeError(f"Prediction field 'preds' not found in {pred_ply}")
        preds = np.asarray(pred_data["preds"], dtype=np.int32)
        if preds.shape[0] != xyz_infer.shape[0]:
            raise RuntimeError(
                f"Prediction length mismatch for {laz_path.name}: preds={preds.shape[0]} points={xyz_infer.shape[0]}"
            )

        infer_new_cls = _preds_to_las_classes(preds, original_cls_infer)
        if sampled_for_inference:
            for cls_val in np.unique(original_cls_infer):
                mask = original_cls_infer == cls_val
                mapped = infer_new_cls[mask]
                if mapped.size == 0:
                    continue
                vals, cnts = np.unique(mapped, return_counts=True)
                class_map[int(cls_val)] = int(vals[np.argmax(cnts)])
            new_cls = original_cls.copy()
            for src, dst in class_map.items():
                new_cls[original_cls == src] = np.uint8(dst)
        else:
            new_cls = infer_new_cls

        las.classification = new_cls

        out_laz = out_dir / f"{laz_path.stem}_reclassified_kpconv.laz"
        las.write(str(out_laz))

        run_summary = {
            "input_laz": str(laz_path),
            "output_laz": str(out_laz),
            "prediction_ply": str(pred_ply),
            "n_points": int(xyz.shape[0]),
            "n_points_inference": int(xyz_infer.shape[0]),
            "sampled_for_inference": bool(sampled_for_inference),
            "class_map_from_sample": {str(k): int(v) for k, v in sorted(class_map.items())},
            "original_class_counts": {str(k): int(v) for k, v in sorted(Counter(original_cls.tolist()).items())},
            "kpconv_pred_counts": {str(k): int(v) for k, v in sorted(Counter(preds.tolist()).items())},
            "new_class_counts": {str(k): int(v) for k, v in sorted(Counter(new_cls.tolist()).items())},
        }
        run_summaries.append(run_summary)
        print(json.dumps(run_summary, indent=2))

    final_summary = {
        "tile_id": args.tile_id,
        "years": years,
        "chosen_log": str(chosen_log),
        "pretrained_variant": args.pretrained_variant,
        "results": run_summaries,
    }

    summary_path = (ROOT / args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
