#!/usr/bin/env python3
"""Simple Tkinter GUI wrapper for label_all_rasters_segmentation and brush labeler.

This provides a minimal, cross-platform GUI to build tile queues (with optional
pre-seg manifests) and launch the existing `scripts/brush_mask_labeler.py` for
manual segmentation. It supports running the underlying commands either
locally (using the running Python interpreter) or inside a Docker image.

This is intentionally small and pragmatic: it shells out to the existing
scripts so we avoid duplicating complex queue/IO logic.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import os
from pathlib import Path
import re
import csv
from typing import Optional

import numpy as np
import rasterio
import tkinter as tk
from tkinter import filedialog, messagebox


class LabelAllRastersGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Label All Rasters — GUI")
        self.geometry("920x600")

        self._build_widgets()

    def _build_widgets(self):
        top = tk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        # CHM dir
        tk.Label(top, text="CHM dir:").grid(row=0, column=0, sticky="w")
        self.chm_entry = tk.Entry(top, width=60)
        self.chm_entry.grid(row=0, column=1, columnspan=4, sticky="w", padx=4)
        tk.Button(top, text="Browse", command=lambda: self._browse_chm()).grid(row=0, column=5, padx=4)

        # Pattern and output dir
        tk.Label(top, text="Pattern:").grid(row=1, column=0, sticky="w")
        self.pattern_entry = tk.Entry(top, width=20)
        self.pattern_entry.insert(0, "*20cm.tif")
        self.pattern_entry.grid(row=1, column=1, sticky="w", padx=4)

        tk.Label(top, text="Output dir:").grid(row=1, column=2, sticky="w")
        self.out_entry = tk.Entry(top, width=40)
        self.out_entry.insert(0, "output/manual_masks_all_rasters")
        self.out_entry.grid(row=1, column=3, columnspan=2, sticky="w", padx=4)
        tk.Button(top, text="Browse", command=lambda: self._browse_out()).grid(row=1, column=5, padx=4)

        # Preseg manifest list
        mid = tk.Frame(self)
        mid.pack(fill="x", padx=8, pady=6)
        tk.Label(mid, text="Pre-seg manifests (optional):").grid(row=0, column=0, sticky="w")
        self.manifest_list = tk.Listbox(mid, height=4, width=80)
        self.manifest_list.grid(row=1, column=0, columnspan=4, sticky="w")
        btn_frame = tk.Frame(mid)
        btn_frame.grid(row=1, column=4, sticky="ns", padx=6)
        tk.Button(btn_frame, text="Add...", command=lambda: self._add_manifest()).pack(fill="x")
        tk.Button(btn_frame, text="Remove", command=lambda: self._remove_manifest()).pack(fill="x", pady=4)

        # Options row
        opt = tk.Frame(self)
        opt.pack(fill="x", padx=8, pady=6)
        tk.Label(opt, text="Chunk size:").grid(row=0, column=0, sticky="w")
        self.chunk_entry = tk.Entry(opt, width=6)
        self.chunk_entry.insert(0, "128")
        self.chunk_entry.grid(row=0, column=1, sticky="w", padx=4)

        tk.Label(opt, text="Overlap:").grid(row=0, column=2, sticky="w")
        self.overlap_entry = tk.Entry(opt, width=6)
        self.overlap_entry.insert(0, "0.5")
        self.overlap_entry.grid(row=0, column=3, sticky="w", padx=4)

        tk.Label(opt, text="Max nodata %:").grid(row=0, column=4, sticky="w")
        self.nodata_entry = tk.Entry(opt, width=6)
        self.nodata_entry.insert(0, "75.0")
        self.nodata_entry.grid(row=0, column=5, sticky="w", padx=4)

        tk.Label(opt, text="Max tiles (0=all):").grid(row=0, column=6, sticky="w")
        self.max_tiles_entry = tk.Entry(opt, width=8)
        self.max_tiles_entry.insert(0, "0")
        self.max_tiles_entry.grid(row=0, column=7, sticky="w", padx=4)

        # Docker/run options
        runopt = tk.Frame(self)
        runopt.pack(fill="x", padx=8, pady=6)
        self.docker_var = tk.BooleanVar(value=False)
        tk.Checkbutton(runopt, text="Run underlying commands in Docker", variable=self.docker_var).grid(row=0, column=0, sticky="w")
        tk.Label(runopt, text="Docker image:").grid(row=0, column=1, sticky="w", padx=6)
        self.docker_entry = tk.Entry(runopt, width=28)
        self.docker_entry.insert(0, "lamapuit-dev")
        self.docker_entry.grid(row=0, column=2, sticky="w")

        # Control buttons
        ctrl = tk.Frame(self)
        ctrl.pack(fill="x", padx=8, pady=6)
        self.build_btn = tk.Button(ctrl, text="Build Queue", command=lambda: self._on_build())
        self.build_btn.pack(side="left")
        self.launch_btn = tk.Button(ctrl, text="Launch Brush Labeler", command=lambda: self._on_launch())
        self.launch_btn.pack(side="left", padx=6)
        tk.Button(ctrl, text="Quit", command=lambda: self.destroy()).pack(side="right")

        # Status/log area
        logf = tk.Frame(self)
        logf.pack(fill="both", expand=True, padx=8, pady=6)
        tk.Label(logf, text="Log: ").pack(anchor="w")
        self.log = tk.Text(logf, height=12)
        self.log.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=(0, 6))
# Insert GUI callbacks and worker threads before helper functions
    def _browse_chm(self):
        d = filedialog.askdirectory()
        if d:
            self.chm_entry.delete(0, tk.END)
            self.chm_entry.insert(0, d)

    def _browse_out(self):
        d = filedialog.askdirectory()
        if d:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, d)

    def _add_manifest(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if p:
            self.manifest_list.insert(tk.END, p)

    def _remove_manifest(self):
        sel = self.manifest_list.curselection()
        for i in reversed(sel):
            self.manifest_list.delete(i)

    def _append_log(self, text: str) -> None:
        def _do():
            self.log.insert(tk.END, text + "\n")
            self.log.see(tk.END)
        self.after(0, _do)

    def _on_build(self) -> None:
        t = threading.Thread(target=self._build_queue_thread, daemon=True)
        t.start()

    def _on_launch(self) -> None:
        t = threading.Thread(target=self._launch_brush_thread, daemon=True)
        t.start()

# --- Internal queue/build helpers (adapted from label_all_rasters_segmentation.py)
    def _build_queue_thread(self) -> None:
        chm = self.chm_entry.get().strip()
        out = self.out_entry.get().strip()
        pattern = self.pattern_entry.get().strip()
        chunk = self.chunk_entry.get().strip() or "128"
        overlap = self.overlap_entry.get().strip() or "0.5"
        nodata = self.nodata_entry.get().strip() or "75.0"
        max_tiles = self.max_tiles_entry.get().strip() or "0"
        manifests = list(self.manifest_list.get(0, tk.END))

        if not chm:
            messagebox.showerror("Missing CHM dir", "Please select a CHM directory to index.")
            return
        if not out:
            messagebox.showerror("Missing output dir", "Please select an output directory.")
            return

        cmd = [
            sys.executable,
            "scripts/label_all_rasters_segmentation.py",
            "--chm-dir",
            chm,
            "--pattern",
            pattern,
            "--output-dir",
            out,
            "--chunk-size",
            str(int(chunk)),
            "--overlap",
            str(float(overlap)),
            "--max-nodata-pct",
            str(float(nodata)),
            "--max-tiles",
            str(int(max_tiles)),
            "--build-only",
        ]

        for m in manifests:
            cmd += ["--preseg-manifest", m]

        self.status_var.set("Building queue...")
        self._append_log("Running: {}".format(shlex.join(cmd)))

        try:
            if self.docker_var.get():
                docker_image = self.docker_entry.get().strip() or "lamapuit-dev"
                cwd = os.getcwd()
                # Activate conda env inside container and run python
                inner = "source /opt/conda/etc/profile.d/conda.sh >/dev/null 2>&1 || true; conda activate cwd-detect >/dev/null 2>&1 || true; " + shlex.join(cmd)
                docker_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{cwd}:{cwd}",
                    "-w",
                    cwd,
                    docker_image,
                    "bash",
                    "-lc",
                    inner,
                ]
                self._append_log("Running in Docker: {}".format(shlex.join(docker_cmd)))
                proc = subprocess.run(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self._append_log(proc.stdout or "(no output)")
                if proc.returncode != 0:
                    self._append_log(f"[error] build failed (rc={proc.returncode})")
                    self.status_var.set("Queue build failed")
                else:
                    self._append_log("[ok] queue build completed")
                    self.status_var.set("Queue built")
            else:
                # Internal build (no external script): enumerate rasters and build queue
                chm_dir = Path(chm)
                rasters = sorted(chm_dir.glob(pattern))
                if not rasters:
                    self._append_log(f"[error] no rasters matched '{pattern}' in {chm_dir}")
                    self.status_var.set("Error")
                    return

                preseg_root = None
                manifest_paths = [Path(p) for p in manifests]
                by_full_path, by_raster_name, by_sample_variant, by_sample = _load_preseg_index(
                    manifest_paths, preseg_root
                )

                rows: list[dict[str, str]] = []
                skipped_nodata = 0
                total_tiles = 0
                seeded_tiles = 0

                for i, raster_path in enumerate(rasters, start=1):
                    nodata_pct = _compute_nodata_pct(raster_path)
                    if nodata_pct is None:
                        self._append_log(f"[{i:3d}/{len(rasters)}] skip read-error: {raster_path.name}")
                        continue
                    if float(nodata) is not None and float(nodata) >= 0 and (nodata_pct is not None) and (nodata_pct >= float(nodata)):
                        skipped_nodata += 1
                        self._append_log(f"[{i:3d}/{len(rasters)}] skip high nodata ({nodata_pct:.1f}%): {raster_path.name}")
                        continue

                    try:
                        with rasterio.open(raster_path) as src:
                            height = int(src.height)
                            width = int(src.width)
                    except Exception as exc:
                        self._append_log(f"[{i:3d}/{len(rasters)}] failed open: {raster_path.name} -> {exc}")
                        continue

                    chunks = _iter_chunks(height, width, int(chunk), float(overlap))
                    self._append_log(
                        f"[{i:3d}/{len(rasters)}] queue {len(chunks):,} tile(s): {raster_path.name} (nodata={nodata_pct:.1f}%)"
                    )

                    for row_off, col_off in chunks:
                        tile_id = f"{raster_path.stem}:{int(row_off)}:{int(col_off)}"
                        output_stem = _safe_stem(f"{raster_path.stem}_{int(row_off)}_{int(col_off)}")
                        row = {
                            "tile_id": tile_id,
                            "output_stem": output_stem,
                            "raster_path": str(raster_path),
                            "row_off": str(int(row_off)),
                            "col_off": str(int(col_off)),
                            "chunk_size": str(int(chunk)),
                            "init_mask": "",
                            "init_cam": "",
                            "hotspot_path": "",
                            "preseg_source": "",
                        }

                        seed = _lookup_seed(
                            raster_path,
                            row_off,
                            col_off,
                            by_full_path,
                            by_raster_name,
                            by_sample_variant,
                            by_sample,
                        )
                        if seed is not None:
                            if getattr(seed, "mask_path", None) is not None and Path(getattr(seed, "mask_path")).exists():
                                row["init_mask"] = str(getattr(seed, "mask_path"))
                            if getattr(seed, "cam_path", None) is not None and Path(getattr(seed, "cam_path")).exists():
                                row["init_cam"] = str(getattr(seed, "cam_path"))
                            if getattr(seed, "hotspot_path", None) is not None and Path(getattr(seed, "hotspot_path")).exists():
                                row["hotspot_path"] = str(getattr(seed, "hotspot_path"))
                            if row["init_mask"] or row["init_cam"]:
                                row["preseg_source"] = getattr(seed, "source", "")
                                seeded_tiles += 1

                        rows.append(row)
                        total_tiles += 1
                        if int(max_tiles) > 0 and total_tiles >= int(max_tiles):
                            self._append_log(f"[queue] reached --max-tiles={max_tiles}")
                            break

                    if int(max_tiles) > 0 and total_tiles >= int(max_tiles):
                        break

                queue_csv = Path(out) / "tile_queue.csv"
                _write_queue_csv(queue_csv, rows)
                self._append_log(
                    f"[queue] built {len(rows):,} row(s), seeded {seeded_tiles:,} row(s), skipped {skipped_nodata} raster(s) by nodata threshold"
                )
                self.status_var.set("Queue built")

        except FileNotFoundError as exc:
            self._append_log(f"[exception] {exc}")
            messagebox.showerror("Execution error", str(exc))
            self.status_var.set("Error")

    def _launch_brush_thread(self) -> None:
        out = self.out_entry.get().strip()
        tile_csv = Path(out) / "tile_queue.csv"
        if not tile_csv.exists():
            messagebox.showerror("Missing queue CSV", f"Queue not found: {tile_csv} — build it first.")
            return

        tile_root = os.getcwd()
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("brush_mask_labeler.py")),
            "--tile-csv",
            str(tile_csv),
            "--tile-root",
            tile_root,
            "--output-dir",
            out,
            "--window-name",
            "CWD Brush Browser (GUI)",
        ]

        self.status_var.set("Launching brush labeler...")
        self._append_log("Launching: {}".format(shlex.join(cmd)))

        try:
            if self.docker_var.get():
                docker_image = self.docker_entry.get().strip() or "lamapuit-dev"
                cwd = os.getcwd()
                inner = "source /opt/conda/etc/profile.d/conda.sh >/dev/null 2>&1 || true; conda activate cwd-detect >/dev/null 2>&1 || true; " + shlex.join(cmd)
                docker_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-e",
                    f"DISPLAY={os.environ.get('DISPLAY','')}",
                    "-v",
                    "/tmp/.X11-unix:/tmp/.X11-unix",
                    "-v",
                    f"{cwd}:{cwd}",
                    "-w",
                    cwd,
                    docker_image,
                    "bash",
                    "-lc",
                    inner,
                ]
                self._append_log("Docker run: {}".format(shlex.join(docker_cmd)))
                proc = subprocess.run(docker_cmd)
            else:
                proc = subprocess.run(cmd)

            if proc.returncode != 0:
                self._append_log(f"[error] brush labeler exited with rc={proc.returncode}")
                self.status_var.set("Labeler failed")
            else:
                self._append_log("[ok] brush labeler exited")
                self.status_var.set("Labeler finished")
        except FileNotFoundError as exc:
            self._append_log(f"[exception] {exc}")
            messagebox.showerror("Execution error", str(exc))
            self.status_var.set("Error")
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_HARMONIZED_RASTER_RE = re.compile(
    r"^(?P<sample>.+)_harmonized_dem_last_(?P<variant>[A-Za-z0-9]+)_chm$",
    re.IGNORECASE,
)


def _safe_stem(text: str) -> str:
    cleaned = _SAFE_STEM_RE.sub("_", str(text)).strip("_")
    return cleaned or "tile"


def _parse_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _norm_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except Exception:
        return str(path)


def _resolve_path(raw_value: str, base_dir: Path, preseg_root: Optional[Path]) -> Optional[Path]:
    value = str(raw_value).strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path

    candidates: list[Path] = [base_dir / path]
    if preseg_root is not None:
        candidates.insert(0, preseg_root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        val = str(row.get(key, "")).strip()
        if val:
            return val
    return ""


def _parse_tile_id_fields(tile_id: str) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    raw = str(tile_id).strip()
    if not raw:
        return None, None, None, None

    parts = raw.split(":")
    if len(parts) < 4:
        return None, None, None, None

    sample_id = ":".join(parts[:-3]).strip() if len(parts) > 4 else parts[0].strip()
    variant = str(parts[-3]).strip().lower() if parts[-3] else None
    row_off = _parse_int(parts[-2])
    col_off = _parse_int(parts[-1])
    if not sample_id:
        sample_id = None
    return sample_id, variant, row_off, col_off


def _sample_variant_from_raster_path(raster_path: Path) -> tuple[str, Optional[str]]:
    stem = raster_path.stem
    m = _HARMONIZED_RASTER_RE.match(stem)
    if m is not None:
        sample = str(m.group("sample")).strip()
        variant = str(m.group("variant")).strip().lower()
        return sample, variant or None

    if "_chm_" in stem:
        sample = stem.split("_chm_", 1)[0].strip()
        if sample:
            return sample, None

    if stem.endswith("_chm"):
        sample = stem[: -len("_chm")].strip("_")
        if sample:
            return sample, None

    return stem, None


def _compute_nodata_pct(path: Path) -> float | None:
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            nodata_val = src.nodata
            if nodata_val is None:
                mask = np.isnan(arr)
            else:
                mask = np.isnan(arr) | (arr == nodata_val)
            return float(100.0 * np.sum(mask) / arr.size)
    except Exception:
        return None


def _iter_chunks(height: int, width: int, chunk_size: int, overlap: float) -> list[tuple[int, int]]:
    stride = max(1, int(chunk_size * (1.0 - overlap)))
    min_gap = chunk_size // 4

    def make_offsets(size: int) -> list[int]:
        if size <= chunk_size:
            return [0]
        offsets = list(range(0, size - chunk_size + 1, stride))
        last_border = size - chunk_size
        gap = size - (offsets[-1] + chunk_size)
        if gap > min_gap and last_border not in offsets:
            offsets.append(last_border)
        return offsets

    rows = make_offsets(height)
    cols = make_offsets(width)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for row_off in rows:
        for col_off in cols:
            key = (row_off, col_off)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _load_preseg_index(manifest_paths: list[Path], preseg_root: Optional[Path]):
    by_full_path = {}
    by_raster_name = {}
    by_sample_variant = {}
    by_sample = {}

    for manifest in manifest_paths:
        if not manifest.exists():
            continue

        loaded = 0
        with manifest.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raster_raw = _first_nonempty(row, ["raster_path", "raster", "chm_path", "image_path"])
                row_off = _parse_int(row.get("row_off", ""))
                col_off = _parse_int(row.get("col_off", ""))
                tile_id = _first_nonempty(row, ["tile_id"])
                tile_sample, tile_variant, tile_row, tile_col = _parse_tile_id_fields(tile_id)

                if row_off is None:
                    row_off = tile_row
                if col_off is None:
                    col_off = tile_col
                if row_off is None or col_off is None:
                    continue

                mask_raw = _first_nonempty(row, ["mask_path", "init_mask", "mask_file"])
                cam_raw = _first_nonempty(row, ["cam_path", "init_cam", "cam_file"])
                hotspot_raw = _first_nonempty(row, ["hotspot_path", "hotspot_file"])

                mask_path = _resolve_path(mask_raw, manifest.parent, preseg_root) if mask_raw else None
                cam_path = _resolve_path(cam_raw, manifest.parent, preseg_root) if cam_raw else None
                hotspot_path = _resolve_path(hotspot_raw, manifest.parent, preseg_root) if hotspot_raw else cam_path

                if mask_path is None and cam_path is None:
                    continue

                source = f"{manifest.name}:{loaded + 1}"
                entry = type("SeedEntry", (), {})()
                entry.mask_path = mask_path
                entry.cam_path = cam_path
                entry.hotspot_path = hotspot_path
                entry.source = source

                if raster_raw:
                    raster_path = _resolve_path(raster_raw, manifest.parent, preseg_root)
                    if raster_path is not None:
                        full_key = (_norm_path(raster_path), int(row_off), int(col_off))
                        name_key = (Path(raster_path).name, int(row_off), int(col_off))
                        by_full_path[full_key] = entry
                        by_raster_name[name_key] = entry

                        inferred_sample, inferred_variant = _sample_variant_from_raster_path(raster_path)
                        if inferred_sample:
                            by_sample[(inferred_sample, int(row_off), int(col_off))] = entry
                            if inferred_variant:
                                by_sample_variant[(inferred_sample, inferred_variant, int(row_off), int(col_off))] = entry

                sample_id = _first_nonempty(row, ["sample_id", "sample", "mapsheet"]).strip()
                if not sample_id and tile_sample:
                    sample_id = tile_sample

                variant = _first_nonempty(row, ["variant", "mode", "kind"]).strip().lower()
                if not variant and tile_variant:
                    variant = tile_variant

                if sample_id:
                    by_sample[(sample_id, int(row_off), int(col_off))] = entry
                    if variant:
                        by_sample_variant[(sample_id, variant, int(row_off), int(col_off))] = entry

                loaded += 1

    return by_full_path, by_raster_name, by_sample_variant, by_sample


def _lookup_seed(raster_path: Path, row_off: int, col_off: int, by_full_path, by_raster_name, by_sample_variant, by_sample):
    full_key = (_norm_path(raster_path), int(row_off), int(col_off))
    entry = by_full_path.get(full_key)
    if entry is not None:
        return entry

    name_key = (raster_path.name, int(row_off), int(col_off))
    entry = by_raster_name.get(name_key)
    if entry is not None:
        return entry

    sample_id, variant = _sample_variant_from_raster_path(raster_path)
    if sample_id and variant:
        sv_key = (sample_id, str(variant).lower(), int(row_off), int(col_off))
        entry = by_sample_variant.get(sv_key)
        if entry is not None:
            return entry

    if sample_id:
        s_key = (sample_id, int(row_off), int(col_off))
        entry = by_sample.get(s_key)
        if entry is not None:
            return entry

    return None


def _write_queue_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile_id",
        "output_stem",
        "raster_path",
        "row_off",
        "col_off",
        "chunk_size",
        "init_mask",
        "init_cam",
        "hotspot_path",
        "preseg_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    


def main():
    app = LabelAllRastersGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
