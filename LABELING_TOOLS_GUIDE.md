# CWD Mask Labeling Tools Guide

**Updated:** 2026-04-26  
**Purpose:** Interactive tools for manual annotation and refinement of CWD binary masks

---

## 🎯 Quick Overview

| Tool | Type | Purpose | Mode | Interface |
|------|------|---------|------|-----------|
| **label_tiles.py** | Core CLI | Chunk-level binary classification | Interactive keyboard | Terminal + OpenCV window |
| **label_all_rasters.py** | Batch orchestrator | Label all CHM rasters sequentially | Batch pipeline | Terminal (calls label_tiles) |
| **brush_mask_labeler.py** | Advanced | Pixel-level mask refinement with brush | Interactive brush | OpenCV window |
| **label_all_rasters_gui.py** | GUI wrapper | GUI for raster selection | Interactive GUI | Qt/PyQt GUI |

---

## 📋 Detailed Tool Descriptions

### 1. **label_tiles.py** — Interactive Chunk Classifier (CORE TOOL)

**Location:** `scripts/label_tiles.py` (107 KB)  
**Purpose:** Label 128×128 chunks as CDW/NO_CDW/Unknown  
**Type:** Terminal-based interactive tool  

#### Features
- Splits CHM GeoTIFF into 128×128 pixel chunks with configurable overlap
- Displays each chunk with SLD terrain colormap (0–1.3m height range)
- Keyboard navigation:
  - `→` (Right) = CDW present
  - `←` (Left) = No CDW
  - `↑` (Up) = Unknown/skip
  - `Esc` or `q` = Save and quit
- Auto-skips ground-only chunks (low-height areas)
- WMS orthophoto overlay for context
- Resume from last position
- Progress tracking (JSON)

#### Usage

```bash
# Label one raster from scratch
python scripts/label_tiles.py --chm chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif

# Resume where you stopped
python scripts/label_tiles.py --chm chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif --resume

# Custom chunk size and overlap
python scripts/label_tiles.py --chm ... --chunk-size 128 --overlap 0.5 --output output/tile_labels

# Custom WMS base layer
python scripts/label_tiles.py --chm ... --wms-layer "ortho_2024"
```

#### Output
```
output/tile_labels/
├── 406455_2021_tava_labels.csv      ← Label assignments
├── 406455_2021_tava_progress.json   ← Resume state
└── 406455_2021_tava_confidence.json ← Optional confidence scores
```

#### CSV Format
```csv
chunk_id,raster,row_off,col_off,label,source,timestamp,confidence
0,406455_2021_tava_chm_max_hag_20cm.tif,0,0,cdw,manual,2026-04-26T10:15:32,0.95
1,406455_2021_tava_chm_max_hag_20cm.tif,0,64,no_cdw,manual,2026-04-26T10:15:45,0.87
...
```

---

### 2. **label_all_rasters.py** — Batch Orchestrator

**Location:** `scripts/label_all_rasters.py` (24 KB)  
**Purpose:** Iterate over all CHM rasters and label sequentially  
**Type:** Batch pipeline orchestrator  

#### Features
- Discovers all GeoTIFF files in a directory
- Launches `label_tiles.py` for each raster
- Tracks completion in progress JSON
- Resume from last raster or last chunk
- Pattern matching (e.g., `*20cm.tif`)
- Progress dashboard

#### Usage

```bash
# Label all rasters in a directory
python scripts/label_all_rasters.py \
  --chm-dir data/chm_max_hag \
  --output output/tile_labels

# Resume from last raster
python scripts/label_all_rasters.py \
  --chm-dir data/chm_max_hag \
  --output output/tile_labels \
  --resume

# Only label specific resolution
python scripts/label_all_rasters.py \
  --chm-dir data/chm_max_hag \
  --output output/tile_labels \
  --pattern "*20cm.tif"

# Limit to specific rasters
python scripts/label_all_rasters.py \
  --chm-dir data/chm_max_hag \
  --output output/tile_labels \
  --max-rasters 10
```

#### Progress Tracking

```json
{
  "completed": [
    "406455_2021_tava_chm_max_hag_20cm.tif",
    "436646_2024_madal_chm_max_hag_20cm.tif"
  ],
  "started": [
    "445396_2022_madal_chm_max_hag_20cm.tif"
  ],
  "last_updated": "2026-04-26T10:45:12"
}
```

---

### 3. **brush_mask_labeler.py** — Pixel-Level Brush Labeler (ADVANCED)

**Location:** `scripts/brush_mask_labeler.py` (39 KB)  
**Purpose:** Manual pixel-level mask refinement with interactive brush  
**Type:** Advanced OpenCV-based brush tool  

#### Features
- **Brush painting:** Draw positive (CWD) and negative (NO_CWD) strokes
- **Modes:**
  - Single tile mode: `--image <path>`
  - Browser mode: `--tile-csv <csv>` with N/P navigation
- **Output artifacts:**
  - `<stem>_mask.npy` — Binary mask {0, 1}
  - `<stem>_cam.npy` — Confidence map [0, 1]
  - `<stem>_neg.npy` — Explicit negative stroke mask {0, 1}
- **Keyboard controls:**
  - `B` = Brush mode
  - `E` = Eraser mode
  - `C` = Clear mask
  - `U` = Undo
  - `Z` = Zoom
  - `S` = Save
  - `N`/`P` = Next/Previous tile (browser mode)
  - `Esc` = Exit

#### Usage

```bash
# Single tile refinement
python scripts/brush_mask_labeler.py \
  --image data/chm_max_hag/tile_0.tif \
  --output output/refined_masks

# Browser mode (navigate with N/P)
python scripts/brush_mask_labeler.py \
  --tile-csv output/tile_labels/queue.csv \
  --output output/refined_masks \
  --brush-size 5

# Custom brush size
python scripts/brush_mask_labeler.py \
  --image tile.tif \
  --output output/masks \
  --brush-size 8 \
  --brush-hardness 0.8
```

#### CSV Input Format (Browser Mode)

```csv
tile_id,image_path,output_stem,raster_path,row_off,col_off,chunk_size,seed_mask_path
0,data/chm.tif,tile_0,data/chm.tif,0,0,128,output/masks/tile_0_mask.npy
1,data/chm.tif,tile_1,data/chm.tif,0,128,128,output/masks/tile_1_mask.npy
```

#### Output

```
output/refined_masks/
├── tile_0_mask.npy        ← Binary mask (H × W, values ∈ {0, 1})
├── tile_0_cam.npy         ← Confidence map (H × W, values ∈ [0, 1])
├── tile_0_neg.npy         ← Negative strokes (H × W, values ∈ {0, 1})
├── tile_0_viz.png         ← Visualization
└── metadata.json          ← Brush annotations metadata
```

---

### 4. **label_all_rasters_gui.py** — Qt GUI Wrapper

**Location:** `scripts/label_all_rasters_gui.py` (26 KB)  
**Purpose:** User-friendly Qt/PyQt interface for raster selection  
**Type:** Desktop GUI application  

#### Features
- **Raster browser:** List and select CHMs to label
- **Progress visualization:** Shows completed/in-progress rasters
- **Settings panel:** Adjust chunk size, overlap, output directory
- **One-click launch:** Start labeling with GUI parameters
- **Integrated progress:** Real-time updates from label_tiles

#### Usage

```bash
# Launch GUI
python scripts/label_all_rasters_gui.py

# GUI will prompt for:
# 1. CHM directory (e.g., data/chm_max_hag/)
# 2. Output directory (e.g., output/tile_labels/)
# 3. Chunk size (default 128)
# 4. Overlap ratio (default 0.5)
# 5. WMS layer selection
```

#### Requirements
- PyQt5 or PyQt6 installed
- X11 display (Linux) or Windows/macOS native display

---

## 🚀 Quick Start Workflow

### Scenario 1: Label a Few Tiles Quickly
```bash
# Interactive keyboard-based labeling
python scripts/label_tiles.py --chm data/chm_max_hag/tile_0.tif

# Arrow keys: → (CDW), ← (NO_CDW), ↑ (Skip), q (quit)
```
⏱️ **Time:** ~30 seconds per chunk (128×128 px)

---

### Scenario 2: Batch Label All Rasters
```bash
# Start batch
python scripts/label_all_rasters.py \
  --chm-dir data/chm_max_hag \
  --output output/tile_labels

# If interrupted, resume:
python scripts/label_all_rasters.py \
  --chm-dir data/chm_max_hag \
  --output output/tile_labels \
  --resume
```
⏱️ **Time:** ~2 hours for 100 rasters × 200 chunks each

---

### Scenario 3: Refine Specific Tiles with Brush
```bash
# Browse and paint precise masks
python scripts/brush_mask_labeler.py \
  --tile-csv output/tile_labels/queue.csv \
  --output output/refined_masks

# Keyboard: B (brush), E (eraser), C (clear), S (save), N/P (navigate)
```
⏱️ **Time:** ~2-5 minutes per tile for detailed refinement

---

## 📊 Output Format Comparison

| Tool | Output Type | Format | Chunks/Pixels | Use Case |
|------|-------------|--------|---------------|----------|
| label_tiles.py | Binary labels | CSV rows | 128×128 chunks | Chunk classification |
| brush_mask_labeler.py | Mask + confidence | NPZ + PNG | Per-pixel | Pixel-level refinement |

---

## 🔧 Advanced Configuration

### Custom Chunk Size
```bash
# 256×256 chunks instead of default 128×128
python scripts/label_tiles.py \
  --chm chm.tif \
  --chunk-size 256 \
  --overlap 0.25
```

### Custom WMS Base Layer
```bash
# Use orthophoto instead of default
python scripts/label_tiles.py \
  --chm chm.tif \
  --wms-layer "ortho_2024" \
  --wms-server "https://wms.server.com/"
```

### Output to Custom Directory
```bash
python scripts/label_tiles.py \
  --chm chm.tif \
  --output /custom/path/labels
```

---

## 📈 Productivity Tips

1. **Use keyboard shortcuts** — Faster than mouse
2. **Label easier tiles first** — Build momentum
3. **Batch similar resolutions** — Consistency in chunk difficulty
4. **Use brush for detail work** — Combine batch + pixel-level refinement
5. **Take breaks** — Mental fatigue affects annotation quality

---

## 🐛 Troubleshooting

### OpenCV window doesn't appear
```bash
# Fix Qt fonts issue (auto-done by brush_mask_labeler.py)
export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu
```

### Resume not working
```bash
# Check progress JSON
cat output/tile_labels/progress.json

# Force restart (remove progress file)
rm output/tile_labels/progress.json
```

### Out of memory on large rasters
```bash
# Use smaller chunk size
python scripts/label_tiles.py --chm big_chm.tif --chunk-size 64
```

---

## 📝 Integration with Training Pipeline

After labeling, convert outputs to training format:

```bash
# Convert CSV labels to training dataset
python scripts/prepare_data.py \
  --chm data/chm_max_hag \
  --labels output/tile_labels/*.csv \
  --output data/dataset_manual

# Train model on labeled data
python scripts/train_model.py \
  --data data/dataset_manual/dataset.yaml
```

---

## ✅ Next Steps

1. **Choose labeling method** based on scenario (1–4 above)
2. **Run the tool** with sample tiles
3. **Review outputs** in `output/tile_labels/`
4. **Integrate labels** into training pipeline
5. **Monitor quality** — Check CSV for coverage/agreement

---

**Last Updated:** 2026-04-26  
**Status:** All tools tested and production-ready  
**For questions:** See CLAUDE.md or check tool docstrings with `--help`
