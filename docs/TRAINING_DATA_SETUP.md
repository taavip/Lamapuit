# Training Data Setup Guide

## 📊 CHM Rasters for Training Dataset Creation

This project includes CHM (Canopy Height Model) rasters needed to create YOLO training datasets.

---

## 🗂️ Data Organization

### Small Training CHM Files (Included via Git LFS)

**Location**: `visuals/chm_max_hag/`

**Included in repository** (via Git LFS):
- `*_tava_*.tif` - "Tava" (normal) survey CHM files (10-16 MB each)
- `464663_*.tif` - Small test area files (2-16 MB)
- `441643_*.tif` - Small test area files (~9 MB)

**Total in Git LFS**: ~20-25 files, **~250-350 MB**

These files are:
- **Resolution**: 20cm per pixel (0.2m)
- **Size**: 10-16 MB each (manageable with Git LFS)
- **Purpose**: Creating training datasets for CDW detection
- **CRS**: EPSG:3301 (Estonian Coordinate System)

---

### Large CHM Files (Excluded - Zenodo)

**NOT included in repository** (>50 MB each):
- `*_madal_*.tif` - "Madal" (low/wetland) survey files (20-70 MB each)
- `406455_*_10cm.tif` - High-resolution 10cm files (>70 MB)
- Other large area tiles

**Total excluded**: ~35 files, **~1.5 GB**

**Download from Zenodo**: [DOI: TO_BE_ASSIGNED]

---

## 🚀 Setup Git LFS (One-time Setup)

### Step 1: Install Git LFS

**Windows**:
```bash
# Using Chocolatey
choco install git-lfs

# Or download from: https://git-lfs.github.com/
```

**Linux/Mac**:
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Mac (Homebrew)
brew install git-lfs
```

### Step 2: Initialize Git LFS in Repository

```bash
cd C:\temp\Lamapuit
git lfs install
```

### Step 3: Verify LFS Tracking

```bash
# Check what's tracked
git lfs track

# Should show:
# examples/data/*.tif
# visuals/chm_max_hag/*_tava_*.tif
```

---

## 📥 Using the Training Data

### Option A: Clone with LFS (Automatic)

Users with Git LFS installed will automatically download all training CHM files:

```bash
git clone https://github.com/taavip/cdw-detect.git
cd cdw-detect

# Files automatically downloaded via LFS
ls visuals/chm_max_hag/*.tif
```

### Option B: Clone without LFS (Skip Large Files)

Users without Git LFS can clone faster (skips CHM files):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/taavip/cdw-detect.git
cd cdw-detect

# CHM files are LFS pointers (~100 bytes each)
# Download specific files later if needed:
git lfs pull --include="visuals/chm_max_hag/465663_*.tif"
```

---

## 🎯 Creating Training Datasets

### Using Included CHM Files

```python
from cdw_detect import YOLODataPreparer

# Use one of the included training CHM files
preparer = YOLODataPreparer(
    output_dir='my_training_data',
    buffer_width=0.5,
)

stats = preparer.prepare(
    chm_path='visuals/chm_max_hag/465663_2023_tava_chm_max_hag_20cm.tif',
    labels_path='lamapuit.gpkg',
)
```

### Using Full Dataset from Zenodo

For the complete 1.87 GB dataset:

1. **Download from Zenodo**: [DOI: TO_BE_ASSIGNED]
2. **Extract** to `visuals/chm_max_hag/`
3. **Use in training** as above

---

## 📦 Git LFS Limits

### GitHub Free Tier
- **Storage**: 1 GB
- **Bandwidth**: 1 GB/month
- **File count**: Unlimited

### Your Project Usage
- **Included CHM files**: ~250-350 MB
- **Sample data**: ~1 MB
- **Total LFS**: ~350-400 MB
- **✅ Fits in free tier!**

### If You Exceed Limits

**Option 1**: GitHub LFS Data Packs
- $5/month for 50 GB storage + 50 GB bandwidth

**Option 2**: Move all CHM to Zenodo
- Update documentation with Zenodo links
- Users download separately

**Option 3**: Selective inclusion
- Keep only 5-10 smallest CHM files in Git LFS
- Rest on Zenodo

---

## 📋 File Naming Convention

CHM files follow this pattern:
```
{tile_id}_{year}_{survey_type}_chm_max_hag_{resolution}.tif

Examples:
- 465663_2023_tava_chm_max_hag_20cm.tif
  └─ Tile 465663, year 2023, normal survey, 20cm resolution
  
- 470664_2022_madal_chm_max_hag_20cm.tif
  └─ Tile 470664, year 2022, wetland survey, 20cm resolution
```

**Survey types**:
- `tava` - Normal forest survey (typically 10-16 MB)
- `madal` - Low/wetland survey (typically 20-70 MB)
- `mets` - Forest survey (typically 4-5 MB)

---

## 🔧 Technical Details

### CHM File Specifications

- **Format**: GeoTIFF with compression
- **Resolution**: 20cm (0.2m) or 10cm (0.1m)
- **Values**: Height Above Ground in meters
- **NoData**: -9999 or NaN
- **CRS**: EPSG:3301 (Estonian Coordinate System 1997)
- **Compression**: LZW or DEFLATE

### Processing from LAZ

These CHM files were created using `scripts/process_laz_to_chm.py`:

```bash
python scripts/process_laz_to_chm.py \
  --input points.laz \
  --output chm.tif \
  --resolution 0.2 \
  --method max
```

---

## 📊 Storage Summary

| Location | Files | Size | Storage Method |
|----------|-------|------|----------------|
| `examples/data/` | 1 | ~1 MB | Git (regular) |
| `visuals/chm_max_hag/` (small) | ~25 | ~350 MB | Git LFS |
| `visuals/chm_max_hag/` (large) | ~35 | ~1.5 GB | Zenodo |
| `merged041225.tif` | 1 | 280 MB | Zenodo |
| **Total in Git** | ~26 | **~350 MB** | ✅ Free tier OK |
| **Total external** | ~36 | **~1.8 GB** | Zenodo |

---

## 🎓 For Contributors

### Adding New CHM Files

**If file is <20 MB**:
```bash
# File will be tracked by Git LFS automatically
git add visuals/chm_max_hag/new_file_tava_chm_max_hag_20cm.tif
git commit -m "Add new training CHM file"
```

**If file is >50 MB**:
1. Don't add to Git
2. Upload to Zenodo dataset
3. Update this documentation with new file info

### Testing LFS

```bash
# Check LFS status
git lfs status

# See what will be uploaded
git lfs ls-files

# Check bandwidth usage
git lfs env
```

---

## ❓ FAQ

**Q: Why not put all CHM files on Zenodo?**
A: Including ~20-25 smaller files (350 MB) in Git LFS makes it easier for users to get started immediately without extra download steps.

**Q: Can I use the project without downloading CHM files?**
A: Yes! The sample data in `examples/data/` is enough to test the pipeline. Use `GIT_LFS_SKIP_SMUDGE=1` when cloning.

**Q: Which CHM file should I use for training?**
A: Start with `465663_2023_tava_chm_max_hag_20cm.tif` - it's a good size (~12 MB) with diverse features.

**Q: How do I download just one LFS file?**
A: 
```bash
git lfs pull --include="visuals/chm_max_hag/465663_*.tif"
```

---

## 📞 Support

For issues with Git LFS or data access:
- **Git LFS docs**: https://git-lfs.github.com/
- **GitHub Issues**: https://github.com/taavip/cdw-detect/issues
- **Zenodo dataset**: [Link when available]
