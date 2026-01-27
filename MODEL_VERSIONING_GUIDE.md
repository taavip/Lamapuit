# Model Versioning Guide for GitHub

## 🎯 Best Practices for ML Model Versioning

### Why Not Commit Large Models to Git?
- ❌ Model files (5-12 MB) bloat repository history
- ❌ Git is optimized for text, not binary files
- ❌ Every version increases clone size permanently
- ✅ Use **Git LFS** or **GitHub Releases** instead

---

## 📦 Recommended Approach: GitHub Releases + Semantic Versioning

### Semantic Versioning (SemVer)

Use **MAJOR.MINOR.PATCH** format:

```
v1.0.0  - Initial release
v1.0.1  - Bug fix (small improvements, same architecture)
v1.1.0  - Minor update (fine-tuning, minor architecture changes)
v2.0.0  - Major update (new architecture, breaking changes)
```

**Examples for CDW Detection**:
- `v1.0.0` - Initial YOLO11n-seg model (50 epochs)
- `v1.0.1` - Fine-tuned +10 epochs (this version)
- `v1.1.0` - Trained with additional data
- `v2.0.0` - Switch to YOLO11m-seg (larger model)

---

## 🚀 Step-by-Step: Release Your Model

### Step 1: Fine-Tune and Tag

```bash
# Fine-tune model (creates versioned output)
python scripts/finetune_model.py \
  --model runs/cdw_detect/cdw_lamapuit_robust/weights/best.pt \
  --data yolo_dataset_lamapuit_robust/dataset.yaml \
  --epochs 10 \
  --name finetune_v1.0.1
```

Output: `runs/cdw_detect/finetune_v1.0.1/weights/best.pt`

### Step 2: Copy Model to Releases Folder

Create a staging area for releases:

```bash
# Create releases directory
mkdir -p models/releases

# Copy with version name
cp runs/cdw_detect/finetune_v1.0.1/weights/best.pt \
   models/releases/cdw_detect_v1.0.1.pt

# Also copy version info
cp runs/cdw_detect/finetune_v1.0.1/version_info.yaml \
   models/releases/cdw_detect_v1.0.1_info.yaml
```

### Step 3: Create Git Tag

```bash
# Create annotated tag
git tag -a v1.0.1 -m "CDW Detection v1.0.1 - Fine-tuned +10 epochs

Improvements:
- Fine-tuned from v1.0.0 with 10 additional epochs
- Improved mAP50(B) from 11.35% to [NEW_VALUE]%
- Lower learning rate (0.001) for stability

Model: cdw_detect_v1.0.1.pt (5.72 MB)
Architecture: YOLO11n-seg (2.8M params)
Dataset: yolo_dataset_lamapuit_robust (448 images)"

# Push tag
git push origin v1.0.1
```

### Step 4: Create GitHub Release

**Option A: GitHub CLI** (Recommended)

```bash
gh release create v1.0.1 \
  models/releases/cdw_detect_v1.0.1.pt \
  models/releases/cdw_detect_v1.0.1_info.yaml \
  --title "CDW Detection v1.0.1" \
  --notes "## CDW Detection Model v1.0.1

### Improvements
- Fine-tuned from v1.0.0 with 10 additional epochs
- Lower learning rate (0.001) for stability
- Improved performance metrics

### Model Details
- **Architecture**: YOLO11n-seg
- **Parameters**: 2.8M
- **Input Size**: 640x640
- **Dataset**: 448 training images
- **Buffer Width**: 0.5m (1m total CDW width)

### Metrics
- Box mAP50: [UPDATE]%
- Mask mAP50: [UPDATE]%

### Download & Usage
\`\`\`bash
# Download model
wget https://github.com/taavip/cdw-detect/releases/download/v1.0.1/cdw_detect_v1.0.1.pt

# Run detection
python scripts/run_detection.py \\
  --model cdw_detect_v1.0.1.pt \\
  --raster your_chm.tif \\
  --output detections.gpkg
\`\`\`

### Changes from v1.0.0
- Fine-tuned with 10 additional epochs
- Learning rate: 0.001 (reduced from 0.01)
- Optimizer: SGD with momentum 0.937

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.4.0+

See [README.md](https://github.com/taavip/cdw-detect#installation) for full installation."
```

**Option B: GitHub Web UI**

1. Go to repository → Releases → "Draft a new release"
2. Choose tag: `v1.0.1`
3. Release title: `CDW Detection v1.0.1`
4. Upload files: Drag `cdw_detect_v1.0.1.pt` and `version_info.yaml`
5. Add release notes (see template above)
6. Publish release

---

## 📋 Model Registry (Track All Versions)

Create `models/MODEL_REGISTRY.md`:

```markdown
# CDW Detection Model Registry

| Version | Date | mAP50(B) | mAP50(M) | Epochs | Notes | Download |
|---------|------|----------|----------|--------|-------|----------|
| v1.0.0 | 2026-01-25 | 11.35% | 8.89% | 50 | Initial release | [Link](https://github.com/user/cdw-detect/releases/v1.0.0) |
| v1.0.1 | 2026-01-25 | [TBD]% | [TBD]% | 60 | Fine-tuned +10 epochs | [Link](https://github.com/user/cdw-detect/releases/v1.0.1) |

## Version Details

### v1.0.1 (Latest)
- **Base**: v1.0.0
- **Training**: +10 epochs fine-tuning
- **Learning Rate**: 0.001 (fine-tuning LR)
- **Improvements**: [Describe improvements]

### v1.0.0
- **Architecture**: YOLO11n-seg
- **Training**: 50 epochs from scratch
- **Dataset**: 448 images with 30% nodata augmentation
- **Buffer**: 0.5m (1m total width)
```

---

## 🔧 Alternative: Git LFS (Large File Storage)

If you prefer to track models in Git:

### Setup Git LFS

```bash
# Install Git LFS (Windows)
# Download from: https://git-lfs.github.com/

# Initialize in repository
git lfs install

# Track model files
git lfs track "models/releases/*.pt"
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model files"
```

### Commit Model with LFS

```bash
# Copy model
mkdir -p models/releases
cp runs/cdw_detect/finetune_v1.0.1/weights/best.pt \
   models/releases/cdw_detect_v1.0.1.pt

# Add and commit
git add models/releases/cdw_detect_v1.0.1.pt
git commit -m "Add model v1.0.1 (fine-tuned +10 epochs)"
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin main --tags
```

**Git LFS Limits** (GitHub free tier):
- Storage: 1 GB
- Bandwidth: 1 GB/month
- Cost: $5/month for 50 GB

---

## 📊 Model Metadata Format

Include `version_info.yaml` with each model:

```yaml
version: "1.0.1"
timestamp: "2026-01-25T16:30:00"
base_model: "runs/cdw_detect/cdw_lamapuit_robust/weights/best.pt"
architecture: "YOLO11n-seg"
parameters: 2800000
epochs_trained: 10
total_epochs: 60
dataset: "yolo_dataset_lamapuit_robust"
training_images: 448

hyperparameters:
  learning_rate: 0.001
  batch_size: 4
  image_size: 640
  optimizer: "SGD"
  momentum: 0.937

metrics:
  box_map50: 0.1135  # Update after training
  mask_map50: 0.0889  # Update after training
  box_map50_95: 0.0XXX
  mask_map50_95: 0.0XXX

training_config:
  buffer_width: 0.5
  nodata_augmentation: 0.3
  patience: 5
  device: "cpu"

notes: "Fine-tuned from v1.0.0 with 10 additional epochs using lower learning rate"
```

---

## 🎯 Recommended Workflow

```bash
# 1. Fine-tune model
python scripts/finetune_model.py \
  --model runs/cdw_detect/cdw_lamapuit_robust/weights/best.pt \
  --data yolo_dataset_lamapuit_robust/dataset.yaml \
  --epochs 10 \
  --name finetune_v1.0.1

# 2. Test model
python scripts/run_detection.py \
  --model runs/cdw_detect/finetune_v1.0.1/weights/best.pt \
  --raster merged041225.tif \
  --output test_v1.0.1.gpkg

# 3. Copy to releases folder
mkdir -p models/releases
cp runs/cdw_detect/finetune_v1.0.1/weights/best.pt models/releases/cdw_detect_v1.0.1.pt
cp runs/cdw_detect/finetune_v1.0.1/version_info.yaml models/releases/cdw_detect_v1.0.1_info.yaml

# 4. Update MODEL_REGISTRY.md with metrics

# 5. Create Git tag
git add models/MODEL_REGISTRY.md
git commit -m "Update model registry for v1.0.1"
git tag -a v1.0.1 -m "Release v1.0.1: Fine-tuned +10 epochs"
git push origin main --tags

# 6. Create GitHub Release
gh release create v1.0.1 \
  models/releases/cdw_detect_v1.0.1.pt \
  models/releases/cdw_detect_v1.0.1_info.yaml \
  --title "CDW Detection v1.0.1" \
  --notes-file models/releases/RELEASE_NOTES_v1.0.1.md
```

---

## 📁 Recommended Repository Structure

```
cdw-detect/
├── models/
│   ├── releases/                    # Staged for GitHub Releases
│   │   ├── cdw_detect_v1.0.0.pt    # Initial model
│   │   ├── cdw_detect_v1.0.1.pt    # Fine-tuned model
│   │   ├── cdw_detect_v1.0.1_info.yaml
│   │   └── RELEASE_NOTES_v1.0.1.md
│   ├── MODEL_REGISTRY.md            # Version tracking
│   └── README.md                    # Model documentation
├── runs/
│   └── cdw_detect/
│       ├── cdw_lamapuit_robust/     # v1.0.0 training
│       └── finetune_v1.0.1/         # v1.0.1 training
└── ...
```

**Do NOT commit `runs/` directory** - only copy final models to `models/releases/`

---

## ✅ Summary: Best Practices

1. ✅ Use **semantic versioning** (v1.0.0, v1.0.1, etc.)
2. ✅ Store models in **GitHub Releases** (not in Git)
3. ✅ Include **version_info.yaml** with metadata
4. ✅ Track versions in **MODEL_REGISTRY.md**
5. ✅ Create **Git tags** for each version
6. ✅ Write **detailed release notes** with metrics
7. ✅ Test model before releasing
8. ❌ Don't commit `runs/` directory
9. ❌ Don't commit model weights directly to Git (unless using LFS)
10. ✅ Stage models in `models/releases/` before releasing

This keeps your repository clean, models accessible, and versions well-documented!
