#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import label_tiles as lt
import torch
import rasterio
from rasterio.windows import Window

MODEL = "output/tile_labels/ensemble_model.pt"
CHM = "chm_max_hag"
# pick a sample tile from debug dir
D = Path("output/heatmap_debug").iterdir()
sample = next(D)
meta = sample / "meta.json"
if not meta.exists():
    print("no meta")
    sys.exit(1)
import json

m = json.loads(meta.read_text())
raster = m["raster"]
row_off = m["row_off"]
col_off = m["col_off"]
print("sample", sample.name)
# load model
pred = lt.CNNPredictor()
pred.load_from_disk(Path(MODEL))
net = pred._net
print("net type", type(net))
# find last conv
last_conv = None
for mo in net.modules():
    import torch.nn as nn

    if isinstance(mo, nn.Conv2d):
        last_conv = mo
print("last conv:", last_conv)
# read raw tile
with rasterio.open(Path(CHM) / raster) as src:
    raw = src.read(
        1, window=Window(col_off, row_off, 128, 128), boundless=True, fill_value=0
    ).astype("float32")
# run forward hook
feat = []


def fh(_, __, out):
    feat.append(out)


fhh = last_conv.register_forward_hook(fh)

t = torch.from_numpy(raw.copy()).float().unsqueeze(0).unsqueeze(0).to(pred._device)
t.requires_grad_(True)
logits = net(t)
score = torch.softmax(logits, dim=1)[0, 1]
print("score", float(score))
print("feat captured?", len(feat))
if feat:
    print("feat[0] shape", feat[0].shape, "requires_grad", feat[0].requires_grad)
try:
    grads = torch.autograd.grad(score, feat[0], retain_graph=False)[0]
    print(
        "grads shape",
        grads.shape,
        "nan?",
        torch.isnan(grads).any().item(),
        "norm",
        grads.abs().sum().item(),
    )
except Exception as e:
    print("grad error", e)
finally:
    fhh.remove()
