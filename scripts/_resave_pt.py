"""One-shot script: reload broken .pt and re-save without wrapper_params."""

import sys

sys.path.insert(0, "/workspace/scripts")

# Import compare_classifiers so _build_deep_cnn_attn is in pickle's namespace
import compare_classifiers  # noqa

# The .pt was pickled with __main__._build_deep_cnn_attn — inject into __main__
import __main__

__main__._build_deep_cnn_attn = compare_classifiers._build_deep_cnn_attn

import torch
from pathlib import Path

pt_path = Path("/workspace/output/tile_labels/ensemble_model.pt")
print("Loading existing .pt ...", flush=True)
ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
print("Keys found:", list(ckpt.keys()), flush=True)

clean = {
    "state_dict": ckpt["state_dict"],
    "build_fn_name": ckpt.get("build_fn_name", "_build_deep_cnn_attn"),
    "meta": ckpt.get("meta", {}),
}
torch.save(clean, pt_path)
print(f"Re-saved clean .pt -> {pt_path}  ({pt_path.stat().st_size // 1024} KB)", flush=True)
