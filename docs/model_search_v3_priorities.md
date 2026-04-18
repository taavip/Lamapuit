# Model Search V3 Priorities (2026-04-07)

## Requested New Models

- `maxvit_small`
- `convnextv2_small`
- `eva02_small`

## Practical Availability

In current `lamapuit:gpu` image:

- `timm` is missing, so timm-only models are unavailable until image rebuild.
- Added `timm>=0.9.16,<1.1` to `docker/requirements.gpu.txt`.

Validation with temporary `timm` install in container:

- `maxvit_small`: builds, ~68.2M params
- `convnextv2_small`: builds, ~49.5M params
- `eva02_small`: builds, ~21.5M params

## Priority Order

1. `convnext_small` (strong and stable v2 winner family)
2. `convnextv2_small` (closest modern upgrade to current winner family)
3. `deep_cnn_attn_dropout_tuned` (strong non-ConvNeXt diversity)
4. `efficientnet_b2` (fast/strong pretrained baseline)
5. `eva02_small` (good model size, promising transfer)
6. `maxvit_small` (highest compute cost; run after above)

Removed from V3 pool:

- `convnext_tiny` (explicitly removed to avoid redundant retest)
- `deep_cnn_attn_headwide` (lower-performing deep variant in prior v2 results)

## Suggested Stage 1 Pool (compact)

- `convnext_small`
- `convnextv2_small`
- `deep_cnn_attn_dropout_tuned`
- `efficientnet_b2`
- `eva02_small`
- `maxvit_small`

## Suggested Stage 2 Variations (slow-path reduction)

Keep only 2 variations per model:

- `mixup_swa`
- `tta`

Avoid full matrix (`full`, `focal`, `mixup_swa`, `tta`) unless a model proves top-3 in Stage 1.

## Leakage Findings

V3 update (2026-04-10):

- Split mode: spatial blocks with place-level year lock
- Same place across years is assigned to only one side (train or test)
- Train/test split target: ~80/20
- Train/test fence buffer: 26m (one full tile)
- CV: spatial-block GroupKFold with block-size probe (candidate meters)

Audit on prepared v2 split (`output/model_search_v2/prepared`):

- Exact key overlap (train vs test): `0`
- Shared 30m spatial-fence cells before filtering: `1805`
- Shared 30m spatial-fence cells after strict fence filtering: `0`

Audit on prepared v3 split (`output/model_search_v3/prepared`):

- Exact key overlap (train vs test): `0`
- Shared 30m spatial-fence cells before filtering: `1825`
- Shared 30m spatial-fence cells after strict fence filtering: `0`

Interpretation:

- Key-level split already avoids direct duplicates.
- Cross-year / same-place leakage exists without coordinate fencing.
- New spatial-fence logic removes this leakage class.

## Run Entry Point

Use:

`python scripts/model_search_v3/model_search_v3.py`

Recommended first run (prepare only):

`python scripts/model_search_v3/model_search_v3.py --prepare-only`
