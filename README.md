# AIS + SAR Anchorage Vessel Activity Prediction (LA–Long Beach)

STATS 201 (Autumn 2026) term project pilot, Keyue Li (@Keyue-Li).

**Research question:** Can Sentinel-1 SAR detections improve short-term prediction of
anchorage vessel activity at the Ports of Los Angeles and Long Beach beyond historical AIS?

**Pilot result (negative, reported honestly):** on a chronological held-out test
(2025-07-01 to 2025-12-31), Ridge MAE is 3.57 (AIS-only) vs 3.61 (AIS+SAR) at h = 1 day,
and 3.68 vs 3.72 at h = 7 days. SAR features add no measurable improvement in this
region and period. See the proposal PDF for interpretation and limits.

## Data

The analysis panel is published on Hugging Face:

- Dataset: `DKU-STATS201-Autumn2026/<dataset-name>` <!-- TODO: 上传后替换为确切数据集名 -->
- Version used in the proposal: <!-- TODO: 填 commit/revision，例如 main @ <commit-hash> -->

Raw AIS/SAR pulls require a Global Fishing Watch API token and are NOT included
(GFW terms, CC BY-NC). Only the aggregated anchorage-day panel is redistributed.

## Environment

```bash
python -m pip install -r requirements.txt
```

Tested with: Python 3.11 (Anaconda), numpy 2.4.6, scipy 1.16.3, scikit-learn 1.7.2,
matplotlib 3.11.2, gfw-api-python-client 1.4.0.

## Reproduce the pilot

1. (Optional, token required) Pull raw data. Set your token first — never commit it:

   ```bash
   export GFW_API_TOKEN="<your token>"
   python scripts/pull_gfw_anchorages.py ais 2024-01-01 2025-12-31
   python scripts/pull_gfw_anchorages.py sar 2024-01-01 2025-12-31
   ```

   Skip this step if you use the released Hugging Face panel instead.

2. Run the analysis and model comparison:

   ```bash
   python scripts/analysis_anchorage.py
   ```

## Expected output / check

The script writes the model comparison tables to `data/` and figures to `figures/`.
You should reproduce these test-set numbers (Ridge):

| horizon | model | MAE | RMSE |
|---|---|---|---|
| 1 | Baseline (persistence) | 4.716 | 6.189 |
| 1 | Model A (Ridge, AIS only) | 3.572 | 4.612 |
| 1 | Model B (Ridge, AIS+SAR) | 3.610 | 4.618 |
| 7 | Baseline (persistence) | 4.927 | 6.304 |
| 7 | Model A (Ridge, AIS only) | 3.680 | 4.684 |
| 7 | Model B (Ridge, AIS+SAR) | 3.715 | 4.760 |

Sanity check: Model A beats the persistence baseline at both horizons; Model B does not
beat Model A. If your run differs beyond rounding, check the chronological split
(train ends 2025-06-30) first.

## Repository layout

- `scripts/pull_gfw_anchorages.py` — monthly-chunked GFW pulls for official anchorages F+G (33 CFR 110.214)
- `scripts/pull_gfw.py` — earlier bounding-box version (kept for reference)
- `scripts/analysis_anchorage.py` — panel construction, leak-free lag features, Ridge/HGB A/B comparison
- `data/` — small result tables (raw pulls excluded; see HF dataset)
- `figures/` — proposal figures

## Code revision

<!-- TODO: 提交后填 commit hash，例如 v1.0.0 或 main @ abc1234 -->
