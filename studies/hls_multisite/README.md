# Two-site satellite-derived bathymetry study: Anza and Ras El Ma

This repository contains the scripts, input data, processed samples, model outputs, and figures used for a two-site comparison of author-designed convolutional neural network (CNN) variants and empirical baselines for shallow-water bathymetry from NASA/USGS Harmonized Landsat and Sentinel-2 (HLS) surface reflectance.

## Study design

The analysis evaluates each coast independently with three buffered alongshore spatial folds. A model is trained, tuned, and tested within the same site. No cross-site model transfer is tested, and elevations from the two sites are never pooled.

| Site | Survey date | HLS acquisition | HLS tile | Local vertical reference | Retained 30 m cells |
|---|---|---|---|---|---:|
| Anza, Agadir (Atlantic coast) | 10 Nov 2022 | 10 Nov 2022 | T29RMP | Hydrographic zero of the Port of Agadir; confirmed by the survey-data owner | 1,373 |
| Ras El Ma / Ras Kebdana (Mediterranean coast) | 8 Aug 2024 | 7 Aug 2024 | T30SWD | Hydrographic zero of the Port of Ras Kebdana; identified in the survey-plan cartouche | 3,817 |

The local port zeros differ. The Ras plan's geodetic `ΔZ = −47 m` parameter is not a levelling connection between the Port of Ras Kebdana and Port of Agadir zeros and was not applied to survey elevations. All z values are in metres relative to the corresponding local reference. The reported metrics are errors against aggregated survey labels, not independent hydrographic accuracy estimates. Neither survey has a documented sounding-time/tide correction or per-point vertical uncertainty. Ras imagery precedes its survey by one day.

## Data and source products

- **Anza survey:** `data/raw/survey/bathy_points.txt`.
- **Ras El Ma survey:** `data/raw/survey/ras_bathy_points.csv`.
- **Site metadata and datum provenance:** `data/raw/survey/site_metadata.json`.
- **Satellite imagery:** HLS S30 Version 2.0, 30 m grid, bands B02 (blue), B03 (green), B04 (red), B08 (near infrared) and Fmask. Product collection DOI: [10.5067/HLS/HLSS30.002](https://doi.org/10.5067/HLS/HLSS30.002). Acquisition and granule provenance are recorded under `data/raw/sentinel2/`.
- **Quality masking:** HLS v2.0 Fmask fill, cloud, adjacent-cloud/shadow, shadow and snow/ice flags are excluded; the water bit is retained as a diagnostic. See the [HLS v2.0 User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).

The HLS input grid is 30 m; it does not preserve Sentinel-2's native 10 m sampling. Survey coordinates are transformed to the HLS grid, survey elevations are aggregated within each 30 m cell, and one 9 × 9 × 4 patch is used per retained center. Mixed-sign shoreline cells and patches containing invalid quality flags or reflectance fill are removed.

## Models and spatial evaluation

The first author designed five CNN variants: `CNN`, `RatioCNN`, `CNN_BiLSTM`, `DenseCNN`, and `DepthBinCNN`. The baseline model is named **CNN**; the label does not contain “SDB”. The comparison also includes a training-zone-calibrated Stumpf-style blue–green log-ratio regression and a random forest using center and neighborhood spectral features.

Three alongshore regions are separated by a patch-width buffer and rotated through training, validation, and test roles. Each retained center receives one out-of-fold prediction. Metrics include RMSE, MAE, mean bias and R²; depth-band and fold-level summaries are descriptive. Seven model implementations are assessed (five CNN variants and two baselines).

## Current within-site results

| Site | Best pooled RMSE | RMSE | Best pooled MAE | MAE |
|---|---|---:|---|---:|
| Anza / Agadir | Stumpf-style log-ratio | 2.788 m | CNN–BiLSTM | 2.253 m |
| Ras El Ma / Ras Kebdana | RatioCNN | 0.500 m | CNN | 0.388 m |

These are site-local results under the current three-zone design. Differences in score between sites must not be interpreted as absolute vertical agreement because the datums, survey support, and acquisition timing are distinct. Per-model summaries and fold-level results are stored in `results/`.

## Scripts

- `code/prepare_hls_data.py` — prepares Anza pixel labels and HLS patches.
- `code/prepare_ras_hls_data.py` — deduplicates Ras survey records and prepares HLS patches.
- `code/models.py` — defines the five CNN variants.
- `code/train_models.py` — fits the CNNs and two baselines across spatial folds.
- `code/evaluate_results.py` — creates out-of-fold predictions and site metrics.
- `code/compare_site_results.py` — builds a site-wise summary and normalized-skill figure without pooling local datums.
- `code/make_figures.py` — creates detailed single-site maps and evaluation figures.
- `code/make_advanced_figures.py` — creates the combined two-site workflow, maps, quality-control and residual-diagnostic figures.
- `code/requirements.txt` — pinned Python package requirements.

## Re-running the analysis

Use Python 3.11 and install the listed packages in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
```

Prepare both site datasets:

```powershell
.\.venv\Scripts\python.exe code\prepare_hls_data.py
.\.venv\Scripts\python.exe code\prepare_ras_hls_data.py
```

The fold predictions and trained weights used for this release are included in `results/`. To retrain the Anza models:

```powershell
.\.venv\Scripts\python.exe code\train_models.py --models all --max-epochs 80 --patience 15 --overwrite
.\.venv\Scripts\python.exe code\evaluate_results.py --site "Anza, Agadir, Morocco" --vertical-reference "Hydrographic zero of the Port of Agadir (confirmed by the survey-data owner)"
.\.venv\Scripts\python.exe code\make_figures.py
```

For Ras El Ma, pass the processed sample and result directories explicitly:

```powershell
.\.venv\Scripts\python.exe code\train_models.py --models all --max-epochs 80 --patience 15 --overwrite --data-path data\processed\ras_el_ma_2024-08-08\samples.npz --results-dir results\hls_ras_2024-08-07\folds --site "Ras El Ma / Ras Kebdana, Morocco" --vertical-reference "Hydrographic zero of the Port of Ras Kebdana (survey-plan cartouche)"
.\.venv\Scripts\python.exe code\evaluate_results.py --data-path data\processed\ras_el_ma_2024-08-08\samples.npz --results-dir results\hls_ras_2024-08-07 --site "Ras El Ma / Ras Kebdana, Morocco" --vertical-reference "Hydrographic zero of the Port of Ras Kebdana (survey-plan cartouche)"
.\.venv\Scripts\python.exe code\make_figures.py --data-path data\processed\ras_el_ma_2024-08-08\samples.npz --results-dir results\hls_ras_2024-08-07 --scene-dir data\raw\sentinel2\HLS_S30_2024-08-07_RasElMa\T30SWD --product HLS.S30.T30SWD.2024220T105031.v2.0 --figures-dir figures\ras_el_ma --utm-zone 30N --scene-date "07 Aug 2024"
.\.venv\Scripts\python.exe code\compare_site_results.py
.\.venv\Scripts\python.exe code\make_advanced_figures.py
```

## Output folders

- `data/processed/` — one HLS patch set per site and a data-preparation summary.
- `results/` — per-fold model weights, held-out predictions, evaluation summaries and cross-site comparison table.
- `figures/` — 300–450 dpi PNG and vector PDF maps, model diagrams and diagnostics. Inter-model spread figures are descriptive disagreement, not calibrated prediction uncertainty.

The scripts use the saved HLS and survey inputs and do not download products during execution. Please cite the HLS Version 2.0 dataset using its DOI when reusing these inputs.

### Literature and figure set

The manuscript bibliography contains 49 DOI-verified topical publications from 2020–2026 plus four foundational references. Ten main figures and one supplementary model-disagreement figure are generated by `code/make_advanced_figures.py`; maps show held-out cells only, use site-specific colour scales where the local vertical zeros differ, and do not interpolate continuous bathymetric surfaces.

Run repository commands from this study directory. The saved fold predictions and model weights are included, so the published scores can be inspected without retraining.
