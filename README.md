# Two-site satellite-derived bathymetry study

This release contains the inputs, scripts, saved outputs, and publication figures for **Spatially Blocked Comparison of CNN Variants and Baseline Models for Satellite-Derived Bathymetry at Two Moroccan Coasts**.

The study compares five CNN variants designed by the first author with a training-zone-calibrated Stumpf-style blue–green log-ratio regression and a random forest. NASA/USGS Harmonized Landsat and Sentinel-2 (HLS) S30 Version 2.0 imagery at 30 m is paired with local bathymetric surveys at Anza, Agadir, and Ras El Ma, near Ras Kebdana.

## Study sites and references

| Site | Survey date | HLS acquisition | Tile | Local vertical reference | Retained cells |
|---|---|---|---|---|---:|
| Anza, Agadir (Atlantic coast) | 10 Nov 2022 | 10 Nov 2022 | T29RMP | Hydrographic zero of the Port of Agadir | 1,373 |
| Ras El Ma / Ras Kebdana (Mediterranean coast) | 8 Aug 2024 | 7 Aug 2024 | T30SWD | Hydrographic zero of the Port of Ras Kebdana | 3,817 |

The z values are in metres relative to each site's own local hydrographic zero. The sites are evaluated independently; no inter-site vertical offset is applied. The experiment does not test cross-site transfer or establish independent hydrographic accuracy. No sounding-time water-level correction or point-level vertical uncertainty was available in the released survey records. Ras El Ma imagery precedes the survey by one day.

## Inputs and processing

- `data/raw/survey/` contains the source bathymetric data and site metadata.
- `data/raw/sentinel2/` contains the HLS S30 bands B02 (blue), B03 (green), B04 (red), B08 (near infrared), and Fmask used for each site. Product identifiers and acquisition dates are listed in `scene_manifest.json`.
- `data/processed/` contains retained sample arrays and dataset summaries.
- HLS reflectance values are scaled by 0.0001. The HLS grid is 30 m; it does not preserve Sentinel-2's native 10 m sampling.
- Survey elevations are aggregated by 30 m HLS cell. Each retained centre has one 9 × 9 × 4 reflectance patch. Invalid Fmask/reflectance windows and cells failing survey-support criteria are excluded.

HLS S30 Version 2.0 is distributed by NASA/USGS LP DAAC. Cite the collection using [DOI 10.5067/HLS/HLSS30.002](https://doi.org/10.5067/HLS/HLSS30.002), acknowledge NASA/USGS LP DAAC, and follow the [HLS v2.0 User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf). HLS products are not relicensed by this repository.

## Models and evaluation

The five study-designed CNN variants are `CNN`, `RatioCNN`, `CNN_BiLSTM`, `DenseCNN`, and `DepthBinCNN`. The comparison baselines are a Stumpf-style log-ratio regression and a random forest. The conventional baseline model is named **CNN**; “SDB” describes the application and is not part of that model's name.

Three buffered alongshore zones rotate through training, validation, and test roles independently within each site. Each retained centre receives one out-of-fold prediction. Metrics include RMSE, MAE, mean bias, and R². The two sites use separate local vertical references and are not pooled as a common elevation scale.

| Site | Lowest-RMSE method | RMSE | Lowest-MAE method | MAE |
|---|---|---:|---|---:|
| Anza / Agadir | Stumpf-style log-ratio | 2.788 m | CNN–BiLSTM | 2.253 m |
| Ras El Ma / Ras Kebdana | RatioCNN | 0.500 m | CNN | 0.388 m |

These scores are residuals against aggregated local survey labels. They are not independent hydrographic accuracy estimates.

## Figures

High-resolution PNG and vector PDF versions are in `figures/rse_14p/`:

1. Study-site imagery, gridded survey depth, and sounding support.
2. Pixel-to-prediction illustration using the actual HLS patch, four-band pixel cube, survey neighbourhood, and saved centre estimate.
3. Buffered spatial zones and the rotating training, validation, and test roles.
4. Out-of-fold agreement density and residual trends across depth.
5. Two-dimensional survey and held-out depth maps with supported 2 m isobaths.
6. Orthographic three-dimensional survey and held-out depth surfaces.
7. Supplementary Figure S1: spatial model disagreement.
8. Supplementary Figure S2: spatial pattern of held-out residuals.

In Figures 5 and 6, interpolated displays are restricted to supported triangles between retained cell centres; triangles with edges longer than 45 m are masked. No surface is extrapolated beyond that support. The displayed surfaces do not add predictions at unsampled cells. Each site keeps its own depth scale.

## Repository contents

- `code/prepare_hls_data.py` and `code/prepare_ras_hls_data.py`: prepare samples from the two survey and HLS inputs.
- `code/models.py`: defines the five CNN variants.
- `code/train_models.py`: fits the CNNs and baseline models across spatial folds.
- `code/evaluate_results.py`: creates site-specific out-of-fold predictions and metrics.
- `code/compare_site_results.py`: summarizes site-wise results without pooling local vertical references.
- `code/make_advanced_figures.py`: creates Figures 1–6 and Supplementary Figures S1–S2 from the saved inputs and predictions.
- `data/`: raw inputs and processed sample arrays.
- `results/`: saved model weights, fold predictions, metric tables, and evaluation summaries.
- `figures/rse_14p/`: publication figures in PNG and PDF formats.

## Regenerating the analysis outputs

Use Python 3.11 and install the packages listed in `code/requirements.txt`. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\requirements.txt
python code\compare_site_results.py
python code\make_advanced_figures.py
```

The released processed arrays, fold predictions, and saved weights are included. The two preparation scripts rebuild the processed arrays from the included input data. `train_models.py` can retrain the models; use its `--help` option to view the available settings. Figure generation uses the released out-of-fold predictions and does not retrain the models.

## Data use and attribution

Code is released under the MIT License in `LICENSE`. Author-provided survey inputs, processed arrays, model outputs, and figures are released under Creative Commons Attribution 4.0 International (CC BY 4.0), subject to the authors' rights in those materials. HLS S30 Version 2.0 imagery remains subject to NASA/USGS LP DAAC terms and is not relicensed here. Give appropriate credit to the study authors and NASA/USGS LP DAAC, link to the relevant license, and indicate changes when redistributing or adapting these materials.
