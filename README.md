# Two-Coast Satellite-Derived Bathymetry in Morocco

This repository contains the analysis inputs, processing and model scripts, saved predictions, evaluation tables, and publication figures for the Anza (Atlantic) and Ras El Ma (Mediterranean) study sites in Morocco.

The study evaluates five author-developed CNN models, a Stumpf-style blue/green log-ratio baseline, and a random-forest baseline using NASA/USGS Harmonized Landsat and Sentinel-2 (HLS) S30 Version 2.0 imagery. Each site is analysed independently using its own local hydrographic zero.

## Where to find each site's files

| Site and survey | Raw bathymetry and imagery | Processed input samples | Saved outputs |
|---|---|---|---|
| Anza / Agadir, survey 10 November 2022 | `data/raw/survey/Anza_Agadir_bathy_points.txt`; `data/raw/sentinel2/HLS_S30_Anza_Agadir_2022-11-10/` | `data/processed/anza_agadir_2022-11-10/` | `results/anza_agadir_2022-11-10/` |
| Ras El Ma / Ras Kebdana, survey 8 August 2024 | `data/raw/survey/ras_bathy_points.csv`; `data/raw/sentinel2/HLS_S30_2024-08-07_RasElMa/T30SWD/` | `data/processed/ras_el_ma_2024-08-08/` | `results/ras_el_ma_2024-08-08/` |

The Anza source survey files are `Anza_Agadir_bathy_points.txt`, `Anza_Agadir_topography_points.txt`, and `Anza_Agadir_survey_plan.dwg`. The Ras El Ma source files include `ras_bathy_points.csv`, the survey-plan DWG, and the corresponding GeoPackage. Site dates, local vertical references, and imagery identifiers are described in `data/raw/survey/site_metadata.json`, `data/raw/sentinel2/scene_manifest.json`, and each processed `dataset_summary.json`.

## Data details

- Both image inputs are HLS S30 at 30 m. The included bands are B02 (blue), B03 (green), B04 (red), B08 (near infrared), and Fmask (quality flags).
- HLS reflectance is stored as scaled integers and converted with a factor of 0.0001. The scripts exclude fill and cloud, cloud-adjacency, shadow, and snow/ice flags.
- Survey elevations are in metres relative to each site's local hydrographic zero: the Port of Agadir for Anza and the Port of Ras Kebdana for Ras El Ma. No vertical tie between the two local zeros is applied, so elevations are not pooled as a shared datum.
- Ras El Ma imagery was acquired on 7 August 2024, one day before its 8 August survey. Anza imagery and survey are dated 10 November 2022.
- The processed NPZ files contain one sample per retained 30 m HLS pixel centre and its 9 × 9 × 4 spectral patch. Dataset summaries record filtering, spatial zones, and sample counts.

NASA/USGS HLS S30 Version 2.0 is distributed under its own terms. Cite the collection using [DOI 10.5067/HLS/HLSS30.002](https://doi.org/10.5067/HLS/HLSS30.002), acknowledge NASA/USGS LP DAAC, and follow the [HLS v2.0 User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf). This repository does not relicense HLS imagery.

## Models and evaluation

The author-developed CNN variants are `CNN`, `RatioCNN`, `CNN_BiLSTM`, `DenseCNN`, and `DepthBinCNN`. Baselines are `StumpfRatio` and `RandomForest`. The baseline neural model is named **CNN**; SDB describes the application and is not part of that model's name.

Three buffered alongshore zones rotate through training, validation, and test roles within each site. Every retained sample receives one out-of-fold prediction. Reported metrics include RMSE, MAE, mean bias, and R². Site-specific outputs are stored separately. The two sites' vertical references are distinct, and the evaluation does not test cross-site model transfer.

## Repository layout

- `code/prepare_hls_data.py` and `code/prepare_ras_hls_data.py`: build the processed samples for Anza and Ras El Ma.
- `code/models.py`: defines the five CNN variants.
- `code/train_models.py`: trains the selected models across spatial folds.
- `code/evaluate_results.py`: assembles out-of-fold predictions and per-site evaluation tables.
- `code/compare_site_results.py`: creates the two-site comparison table and relative-skill figure without pooling the local elevations.
- `code/make_advanced_figures.py`: creates Figures 1–7 and Supplementary Figures S1–S2 from the saved samples and predictions. Site-specific 3D Figures 6 and 7 show survey and held-out XYZ surfaces separately at larger scale, with numeric UTM X/Y coordinates, local survey elevation Z in metres, and supported isobath projections. Their vertical dimension uses a fixed 40× exaggeration; amber marks estimates above the local zero.
- `data/raw/`: source bathymetry and HLS imagery used by the study.
- `data/processed/`: site-specific sample arrays and processing summaries.
- `results/`: trained model outputs, out-of-fold predictions, and evaluation tables, kept in separate site folders.
- `figures/rse_14p/`: manuscript figures in high-resolution PNG and vector PDF formats.

## Recreate the processing and figures

Use Python 3.11 and install the packages listed in `code/requirements.txt`. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\requirements.txt
python code\prepare_hls_data.py
python code\prepare_ras_hls_data.py
python code\compare_site_results.py
python code\make_advanced_figures.py
```

The repository includes processed samples, saved fold predictions, model weights, and evaluation outputs. Figure generation uses the released predictions; it does not retrain the models. To retrain both sites, run the following commands separately. Training can take substantial time, especially with all CNNs and folds.

```powershell
python code\train_models.py --data-path data\processed\anza_agadir_2022-11-10\samples.npz --results-dir results\anza_agadir_2022-11-10\folds --site "Anza, Agadir, Morocco" --vertical-reference "Hydrographic zero of the Port of Agadir"
python code\train_models.py --data-path data\processed\ras_el_ma_2024-08-08\samples.npz --results-dir results\ras_el_ma_2024-08-08\folds --site "Ras El Ma, Morocco" --vertical-reference "Hydrographic zero of the Port of Ras Kebdana"
```

To regenerate site-level evaluation files after training, call `code/evaluate_results.py` with the matching processed sample and results paths. The script accepts `--data-path`, `--results-dir`, `--site`, and `--vertical-reference` arguments.

## Data and code terms

The code is released under the MIT License in `LICENSE`. Author-provided survey inputs, derived samples, model outputs, and figures are released under CC BY 4.0, subject to the authors' rights in those materials. HLS imagery remains subject to NASA/USGS LP DAAC terms. Credit the study authors and NASA/USGS LP DAAC, link to the applicable license, and identify changes when redistributing or adapting these materials.
