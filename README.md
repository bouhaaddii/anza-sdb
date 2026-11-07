# Spatially Blocked Comparison of CNN Variants and Baseline Models for Satellite-Derived Bathymetry at Two Moroccan Coasts

## Study code, inputs, and outputs for Anza and Ras El Ma, Morocco

This repository contains the analysis code, source inputs, prepared samples, saved spatial-holdout outputs, and figures for a study of shallow-water bathymetry at two Moroccan coasts. It compares five convolutional neural network variants designed by the first author with a Stumpf-style blue–green log-ratio baseline and a random-forest baseline using NASA/USGS Harmonized Landsat and Sentinel-2 (HLS) S30 imagery.

The repository now contains only the materials for this Anza–Ras El Ma study. The article manuscript and submission correspondence are not part of this code-and-data release.

## Study sites and inputs

| Site | Bathymetric survey | HLS acquisition | Tile and CRS | Local vertical reference | Retained 30 m cells |
| --- | --- | --- | --- | --- | ---: |
| Anza / Agadir, Atlantic coast | 10 November 2022 | 10 November 2022 | T29RMP, EPSG:32629 | Hydrographic zero of the Port of Agadir | 1,373 |
| Ras El Ma / Ras Kebdana, Mediterranean coast | 8 August 2024 | 7 August 2024 | T30SWD, EPSG:32630 | Hydrographic zero of the Port of Ras Kebdana | 3,817 |

Satellite inputs are the 30 m HLS S30 v2.0 reflectance bands B02 (blue), B03 (green), B04 (red), and B08 (near infrared), together with Fmask. HLS reflectance values are scaled by 0.0001. Fmask fill, cloud, adjacent cloud/shadow, shadow, and snow/ice flags are excluded; the water flag is retained as a diagnostic. Product identifiers, acquisition dates, and source details are in [`data/raw/sentinel2/scene_manifest.json`](data/raw/sentinel2/scene_manifest.json). Cite the HLS S30 collection using [DOI 10.5067/HLS/HLSS30.002](https://doi.org/10.5067/HLS/HLSS30.002) and acknowledge NASA/USGS LP DAAC. See the official [HLS v2.0 User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).

The survey sources are in `data/raw/survey/`. The Ras El Ma survey drawing is included as supporting provenance for its coordinate and vertical-reference notes. Survey coordinates were transformed to each HLS grid and aggregated by 30 m cell. Each retained cell has a 9 × 9 × 4 reflectance patch. Cells with inadequate sounding support, mixed land/water signs, or invalid patch quality were removed.

**The two vertical references are separate local port hydrographic zeros.** Anza uses the Port of Agadir zero; Ras El Ma uses the Port of Ras Kebdana zero. The Ras drawing's ΔZ = −47 m geodetic datum-translation parameter is not a levelling connection between these port zeros and was not applied as one. The results are evaluated separately within each site; the experiment does not test transfer from one coast to the other. Ras imagery is one day older than its survey. Neither dataset has a documented sounding-time/tide correction or point-level vertical uncertainty.

## Models and evaluation

The five author-designed CNN variants are `CNN`, `RatioCNN`, `CNN_BiLSTM`, `DenseCNN`, and `DepthBinCNN`. The conventional baseline is called **CNN**; “SDB” describes the application and is not part of that model's name. The comparisons also include a training-zone-calibrated Stumpf-style log-ratio regression and a random forest. Three buffered alongshore zones rotate through training, validation, and held-out test roles independently at each site.

The archived out-of-fold summaries currently report:

| Site | Lowest-RMSE method | RMSE | Lowest-MAE method | MAE |
| --- | --- | ---: | --- | ---: |
| Anza / Agadir | Stumpf-style log-ratio | 2.788 m | CNN–BiLSTM | 2.253 m |
| Ras El Ma / Ras Kebdana | RatioCNN | 0.500 m | CNN | 0.388 m |

These are errors against aggregated local survey labels in metres, not independent hydrographic accuracy measurements. Differences between sites must not be interpreted as absolute vertical agreement because the reference zeros, survey support, and acquisition timing differ. Fold counts and per-model values are in the site-specific `results/` folders.

## Cartographic outputs

The five main figures and two supplementary figures are in [`figures/rse_14p/`](figures/rse_14p/), each as a high-resolution PNG and a vector PDF:

1. Site, satellite-image, survey-depth, and sounding-support overview.
2. Map-led illustration of the actual HLS patch, four-band 9 × 9 pixel cube, and local survey support with one saved held-out centre estimate.
3. Buffered spatial folds and rotating training, validation, and test roles.
4. Modern held-out validation panels: observed-versus-predicted density and residual trends by depth.
5. Survey and held-out depth isobaths with supported 3D views at both sites.
6. Supplementary inter-model disagreement maps.
7. Supplementary spatial residual maps.

The map panels use an eight-point compass rose with N, E, S, and W labels. Depth contours are drawn at 2 m intervals. The 3D and filled-surface panels are cartographic displays built from linear triangles between neighbouring 30 m cell centres; triangles with an edge longer than 45 m are masked, and no surface is extrapolated beyond supported cells. These displays are not extra model predictions. The vertical axis is exaggerated for visual clarity, and each site keeps its own depth scale and datum.

## Repository contents

- `code/prepare_hls_data.py` and `code/prepare_ras_hls_data.py`: build one sample per eligible HLS cell from each survey and scene.
- `code/models.py`: defines the five CNN variants and masked patch loss.
- `code/train_models.py`: trains the CNNs and baselines using buffered spatial folds.
- `code/evaluate_results.py`: creates site-specific out-of-fold predictions and evaluation summaries.
- `code/compare_site_results.py`: summarizes site-local metrics without pooling the local vertical zeros.
- `code/make_advanced_figures.py`: builds the five main and two supplementary map-led figures from the saved inputs and predictions.
- `code/requirements.txt`: Python package versions used by the analysis.
- `data/`: selected HLS bands, survey inputs, site metadata, and prepared sample arrays.
- `results/`: saved model weights, fold predictions, metrics, and site-comparison tables.
- `figures/rse_14p/`: final figures in PNG and PDF formats.

## Run the analysis

Use Python 3.11. From the repository root, create an environment and install the listed packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\requirements.txt
```

The released prepared arrays and fold outputs are already included. Regenerate the current figures and the site-wise comparison table with:

```powershell
python code\compare_site_results.py
python code\make_advanced_figures.py
```

To rebuild prepared arrays from the source inputs:

```powershell
python code\prepare_hls_data.py
python code\prepare_ras_hls_data.py
```

To retrain, run `python code\train_models.py --help` for the available options. The saved fold predictions and weights are provided so the reported comparisons can be inspected without training again.

## Data use and attribution

Code is released under the MIT License in [`LICENSE`](LICENSE). Author-created survey tables, prepared arrays, model outputs, and figures for which the authors hold the relevant rights are released under Creative Commons Attribution 4.0 International (CC BY 4.0). You may share and adapt those materials, including for commercial purposes, provided that you give appropriate credit, link to the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/), and indicate whether changes were made. HLS S30 v2.0 imagery remains subject to NASA/USGS LP DAAC reuse terms and is not relicensed here; cite the collection using [DOI 10.5067/HLS/HLSS30.002](https://doi.org/10.5067/HLS/HLSS30.002), acknowledge NASA/USGS LP DAAC, and retain the source-product attribution.
