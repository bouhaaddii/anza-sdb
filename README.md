# Anza Coastal Study: Code and Data

This repository provides the research code, study inputs, prepared datasets, archived model outputs, and visualization scripts for the Anza coastal remote-sensing study. It contains the materials used to prepare data, reproduce retained results, run the documented spatial experiment, and create figures. The article manuscript is not included.

## Repository contents

- `original_scripts/`: source scripts supplied for the study, including the original model and plotting implementations.
- `src/`: shared data preparation, model, metric, and plotting functions used by the command-line pipeline.
- `run.py` and `pipeline.py`: commands to check source identity, verify the data bundle, prepare data, reproduce results, train models, and create figures.
- `SDB_Visualisation_Avancee_Corrigee.py`: detailed evaluation and result visualizations.
- `SDB_Visualisation_Publication.py`, `SDB_Cartes_Publication.py`, and `result_figures.py`: publication figures and maps from saved outputs.
- `data/anza_study/reference/`: supplied imagery, survey material, and saved reference arrays.
- `data/anza_study/prepared/`: prepared patches, labels, and train/validation/test memberships.
- `data/anza_study/experiments/`: saved spatial-experiment weights, histories, predictions, and metrics.
- `data/anza_study/reproduced/`: retained predictions and paired reference–prediction tables.
- `data/anza_study/tables/` and `data/anza_study/audit/`: derived tables, provenance notes, and validation checks.
- `data/anza_study/file_manifest.json`: sizes and SHA-256 hashes for the released data files.
- `studies/hls_multisite/`: a separate HLS S30 two-site experiment, including its executed scripts, selected source inputs, prepared samples, saved fold outputs, metrics, and figures. It does not replace the original 2021 RGB study above.

## Models

The study compares CNN, Pi-CNN, CNN-BiLSTM, FCNN, and SeaBed-Net. The model historically named `sdb_cnn` is a conventional CNN architecture; it is not a separate model called “SDB-CNN.” Use `cnn` as its identifier for a new training run. `sdb_cnn` remains accepted only as a legacy identifier for archived results.

## Verify the release and create figures

Use Python 3.10 or 3.11. From the repository root, run:

```powershell
python run.py check-code
python run.py verify --data-root data/anza_study
python run.py figures --data-root data/anza_study --work-root work/anza
```

The launcher places generated files in the work directory and leaves the released data bundle unchanged. Run `python run.py --help`, or add `--help` to a command, to see other options.

The advanced visualization scripts can also be run directly in an environment with the dependencies in `requirements.txt`:

```powershell
python SDB_Visualisation_Avancee_Corrigee.py --data-root data/anza_study --out work/SDB_Resultats_Corriges
python SDB_Visualisation_Publication.py --results work/SDB_Resultats_Corriges --data-root data/anza_study --out work/SDB_Resultats_Figures
python SDB_Cartes_Publication.py --results work/SDB_Resultats_Corriges --data-root data/anza_study --out work/SDB_Cartes
```

Install dependencies in an isolated Python 3.11 environment on Windows with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data provenance and interpretation

The released input image is an exact pixel-for-pixel crop of the 10 m TCI true-colour component in Sentinel-2A MSI Level-2A product `S2A_MSIL2A_20211210T112451_N0500_R037_T29RMP_20221228T204212.SAFE`. Its SAFE metadata verify the 10 December 2021 acquisition date, processing baseline 05.00 and generation date of 28 December 2022. The released 202 × 210 GeoTIFF matches the source TCI at row offset 2985 and column offset 3475 in tile T29RMP (EPSG:32629). The model uses the product’s 8-bit display RGB composite, not its calibrated BOA reflectance bands. Product-level quality reports pass general, geometric, radiometric and sensor checks; product metadata report 0.016484% cloudy pixels. In the cropped 20 m scene-classification layer, 27 cells are labelled medium-probability cloud, 3 cloud shadow and 494 snow/ice. These classifier labels were not used to mask the model inputs. The six prepared channels are interpolated combinations of RGB and are not six measured spectral bands. Because the input is clipped from Sentinel data, the required attribution is “Contains modified Copernicus Sentinel data 2021.” See `data/anza_study/audit/image_quality.json` for the source metadata and crop check.

The survey plan identifies GEOMAITRISE SARL AU as producer and TANGER MED ENGINEERING as client. The available plan does not establish field-survey dates, the target-generation workflow, confirmed vertical units, or a vertical datum. Targets are signed elevations in nominal metres, include positive land values, and must not automatically be interpreted as water depth. See `data/anza_study/dataset_metadata.json` and the audit files for the available provenance and its limitations.

The spatial partition prevents shared input pixels between training, validation, and test subsets. It covers one scene, one seed, and one partition and does not provide external validation. Historical results are retained for comparison; the original historical test partition contains patches that share input pixels with training. The README and archived metrics describe these limitations so the results are not mistaken for independent external validation.

## Additional study: two-site HLS coastal elevation experiment

The separate [`studies/hls_multisite/`](studies/hls_multisite/) folder contains the complete executed HLS experiment for Anza and Ras El Ma/Ras Kebdana. It includes only the scripts used in this study, the selected input files, prepared data, saved fold predictions and weights, evaluation tables, and code-generated figures. The earlier 2021 RGB study at the repository root remains a separate project.

The study-specific English README at [`studies/hls_multisite/README.md`](studies/hls_multisite/README.md) documents its inputs, local vertical references, methods, figures, saved results, and commands for regenerating the analysis.

### Inputs and provenance

| Site | Survey date and label reference | HLS input | Retained samples |
| --- | --- | --- | ---: |
| Anza / Agadir | 10 November 2022; z in metres; Port of Agadir hydrographic zero reported by the user | HLS S30 v2.0, tile T29RMP, acquired 10 November 2022 | 1,373 |
| Ras El Ma / Ras Kebdana | 8 August 2024; z in metres; Port of Ras Kebdana hydrographic zero stated in the survey-plan cartouche | HLS S30 v2.0, tile T30SWD, acquired 7 August 2024 | 3,817 |

Each scene uses the 30 m B02, B03, B04, and B08 reflectance bands and Fmask. HLS reflectance integers are multiplied by 0.0001. The scripts exclude fill, cloud, adjacent cloud/shadow, shadow, and snow/ice flags; the water flag is retained as a diagnostic. See the [NASA/USGS HLS v2.0 User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf), the [HLS S30 product DOI](https://doi.org/10.5067/HLS/HLSS30.002), and `studies/hls_multisite/data/raw/sentinel2/scene_manifest.json` for product identifiers and selected files.

Survey labels are aggregated to one observation per HLS pixel. The 9 × 9 input patches have a 120 m radius. Cells with fewer than three survey points and mixed-sign elevation cells are excluded. Ras El Ma's 10,595 exact duplicate XYZ records were removed; repeated XY positions are represented by their median z, with the observed 0.03 m spread documented in the processed-data metadata. Three alongshore spatial zones are separated by a patch-footprint buffer and rotated through the folds.

The two surveys use distinct local port hydrographic zeros. Anza's vertical reference was reported by the user and is not documented in the supplied plan extract. Ras El Ma's HLS scene predates its survey by one day. There is no documented water-level correction, point-level uncertainty model, or vertical tie between the two port zeros. Therefore, the reported metrics are separate site-specific spatial evaluations; they do not demonstrate cross-site transfer, and absolute elevations or errors must not be pooled between sites.

### Saved spatial holdout results

| Site | Model with lowest pooled out-of-fold RMSE | RMSE (m) | MAE (m) |
| --- | --- | ---: | ---: |
| Anza / Agadir | StumpfRatio | 2.79 | 2.42 |
| Ras El Ma / Ras Kebdana | RatioCNN | 0.50 | 0.40 |

These are out-of-fold predictions from three buffered spatial folds within each site. They are not an independent external validation. Refer to the site-specific `evaluation_summary.json` files for the full seven-model results and fold variability, and to the figures for spatial patterns and error distributions. The metrics are errors against the aggregated survey labels in metres, not a claim of universally accurate bathymetry.

### Run the scripts

From the `studies/hls_multisite/` directory, create a Python 3.11 environment and install `code/requirements.txt`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code/requirements.txt
```

Prepare both site datasets, train the seven models in three spatial folds, and regenerate the evaluations:

```powershell
python code/prepare_hls_data.py
python code/prepare_ras_hls_data.py
python code/train_models.py --data-path data/processed/samples.npz --results-dir results/hls_2022-11-10/folds --site "Anza, Agadir, Morocco" --vertical-reference "Local hydrographic zero of the Port of Agadir (user-reported)"
python code/evaluate_results.py --data-path data/processed/samples.npz --results-dir results/hls_2022-11-10 --site "Anza, Agadir, Morocco" --vertical-reference "Local hydrographic zero of the Port of Agadir (user-reported)"
python code/train_models.py --data-path data/processed/ras_el_ma_2024-08-08/samples.npz --results-dir results/hls_ras_2024-08-07/folds --site "Ras El Ma / Ras Kebdana, Morocco" --vertical-reference "Local hydrographic zero of the Port of Ras Kebdana (survey-plan cartouche)"
python code/evaluate_results.py --data-path data/processed/ras_el_ma_2024-08-08/samples.npz --results-dir results/hls_ras_2024-08-07 --site "Ras El Ma / Ras Kebdana, Morocco" --vertical-reference "Local hydrographic zero of the Port of Ras Kebdana (survey-plan cartouche)"
python code/make_figures.py
python code/make_figures.py --data-path data/processed/ras_el_ma_2024-08-08/samples.npz --results-dir results/hls_ras_2024-08-07 --scene-dir data/raw/sentinel2/HLS_S30_2024-08-07_RasElMa/T30SWD --product HLS.S30.T30SWD.2024220T105031.v2.0 --figures-dir figures/ras_el_ma --utm-zone 30N --scene-date "7 Aug 2024"
python code/compare_site_results.py
```

Training may take substantial time on CPU. The archived fold weights, predictions, metrics, and figures are included, so the current results can be inspected without rerunning model fitting.

## Citation and licenses

Suggested citation for this release:

> Bouhadi, A., Khali Issa, L., Raissouni, A., & El Arrim, A. (2026). *Anza Coastal Study: Code and Data* (Version 1.0.0) [Research code and data]. GitHub. https://github.com/bouhaaddii/anza-sdb

The software is released under the MIT License in `LICENSE`. Author-created study materials and model-derived results for which the authors hold the relevant rights are released under Creative Commons Attribution 4.0 International (CC BY 4.0). Sentinel-derived image files and image inputs are identified individually in the data manifest and remain subject to the official [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice); the CC BY notice does not replace those terms. The required attribution for modified Copernicus data is “Contains modified Copernicus Sentinel data 2021.” The ESA data citation DOI is [10.5270/S2_-znk9xsj](https://doi.org/10.5270/S2_-znk9xsj). Please cite the article when its final bibliographic details become available. No DOI is assigned to this GitHub repository version.

## Versioning

The repository does not currently have a Git tag or DOI. The data integrity manifest covers files under `data/anza_study/`, excluding the manifest itself; it does not cover the separate HLS study folder. Recompute hashes after changing the corresponding data files.
