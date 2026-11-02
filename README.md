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

The author identifies the supplied image as Sentinel-2 `S2_20211210`, acquired on 10 December 2021. The original Sentinel-2 product identifier and product metadata are not included in the released files, so this date and sensor identification are author-reported. The raster contains three 8-bit RGB channels; reflectance calibration is not documented. The six channels in the prepared patches are interpolated from the RGB data and are not six measured spectral bands.

The survey plan identifies GEOMAITRISE SARL AU as producer and TANGER MED ENGINEERING as client. The available plan does not establish field-survey dates, the target-generation workflow, confirmed vertical units, or a vertical datum. Targets are signed elevations in nominal metres, include positive land values, and must not automatically be interpreted as water depth. See `data/anza_study/dataset_metadata.json` and the audit files for the available provenance and its limitations.

The spatial partition prevents shared input pixels between training, validation, and test subsets. It covers one scene, one seed, and one partition and does not provide external validation. Historical results are retained for comparison; the original historical test partition contains patches that share input pixels with training. The README and archived metrics describe these limitations so the results are not mistaken for independent external validation.

## Citation and licenses

Suggested citation for this release:

> Bouhadi, A., Khali Issa, L., Raissouni, A., & El Arrim, A. (2026). *Anza Coastal Study: Code and Data* (Version 1.0.0) [Research code and data]. GitHub. https://github.com/bouhaaddii/anza-sdb

The software is released under the MIT License in `LICENSE`. Study data and derived data are released under Creative Commons Attribution 4.0 International (CC BY 4.0); see `DATA_LICENSE.md`. Please cite the article when its final bibliographic details become available. No DOI is assigned to this GitHub repository version.

## Versioning

The public release is tagged `v1.0.0`. The data integrity manifest covers files under `data/anza_study/`, excluding the manifest itself. Recompute hashes after any changes to those data files.
