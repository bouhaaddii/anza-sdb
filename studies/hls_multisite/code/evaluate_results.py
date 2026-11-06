"""Summarize site-specific HLS spatial out-of-fold predictions without IID assumptions."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "hls_2022-11-10"
DEFAULT_DATA = ROOT / "data" / "processed" / "samples.npz"


def score(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n_pixels": 0, "rmse_m": None, "mae_m": None,
                "mean_bias_m": None, "median_absolute_error_m": None,
                "p95_absolute_error_m": None, "r2": None}
    obs = frame["reference_m"].to_numpy(dtype=float)
    pred = frame["prediction_m"].to_numpy(dtype=float)
    err = pred - obs
    denom = np.sum((obs - obs.mean()) ** 2)
    return {
        "n_pixels": int(len(frame)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "mae_m": float(np.mean(np.abs(err))),
        "mean_bias_m": float(np.mean(err)),
        "median_absolute_error_m": float(np.median(np.abs(err))),
        "p95_absolute_error_m": float(np.percentile(np.abs(err), 95)),
        "r2": float(1.0 - np.sum(err ** 2) / denom) if denom else None,
    }


def refresh_run_metadata(folds_dir: Path, site: str, vertical_reference: str) -> None:
    for config_path in folds_dir.glob("fold_*/**/run_config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["site"] = site
        config["label_unit"] = "m"
        config["vertical_reference"] = vertical_reference
        config["measurement_note"] = (
            "Errors are against the site's aggregated survey z values. The site-local vertical reference is not tied to the other study site's port datum."
        )
        for key in ("scope", "important_scope"):
            if key in config:
                config[key] = (
                    "One observation per 30 m HLS pixel; spatial calibration and testing use distinct buffered alongshore zones. "
                    "Scores are site-specific and do not establish cross-site transfer or absolute-depth agreement between different port datums."
                )
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--site", default="Anza, Agadir, Morocco")
    parser.add_argument("--vertical-reference", default="Local hydrographic zero of the Port of Agadir (user-reported)")
    args = parser.parse_args()
    results_dir = args.results_dir
    folds_dir = results_dir / "folds"
    tables = []
    model_names = ["CNN", "RatioCNN", "CNN_BiLSTM", "DenseCNN", "DepthBinCNN", "StumpfRatio", "RandomForest"]
    for name in model_names:
        for fold in (1, 2, 3):
            f = folds_dir / f"fold_{fold}" / name / "test_predictions.csv"
            if not f.exists():
                raise FileNotFoundError(f"Missing completed fold: {f}")
            part = pd.read_csv(f)
            # Read old Anza runs as well as newly generated metre-labelled outputs.
            aliases = {
                "reference_m": "reference_nominal_units",
                "prediction_m": "prediction_nominal_units",
                "residual_m": "residual_nominal_units",
            }
            for current, legacy in aliases.items():
                if current not in part.columns and legacy in part.columns:
                    part[current] = part[legacy]
            part = part.drop(columns=[legacy for legacy in aliases.values() if legacy in part.columns])
            part["model"] = name
            tables.append(part)
    oof = pd.concat(tables, ignore_index=True)
    with np.load(args.data_path, allow_pickle=False) as pack:
        rows, cols = pack["rows"], pack["cols"]
        points = pd.DataFrame({
            "pixel_id": pack["pixel_ids"].astype(str),
            "row": rows.astype(int), "col": cols.astype(int),
            "x_utm": pack["x_utm"], "y_utm": pack["y_utm"],
        })
    oof = oof.merge(points, on=["pixel_id", "row", "col"], how="left", validate="many_to_one")
    oof.to_csv(results_dir / "out_of_fold_predictions.csv", index=False)
    refresh_run_metadata(folds_dir, args.site, args.vertical_reference)

    summary_rows, fold_rows, stratified_rows = [], [], []
    for name, group in oof.groupby("model", sort=False):
        overall = {"model": name, **score(group)}
        fold_scores = []
        for fold, fg in group.groupby("fold", sort=True):
            rec = {"model": name, "fold": int(fold), **score(fg)}
            fold_rows.append(rec)
            fold_scores.append(rec)
            for band, condition in [
                ("shallow_0_to_5", (fg["reference_m"] >= -5.0) & (fg["reference_m"] < 0.0)),
                ("mid_5_to_15", (fg["reference_m"] >= -15.0) & (fg["reference_m"] < -5.0)),
                ("deep_gt_15", fg["reference_m"] < -15.0),
            ]:
                band_frame = fg.loc[condition]
                stratified_rows.append({"model": name, "fold": int(fold), "depth_band": band, **score(band_frame)})
        overall["fold_rmse_mean_m"] = float(np.mean([r["rmse_m"] for r in fold_scores]))
        overall["fold_rmse_sd_m"] = float(np.std([r["rmse_m"] for r in fold_scores], ddof=1))
        overall["fold_rmse_min_m"] = float(np.min([r["rmse_m"] for r in fold_scores]))
        overall["fold_rmse_max_m"] = float(np.max([r["rmse_m"] for r in fold_scores]))
        summary_rows.append(overall)

    pd.DataFrame(summary_rows).to_csv(results_dir / "model_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(results_dir / "metrics_by_fold.csv", index=False)
    pd.DataFrame(stratified_rows).to_csv(results_dir / "metrics_by_depth_band.csv", index=False)
    payload = {
        "site": args.site,
        "label_unit": "m",
        "vertical_reference": args.vertical_reference,
        "scope": "Out-of-fold spatial prediction within one NASA HLS S30 scene at 30 m. Survey labels are averaged by HLS pixel; the spatial folds are internal to this site and do not establish cross-site transfer.",
        "warning": "The three spatial zones are the geographic units. Per-pixel confidence intervals and IID significance tests are not reported. Each site's error metrics are calculated against its own port-datum labels; these evaluations do not test cross-site transfer and the absolute z labels must not be pooled without a documented vertical connection.",
        "summary": summary_rows,
        "fold_metrics": fold_rows,
        "depth_band_metrics": stratified_rows,
    }
    (results_dir / "evaluation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
