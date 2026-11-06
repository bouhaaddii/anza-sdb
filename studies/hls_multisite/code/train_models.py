"""Fit five author-developed CNN variants and two baselines on site-specific HLS data."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from models import BUILDERS, masked_patch_mse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "samples.npz"
DEFAULT_OUT = ROOT / "results" / "hls_2022-11-10" / "folds"
NEURAL_MODELS = list(BUILDERS)
MODELS = NEURAL_MODELS + ["StumpfRatio", "RandomForest"]


def dense_targets(pack, ids: np.ndarray) -> np.ndarray:
    return np.concatenate([pack["target_patches"][ids], pack["target_weights"][ids]], axis=-1)


def inputs_for(name: str, pack, ids: np.ndarray, mean: np.ndarray, std: np.ndarray):
    x = ((pack["X"][ids] - mean) / std).astype("float32")
    if name == "RatioCNN":
        return [x, pack["ratio"][ids].astype("float32")]
    return x


def labels_for(name: str, pack, ids: np.ndarray):
    z = pack["y"][ids].astype("float32")
    if name == "DenseCNN":
        return dense_targets(pack, ids).astype("float32")
    if name == "DepthBinCNN":
        # Auxiliary classes derive from the regression target; they are depth bands, not seabed types.
        bands = np.digitize(z[:, 0], bins=[-15.0, -5.0], right=False).astype("int32")
        return {"z": z, "depth_bin": bands}
    return z


def split_for_fold(zones: np.ndarray, fold: int) -> dict[str, np.ndarray]:
    test_zone = fold
    val_zone = (fold + 1) % 3
    train_zone = (fold + 2) % 3
    split = {
        "train": np.flatnonzero(zones == train_zone),
        "validation": np.flatnonzero(zones == val_zone),
        "test": np.flatnonzero(zones == test_zone),
    }
    if any(len(v) == 0 for v in split.values()):
        raise ValueError(f"Empty split in fold {fold}: { {k: len(v) for k, v in split.items()} }")
    # Each region is separated by a strip boundary; complete 9 x 9 windows stay within one zone.
    row_ranges = {}
    for role, ids in split.items():
        row_ranges[role] = (int(np.min(zones[ids])), int(np.max(zones[ids])))
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if row_ranges[a][0] == row_ranges[b][0]:
            raise AssertionError(f"Two split roles share spatial zone: {a}, {b}")
    return split


def compile_model(name: str):
    model = BUILDERS[name](input_shape=(9, 9, 4), dropout=0.25)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    if name == "DenseCNN":
        model.compile(optimizer=optimizer, loss=masked_patch_mse)
    elif name == "DepthBinCNN":
        model.compile(
            optimizer=optimizer,
            loss={"z": "mse", "depth_bin": "sparse_categorical_crossentropy"},
            loss_weights={"z": 1.0, "depth_bin": 0.2},
            metrics={
                "z": [tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
                "depth_bin": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            },
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
    return model


def baseline_features(pack, ids: np.ndarray) -> np.ndarray:
    patch = pack["X"][ids].astype("float32")
    center = patch[:, 4, 4, :]
    mean = patch[:, 3:6, 3:6, :].mean(axis=(1, 2))
    std = patch.std(axis=(1, 2))
    ratio = pack["ratio"][ids]
    return np.concatenate([center, mean, std, ratio], axis=1).astype("float32")


def fit_baseline(name: str, fold: int, pack, split: dict[str, np.ndarray],
                 out_dir: Path, site: str, vertical_reference: str) -> dict:
    seed = 20261106 + fold
    tr, va, te = split["train"], split["validation"], split["test"]
    y_train = pack["y"][tr, 0].astype("float32")
    y_test = pack["y"][te, 0].astype("float32")
    run_dir = out_dir / f"fold_{fold + 1}" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if name == "StumpfRatio":
        # The classic log-ratio relationship is calibrated on training locations only.
        b2 = np.maximum(pack["X"][:, 4, 4, 0], 1e-5)
        b3 = np.maximum(pack["X"][:, 4, 4, 1], 1e-5)
        n = 1000.0
        ratio = (np.log(n * b2) / np.log(n * b3)).reshape(-1, 1)
        estimator = LinearRegression().fit(ratio[tr], y_train)
        pred = estimator.predict(ratio[te])
        val_pred = estimator.predict(ratio[va])
        config = {"formula": "z = intercept + slope * ln(1000*B02)/ln(1000*B03)", "intercept": float(estimator.intercept_), "slope": float(estimator.coef_[0]), "calibration": "training zone only"}
    else:
        x_train = baseline_features(pack, tr)
        x_val = baseline_features(pack, va)
        x_test = baseline_features(pack, te)
        estimator = RandomForestRegressor(n_estimators=300, max_features=0.75, min_samples_leaf=2, n_jobs=4, random_state=seed)
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        val_pred = estimator.predict(x_val)
        config = {"features": "centre B02/B03/B04/B08, central 3x3 means, 9x9 standard deviations, log(B02/B03)", "n_estimators": 300, "max_features": 0.75, "min_samples_leaf": 2, "random_state": seed}

    result = pd.DataFrame({
        "pixel_id": pack["pixel_ids"][te].astype(str), "fold": fold + 1,
        "test_zone": pack["zones"][te].astype(int), "row": pack["rows"][te].astype(int),
        "col": pack["cols"][te].astype(int), "reference_m": y_test,
        "prediction_m": pred, "residual_m": pred - y_test,
    })
    result.to_csv(run_dir / "test_predictions.csv", index=False)
    error = pred - y_test
    metrics = {"model": name, "fold": fold + 1, "n": int(len(y_test)), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)), "r2": float(1 - np.sum(error ** 2) / np.sum((y_test - y_test.mean()) ** 2)), "best_epoch": None, "seconds": round(time.time() - t0, 3)}
    config.update({"train_n": int(len(tr)), "validation_n": int(len(va)), "test_n": int(len(te)), "fold": fold + 1, "model": name, "seed": seed, "metrics": metrics, "site": site, "label_unit": "m", "vertical_reference": vertical_reference, "scope": "One observation per 30 m HLS pixel; calibration and testing use distinct alongshore zones; errors are relative to site-specific survey labels."})
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "fold_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def fit_one(name: str, fold: int, pack, split: dict[str, np.ndarray], max_epochs: int,
            patience: int, out_dir: Path, site: str, vertical_reference: str) -> dict:
    tf.keras.backend.clear_session()
    seed = 20261106 + fold
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    train_ids = split["train"]
    mean = pack["X"][train_ids].mean(axis=(0, 1, 2), dtype=np.float64).astype("float32")
    std = pack["X"][train_ids].std(axis=(0, 1, 2), dtype=np.float64).astype("float32")
    std = np.maximum(std, 1e-6)
    model = compile_model(name)

    fit_kwargs = {"epochs": max_epochs, "batch_size": 128, "verbose": 0}
    x_train = inputs_for(name, pack, train_ids, mean, std)
    y_train = labels_for(name, pack, train_ids)
    val_ids = split["validation"]
    x_val = inputs_for(name, pack, val_ids, mean, std)
    y_val = labels_for(name, pack, val_ids)
    val_data = (x_val, y_val)

    run_dir = out_dir / f"fold_{fold + 1}" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(weights_path), monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, min_delta=1e-4),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=max(5, patience // 2), min_lr=1e-6, verbose=0),
    ]
    t0 = time.time()
    hist = model.fit(x_train, y_train, validation_data=val_data, callbacks=callbacks, **fit_kwargs)
    model.load_weights(weights_path)
    test_ids = split["test"]
    predictions = model.predict(inputs_for(name, pack, test_ids, mean, std), batch_size=256, verbose=0)
    if name == "DenseCNN":
        pred = predictions[:, 4, 4, 0]
    elif name == "DepthBinCNN":
        if isinstance(predictions, dict):
            pred = np.asarray(predictions["z"]).reshape(-1)
        elif isinstance(predictions, (list, tuple)):
            pred = np.asarray(predictions[model.output_names.index("z")]).reshape(-1)
        else:
            raise TypeError("Depth-bin model did not return named outputs")
    else:
        pred = np.asarray(predictions).reshape(-1)

    y_test = pack["y"][test_ids, 0]
    result = pd.DataFrame({
        "pixel_id": pack["pixel_ids"][test_ids].astype(str),
        "fold": fold + 1,
        "test_zone": pack["zones"][test_ids].astype(int),
        "row": pack["rows"][test_ids].astype(int),
        "col": pack["cols"][test_ids].astype(int),
        "reference_m": y_test,
        "prediction_m": pred,
        "residual_m": pred - y_test,
    })
    result.to_csv(run_dir / "test_predictions.csv", index=False)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    config = {
        "model": name,
        "fold": fold + 1,
        "seed": seed,
        "train_zone": int(pack["zones"][train_ids][0]),
        "validation_zone": int(pack["zones"][val_ids][0]),
        "test_zone": int(pack["zones"][test_ids][0]),
        "counts": {k: int(len(v)) for k, v in split.items()},
        "input_channels": ["B02", "B03", "B04", "B08"],
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "max_epochs": max_epochs,
        "epochs_run": len(history["loss"]),
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "runtime_seconds": round(time.time() - t0, 3),
        "parameter_count": int(model.count_params()),
        "model_definition": "code/models.py",
        "site": site,
        "label_unit": "m",
        "vertical_reference": vertical_reference,
        "important_scope": "one observation per 30 m HLS pixel; negative survey z and clear Fmask pixels; single acquisition; scores use a local site-specific vertical reference and are not directly comparable as absolute elevations across sites without a documented vertical tie",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"model": name, "fold": fold + 1, "n": int(len(y_test)), "rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))), "mae": float(np.mean(np.abs(pred - y_test))), "bias": float(np.mean(pred - y_test)), "r2": float(1 - np.sum((pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)), "best_epoch": config["best_epoch"], "seconds": config["runtime_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS + ["all"], default=["all"])
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--folds", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--site", default="Anza, Agadir, Morocco")
    parser.add_argument("--vertical-reference", default="Hydrographic zero of the Port of Agadir (confirmed by the survey-data owner)")
    args = parser.parse_args()
    pack = np.load(args.data_path, allow_pickle=False)
    selected = MODELS if "all" in args.models else args.models
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold_number in args.folds:
        split = split_for_fold(pack["zones"], fold_number - 1)
        for name in selected:
            run_dir = args.results_dir / f"fold_{fold_number}" / name
            metrics_path = run_dir / "fold_metrics.json"
            if metrics_path.exists() and not args.overwrite:
                summaries.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                print(f"RETAINED {name} fold {fold_number}", flush=True)
                continue
            print(f"START {name} fold {fold_number}; n={ {k: len(v) for k, v in split.items()} }", flush=True)
            if name in NEURAL_MODELS:
                result = fit_one(name, fold_number - 1, pack, split, args.max_epochs,
                                 args.patience, args.results_dir, args.site, args.vertical_reference)
            else:
                result = fit_baseline(name, fold_number - 1, pack, split,
                                      args.results_dir, args.site, args.vertical_reference)
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            summaries.append(result)
            print("DONE " + json.dumps(result), flush=True)
    pd.DataFrame(summaries).to_csv(args.results_dir.parent / "fold_metrics.csv", index=False)


if __name__ == "__main__":
    main()
