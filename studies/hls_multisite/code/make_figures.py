"""Create paper-ready figures from one site's 30 m HLS experiment."""
from __future__ import annotations

import json
import argparse
from contextlib import ExitStack
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "hls_2022-11-10"
FIG = ROOT / "figures"
SAMPLES = ROOT / "data" / "processed" / "samples.npz"
SCENE = RAW / "sentinel2" / "HLS_S30_2022-11-10"
PRODUCT = "HLS.S30.T29RMP.2022314T112209.v2.0"
UTM_ZONE = "29N"
SCENE_DATE = "10 Nov 2022"
PALETTE = {
    "navy": "#17324D", "blue": "#2F6BFF", "teal": "#13A89E",
    "orange": "#E68A00", "purple": "#815AC0", "rose": "#D44762",
    "slate": "#65758B", "green": "#2E8B57", "ink": "#172033",
    "grid": "#DFE6EE", "pale": "#F3F6FA",
}
NEURAL_MODELS = ["CNN", "RatioCNN", "CNN_BiLSTM", "DenseCNN", "DepthBinCNN"]
MODELS = NEURAL_MODELS + ["StumpfRatio", "RandomForest"]
PRETTY = {
    "CNN": "CNN", "RatioCNN": "Ratio-augmented CNN",
    "CNN_BiLSTM": "CNN–BiLSTM", "DenseCNN": "Dense-decoder CNN",
    "DepthBinCNN": "Depth-bin CNN", "StumpfRatio": "Log-ratio regression",
    "RandomForest": "Random forest",
}
COLORS = {
    "CNN": PALETTE["blue"], "RatioCNN": PALETTE["teal"],
    "CNN_BiLSTM": PALETTE["orange"], "DenseCNN": PALETTE["purple"],
    "DepthBinCNN": PALETTE["rose"], "StumpfRatio": PALETTE["slate"],
    "RandomForest": PALETTE["green"],
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#C8D2DE", "axes.linewidth": 0.8,
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"],
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6, "legend.frameon": False,
    })


def save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_results():
    oof = pd.read_csv(RESULTS / "out_of_fold_predictions.csv")
    summary = pd.read_csv(RESULTS / "model_summary.csv").set_index("model")
    folds = pd.read_csv(RESULTS / "metrics_by_fold.csv")
    strata = pd.read_csv(RESULTS / "metrics_by_depth_band.csv")
    return oof, summary, folds, strata


def read_context(pack, padding_m: float = 450.0):
    rows, cols = pack["rows"], pack["cols"]
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(SCENE / f"{PRODUCT}.B02.tif"))
        pad = int(np.ceil(padding_m / abs(src.transform.a)))
        r0 = max(0, int(rows.min()) - pad)
        r1 = min(src.height, int(rows.max()) + pad + 1)
        c0 = max(0, int(cols.min()) - pad)
        c1 = min(src.width, int(cols.max()) + pad + 1)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        left, bottom, right, top = src.window_bounds(window)
        extent = [left / 1000, right / 1000, bottom / 1000, top / 1000]
        transform = src.window_transform(window)
        bands = {}
        for name in ("B02", "B03", "B04", "B08"):
            layer = stack.enter_context(rasterio.open(SCENE / f"{PRODUCT}.{name}.tif"))
            bands[name] = layer.read(1, window=window).astype("float32") * 0.0001
        qa_src = stack.enter_context(rasterio.open(SCENE / f"{PRODUCT}.Fmask.tif"))
        qa = qa_src.read(1, window=window)
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    return rgb, qa, bands, extent, (r0, r1, c0, c1), transform


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    output = np.zeros_like(rgb, dtype="float32")
    for channel in range(3):
        band = rgb[..., channel]
        good = np.isfinite(band) & (band > -0.5)
        lo, hi = np.percentile(band[good], [2, 98])
        if hi <= lo:
            hi = lo + 1e-5
        output[..., channel] = np.clip((band - lo) / (hi - lo), 0, 1)
    return output


def figure_inputs() -> None:
    pack = np.load(SAMPLES, allow_pickle=False)
    rgb, qa, _, extent, window, _ = read_context(pack)
    r0, r1, c0, c1 = window
    target = pack["target_grid"][r0:r1, c0:c1]
    target_mask = pack["target_grid_mask"][r0:r1, c0:c1]
    z = np.ma.masked_where(~target_mask, target)
    water = np.ma.masked_where((qa & (1 << 5)) == 0, np.ones(qa.shape))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)
    axes[0].imshow(stretch_rgb(rgb), extent=extent, origin="upper")
    axes[0].set_title(f"(a) HLS true-colour reflectance · {SCENE_DATE}")
    axes[1].imshow(water, extent=extent, origin="upper", cmap="Blues", vmin=0.5, vmax=1.5)
    axes[1].set_title("(b) HLS Fmask water bit")
    depth_min = float(np.percentile(pack["y"][:, 0], 1))
    depth_img = axes[2].imshow(z, extent=extent, origin="upper", cmap="viridis_r", vmin=depth_min, vmax=0)
    axes[2].set_title("(c) Aggregated survey z labels")
    cb = fig.colorbar(depth_img, ax=axes[2], fraction=.046, pad=.04)
    cb.set_label("Survey z relative to the local vertical reference (m)")
    for ax in axes:
        ax.set_xlabel(f"Easting (km; UTM {UTM_ZONE})")
        ax.set_ylabel(f"Northing (km; UTM {UTM_ZONE})")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12)
    save(fig, "Figure_1_study_area_and_inputs")


def figure_folds() -> None:
    pack = np.load(SAMPLES, allow_pickle=False)
    rgb, _, _, extent, _, _ = read_context(pack)
    zones = pack["zones"].astype(int)
    x, y = pack["x_utm"] / 1000, pack["y_utm"] / 1000
    roles = ["Training", "Validation", "Test"]
    role_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["rose"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for fold, ax in enumerate(axes):
        ax.imshow(stretch_rgb(rgb), extent=extent, origin="upper", alpha=.68)
        zone_roles = {((fold + 2) % 3): 0, ((fold + 1) % 3): 1, fold: 2}
        for zone, role_index in zone_roles.items():
            m = zones == zone
            ax.scatter(x[m], y[m], s=4, c=role_colors[role_index], linewidths=0,
                       label=f"{roles[role_index]} ({int(m.sum())} pixels)", rasterized=True)
        ax.set_title(f"Spatial fold {fold + 1}")
        ax.set_xlabel(f"Easting (km; UTM {UTM_ZONE})")
        ax.set_ylabel(f"Northing (km; UTM {UTM_ZONE})")
        ax.set_aspect("equal", adjustable="box")
        legend = ax.legend(loc="lower left", fontsize=7, markerscale=1.8,
                           frameon=True, facecolor="white", framealpha=.96,
                           edgecolor="#C8D2DE")
        legend.set_zorder(20)
        ax.grid(alpha=.12)
    save(fig, "Figure_3_spatial_folds")


def figure_metrics() -> None:
    _, summary, fold_metrics, _ = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.7), constrained_layout=True)
    for ax, metric, title in zip(
        axes, ["rmse_m", "mae_m"],
        ["Spatial out-of-fold RMSE", "Spatial out-of-fold MAE"],
    ):
        values = [float(summary.loc[m, metric]) for m in MODELS]
        x = np.arange(len(MODELS))
        ax.bar(x, values, color=[COLORS[m] for m in MODELS], width=.68, alpha=.86, zorder=2)
        for i, name in enumerate(MODELS):
            subset = fold_metrics.loc[fold_metrics.model == name, metric].to_numpy(float)
            ax.scatter(np.full(len(subset), i) + np.linspace(-.1, .1, len(subset)),
                       subset, s=24, c=PALETTE["ink"], edgecolors="white",
                       linewidths=.5, zorder=3)
            if metric == "rmse_m":
                mean = float(summary.loc[name, "fold_rmse_mean_m"])
                sd = float(summary.loc[name, "fold_rmse_sd_m"])
                ax.errorbar(i, mean, yerr=sd, fmt="none", ecolor=PALETTE["ink"],
                            capsize=3, lw=1, zorder=4)
        ax.set_xticks(x, [PRETTY[m] for m in MODELS], rotation=24, ha="right")
        ax.set_ylabel("Error (m)")
        ax.set_title(title, weight="bold", color=PALETTE["navy"])
        ax.grid(axis="y", zorder=0)
    fig.suptitle("Errors on held-out 30 m alongshore zones", fontsize=13,
                 weight="bold", color=PALETTE["navy"])
    save(fig, "Figure_4_spatial_model_comparison")


def figure_scatter() -> None:
    oof, summary, _, _ = load_results()
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.8), constrained_layout=True,
                             sharex=True, sharey=True)
    lo = float(min(oof.reference_m.min(), oof.prediction_m.min()))
    hi = float(max(oof.reference_m.max(), oof.prediction_m.max()))
    last = None
    for ax, name in zip(axes.flat, MODELS):
        d = oof[oof.model == name]
        last = ax.hexbin(d.reference_m, d.prediction_m,
                         gridsize=25, mincnt=1, cmap="Blues", bins="log", linewidths=0)
        ax.plot([lo, hi], [lo, hi], ls="--", color=PALETTE["rose"], lw=1)
        ax.set_title(f"{PRETTY[name]}\nRMSE {summary.loc[name, 'rmse_m']:.2f} m",
                     fontsize=8.5, weight="bold")
        ax.grid(alpha=.13)
    axes.flat[-1].axis("off")
    axes[1, 0].set_xlabel("Survey z (m)")
    axes[1, 1].set_xlabel("Survey z (m)")
    axes[1, 2].set_xlabel("Survey z (m)")
    axes[0, 0].set_ylabel("Predicted z (m)")
    axes[1, 0].set_ylabel("Predicted z (m)")
    if last is not None:
        cb = fig.colorbar(last, ax=axes, pad=.015, shrink=.78)
        cb.set_label("Unique HLS pixel-centre density (log count)")
    save(fig, "Figure_5_out_of_fold_predictions")


def figure_maps() -> None:
    oof, _, _, _ = load_results()
    pack = np.load(SAMPLES, allow_pickle=False)
    rgb, _, _, extent, window, _ = read_context(pack)
    r0, r1, c0, c1 = window
    ref_grid = np.ma.masked_where(
        ~pack["target_grid_mask"][r0:r1, c0:c1],
        pack["target_grid"][r0:r1, c0:c1],
    )
    lower = float(np.percentile(pack["y"][:, 0], 1))
    upper = 0.0
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.6), constrained_layout=True)
    axes[0, 0].imshow(ref_grid, extent=extent, origin="upper", cmap="viridis_r",
                     vmin=lower, vmax=upper)
    axes[0, 0].set_title("Survey reference")
    last = None
    for ax, name in zip(axes.flat[1:8], MODELS):
        d = oof[oof.model == name]
        last = ax.scatter(d.x_utm / 1000, d.y_utm / 1000,
                          c=d.prediction_m, s=5, cmap="viridis_r",
                          vmin=lower, vmax=upper, linewidths=0, rasterized=True)
        ax.set_title(PRETTY[name])
    for ax in axes.flat[:8]:
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12)
    cb = fig.colorbar(last, ax=axes, fraction=.025, pad=.02)
    cb.set_label("z relative to local vertical reference (m)")
    save(fig, "Figure_6_out_of_fold_depth_maps")

    limit = max(1.0, float(np.percentile(np.abs(oof.residual_m), 98)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.6), constrained_layout=True)
    last = None
    for ax, name in zip(axes.flat, MODELS):
        d = oof[oof.model == name]
        last = ax.scatter(d.x_utm / 1000, d.y_utm / 1000,
                          c=d.residual_m, s=5, cmap="RdBu_r",
                          norm=norm, linewidths=0, rasterized=True)
        ax.set_title(PRETTY[name])
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12)
    axes.flat[-1].axis("off")
    cb = fig.colorbar(last, ax=axes, fraction=.025, pad=.02)
    cb.set_label("Prediction − survey (m)")
    save(fig, "Figure_7_spatial_residuals")


def figure_depth_strata() -> None:
    oof, _, _, _ = load_results()
    d = oof.copy()
    d["depth_magnitude"] = d["reference_m"].abs()
    d["depth_band"] = pd.cut(
        d["depth_magnitude"], bins=[0, 5, 15, np.inf],
        labels=["0–5", "5–15", ">15"], include_lowest=True,
    )
    bands = ["0–5", "5–15", ">15"]
    grouped = d.groupby(["model", "depth_band"], observed=False).agg(
        n=("residual_m", "size"),
        rmse=("residual_m", lambda v: float(np.sqrt(np.mean(np.square(v))))),
        bias=("residual_m", "mean"),
    ).reset_index()
    counts = d[d.model == MODELS[0]].groupby("depth_band", observed=False).size()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    x = np.arange(len(bands))
    for ax, metric, ylabel, title in [
        (axes[0], "rmse", "RMSE against supplied labels (m)", "Error magnitude"),
        (axes[1], "bias", "Mean prediction − survey (m)", "Signed bias"),
    ]:
        for name in MODELS:
            values = []
            for band in bands:
                row = grouped[(grouped.model == name) & (grouped.depth_band.astype(str) == band)]
                values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            ax.plot(x, values, marker="o", markersize=5, linewidth=1.8,
                    color=COLORS[name], label=PRETTY[name])
        ax.set_xticks(x, [f"{band}\n(n={int(counts.get(band, 0))})" for band in bands])
        ax.set_xlabel("Absolute supplied survey z value")
        ax.set_ylabel(ylabel)
        ax.set_title(title, weight="bold", color=PALETTE["navy"])
        ax.grid(axis="y", alpha=.8)
    axes[1].axhline(0, color=PALETTE["ink"], linewidth=.9, linestyle="--", zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8,
               frameon=True, facecolor="white", edgecolor="#C8D2DE")
    fig.suptitle("Performance across site-local |z| ranges",
                 fontsize=13, weight="bold", color=PALETTE["navy"])
    save(fig, "Figure_8_depth_stratified_error")


def figure_training() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    for ax, name in zip(axes.flat, NEURAL_MODELS):
        for fold in (1, 2, 3):
            path = RESULTS / "folds" / f"fold_{fold}" / name / "history.json"
            if not path.exists():
                continue
            history = json.loads(path.read_text(encoding="utf-8"))
            key = "val_z_loss" if name == "DepthBinCNN" and "val_z_loss" in history else "val_loss"
            values = history.get(key, [])
            if values:
                ax.plot(np.arange(1, len(values) + 1), values,
                        label=f"Fold {fold}", lw=1.3, color=COLORS[name], alpha=.45 + .15 * fold)
        ax.set_title(PRETTY[name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Validation learning curves across buffered spatial folds",
                 fontsize=13, weight="bold", color=PALETTE["navy"])
    save(fig, "Figure_S1_training_histories")


def main() -> None:
    global SAMPLES, RESULTS, SCENE, PRODUCT, FIG, UTM_ZONE, SCENE_DATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=SAMPLES)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--scene-dir", type=Path, default=SCENE)
    parser.add_argument("--product", default=PRODUCT)
    parser.add_argument("--figures-dir", type=Path, default=FIG)
    parser.add_argument("--utm-zone", default=UTM_ZONE)
    parser.add_argument("--scene-date", default=SCENE_DATE)
    args = parser.parse_args()
    SAMPLES = args.data_path
    RESULTS = args.results_dir
    SCENE = args.scene_dir
    PRODUCT = args.product
    FIG = args.figures_dir
    UTM_ZONE = args.utm_zone
    SCENE_DATE = args.scene_date
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    figure_inputs()
    figure_folds()
    figure_metrics()
    figure_scatter()
    figure_maps()
    figure_depth_strata()
    figure_training()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
