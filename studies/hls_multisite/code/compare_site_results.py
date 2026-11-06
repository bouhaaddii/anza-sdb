"""Create a cautious, site-wise comparison of spatial holdout metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITES = [
    {
        "name": "Anza / Agadir (2022)",
        "results": ROOT / "results" / "hls_2022-11-10",
        "vertical_reference": "Port of Agadir hydrographic zero (user-reported)",
        "color": "#2F6BFF",
    },
    {
        "name": "Ras El Ma / Ras Kebdana (2024)",
        "results": ROOT / "results" / "hls_ras_2024-08-07",
        "vertical_reference": "Port of Ras Kebdana hydrographic zero (survey-plan cartouche)",
        "color": "#E68A00",
    },
]
MODELS = ["CNN", "RatioCNN", "CNN_BiLSTM", "DenseCNN", "DepthBinCNN", "StumpfRatio", "RandomForest"]
PRETTY = {
    "CNN": "CNN", "RatioCNN": "Ratio-augmented CNN", "CNN_BiLSTM": "CNN–BiLSTM",
    "DenseCNN": "Dense-decoder CNN", "DepthBinCNN": "Depth-bin CNN",
    "StumpfRatio": "Log-ratio regression", "RandomForest": "Random forest",
}


def summarize() -> pd.DataFrame:
    rows = []
    for site in SITES:
        pred = pd.read_csv(site["results"] / "out_of_fold_predictions.csv")
        for model in MODELS:
            part = pred[pred.model == model]
            obs = part["reference_m"].to_numpy(dtype=float)
            estimate = part["prediction_m"].to_numpy(dtype=float)
            residual = estimate - obs
            sd = float(np.std(obs, ddof=1)) if len(obs) > 1 else float("nan")
            rows.append({
                "site": site["name"],
                "scene_date": "2022-11-10" if "Anza" in site["name"] else "2024-08-07",
                "vertical_reference": site["vertical_reference"],
                "model": model,
                "n_pixels": int(len(part)),
                "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
                "mae_m": float(np.mean(np.abs(residual))),
                "mean_bias_m": float(np.mean(residual)),
                "r2": float(1 - np.sum(residual ** 2) / np.sum((obs - obs.mean()) ** 2)),
                "label_standard_deviation_m": sd,
                "normalized_rmse_by_label_sd": float(np.sqrt(np.mean(residual ** 2)) / sd) if sd else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = ROOT / "results"
    figure_dir = ROOT / "figures"
    table = summarize()
    table.to_csv(output_dir / "multisite_model_comparison.csv", index=False)
    (output_dir / "multisite_comparison_metadata.json").write_text(json.dumps({
        "scope": "Independent spatial holdout evaluation within each study site; these are not cross-site transfer scores.",
        "caution": "The sites use separate local port hydrographic zeros. RMSE and MAE describe errors against each site's own labels. Do not interpret differences as absolute vertical agreement or pooled-depth performance. Ras HLS imagery is from one day before its survey.",
        "sites": [{"name": x["name"], "vertical_reference": x["vertical_reference"]} for x in SITES],
    }, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    x = np.arange(len(MODELS))
    width = .36
    for site_index, site in enumerate(SITES):
        values = table[table.site == site["name"]].set_index("model").loc[MODELS]
        offset = (-width / 2) if site_index == 0 else (width / 2)
        axes[0].bar(x + offset, values.rmse_m, width, color=site["color"], label=site["name"], alpha=.88)
        axes[1].bar(x + offset, values.mae_m, width, color=site["color"], label=site["name"], alpha=.88)
    axes[0].set_title("Spatial holdout RMSE", weight="bold")
    axes[1].set_title("Spatial holdout MAE", weight="bold")
    for ax in axes:
        ax.set_ylabel("Error against site-local labels (m)")
        ax.set_xticks(x, [PRETTY[m] for m in MODELS], rotation=24, ha="right")
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Model performance within each coastal site\nIndependent spatial folds; no cross-site transfer", fontsize=13, weight="bold")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "Figure_9_multisite_spatial_skill.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "Figure_9_multisite_spatial_skill.pdf", bbox_inches="tight")
    plt.close(fig)
    print(table[["site", "model", "n_pixels", "rmse_m", "mae_m", "r2", "normalized_rmse_by_label_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
