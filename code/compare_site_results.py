"""Summarize and visualize independent, site-local spatial holdout results."""
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
    {"name":"Anza / Agadir (2022)","results":ROOT/"results"/"anza_agadir_2022-11-10","vertical_reference":"Hydrographic zero of the Port of Agadir","color":"#2864A5"},
    {"name":"Ras El Ma / Ras Kebdana (2024)","results":ROOT/"results"/"ras_el_ma_2024-08-08","vertical_reference":"Hydrographic zero of the Port of Ras Kebdana","color":"#D97742"},
]
MODELS=["CNN","RatioCNN","CNN_BiLSTM","DenseCNN","DepthBinCNN","StumpfRatio","RandomForest"]
PRETTY={"CNN":"CNN","RatioCNN":"RatioCNN","CNN_BiLSTM":"CNN–BiLSTM","DenseCNN":"DenseCNN","DepthBinCNN":"DepthBinCNN","StumpfRatio":"Stumpf log-ratio","RandomForest":"Random forest"}

def summarize() -> pd.DataFrame:
    rows=[]
    for site in SITES:
        pred=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
        folds=pd.read_csv(site["results"]/"metrics_by_fold.csv")
        for model in MODELS:
            d=pred[pred.model==model]
            obs=d.reference_m.to_numpy(float); estimate=d.prediction_m.to_numpy(float); err=estimate-obs
            sd=float(np.std(obs,ddof=1))
            row={"site":site["name"],"scene_date":"2022-11-10" if "Anza" in site["name"] else "2024-08-07","vertical_reference":site["vertical_reference"],"model":model,"n_pixels":int(len(d)),"rmse_m":float(np.sqrt(np.mean(err**2))),"mae_m":float(np.mean(np.abs(err))),"mean_bias_m":float(np.mean(err)),"r2":float(1-np.sum(err**2)/np.sum((obs-obs.mean())**2)),"label_standard_deviation_m":sd,"normalized_rmse_by_label_sd":float(np.sqrt(np.mean(err**2))/sd),"normalized_mae_by_label_sd":float(np.mean(np.abs(err))/sd)}
            for metric in ("rmse_m","mae_m"):
                vals=folds.loc[folds.model==model,metric].to_numpy(float)
                row[f"fold_{metric}_mean_m"]=float(np.mean(vals)); row[f"fold_{metric}_sd_m"]=float(np.std(vals,ddof=1))
                row[f"fold_{metric}_mean_normalized"]=row[f"fold_{metric}_mean_m"]/sd
            rows.append(row)
    return pd.DataFrame(rows)

def main() -> None:
    outdir=ROOT/"results"; figdir=ROOT/"figures"; figdir.mkdir(exist_ok=True)
    table=summarize(); table.to_csv(outdir/"multisite_model_comparison.csv",index=False)
    (outdir/"multisite_comparison_metadata.json").write_text(json.dumps({
        "scope":"Independent, within-site spatial holdout evaluation; no model was trained on one site and tested on the other.",
        "normalization":"Normalized errors divide site-specific RMSE or MAE by the standard deviation of that site's aggregated survey labels. They compare relative error to local label spread, not absolute vertical accuracy across datums.",
        "caution":"Anza labels refer to the Port of Agadir hydrographic zero; Ras El Ma labels refer to the Port of Ras Kebdana hydrographic zero. The two local zeros are not tied. Ras imagery is one day before its survey. Do not pool elevations or interpret normalized metrics as cross-site transfer.",
        "sites":[{"name":s["name"],"vertical_reference":s["vertical_reference"]} for s in SITES]
    },indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#CBD5E1","axes.labelcolor":"#172033","text.color":"#172033","figure.facecolor":"white","axes.facecolor":"white","grid.color":"#E2E8F0"})
    fig,axes=plt.subplots(1,2,figsize=(6.65,5.4),sharey=True,constrained_layout=True)
    y=np.arange(len(MODELS)); offsets={SITES[0]["name"]:-.14,SITES[1]["name"]:.14}
    for ax,metric,title in zip(axes,["normalized_rmse_by_label_sd","normalized_mae_by_label_sd"],["Normalized RMSE","Normalized MAE"]):
        for site in SITES:
            part=table[table.site==site["name"]].set_index("model").loc[MODELS]
            x=part[metric].to_numpy(float); ysite=y+offsets[site["name"]]
            ax.hlines(ysite,0,x,color=site["color"],alpha=.18,lw=2)
            ax.scatter(x,ysite,s=47,c=site["color"],label=site["name"],edgecolors="white",linewidths=.65,zorder=3)
            for i,val in enumerate(x): ax.text(val+.012,ysite[i],f"{val:.2f}",va="center",fontsize=7.4,color=site["color"])
        ax.set_title(title,weight="bold"); ax.set_xlabel("Error / SD of local survey labels")
        ax.set_xlim(left=0); ax.grid(axis="x",alpha=.65); ax.set_axisbelow(True)
        ax.set_yticks(y,[PRETTY[m] for m in MODELS])
    axes[0].legend(frameon=False,loc="lower right",fontsize=8)
    fig.suptitle("Relative model skill under independent spatial holdouts",fontsize=11.5,weight="bold",color="#17324D")
    fig.text(.5,-.015,"Each site is normalized to its own label distribution. The local port zeros differ; these scores do not test cross-site transfer or absolute vertical agreement.",ha="center",fontsize=8,color="#526173")
    fig.savefig(figdir/"Figure_07_relative_skill.png",dpi=450,bbox_inches="tight")
    fig.savefig(figdir/"Figure_07_relative_skill.pdf",bbox_inches="tight")
    plt.close(fig)
    print(table[["site","model","n_pixels","rmse_m","mae_m","normalized_rmse_by_label_sd","normalized_mae_by_label_sd"]].to_string(index=False))
if __name__=="__main__": main()

