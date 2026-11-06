"""Publication-size two-site figures generated from saved HLS samples and OOF results."""
from __future__ import annotations
import json
from pathlib import Path
from contextlib import ExitStack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"/"manuscript"; OUT.mkdir(parents=True,exist_ok=True)
DEPTH_CMAP=LinearSegmentedColormap.from_list("coast_depth",["#e8f4f2","#85d1c6","#2f8ea8","#17466c","#10253f"])
SITES=[
 {"name":"Anza / Agadir","coast":"Atlantic coast","date":"10 Nov 2022","sample":ROOT/"data"/"processed"/"samples.npz","summary":ROOT/"data"/"processed"/"dataset_summary.json","scene":ROOT/"data"/"raw"/"sentinel2"/"HLS_S30_2022-11-10","product":"HLS.S30.T29RMP.2022314T112209.v2.0","results":ROOT/"results"/"hls_2022-11-10","utm":"29N","datum":"Port of Agadir hydrographic zero","color":"#2864A5"},
 {"name":"Ras El Ma / Ras Kebdana","coast":"Mediterranean coast","date":"07 Aug image · 08 Aug survey 2024","sample":ROOT/"data"/"processed"/"ras_el_ma_2024-08-08"/"samples.npz","summary":ROOT/"data"/"processed"/"ras_el_ma_2024-08-08"/"dataset_summary.json","scene":ROOT/"data"/"raw"/"sentinel2"/"HLS_S30_2024-08-07_RasElMa"/"T30SWD","product":"HLS.S30.T30SWD.2024220T105031.v2.0","results":ROOT/"results"/"hls_ras_2024-08-07","utm":"30N","datum":"Port of Ras Kebdana hydrographic zero","color":"#D97742"},
]
MODELS=["CNN","RatioCNN","CNN_BiLSTM","DenseCNN","DepthBinCNN","StumpfRatio","RandomForest"]
PRETTY={"CNN":"CNN","RatioCNN":"RatioCNN","CNN_BiLSTM":"CNN–BiLSTM","DenseCNN":"DenseCNN","DepthBinCNN":"DepthBinCNN","StumpfRatio":"Stumpf log-ratio","RandomForest":"Random forest"}
MODEL_COLORS={"CNN":"#277DA1","RatioCNN":"#43AA8B","CNN_BiLSTM":"#F8961E","DenseCNN":"#845EC2","DepthBinCNN":"#D45087","StumpfRatio":"#64748B","RandomForest":"#90BE6D"}
ZONE_COLORS={0:"#2864A5",1:"#F2A541",2:"#369A78"}

def style():
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.2,"axes.titlesize":8.8,"axes.labelsize":7.8,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#B8C4D0","axes.linewidth":.7,"axes.labelcolor":"#172033","xtick.color":"#334155","ytick.color":"#334155","text.color":"#172033","figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","grid.color":"#DCE4EC","grid.linewidth":.5,"legend.frameon":False})
def save(fig,name):
 fig.savefig(OUT/f"{name}.png",dpi=450,bbox_inches="tight",facecolor="white")
 fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",facecolor="white")
 plt.close(fig)
def rgb_stretch(a):
 o=np.zeros_like(a,dtype=float)
 for k in range(3):
  x=a[...,k]; g=np.isfinite(x)&(x>-.5)
  lo,hi=np.nanpercentile(x[g],[2,98]) if g.any() else (0,1)
  o[...,k]=np.clip((x-lo)/max(hi-lo,1e-6),0,1)
 return o
def context(site,pad=24):
 pack=np.load(site["sample"],allow_pickle=False); rows=pack["rows"].astype(int); cols=pack["cols"].astype(int)
 with ExitStack() as stack:
  b2=stack.enter_context(rasterio.open(site["scene"]/f"{site['product']}.B02.tif"))
  r0=max(0,int(rows.min())-pad); r1=min(b2.height,int(rows.max())+pad+1); c0=max(0,int(cols.min())-pad); c1=min(b2.width,int(cols.max())+pad+1)
  win=Window(c0,r0,c1-c0,r1-r0); left,bottom,right,top=b2.window_bounds(win); extent=[left/1000,right/1000,bottom/1000,top/1000]
  bands=[]
  for b in ("B04","B03","B02"):
   src=stack.enter_context(rasterio.open(site["scene"]/f"{site['product']}.{b}.tif")); bands.append(src.read(1,window=win).astype("float32")*.0001)
 return pack,rgb_stretch(np.stack(bands,axis=-1)),extent,(r0,r1,c0,c1)
def map_axes(ax,site,extent):
 ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3]); ax.set_aspect("equal",adjustable="box")
 ax.set_xlabel("UTM easting (km)",fontsize=7); ax.set_ylabel(f"UTM northing (km; {site['utm']})",fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(color="white",alpha=.22,lw=.4)
def north(ax):
 ax.annotate("N",xy=(.92,.93),xytext=(.92,.82),xycoords="axes fraction",textcoords="axes fraction",ha="center",va="center",fontsize=7,weight="bold",color="white",arrowprops={"arrowstyle":"-|>","color":"white","lw":1},bbox={"boxstyle":"round,pad=.12","facecolor":"#172033","alpha":.55,"edgecolor":"none"})
def scalebar(ax,extent,length=.5):
 x=extent[0]+.08*(extent[1]-extent[0]); y=extent[2]+.08*(extent[3]-extent[2]); x1=x+length
 if x1<extent[1]-.05*(extent[1]-extent[0]):
  ax.plot([x,x1],[y,y],color="white",lw=3,zorder=20); ax.plot([x,x1],[y,y],color="#172033",lw=.9,zorder=21)
  ax.text((x+x1)/2,y+.03*(extent[3]-extent[2]),f"{length:g} km",ha="center",va="bottom",fontsize=6.5,color="white",weight="bold",bbox={"facecolor":"#172033","alpha":.5,"edgecolor":"none","pad":1},zorder=22)
def box(ax,xy,wh,text,face,edge,fontsize=8.2,bold=False):
 x,y=xy; w,h=wh; ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.014,rounding_size=.02",facecolor=face,edgecolor=edge,lw=1.0))
 ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,weight="bold" if bold else "normal",wrap=True,color="#172033")
def connect(ax,p1,p2,color="#64748B"):
 ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=11,lw=1.1,color=color))

def fig01_data_overview():
 fig,ax=plt.subplots(3,2,figsize=(6.8,7.5),constrained_layout=True)
 for c,site in enumerate(SITES):
  pack,rgb,extent,(r0,r1,c0,c1)=context(site)
  ax[0,c].imshow(rgb,extent=extent,origin="upper",interpolation="nearest"); ax[0,c].set_title(f"({chr(97+c)}) {site['name']} · {site['date']}\nHLS true colour ({site['coast']})",loc="left",weight="bold")
  depth=np.ma.masked_where(~pack["target_grid_mask"][r0:r1,c0:c1],-pack["target_grid"][r0:r1,c0:c1]); vmax=max(5,float(np.percentile(-pack["y"][:,0],98)))
  im=ax[1,c].imshow(depth,extent=extent,origin="upper",cmap=DEPTH_CMAP,vmin=0,vmax=vmax,interpolation="nearest"); ax[1,c].set_title(f"({chr(99+c)}) Aggregated survey depth below local zero",loc="left",weight="bold")
  cb=fig.colorbar(im,ax=ax[1,c],fraction=.04,pad=.015); cb.set_label("Depth (m)",fontsize=7); cb.ax.tick_params(labelsize=6.5)
  sc=ax[2,c].scatter(pack["x_utm"]/1000,pack["y_utm"]/1000,c=pack["survey_points_per_pixel"],s=10+np.clip(pack["survey_points_per_pixel"],0,100)*.12,cmap="magma",linewidths=0,rasterized=True)
  ax[2,c].set_title(f"({chr(101+c)}) Survey support · n={len(pack['y']):,} retained HLS cells",loc="left",weight="bold")
  cb=fig.colorbar(sc,ax=ax[2,c],fraction=.04,pad=.015); cb.set_label("Soundings per 30 m cell",fontsize=7); cb.ax.tick_params(labelsize=6.5)
  for a in ax[:,c]: map_axes(a,site,extent)
  north(ax[0,c]); scalebar(ax[0,c],extent)
 ax[0,0].set_ylabel("True-colour image\n\nUTM northing (km; 29N)",fontsize=7)
 ax[1,0].set_ylabel("Survey depth\n\nUTM northing (km; 29N)",fontsize=7)
 ax[2,0].set_ylabel("Survey support\n\nUTM northing (km; 29N)",fontsize=7)
 fig.suptitle("Study sites, input imagery, survey depth, and retained sample support",fontsize=11.5,weight="bold",color="#17324D")
 save(fig,"Figure_01_data_overview")

def fig02_workflow():
 fig,ax=plt.subplots(figsize=(6.8,5.4)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
 ax.text(.5,.975,"Shared processing · separate site evaluations",ha="center",va="top",fontsize=10.2,weight="bold",color="#17324D")
 box(ax,(.04,.75),(.43,.14),"HLS S30 v2.0 · 30 m\nB02 / B03 / B04 / B08\nFmask cloud and shadow screen","#E6F1FA","#7CA7C9",7.1,True)
 box(ax,(.53,.75),(.43,.14),"Survey labels z (m)\nAnza: Agadir port zero\nRas: Ras Kebdana port zero","#FCEFE7","#DB9A6D",7.1,True)
 box(ax,(.10,.53),(.80,.14),"Align grids · aggregate z by 30 m cell\nExclude mixed shoreline cells · apply Fmask\nBuild 9 × 9 × 4 spectral patches","#E8F5F2","#72B5A8",7.3,True)
 connect(ax,(.255,.75),(.39,.67),"#4E7D9C"); connect(ax,(.745,.75),(.61,.67),"#B9794D")
 box(ax,(.04,.29),(.43,.15),"Five author-designed CNNs\nCNN · RatioCNN · CNN–BiLSTM\nDenseCNN · DepthBinCNN","#F1EBFA","#9A82BF",7.0,True)
 box(ax,(.53,.29),(.43,.15),"Two baselines\nStumpf blue–green log-ratio\nRandom forest","#F0F3F6","#8A98A8",7.0,True)
 connect(ax,(.5,.53),(.25,.44),"#6D5B91"); connect(ax,(.5,.53),(.75,.44),"#78899A")
 box(ax,(.10,.07),(.80,.14),"Three buffered alongshore zones rotate train / validation / test roles\nOut-of-fold predictions → errors, residuals, depth diagnostics","#FFF5DC","#D3A434",7.2,True)
 connect(ax,(.255,.29),(.4,.235),"#6D5B91"); connect(ax,(.745,.29),(.6,.235),"#78899A")
 ax.text(.5,.018,"Local vertical zeros stay separate · no cross-site transfer test",ha="center",va="bottom",fontsize=6.6,color="#526173")
 save(fig,"Figure_03_processing_workflow")

def fig03_architecture():
 fig,ax=plt.subplots(figsize=(6.8,6.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
 ax.text(.5,.985,"Author-designed CNN variants: input to output",ha="center",va="top",fontsize=10.2,weight="bold",color="#17324D")
 box(ax,(.08,.80),(.84,.09),"Input: 9 × 9 × 4 HLS patch · four bands · 30 m\nOutput: survey z (m), relative to the site's port zero","#E6F1FA","#7CA7C9",6.8,True)
 rows=[
  ("CNN","3 × 3 conv encoder · 32 → 64 → 128\nGlobal average pooling → dense head → scalar z","#277DA1"),
  ("RatioCNN","CNN encoder + centre ln(B02/B03)\nDense head → scalar z","#43AA8B"),
  ("CNN–BiLSTM","CNN encoder → four spatial tokens\nBidirectional LSTM → scalar z","#F8961E"),
  ("DenseCNN","CNN encoder → transpose-convolution decoder\n9 × 9 z field → score centre cell","#845EC2"),
  ("DepthBinCNN","CNN encoder → z regression + 3-bin head\nBins use the same survey z labels","#D45087"),
 ]
 y=.715
 for name,detail,color in rows:
  ax.add_patch(FancyBboxPatch((.06,y-.05),.88,.10,boxstyle="round,pad=.008,rounding_size=.015",facecolor="#FAFBFD",edgecolor=color,lw=1.0))
  ax.add_patch(FancyBboxPatch((.06,y-.05),.20,.10,boxstyle="round,pad=.008,rounding_size=.015",facecolor=color,edgecolor=color,lw=0))
  ax.text(.16,y,name,ha="center",va="center",fontsize=7.0,weight="bold",color="white")
  ax.text(.29,y,detail,ha="left",va="center",fontsize=6.3,color="#172033",linespacing=1.1)
  y-=.125
 box(ax,(.08,.045),(.84,.10),"Baselines: Stumpf blue–green log-ratio and random forest\nBoth are fitted within each training fold","#F0F3F6","#8A98A8",6.4,True)
 save(fig,"Figure_04_model_architectures")

def fig04_spatial_validation():
 fig=plt.figure(figsize=(6.8,5.4),constrained_layout=True); gs=fig.add_gridspec(2,2,height_ratios=[3.5,1.25])
 for c,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=21); ax=fig.add_subplot(gs[0,c]); ax.imshow(rgb,extent=extent,origin="upper",alpha=.78,interpolation="nearest")
  for z in range(3):
   m=pack["zones"]==z; ax.scatter(pack["x_utm"][m]/1000,pack["y_utm"][m]/1000,s=6,c=ZONE_COLORS[z],edgecolors="white",linewidths=.12,label=f"Zone {z}",rasterized=True)
  ax.set_title(site["name"],weight="bold",fontsize=8.8); map_axes(ax,site,extent); ax.legend(loc="lower left",fontsize=6.5,ncol=3,frameon=True,facecolor="white",framealpha=.9)
  north(ax); scalebar(ax,extent)
 tabax=fig.add_subplot(gs[1,:]); tabax.axis("off")
 cells=[["Fold","Zone 0","Zone 1","Zone 2"],["1 · test 0","Test","Validation","Training"],["2 · test 1","Training","Test","Validation"],["3 · test 2","Validation","Training","Test"]]
 tab=tabax.table(cellText=[r[1:] for r in cells[1:]],colLabels=cells[0],cellLoc="center",loc="center",colWidths=[.18,.25,.28,.25])
 tab.auto_set_font_size(False); tab.set_fontsize(7.2); tab.scale(1,.95)
 for (row,col),cell in tab.get_celld().items():
  if row==0: cell.set_facecolor("#E8EEF4"); cell.set_text_props(weight="bold",color="#17324D")
  elif row>0:
   role=cell.get_text().get_text(); color={"Training":"#E6F1FA","Validation":"#FFF1D6","Test":"#FBE7E5"}.get(role,"white"); cell.set_facecolor(color)
 fig.suptitle("Within-site buffered spatial validation: zone map and role rotation",fontsize=11.3,weight="bold",color="#17324D")
 fig.text(.5,.005,"A patch-width buffer separates alongshore zones. The test fold is never used to fit or select model parameters; sites have separate coordinate systems and local vertical references.",ha="center",fontsize=7.2,color="#526173")
 save(fig,"Figure_05_spatial_validation")

def fig05_errors():
 fig,axes=plt.subplots(2,2,figsize=(6.8,7.8),constrained_layout=True)
 for r,site in enumerate(SITES):
  summary=pd.read_csv(site["results"]/"model_summary.csv").set_index("model").loc[MODELS]; folds=pd.read_csv(site["results"]/"metrics_by_fold.csv")
  for c,metric in enumerate(("rmse_m","mae_m")):
   ax=axes[r,c]; y=np.arange(len(MODELS)); vals=summary[metric].to_numpy(float)
   ax.barh(y,vals,color=[MODEL_COLORS[m] for m in MODELS],height=.58,alpha=.83,zorder=2)
   for j,m in enumerate(MODELS):
    f=folds.loc[folds.model==m,metric].to_numpy(float); ax.scatter(f,y[j]+np.linspace(-.10,.10,len(f)),s=22,c="#172033",edgecolors="white",linewidths=.45,zorder=3)
    ax.text(vals[j]+max(vals)*.02,y[j],f"{vals[j]:.2f}",va="center",fontsize=6.7,color="#334155")
   ax.set_yticks(y,[PRETTY[m] for m in MODELS]); ax.invert_yaxis(); ax.set_xlim(0,max(vals)*1.32); ax.set_xlabel(f"{metric.replace('_m','').upper()} (m; local datum)",fontsize=7)
   ax.set_title(f"{site['name']} · {metric.replace('_m','').upper()}",loc="left",fontsize=8.4,weight="bold"); ax.grid(axis="x",alpha=.5); ax.set_axisbelow(True); ax.tick_params(labelsize=6.6)
 fig.suptitle("Pooled out-of-fold errors and spatial-fold variation",fontsize=11.3,weight="bold",color="#17324D")
 fig.text(.5,-.005,"Bars show pooled errors; dots show the three spatial test zones. Axes are site-specific. Errors are relative to local survey labels, not a cross-site transfer evaluation.",ha="center",fontsize=7.2,color="#526173")
 save(fig,"Figure_06_sitewise_model_errors")

def fig06_prediction_maps():
 fig=plt.figure(figsize=(7.2,6.25)); gs=fig.add_gridspec(2,3,width_ratios=[1,1,.045],left=.08,right=.94,top=.88,bottom=.10,wspace=.24,hspace=.34)
 axes=np.array([[fig.add_subplot(gs[r,c]) for c in range(2)] for r in range(2)])
 for r,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=22)
  d=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=d.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin()); d=d[d.model==model]
  lim=max(1.0,float(np.percentile(np.abs(d.reference_m),98)),float(np.percentile(np.abs(d.prediction_m),98)))
  norm=plt.Normalize(vmin=0,vmax=lim)
  for c,(field,label) in enumerate((("reference_m","Aggregated survey z"),("prediction_m","Held-out prediction"))):
   ax=axes[r,c]; ax.imshow(rgb,extent=extent,origin="upper",alpha=.35,interpolation="nearest")
   sc=ax.scatter(d.x_utm/1000,d.y_utm/1000,c=-d[field],s=9 if len(d)>2000 else 13,cmap=DEPTH_CMAP,norm=norm,linewidths=.18,edgecolors="white",rasterized=True)
   ax.set_title(f"{site['name']} · {label}\n{PRETTY[model]} · RMSE {rm[model]:.2f} m",loc="left",fontsize=7.8,weight="bold")
   map_axes(ax,site,extent); north(ax); scalebar(ax,extent)
  cax=fig.add_subplot(gs[r,2]); cb=fig.colorbar(sc,cax=cax)
  cb.set_label("Depth below local port zero (m)",fontsize=6.9); cb.ax.tick_params(labelsize=6.3)
 fig.suptitle("Held-out depth predictions at retained 30 m centres",fontsize=10.3,weight="bold",color="#17324D",y=.97)
 fig.text(.5,.025,"Survey labels and one out-of-fold prediction per cell; no continuous surface is interpolated. Colour limits are site-specific because local vertical zeros differ.",ha="center",fontsize=6.6,color="#526173")
 save(fig,"Figure_07_oof_depth_maps")

def fig06_residual_maps():
 fig,axes=plt.subplots(1,2,figsize=(6.8,4.0),constrained_layout=True)
 for ax,site in zip(axes,SITES):
  pack,rgb,extent,_=context(site,pad=22); d=pd.read_csv(site["results"]/"out_of_fold_predictions.csv"); rm=d.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2)))); model=str(rm.idxmin()); d=d[d.model==model]
  lim=max(.5,float(np.percentile(np.abs(d.residual_m),98))); ax.imshow(rgb,extent=extent,origin="upper",alpha=.60,interpolation="nearest")
  sc=ax.scatter(d.x_utm/1000,d.y_utm/1000,c=d.residual_m,s=12,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),linewidths=.15,edgecolors="white",rasterized=True)
  ax.set_title(f"{site['name']} · {PRETTY[model]} · RMSE {rm[model]:.2f} m",fontsize=8.2,weight="bold"); map_axes(ax,site,extent); north(ax); scalebar(ax,extent)
  cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.025); cb.set_label("z residual: prediction − survey (m)",fontsize=7); cb.ax.tick_params(labelsize=6.5)
 fig.suptitle("Spatial pattern of residuals for the lowest-RMSE model at each site",fontsize=11.2,weight="bold",color="#17324D")
 fig.text(.5,-.015,"Only held-out predictions are mapped. Positive residuals are less-negative than survey labels. Each site uses a local zero and its own colour scale; no residual surface is interpolated.",ha="center",fontsize=7.2,color="#526173")
 save(fig,"Figure_08_best_model_residual_maps")

def fig07_diagnostics():
 fig,axes=plt.subplots(2,2,figsize=(6.8,6.8),constrained_layout=True)
 for r,site in enumerate(SITES):
  data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv"); rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2)))); model=str(rm.idxmin()); d=data[data.model==model]; obs=d.reference_m.to_numpy(float); pred=d.prediction_m.to_numpy(float); res=pred-obs
  ax=axes[r,0]; lo=min(float(np.percentile(obs,1)),float(np.percentile(pred,1))); hi=max(float(np.percentile(obs,99)),float(np.percentile(pred,99)))
  hb=ax.hexbin(obs,pred,gridsize=24,mincnt=1,bins="log",cmap="Blues",linewidths=0); ax.plot([lo,hi],[lo,hi],ls="--",lw=1,color="#C44536"); ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Survey z (m; local zero)",ylabel="Predicted z (m)"); ax.set_title(f"{site['name']} · {PRETTY[model]}\nRMSE {rm[model]:.2f} m",loc="left",fontsize=8.1,weight="bold"); ax.tick_params(labelsize=6.5)
  cb=fig.colorbar(hb,ax=ax,fraction=.045,pad=.02); cb.set_label("Pixel count (log scale)",fontsize=6.8); cb.ax.tick_params(labelsize=6.2)
  ax=axes[r,1]; groups=[d.loc[d.fold==f,"residual_m"].to_numpy(float) for f in sorted(d.fold.unique())]
  bp=ax.boxplot(groups,patch_artist=True,showfliers=False,widths=.55,medianprops={"color":"#172033","linewidth":1.2},whiskerprops={"color":site["color"]},capprops={"color":site["color"]})
  for patch in bp["boxes"]: patch.set_facecolor(site["color"]); patch.set_alpha(.58)
  ax.axhline(0,color="#C44536",ls="--",lw=.9); ax.set_xticks([1,2,3],["Zone 0","Zone 1","Zone 2"]); ax.set_ylabel("z residual (prediction − survey; m)",fontsize=7); ax.set_title("Residuals by held-out zone",loc="left",fontsize=8.1,weight="bold"); ax.tick_params(labelsize=6.5); ax.grid(axis="y",alpha=.4)
 fig.suptitle("Best-RMSE model: agreement and geographic variation",fontsize=11.3,weight="bold",color="#17324D")
 fig.text(.5,-.01,"Hexbin panels summarize observed–predicted density; boxplots show the distribution of residuals in each held-out spatial zone. Pixel observations are spatially dependent.",ha="center",fontsize=7.2,color="#526173")
 save(fig,"Figure_09_residual_diagnostics")

def fig08_depth_stratified():
 fig=plt.figure(figsize=(6.8,5.9)); axes=[fig.add_axes([.15,.57,.80,.27]),fig.add_axes([.15,.22,.80,.27])]; labels=["0–5 m","5–15 m",">15 m"]
 for ax,site in zip(axes,SITES):
  d=pd.read_csv(site["results"]/"out_of_fold_predictions.csv"); depth=np.abs(d.reference_m.to_numpy(float)); bins=[(depth<5),(depth>=5)&(depth<15),(depth>=15)]; counts=[int(m.sum()/len(MODELS)) for m in bins]
  for model in MODELS:
   part=d[d.model==model]; dep=np.abs(part.reference_m.to_numpy(float)); err=part.residual_m.to_numpy(float); vals=[]
   for low,high in [(0,5),(5,15),(15,np.inf)]:
    m=(dep>=low)&(dep<high); vals.append(np.sqrt(np.mean(err[m]**2)) if m.any() else np.nan)
   ax.plot(np.arange(3),vals,marker="o",lw=1.25,ms=3.6,color=MODEL_COLORS[model],label=PRETTY[model])
  ax.set_xticks(np.arange(3),[f"{lab}\nn={n:,}" for lab,n in zip(labels,counts)]); ax.set_ylabel("RMSE (m; local datum)",fontsize=7.3); ax.set_title(site["name"],loc="left",fontsize=8.7,weight="bold"); ax.grid(alpha=.4); ax.tick_params(labelsize=7)
 handles=[plt.Line2D([0],[0],color=MODEL_COLORS[m],marker="o",lw=1.2,ms=3.5,label=PRETTY[m]) for m in MODELS]
 fig.legend(handles=handles,ncol=4,loc="lower center",bbox_to_anchor=(.5,.055),fontsize=6.8,columnspacing=1.3,handletextpad=.4)
 fig.suptitle("Out-of-fold error by depth range",fontsize=11.3,weight="bold",color="#17324D",y=.97)
 fig.text(.5,.012,"Depth bands use |survey z| relative to each site's own port zero. Counts are unique HLS centres. Trends are descriptive; the shallowest Anza band has limited support.",ha="center",va="bottom",fontsize=6.6,color="#526173")
 save(fig,"Figure_10_depth_stratified_errors")

def fig09_quality():
 steps=[[("Survey rows",24508),("Negative-z\nrecords",23170),("Pixels with\nnegative labels",2660),("Eligible pixels\nbefore patch QC",2535),("Retained 30 m\ncentres",1373)],[("Survey rows",383259),("Unique XYZ\nrows",372664),("Unique XY\ngroups",372663),("Eligible pixels\nbefore split",5483),("Retained 30 m\ncentres",3817)]]
 face=["#E6F1FA","#E8F5F2","#F1EBFA","#FFF5DC","#FCEFE7"]; edge=["#7CA7C9","#72B5A8","#9A82BF","#D3A434","#DB9A6D"]
 fig,axes=plt.subplots(2,1,figsize=(6.8,5.1));
 for r,(site,arr,ax) in enumerate(zip(SITES,steps,axes)):
  ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); ax.text(.015,.96,f"{site['name']} · {site['coast']}",ha="left",va="top",fontsize=9,weight="bold",color=site["color"])
  xs=np.linspace(.025,.82,len(arr)); width=.155; y=.40; h=.34
  for i,((label,val),x) in enumerate(zip(arr,xs)):
   box(ax,(x,y),(width,h),f"{label}\n\n{val:,}",face[i],edge[i],7.3,True)
   if i<len(arr)-1: connect(ax,(x+width,y+h/2),(xs[i+1]-.012,y+h/2),site["color"])
  note="85 quality-flagged negative cells, 38 with <3 negative points, 49 mixed-sign cells; these reasons may overlap. 205 candidate patches failed full-window Fmask QC." if r==0 else "10,595 exact duplicate XYZ rows removed; one repeated XY group median-collapsed (0.03 m maximum z spread); 351 candidate patches failed Fmask QC."
  ax.text(.03,.16,note,ha="left",va="center",fontsize=7.1,color="#526173",wrap=True)
 fig.suptitle("Survey-to-pixel quality-control funnel",fontsize=11.5,weight="bold",color="#17324D")
 fig.text(.5,.01,"Counts transition from point records to spatial groups and then 30 m HLS cells; units are indicated at each stage.",ha="center",fontsize=7.2,color="#526173")
 fig.tight_layout(rect=[0,.035,1,.95]); save(fig,"Figure_02_data_quality_funnel")

def fig_s1_disagreement():
 fig,axes=plt.subplots(1,2,figsize=(6.8,4.0),constrained_layout=True)
 for ax,site in zip(axes,SITES):
  d=pd.read_csv(site["results"]/"out_of_fold_predictions.csv"); wide=d.pivot_table(index=["pixel_id","x_utm","y_utm"],columns="model",values="prediction_m",aggfunc="first").reindex(columns=MODELS); spread=wide.std(axis=1,ddof=0).to_numpy(float); xy=wide.index.to_frame(index=False)
  sc=ax.scatter(xy.x_utm/1000,xy.y_utm/1000,c=spread,s=9,cmap="cividis",vmin=0,vmax=max(.8,float(np.percentile(spread,98))),linewidths=0,rasterized=True)
  ax.set_title(site["name"],fontsize=8.8,weight="bold"); ax.set_xlabel("UTM easting (km)",fontsize=7); ax.set_ylabel(f"UTM northing (km; {site['utm']})",fontsize=7); ax.set_aspect("equal",adjustable="box"); ax.tick_params(labelsize=6.5); ax.grid(alpha=.35)
  cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.025); cb.set_label("Across-model SD (m)",fontsize=7); cb.ax.tick_params(labelsize=6.5)
 fig.suptitle("Supplementary: spatial variation in model disagreement",fontsize=11.3,weight="bold",color="#17324D")
 fig.text(.5,-.01,"Standard deviation across seven out-of-fold model predictions at the same cell. This is descriptive model disagreement, not calibrated predictive uncertainty.",ha="center",fontsize=7.2,color="#526173")
 save(fig,"Figure_S1_model_disagreement")

def main():
 style(); fig01_data_overview(); fig02_workflow(); fig03_architecture(); fig04_spatial_validation(); fig05_errors(); fig06_prediction_maps(); fig06_residual_maps(); fig07_diagnostics(); fig08_depth_stratified(); fig09_quality(); fig_s1_disagreement()
 print("Saved readable two-site figures to",OUT)
if __name__=="__main__": main()
