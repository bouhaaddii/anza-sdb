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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
import matplotlib.patheffects as path_effects
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"/"rse_14p"; OUT.mkdir(parents=True,exist_ok=True)
DEPTH_CMAP=LinearSegmentedColormap.from_list("coast_depth",["#e8f4f2","#85d1c6","#2f8ea8","#17466c","#10253f"])
SITES=[
 {"name":"Anza / Agadir","coast":"Atlantic coast","date":"10 Nov 2022","sample":ROOT/"data"/"processed"/"samples.npz","summary":ROOT/"data"/"processed"/"dataset_summary.json","scene":ROOT/"data"/"raw"/"sentinel2"/"HLS_S30_2022-11-10","product":"HLS.S30.T29RMP.2022314T112209.v2.0","results":ROOT/"results"/"hls_2022-11-10","utm":"29N","datum":"Port of Agadir hydrographic zero","color":"#2864A5"},
 {"name":"Ras El Ma / Ras Kebdana","coast":"Mediterranean coast","date":"07 Aug image · 08 Aug survey 2024","sample":ROOT/"data"/"processed"/"ras_el_ma_2024-08-08"/"samples.npz","summary":ROOT/"data"/"processed"/"ras_el_ma_2024-08-08"/"dataset_summary.json","scene":ROOT/"data"/"raw"/"sentinel2"/"HLS_S30_2024-08-07_RasElMa"/"T30SWD","product":"HLS.S30.T30SWD.2024220T105031.v2.0","results":ROOT/"results"/"hls_ras_2024-08-07","utm":"30N","datum":"Port of Ras Kebdana hydrographic zero","color":"#D97742"},
]
MODELS=["CNN","RatioCNN","CNN_BiLSTM","DenseCNN","DepthBinCNN","StumpfRatio","RandomForest"]
PRETTY={"CNN":"CNN","RatioCNN":"RatioCNN","CNN_BiLSTM":"CNN–BiLSTM","DenseCNN":"DenseCNN","DepthBinCNN":"DepthBinCNN","StumpfRatio":"Stumpf log-ratio","RandomForest":"Random forest"}
MODEL_COLORS={"CNN":"#277DA1","RatioCNN":"#43AA8B","CNN_BiLSTM":"#F8961E","DenseCNN":"#845EC2","DepthBinCNN":"#D45087","StumpfRatio":"#64748B","RandomForest":"#90BE6D"}
ZONE_COLORS={0:"#2864A5",1:"#F2A541",2:"#369A78"}
MAP_DOT_SIZE=1.8  # Match the small 30 m cell markers in Supplementary Figures S1–S2.
SOUNDING_SUPPORT_DOT_SIZE=0.6  # Smaller symbols for dense sounding counts in Figure 1e–f.

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
def compact_map_axes(ax,site,extent):
 ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3]); ax.set_aspect("equal",adjustable="box")
 ax.set_xlabel("E (km)",fontsize=5.0,labelpad=1); ax.set_ylabel("N (km)",fontsize=5.0,labelpad=1)
 ax.tick_params(labelsize=4.6,pad=1); ax.grid(color="white",alpha=.22,lw=.35)
def north(ax):
 """Add a compact, vector-drawn eight-point compass rose to a map."""
 cx,cy,r=.90,.88,.047
 ax.add_patch(Circle((cx,cy),r*1.32,transform=ax.transAxes,facecolor="white",
                     edgecolor="#17324D",linewidth=.55,alpha=.96,zorder=40))
 center=np.array([cx,cy])
 for k in range(8):
  theta=np.deg2rad(90-45*k); direction=np.array([np.cos(theta),np.sin(theta)])
  perpendicular=np.array([-direction[1],direction[0]])
  cardinal=(k%2==0); outer=r*(1.0 if cardinal else .70)
  inner=r*.16; half=r*(.20 if cardinal else .14)
  p0=center+direction*outer
  p1=center+direction*inner+perpendicular*half
  p2=center-direction*inner
  p3=center+direction*inner-perpendicular*half
  color="#17324D" if k in (0,2,4,6) else "#F8FAFC"
  ax.add_patch(Polygon([p0,p1,p2,p3],closed=True,transform=ax.transAxes,
                       facecolor=color,edgecolor="#17324D",linewidth=.42,zorder=42))
 ax.add_patch(Circle((cx,cy),r*.12,transform=ax.transAxes,facecolor="#E5A33B",
                     edgecolor="#17324D",linewidth=.4,zorder=43))
 ax.text(cx,cy+r*1.48,"N",transform=ax.transAxes,ha="center",va="center",
         fontsize=5.4,weight="bold",color="#17324D",
         bbox={"facecolor":"white","edgecolor":"none","alpha":.94,"pad":.3},zorder=44)
 for label,dx,dy in (("E",1.52,0),("S",0,-1.50),("W",-1.52,0)):
  ax.text(cx+dx*r,cy+dy*r,label,transform=ax.transAxes,ha="center",va="center",
          fontsize=3.6,weight="bold",color="#17324D",
          bbox={"facecolor":"white","edgecolor":"none","alpha":.94,"pad":.15},zorder=44)
def scalebar(ax,extent,length=.5):
 x=extent[0]+.08*(extent[1]-extent[0]); y=extent[2]+.08*(extent[3]-extent[2]); x1=x+length
 if x1<extent[1]-.05*(extent[1]-extent[0]):
  ax.plot([x,x1],[y,y],color="white",lw=3,zorder=20); ax.plot([x,x1],[y,y],color="#172033",lw=.9,zorder=21)
  ax.text((x+x1)/2,y+.03*(extent[3]-extent[2]),f"{length:g} km",ha="center",va="bottom",fontsize=6.5,color="white",weight="bold",bbox={"facecolor":"#172033","alpha":.5,"edgecolor":"none","pad":1},zorder=22)
def supported_triangulation(x,y,max_edge_km=.045):
 """Triangulation limited to neighbouring 30 m cell centres; no long-gap bridging."""
 tri=mtri.Triangulation(np.asarray(x,float),np.asarray(y,float))
 t=tri.triangles; pts=np.column_stack([tri.x,tri.y])
 edge_lengths=np.maximum.reduce([
  np.linalg.norm(pts[t[:,0]]-pts[t[:,1]],axis=1),
  np.linalg.norm(pts[t[:,1]]-pts[t[:,2]],axis=1),
  np.linalg.norm(pts[t[:,2]]-pts[t[:,0]],axis=1)])
 tri.set_mask(edge_lengths>max_edge_km)
 return tri
def depth_levels(depth,step=2.0):
 v=np.asarray(depth,float); v=v[np.isfinite(v)]
 if not len(v): return np.array([])
 hi=float(np.nanpercentile(v,98))
 levels=np.arange(step,np.floor(hi/step)*step+.01,step)
 if not len(levels) and hi>1: levels=np.array([round(hi/2,1)])
 return levels
def add_depth_map(ax,x,y,depth,site,extent,vmax,show_rose=True):
 """Draw a labelled isobath map using only supported triangles between cell centres."""
 x=np.asarray(x,float);y=np.asarray(y,float);depth=np.asarray(depth,float)
 tri=supported_triangulation(x,y)
 if tri.mask is None or np.any(~tri.mask):
  fill_levels=np.linspace(0,vmax,18)
  cf=ax.tricontourf(tri,depth,levels=fill_levels,cmap=DEPTH_CMAP,
                    norm=Normalize(0,vmax),extend="max",antialiased=True)
  levels=depth_levels(depth)
  if len(levels):
   cs=ax.tricontour(tri,depth,levels=levels,colors="#F8FAFC",linewidths=.55,alpha=.95)
   cs.set_path_effects([path_effects.Stroke(linewidth=1.05,foreground="#17324D",alpha=.8),
                        path_effects.Normal()])
   ax.clabel(cs,levels=levels[::2],fmt=lambda z:f"{z:.0f}",fontsize=5.0,
             inline=True,inline_spacing=2)
 else:
  cf=ax.scatter(x,y,c=depth,s=2.2,cmap=DEPTH_CMAP,norm=Normalize(0,vmax),linewidths=0)
 ax.scatter(x,y,s=1.15,c="#102A43",alpha=.42,linewidths=0,rasterized=True,zorder=5)
 compact_map_axes(ax,site,extent)
 if show_rose:north(ax)
 return cf,tri
def set_3d_panes(ax):
 for pane in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane):
  pane.set_facecolor((.96,.98,.99,.72));pane.set_edgecolor("#B8C4D0")
 ax.grid(True,alpha=.16)
def method_band_cube(ax,patch):
 """Plot the actual four-band 9 x 9 reflectance patch as a stacked raster cube."""
 xx,yy=np.meshgrid(np.arange(9),np.arange(9))
 palettes=("Blues","Greens","Reds","Purples")
 for k,palette in enumerate(palettes):
  values=np.asarray(patch[:,:,k],float)
  lo,hi=np.nanpercentile(values,[2,98]); hi=max(float(hi),float(lo)+1e-6)
  colors=plt.get_cmap(palette)(np.clip((values-lo)/(hi-lo),0,1))
  ax.plot_surface(xx,yy,np.full_like(xx,float(k)),facecolors=colors,
                  edgecolor="#F8FAFC",linewidth=.28,shade=False,antialiased=False)
 ax.set(xlim=(0,8),ylim=(0,8),zlim=(-.25,3.35),xlabel="Col.",ylabel="Row")
 ax.set_xticks([0,4,8]);ax.set_yticks([0,4,8]);ax.set_zticks([0,1,2,3],["B02","B03","B04","B08"])
 ax.tick_params(labelsize=4.6,pad=0);ax.view_init(elev=26,azim=-58);set_3d_panes(ax)
 ax.set_title("Measured 9 × 9 × 4 reflectance cube",fontsize=6.1,weight="bold",pad=0)
def method_depth_patch(ax,pack,idx,prediction,model):
 """Show the true 9 x 9 local labels and the model's one central held-out output."""
 z=-np.asarray(pack["target_patches"][idx,:,:,0],float)
 weights=np.asarray(pack["target_weights"][idx,:,:,0],float)
 z=np.where(weights>0,z,np.nan)
 xx,yy=np.meshgrid(np.arange(9),np.arange(9))
 vmin=max(0,float(np.nanmin(z))-.5);vmax=max(float(np.nanmax(z)),float(-prediction))+.5
 colors=DEPTH_CMAP(Normalize(vmin,vmax)(np.nan_to_num(z,nan=vmin)))
 ax.plot_surface(xx,yy,z,facecolors=colors,edgecolor="#F8FAFC",linewidth=.26,
                 antialiased=True,shade=False)
 observed=float(-pack["y"][idx,0]);predicted=float(-prediction)
 ax.plot([4,4],[4,4],[min(observed,predicted),max(observed,predicted)],
         color="#D95F02",linewidth=1.25,zorder=10)
 ax.scatter([4],[4],[observed],s=16,marker="o",c="#17324D",edgecolors="white",linewidths=.35,zorder=11)
 ax.scatter([4],[4],[predicted],s=30,marker="*",c="#F28E2B",edgecolors="#17324D",linewidths=.35,zorder=12)
 ax.set(xlim=(0,8),ylim=(0,8),zlim=(vmin,vmax),xlabel="Col.",ylabel="Row",
        zlabel="Depth")
 ax.set_xticks([0,4,8]);ax.set_yticks([0,4,8]);ax.tick_params(labelsize=4.7,pad=0)
 ax.view_init(elev=26,azim=-58);set_3d_panes(ax)
 ax.set_title(f"9 × 9 survey support · {PRETTY[model]} centre estimate",
              fontsize=6.0,weight="bold",pad=0)
 handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor="#17324D",
                 markeredgecolor="white",markersize=4,label=f"Survey {observed:.1f} m"),
          Line2D([0],[0],marker="*",color="none",markerfacecolor="#F28E2B",
                 markeredgecolor="#17324D",markersize=6,label=f"Estimate {predicted:.1f} m")]
 ax.legend(handles=handles,loc="upper left",fontsize=4.5,frameon=True,
           facecolor="white",framealpha=.92,borderpad=.2,handletextpad=.3)
def results_surface_3d(ax,x,y,depth,site,extent,vmax,title):
 tri=supported_triangulation(x,y)
 surf=ax.plot_trisurf(tri,depth,cmap=DEPTH_CMAP,norm=Normalize(0,vmax),
                      linewidth=0,antialiased=True,shade=True)
 ax.set_xlim(extent[0],extent[1]);ax.set_ylim(extent[2],extent[3]);ax.set_zlim(0,vmax*1.08)
 # The depth colour bar carries the quantitative scale. Suppress 3D z tick
 # labels here because Matplotlib projects them into the compact map panels,
 # where they collide with the projected northing labels.
 ax.set_zticks([])
 ax.set_xlabel("Easting (km)",fontsize=5.3,labelpad=0)
 ax.set_ylabel("Northing (km)",fontsize=5.3,labelpad=0)
 ax.set_zlabel("",fontsize=5.3,labelpad=0)
 ax.tick_params(labelsize=4.2,pad=0)
 ax.view_init(elev=31,azim=-132);set_3d_panes(ax)
 spanx=max(extent[1]-extent[0],.05);spany=max(extent[3]-extent[2],.05)
 ax.set_box_aspect((spanx,spany,max(vmax/1000*38,.15)))
 ax.set_title(title,fontsize=6.2,weight="bold",pad=0)
 return surf

def fig01_data_overview():
 fig,ax=plt.subplots(3,2,figsize=(6.8,7.5),constrained_layout=True)
 for c,site in enumerate(SITES):
  pack,rgb,extent,(r0,r1,c0,c1)=context(site)
  ax[0,c].imshow(rgb,extent=extent,origin="upper",interpolation="nearest")
  scene_label=("Anza / Agadir · 10 Nov 2022" if c==0 else "Ras El Ma · image 7 Aug 2024")
  ax[0,c].set_title(f"({chr(97+c)}) {scene_label}",loc="left",weight="bold")
  depth=np.ma.masked_where(~pack["target_grid_mask"][r0:r1,c0:c1],
                           -pack["target_grid"][r0:r1,c0:c1])
  vmax=max(5,float(np.percentile(-pack["y"][:,0],98)))
  im=ax[1,c].imshow(depth,extent=extent,origin="upper",cmap=DEPTH_CMAP,
                    vmin=0,vmax=vmax,interpolation="nearest")
  ax[1,c].set_title(f"({chr(99+c)}) Survey depth · {('Anza' if c==0 else 'Ras El Ma')}",
                    loc="left",weight="bold")
  nrows,ncols=depth.shape
  gx=extent[0]+(np.arange(ncols)+.5)*(extent[1]-extent[0])/ncols
  gy=extent[3]-(np.arange(nrows)+.5)*(extent[3]-extent[2])/nrows
  xx,yy=np.meshgrid(gx,gy);levels=depth_levels(depth.compressed())
  if len(levels):
   contours=ax[1,c].contour(xx,yy,depth,levels=levels,colors="#F8FAFC",linewidths=.5)
   ax[1,c].clabel(contours,fmt=lambda z:f"{z:.0f}",fontsize=5.0,inline=True)
  cb=fig.colorbar(im,ax=ax[1,c],fraction=.04,pad=.015)
  cb.set_label("Depth below local zero (m)",fontsize=6.5);cb.ax.tick_params(labelsize=5.8)
  sc=ax[2,c].scatter(pack["x_utm"]/1000,pack["y_utm"]/1000,
                     c=pack["survey_points_per_pixel"],s=SOUNDING_SUPPORT_DOT_SIZE,cmap="magma",
                     linewidths=0,alpha=.88,rasterized=True)
  ax[2,c].set_title(f"({chr(101+c)}) Sounding support · n={len(pack['y']):,} cells",
                    loc="left",weight="bold")
  cb=fig.colorbar(sc,ax=ax[2,c],fraction=.04,pad=.015)
  cb.set_label("Soundings per 30 m cell",fontsize=6.5);cb.ax.tick_params(labelsize=5.8)
  for a in ax[:,c]:map_axes(a,site,extent)
  for a in ax[:,c]:north(a)
  scalebar(ax[0,c],extent)
 ax[0,0].set_ylabel("HLS image · UTM northing (km; 29N)",fontsize=7)
 ax[1,0].set_ylabel("Depth · UTM northing (km; 29N)",fontsize=7)
 ax[2,0].set_ylabel("Sounding count · UTM northing (km; 29N)",fontsize=7)
 fig.suptitle("Anza and Ras El Ma: imagery, survey depth, and sounding support",
              fontsize=11.5,weight="bold",color="#17324D")
 save(fig,"Figure_01_data_overview")
def fig02_cartographic_method():
 """Map-led 3D explanation of the observed 9 x 9 spectral cube and single-cell output."""
 fig=plt.figure(figsize=(7.25,5.9),facecolor="white")
 gs=fig.add_gridspec(2,3,left=.055,right=.98,top=.83,bottom=.135,
                    width_ratios=[1.25,1.18,1.40],wspace=.20,hspace=.37)
 headers=("Coastal scene and selected pixel patch",
          "Nine-by-nine input · four HLS bands",
          "Nine-by-nine survey support · one centre output")
 for c,title in enumerate(headers):
  fig.text([.205,.535,.835][c],.887,title,ha="center",va="center",
           fontsize=5.8,weight="bold",color="#17324D")
 for r,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=25)
  pred=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  errors=pred.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(errors.idxmin()); candidates=np.flatnonzero(pack["zones"]==1)
  if not len(candidates):candidates=np.arange(len(pack["y"]))
  mid=np.array([np.median(pack["x_utm"][candidates]),np.median(pack["y_utm"][candidates])])
  idx=int(candidates[np.argmin(np.sum((np.column_stack([pack["x_utm"][candidates],
       pack["y_utm"][candidates]])-mid)**2,axis=1))])
  pid=str(pack["pixel_ids"][idx])
  row=pred[(pred.model==model)&(pred.pixel_id.astype(str)==pid)]
  if row.empty:row=pred[pred.model==model].iloc[[int(np.argmin((pred.x_utm-pack["x_utm"][idx])**2+
                                                               (pred.y_utm-pack["y_utm"][idx])**2))]]
  estimate=float(row.iloc[0].prediction_m)
  ax=fig.add_subplot(gs[r,0]);ax.imshow(rgb,extent=extent,origin="upper",interpolation="nearest")
  ax.scatter(pack["x_utm"]/1000,pack["y_utm"]/1000,s=1.25,c="white",alpha=.62,
             linewidths=0,rasterized=True)
  cx=pack["x_utm"][idx]/1000;cy=pack["y_utm"][idx]/1000;half=.135
  ax.add_patch(plt.Rectangle((cx-half,cy-half),2*half,2*half,fill=False,
                 edgecolor="#FFD166",linewidth=1.1,zorder=10))
  ax.scatter([cx],[cy],marker="+",s=18,c="#17324D",linewidths=.9,zorder=11)
  map_axes(ax,site,extent);north(ax);scalebar(ax,extent)
  row_label=("Anza / Agadir · image 10 Nov 2022" if r==0 else "Ras El Ma · image 07 Aug 2024")
  ax.set_title(row_label,loc="left",fontsize=5.7,weight="bold")
  cube=fig.add_subplot(gs[r,1],projection="3d");method_band_cube(cube,pack["X"][idx])
  out=fig.add_subplot(gs[r,2],projection="3d")
  method_depth_patch(out,pack,idx,estimate,model)
 fig.suptitle("From a 30 m image patch to one site-local depth estimate",
              y=.963,fontsize=10.1,weight="bold",color="#17324D")
 fig.text(.51,.048,
  "The outlined image window is 270 × 270 m (9 × 9 HLS pixels). Band colours show measured reflectance, scaled separately within each band. The 3D neighbourhood shows aggregated survey labels; only the central star is a saved held-out model estimate. Vertical axes are schematic and exaggerated.",
  ha="center",va="center",wrap=True,fontsize=5.6,color="#526173")
 save(fig,"Figure_02_cartographic_method")
def fig04_spatial_validation():
 fig=plt.figure(figsize=(6.8,5.4),constrained_layout=True); gs=fig.add_gridspec(2,2,height_ratios=[3.5,1.25])
 for c,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=21); ax=fig.add_subplot(gs[0,c]); ax.imshow(rgb,extent=extent,origin="upper",alpha=.78,interpolation="nearest")
  for z in range(3):
   m=pack["zones"]==z; ax.scatter(pack["x_utm"][m]/1000,pack["y_utm"][m]/1000,s=MAP_DOT_SIZE,c=ZONE_COLORS[z],linewidths=0,label=f"Zone {z}",rasterized=True)
  ax.set_title(site["name"],weight="bold",fontsize=8.8); map_axes(ax,site,extent); north(ax); ax.legend(loc="lower left",fontsize=6.5,ncol=3,frameon=True,facecolor="white",framealpha=.9)
  scalebar(ax,extent)
 tabax=fig.add_subplot(gs[1,:]); tabax.axis("off")
 cells=[["Held-out zone","Zone 0","Zone 1","Zone 2"],["Zone 0","Test","Validation","Training"],["Zone 1","Training","Test","Validation"],["Zone 2","Validation","Training","Test"]]
 tab=tabax.table(cellText=cells[1:],colLabels=cells[0],cellLoc="center",loc="center",colWidths=[.24,.23,.23,.23])
 tab.auto_set_font_size(False); tab.set_fontsize(7.2); tab.scale(1,.95)
 for (row,col),cell in tab.get_celld().items():
  if row==0: cell.set_facecolor("#E8EEF4"); cell.set_text_props(weight="bold",color="#17324D")
  elif row>0:
   role=cell.get_text().get_text(); color={"Training":"#E6F1FA","Validation":"#FFF1D6","Test":"#FBE7E5"}.get(role,"white"); cell.set_facecolor(color)
 fig.suptitle("Within-site buffered spatial validation: zone map and role rotation",fontsize=11.3,weight="bold",color="#17324D")
 fig.text(.5,.005,"A patch-width buffer separates alongshore zones. The test fold is never used to fit or select model parameters; sites have separate coordinate systems and local vertical references.",ha="center",fontsize=7.2,color="#526173")
 save(fig,"Figure_03_spatial_validation")

def fig05_errors():
 """Contemporary validation panels: density agreement and median residual trend by depth."""
 fig,axes=plt.subplots(2,2,figsize=(6.9,6.15),constrained_layout=True)
 for r,site in enumerate(SITES):
  data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin());d=data[data.model==model]
  obs=d.reference_m.to_numpy(float);pred=d.prediction_m.to_numpy(float);res=d.residual_m.to_numpy(float)
  lo=min(float(np.percentile(obs,.2)),float(np.percentile(pred,.2)))
  hi=max(float(np.percentile(obs,99.8)),float(np.percentile(pred,99.8)))
  ax=axes[r,0]
  hb=ax.hexbin(obs,pred,gridsize=28,mincnt=1,bins="log",cmap="viridis",
               linewidths=0,extent=(lo,hi,lo,hi),rasterized=True)
  ax.plot([lo,hi],[lo,hi],ls="--",lw=1.0,color="#172033",label="1:1")
  slope,intercept=np.polyfit(obs,pred,1);xx=np.linspace(lo,hi,100)
  ax.plot(xx,slope*xx+intercept,color=site["color"],lw=1.35,
          label=f"Held-out fit · slope {slope:.2f}")
  r2=1-float(np.sum(res**2))/float(np.sum((obs-obs.mean())**2))
  ax.text(.035,.965,f"{PRETTY[model]}\nRMSE {np.sqrt(np.mean(res**2)):.2f} m · "
          f"MAE {np.mean(np.abs(res)):.2f} m\nBias {np.mean(res):+.2f} m · "
          f"R² {r2:.2f} · n={len(d):,}",transform=ax.transAxes,ha="left",va="top",
          fontsize=5.6,color="#172033",
          bbox={"facecolor":"white","edgecolor":"#D5DEE8","alpha":.92,"boxstyle":"round,pad=.25"})
  ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Survey z (m; local zero)",
         ylabel="Predicted z (m)")
  ax.set_title(f"{site['name']} · held-out agreement",loc="left",fontsize=7.6,weight="bold")
  ax.legend(loc="lower right",fontsize=5.6,frameon=True,facecolor="white",framealpha=.92)
  ax.grid(alpha=.3)
  cb=fig.colorbar(hb,ax=ax,fraction=.045,pad=.025);cb.set_label("Cell density (log count)",fontsize=5.8);cb.ax.tick_params(labelsize=5.2)
  ax=axes[r,1];depth=-obs
  max_depth=float(np.percentile(depth,98));bins=np.linspace(max(0,float(depth.min())),max_depth,10)
  keep=depth<=max_depth;depth_view=depth[keep];res_view=res[keep]
  ax.scatter(depth_view,res_view,s=2.0,c=np.abs(res_view),cmap="magma",alpha=.40,
             linewidths=0,rasterized=True)
  centres=[];med=[];q25=[];q75=[];counts=[]
  for a,b in zip(bins[:-1],bins[1:]):
   m=(depth_view>=a)&(depth_view<(b if b<bins[-1] else b+1e-9))
   if int(m.sum())>=8:
    centres.append(float(np.median(depth_view[m])));med.append(float(np.median(res_view[m])))
    q25.append(float(np.percentile(res_view[m],25)));q75.append(float(np.percentile(res_view[m],75)))
    counts.append(int(m.sum()))
  if centres:
   ax.fill_between(centres,q25,q75,color=site["color"],alpha=.18,label="25th–75th percentile")
   ax.plot(centres,med,color=site["color"],marker="o",markersize=2.7,lw=1.3,
           label="Median residual")
  ax.axhline(0,color="#172033",ls="--",lw=.8)
  ax.set(xlabel="Survey depth below local zero (m)",ylabel="Prediction − survey z (m)")
  ax.set_title("Residual trend across depth",loc="left",fontsize=7.6,weight="bold")
  ax.legend(loc="best",fontsize=5.4,frameon=True,facecolor="white",framealpha=.9)
  ax.grid(alpha=.3)
 fig.suptitle("Held-out validation: density agreement and depth-dependent residuals",
              fontsize=10.8,weight="bold",color="#17324D")
 fig.text(.5,-.01,
  "Each site uses its lowest pooled-RMSE method. The fitted line is descriptive; the dashed line is 1:1. Residual trend lines show binned medians and interquartile ranges; all scores use each site's own survey zero.",
  ha="center",fontsize=6.2,color="#526173")
 save(fig,"Figure_04_validation_trends")
def fig05_results_maps():
 """Two-dimensional isobaths and supported 3D surfaces of held-out predictions."""
 fig=plt.figure(figsize=(7.25,5.8),facecolor="white")
 gs=fig.add_gridspec(2,3,left=.055,right=.875,top=.81,bottom=.14,
                    width_ratios=[1.0,1.0,1.30],wspace=.22,hspace=.39)
 headers=("Survey labels · isobaths","Held-out estimates · isobaths","3D held-out depth surface")
 for c,title in enumerate(headers):
  fig.text([.19,.465,.735][c],.875,title,ha="center",va="center",
           fontsize=6.1,weight="bold",color="#17324D")
 for r,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=22)
  data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin());d=data[data.model==model]
  x=d.x_utm.to_numpy(float)/1000;y=d.y_utm.to_numpy(float)/1000
  observed=-d.reference_m.to_numpy(float);predicted=-d.prediction_m.to_numpy(float)
  vmax=max(2.,float(np.ceil(max(np.percentile(observed,98),
                                  np.percentile(predicted,98))/2)*2))
  ax1=fig.add_subplot(gs[r,0]);ax1.imshow(rgb,extent=extent,origin="upper",alpha=.17,interpolation="nearest")
  cf,_=add_depth_map(ax1,x,y,observed,site,extent,vmax)
  ax1.set_title(f"{site['name']} · {PRETTY[model]}",
                loc="left",fontsize=5.7,weight="bold",color=site["color"],pad=2)
  scalebar(ax1,extent,length=.5)
  ax2=fig.add_subplot(gs[r,1]);ax2.imshow(rgb,extent=extent,origin="upper",alpha=.17,interpolation="nearest")
  cf2,_=add_depth_map(ax2,x,y,predicted,site,extent,vmax)
  ax2.set_title(f"OOF estimate · RMSE {rm[model]:.2f} m",
                loc="left",fontsize=5.7,weight="bold",color=site["color"],pad=2)
  scalebar(ax2,extent,length=.5)
  ax3=fig.add_subplot(gs[r,2],projection="3d")
  surf=results_surface_3d(ax3,x,y,predicted,site,extent,vmax,
                         f"{PRETTY[model]} · depth below {site['name'].split('/')[0].strip()} zero")
  cax=fig.add_axes([.889,.525 if r==0 else .17,.016,.23])
  cb=fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap=DEPTH_CMAP),cax=cax)
  cb.set_label("Depth (m)",fontsize=5.4);cb.ax.tick_params(labelsize=4.7,pad=1)
 fig.suptitle("Survey-derived and CNN-estimated bathymetry at held-out cells",
              y=.967,fontsize=10.2,weight="bold",color="#17324D")
 fig.text(.46,.045,
  "Filled depth and 3D surfaces are linear-triangle displays between 30 m cell centres; triangles with an edge longer than 45 m are masked. No extrapolation is shown. Contours are 2 m isobaths. The vertical axis is exaggerated 38× for legibility; site datums remain separate.",
  ha="center",va="center",wrap=True,fontsize=5.5,color="#526173")
 save(fig,"Figure_05_results_maps")
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
   sc=ax.scatter(d.x_utm/1000,d.y_utm/1000,c=-d[field],s=MAP_DOT_SIZE,cmap=DEPTH_CMAP,norm=norm,linewidths=0,rasterized=True)
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
  sc=ax.scatter(d.x_utm/1000,d.y_utm/1000,c=d.residual_m,s=MAP_DOT_SIZE,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),linewidths=.15,edgecolors="white",rasterized=True)
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

def fig_s2_residual_maps():
 fig,axes=plt.subplots(1,2,figsize=(6.8,3.9),constrained_layout=True)
 for ax,site in zip(axes,SITES):
  pack,rgb,extent,_=context(site,pad=22)
  data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin());d=data[data.model==model]
  lim=max(.5,float(np.percentile(np.abs(d.residual_m),98)))
  ax.imshow(rgb,extent=extent,origin="upper",alpha=.2,interpolation="nearest")
  sc=ax.scatter(d.x_utm/1000,d.y_utm/1000,c=d.residual_m,s=MAP_DOT_SIZE,cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),
                linewidths=0,rasterized=True,alpha=.9)
  ax.set_title(f"{site['name']} · {PRETTY[model]}",fontsize=7.2,weight="bold")
  map_axes(ax,site,extent);north(ax);scalebar(ax,extent)
  cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.025)
  cb.set_label("Prediction − survey (m)",fontsize=6.2);cb.ax.tick_params(labelsize=5.7)
 fig.suptitle("Supplementary Figure S2 · spatial pattern of held-out residuals",
              fontsize=9.8,weight="bold",color="#17324D")
 fig.text(.5,-.012,
  "Only one out-of-fold residual per retained cell is shown. The two panels use local port vertical zeros and independent colour ranges; small points mark 30 m cell centres.",
  ha="center",fontsize=6.5,color="#526173")
 save(fig,"Figure_S2_residual_maps")

def fig_s1_disagreement():
 fig,axes=plt.subplots(1,2,figsize=(6.8,4.0),constrained_layout=True)
 for ax,site in zip(axes,SITES):
  d=pd.read_csv(site["results"]/"out_of_fold_predictions.csv"); wide=d.pivot_table(index=["pixel_id","x_utm","y_utm"],columns="model",values="prediction_m",aggfunc="first").reindex(columns=MODELS); spread=wide.std(axis=1,ddof=0).to_numpy(float); xy=wide.index.to_frame(index=False)
  sc=ax.scatter(xy.x_utm/1000,xy.y_utm/1000,c=spread,s=MAP_DOT_SIZE,cmap="cividis",vmin=0,vmax=max(.8,float(np.percentile(spread,98))),linewidths=0,rasterized=True)
  ax.set_title(site["name"],fontsize=8.8,weight="bold"); ax.set_xlabel("UTM easting (km)",fontsize=7); ax.set_ylabel(f"UTM northing (km; {site['utm']})",fontsize=7); ax.set_aspect("equal",adjustable="box"); ax.tick_params(labelsize=6.5); ax.grid(alpha=.35); north(ax)
  cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.025); cb.set_label("Across-model SD (m)",fontsize=7); cb.ax.tick_params(labelsize=6.5)
 fig.suptitle("Supplementary Figure S1 · spatial variation in model disagreement",fontsize=11.3,weight="bold",color="#17324D")
 fig.text(.5,-.01,"Standard deviation across seven out-of-fold model predictions at the same cell. This is descriptive model disagreement, not calibrated predictive uncertainty.",ha="center",fontsize=7.2,color="#526173")
 save(fig,"Figure_S1_model_disagreement")

def main():
 style(); fig01_data_overview(); fig02_cartographic_method(); fig04_spatial_validation(); fig05_errors(); fig05_results_maps(); fig_s1_disagreement(); fig_s2_residual_maps()
 print("Saved five main cartographic/results figures and two supplementary maps to",OUT)
if __name__=="__main__": main()
