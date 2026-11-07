"""Publication-size two-site figures generated from saved HLS samples and OOF results."""
from __future__ import annotations
import json
import sys
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
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator
import matplotlib.tri as mtri
import matplotlib.patheffects as path_effects
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"/"rse_14p"; OUT.mkdir(parents=True,exist_ok=True)
DEPTH_CMAP=LinearSegmentedColormap.from_list("coast_depth",["#e8f4f2","#85d1c6","#2f8ea8","#17466c","#10253f"])
SITES=[
 {"name":"Anza / Agadir","coast":"Atlantic coast","date":"10 Nov 2022","sample":ROOT/"data"/"processed"/"anza_agadir_2022-11-10"/"samples.npz","summary":ROOT/"data"/"processed"/"anza_agadir_2022-11-10"/"dataset_summary.json","scene":ROOT/"data"/"raw"/"sentinel2"/"HLS_S30_Anza_Agadir_2022-11-10","product":"HLS.S30.T29RMP.2022314T112209.v2.0","results":ROOT/"results"/"anza_agadir_2022-11-10","utm":"29N","datum":"Port of Agadir hydrographic zero","color":"#2864A5"},
 {"name":"Ras El Ma / Ras Kebdana","coast":"Mediterranean coast","date":"07 Aug image · 08 Aug survey 2024","sample":ROOT/"data"/"processed"/"ras_el_ma_2024-08-08"/"samples.npz","summary":ROOT/"data"/"processed"/"ras_el_ma_2024-08-08"/"dataset_summary.json","scene":ROOT/"data"/"raw"/"sentinel2"/"HLS_S30_2024-08-07_RasElMa"/"T30SWD","product":"HLS.S30.T30SWD.2024220T105031.v2.0","results":ROOT/"results"/"ras_el_ma_2024-08-08","utm":"30N","datum":"Port of Ras Kebdana hydrographic zero","color":"#D97742"},
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
def panel_tag(ax,label,fontsize=8,xy=(.025,.965)):
 """Add a small panel key; explanatory wording belongs in the article caption."""
 if hasattr(ax,"text2D"):
  ax.text2D(xy[0],xy[1],f"({label})",transform=ax.transAxes,ha="left",va="top",
            fontsize=fontsize,weight="bold",color="#172033",
            bbox={"facecolor":"white","edgecolor":"#D5DEE8","alpha":.95,
                  "boxstyle":"round,pad=.18"},zorder=50)
 else:
  ax.text(xy[0],xy[1],f"({label})",transform=ax.transAxes,ha="left",va="top",
          fontsize=fontsize,weight="bold",color="#172033",
          bbox={"facecolor":"white","edgecolor":"#D5DEE8","alpha":.95,
                "boxstyle":"round,pad=.18"},zorder=50)

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
def method_band_map(ax, values, band, norm):
 """Display one measured 9 x 9 HLS reflectance layer on the shared scale."""
 edges=np.arange(10,dtype=float)-.5
 image=ax.pcolormesh(edges,edges,np.asarray(values,float),cmap=plt.get_cmap("cividis"),
                     norm=norm,shading="flat",edgecolors="#FFFFFF",linewidth=.22)
 ax.add_patch(Rectangle((3.5,3.5),1,1,fill=False,edgecolor="#172033",linewidth=.8,zorder=5))
 ax.set(xlim=(-.5,8.5),ylim=(8.5,-.5),aspect="equal")
 ax.set_xticks([]);ax.set_yticks([])
 ax.set_title(band,fontsize=7.0,weight="bold",pad=1.1)
 ax.text(.5,-.055,f"mean {np.nanmean(values):.3f}",transform=ax.transAxes,
         ha="center",va="top",fontsize=5.7,color="#334155")
 return image

def method_depth_pixels(ax,pack,idx,prediction,model):
 """Map the measured 9 x 9 depth labels and mark the held-out centre estimate."""
 z=-np.asarray(pack["target_patches"][idx,:,:,0],float)
 weights=np.asarray(pack["target_weights"][idx,:,:,0],float)
 z=np.where(weights>0,z,np.nan)
 observed=float(-pack["y"][idx,0]);predicted=float(-prediction)
 vals=np.r_[z[np.isfinite(z)],observed,predicted]
 vmin=max(0.0,float(np.nanmin(vals)));vmax=float(np.nanmax(vals))
 if vmax<=vmin:vmax=vmin+0.1
 edges=np.arange(10,dtype=float)-.5
 image=ax.pcolormesh(edges,edges,np.ma.masked_invalid(z),cmap=DEPTH_CMAP,
                     vmin=vmin,vmax=vmax,shading="flat",edgecolors="#FFFFFF",linewidth=.28)
 ax.add_patch(Rectangle((3.5,3.5),1,1,fill=False,edgecolor="#172033",linewidth=.9,zorder=5))
 # Both symbols refer to the same 30 m centre cell. A small within-cell offset
 # makes the observed label and held-out estimate distinguishable at print size.
 ax.scatter([3.73],[4],s=48,marker="o",c="#17324D",edgecolors="white",
            linewidths=.8,zorder=8)
 ax.scatter([4.27],[4],s=100,marker="*",c="#F28E2B",edgecolors="#17324D",
            linewidths=.65,zorder=9)
 short_model=PRETTY.get(model,model)
 ax.set_title(f"Survey {observed:.2f} m\n{short_model} {predicted:.2f} m",
              fontsize=5.9,pad=2.2)
 ax.set(xlim=(-.5,8.5),ylim=(8.5,-.5),aspect="equal",xlabel="Pixel column",ylabel="Pixel row")
 ax.xaxis.label.set_size(6.4);ax.yaxis.label.set_size(6.4)
 ax.set_xticks([0,4,8]);ax.set_yticks([0,4,8]);ax.tick_params(labelsize=4.8,pad=1)
 cb=plt.gcf().colorbar(image,ax=ax,fraction=.045,pad=.035,aspect=24)
 cb.set_label("Depth below local zero (m)",fontsize=5.4,labelpad=1)
 cb.ax.tick_params(labelsize=5.0,length=1.5,pad=1)
 return image
def results_surface_3d(ax,x,y,elevation,site,vmax):
 """Map the local XYZ coordinates with a depth-coloured surface and floor contours."""
 x=np.asarray(x,float);y=np.asarray(y,float);elevation=np.asarray(elevation,float)
 finite=np.isfinite(x)&np.isfinite(y)&np.isfinite(elevation)
 x=x[finite];y=y[finite];elevation=elevation[finite]
 depth=-elevation
 tri=supported_triangulation(x,y)
 triangles=tri.get_masked_triangles()
 if not len(triangles):
  raise ValueError("No supported 30 m triangles are available for the 3D surface.")
 # The surface uses true local elevations for Z. A piecewise colormap maps
 # negative Z to positive-down depth colours and predictions above zero to amber.
 zmax=float(elevation.max())
 if zmax>0:
  transition=vmax/(vmax+zmax)
  surface_cmap=LinearSegmentedColormap.from_list(
   "local_elevation_depth",
   [(0.0,DEPTH_CMAP(1.0)),(transition,DEPTH_CMAP(0.0)),(1.0,"#D7A54A")])
  surface_norm=Normalize(-vmax,zmax)
 else:
  surface_cmap=DEPTH_CMAP.reversed()
  surface_norm=Normalize(-vmax,0.0)
 surf=ax.plot_trisurf(tri,elevation,cmap=surface_cmap,norm=surface_norm,
                      linewidth=.035,antialiased=True,shade=True)
 sx=max(float(np.ptp(x)),.12);sy=max(float(np.ptp(y)),.12)
 padx=max(.035,sx*.045);pady=max(.035,sy*.045)
 xlo=float(x.min()-padx);xhi=float(x.max()+padx)
 ylo=float(y.min()-pady);yhi=float(y.max()+pady)
 zlo=min(float(elevation.min())-.5,-vmax*1.12)
 zhi=max(float(elevation.max())+.5,.5)
 ax.set_xlim(xlo,xhi);ax.set_ylim(ylo,yhi);ax.set_zlim(zlo,zhi)
 # Project isobaths onto the plot floor to make depth intervals legible without
 # interpolating beyond the supported cell-centre neighbourhoods.
 contour_step=5.0 if vmax>15 else 2.0
 levels=np.arange(contour_step,np.floor(vmax/contour_step)*contour_step+.01,contour_step)
 if len(levels):
  ax.tricontour(tri,depth,levels=levels,zdir="z",offset=zlo,
                colors="#40566B",linewidths=.48,alpha=.78)
 ax.set_xlabel("X (km)",fontsize=9.0,labelpad=2)
 ax.set_ylabel("Y (km)",fontsize=9.0,labelpad=2)
 # Keep the vertical coordinate explicit without letting the right-hand
 # 3D-axis label collide with the figure edge or the panel color scale.
 ax.set_zlabel("")
 ax.text2D(.875,.53,"Z (m)",transform=ax.transAxes,rotation=90,
           ha="center",va="center",fontsize=9.0,color="#172033")
 xticks=xlo+np.array([.25,.75])*(xhi-xlo)
 yticks=ylo+np.array([.25,.75])*(yhi-ylo)
 ax.set_xticks(xticks);ax.set_yticks(yticks)
 ax.set_xticklabels([f"{v:.3f}" for v in xticks],fontsize=7.5)
 ax.set_yticklabels([f"{v:.3f}" for v in yticks],fontsize=7.5)
 ax.zaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,2.5,5,10]))
 ax.tick_params(axis="z",labelsize=7.2,pad=1)
 ax.tick_params(axis="x",pad=1);ax.tick_params(axis="y",pad=1)
 ax.view_init(elev=31,azim=-54);ax.set_proj_type("ortho");set_3d_panes(ax)
 # Vertical exaggeration is fixed across both sites and disclosed in the caption.
 vertical_exaggeration=40.0
 ax.set_box_aspect((max(xhi-xlo,.12),max(yhi-ylo,.12),
                    max((zhi-zlo)/1000*vertical_exaggeration,.08)))
 return surf
def fig01_data_overview():
 fig,ax=plt.subplots(3,2,figsize=(6.8,7.45),constrained_layout=True)
 for c,site in enumerate(SITES):
  pack,rgb,extent,(r0,r1,c0,c1)=context(site)
  ax[0,c].imshow(rgb,extent=extent,origin="upper",interpolation="nearest")
  panel_tag(ax[0,c],chr(97+c))
  depth=np.ma.masked_where(~pack["target_grid_mask"][r0:r1,c0:c1],
                           -pack["target_grid"][r0:r1,c0:c1])
  vmax=max(5,float(np.percentile(-pack["y"][:,0],98)))
  im=ax[1,c].imshow(depth,extent=extent,origin="upper",cmap=DEPTH_CMAP,
                    vmin=0,vmax=vmax,interpolation="nearest")
  panel_tag(ax[1,c],chr(99+c))
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
                     c=pack["survey_points_per_pixel"],s=SOUNDING_SUPPORT_DOT_SIZE,
                     cmap="magma",linewidths=0,alpha=.88,rasterized=True)
  panel_tag(ax[2,c],chr(101+c))
  cb=fig.colorbar(sc,ax=ax[2,c],fraction=.04,pad=.015)
  cb.set_label("Soundings per 30 m cell",fontsize=6.5);cb.ax.tick_params(labelsize=5.8)
  for a in ax[:,c]:map_axes(a,site,extent);north(a)
  scalebar(ax[0,c],extent)
 save(fig,"Figure_01_data_overview")
def fig02_cartographic_method():
 """Locate the source pixels, compare all measured HLS bands, and show each held-out output."""
 fig=plt.figure(figsize=(6.2,4.1),facecolor="white")
 gs=fig.add_gridspec(2,3,left=.035,right=.985,top=.97,bottom=.07,
                     width_ratios=[1.08,2.35,1.30],wspace=.20,hspace=.10)
 inner=gs[:,1].subgridspec(3,4,height_ratios=[1,1,.075],hspace=.22,wspace=.13)
 reflectance_norm=Normalize(vmin=-.01,vmax=.03)
 reflectance_images=[]
 for r,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=25)
  pred=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  errors=pred.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(errors.idxmin());candidates=np.flatnonzero(pack["zones"]==1)
  if not len(candidates):candidates=np.arange(len(pack["y"]))
  mid=np.array([np.median(pack["x_utm"][candidates]),np.median(pack["y_utm"][candidates])])
  xy=np.column_stack([pack["x_utm"][candidates],pack["y_utm"][candidates]])
  idx=int(candidates[np.argmin(np.sum((xy-mid)**2,axis=1))])
  pid=str(pack["pixel_ids"][idx]);row=pred[(pred.model==model)&(pred.pixel_id.astype(str)==pid)]
  if row.empty:
   row=pred[pred.model==model].iloc[[int(np.argmin((pred.x_utm-pack["x_utm"][idx])**2+
                                                    (pred.y_utm-pack["y_utm"][idx])**2))]]
  estimate=float(row.iloc[0].prediction_m)

  # Site map: all retained cells are shown; the outlined square is the 9 x 9 input window.
  ax=fig.add_subplot(gs[r,0]);ax.imshow(rgb,extent=extent,origin="upper",interpolation="nearest")
  ax.scatter(pack["x_utm"]/1000,pack["y_utm"]/1000,s=.65,c="white",alpha=.62,
             linewidths=0,rasterized=True)
  cx=pack["x_utm"][idx]/1000;cy=pack["y_utm"][idx]/1000;half=.135
  ax.add_patch(Rectangle((cx-half,cy-half),2*half,2*half,fill=False,
                         edgecolor="#FFD166",linewidth=1.1,zorder=10))
  ax.scatter([cx],[cy],marker="+",s=20,c="#17324D",linewidths=1.0,zorder=11)
  compact_map_axes(ax,site,extent);north(ax);scalebar(ax,extent)
  panel_tag(ax,"a" if r==0 else "d",fontsize=7.8)
  ax.text(.96,.04,site["name"],transform=ax.transAxes,ha="right",va="bottom",fontsize=5.7,
          color="#172033",bbox={"facecolor":"white","edgecolor":"none","alpha":.91,"pad":1.6})

  # Four measured reflectance rasters per site share one numeric colour scale.
  bandax=None
  for b,band in enumerate(("B02","B03","B04","B08")):
   bandax=fig.add_subplot(inner[r,b])
   reflectance_images.append(method_band_map(bandax,pack["X"][idx,:,:,b],band,reflectance_norm))
   if b==0:
    bandax.text(-.15,.5,site["name"],transform=bandax.transAxes,rotation=90,ha="right",va="center",
                fontsize=6.0,weight="bold",color="#17324D")
    if r==0:panel_tag(bandax,"b",fontsize=7.2,xy=(.015,1.26))
  # Exact 30 m cell labels are shown top-down; star and circle share the centre cell.
  out=fig.add_subplot(gs[r,2]);method_depth_pixels(out,pack,idx,estimate,model)
  panel_tag(out,"c" if r==0 else "e",fontsize=7.8,xy=(.015,1.19))

 # Shared scale makes the Anza and Ras El Ma reflectance values directly comparable.
 cax=fig.add_subplot(inner[2,:])
 cb=fig.colorbar(reflectance_images[-1],cax=cax,orientation="horizontal")
 cb.set_label("HLS surface reflectance (unitless)",fontsize=5.8,labelpad=1)
 cb.set_ticks([-.01,0,.01,.02,.03]);cb.ax.tick_params(labelsize=5.2,length=1.5,pad=1)
 save(fig,"Figure_02_cartographic_method")
def fig04_spatial_validation():
 fig=plt.figure(figsize=(6.8,5.4),facecolor="white")
 gs=fig.add_gridspec(2,2,height_ratios=[3.45,1.35],left=.065,right=.975,top=.985,bottom=.04,
                    hspace=.08,wspace=.12)
 for c,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=21);ax=fig.add_subplot(gs[0,c])
  ax.imshow(rgb,extent=extent,origin="upper",alpha=.78,interpolation="nearest")
  for z in range(3):
   m=pack["zones"]==z
   ax.scatter(pack["x_utm"][m]/1000,pack["y_utm"][m]/1000,s=MAP_DOT_SIZE,
              c=ZONE_COLORS[z],linewidths=0,label=f"Zone {z}",rasterized=True)
  map_axes(ax,site,extent);north(ax);panel_tag(ax,chr(97+c))
  ax.legend(loc="lower left",fontsize=6.5,ncol=3,frameon=True,facecolor="white",framealpha=.9)
  scalebar(ax,extent)
 tabax=fig.add_subplot(gs[1,:]);tabax.axis("off")
 cells=[["Held-out zone","Zone 0","Zone 1","Zone 2"],["Zone 0","Test","Validation","Training"],
        ["Zone 1","Training","Test","Validation"],["Zone 2","Validation","Training","Test"]]
 tab=tabax.table(cellText=cells[1:],colLabels=cells[0],cellLoc="center",loc="center",
                 colWidths=[.24,.23,.23,.23])
 tab.auto_set_font_size(False);tab.set_fontsize(7.2);tab.scale(1,.95)
 for (row,col),cell in tab.get_celld().items():
  if row==0:cell.set_facecolor("#E8EEF4");cell.set_text_props(weight="bold",color="#17324D")
  elif row>0:
   role=cell.get_text().get_text();cell.set_facecolor({"Training":"#E6F1FA","Validation":"#FFF1D6","Test":"#FBE7E5"}.get(role,"white"))
 save(fig,"Figure_03_spatial_validation")
def fig05_errors():
 """Held-out agreement density and residual dependence on survey depth."""
 fig,axes=plt.subplots(2,2,figsize=(6.9,6.1),constrained_layout=True)
 for r,site in enumerate(SITES):
  data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin());d=data[data.model==model]
  obs=d.reference_m.to_numpy(float);pred=d.prediction_m.to_numpy(float);res=d.residual_m.to_numpy(float)
  lo=min(float(np.percentile(obs,.2)),float(np.percentile(pred,.2)))
  hi=max(float(np.percentile(obs,99.8)),float(np.percentile(pred,99.8)))
  ax=axes[r,0];hb=ax.hexbin(obs,pred,gridsize=28,mincnt=1,bins="log",cmap="viridis",
                            linewidths=0,extent=(lo,hi,lo,hi),rasterized=True)
  ax.plot([lo,hi],[lo,hi],ls="--",lw=1.0,color="#172033",label="1:1 reference")
  slope,intercept=np.polyfit(obs,pred,1);xx=np.linspace(lo,hi,100)
  ax.plot(xx,slope*xx+intercept,color=site["color"],lw=1.35,label="Linear fit")
  ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Survey z (m; local zero)",ylabel="Predicted z (m)")
  panel_tag(ax,chr(97+r*2));ax.legend(loc="lower right",fontsize=5.6,frameon=True,facecolor="white",framealpha=.92)
  ax.grid(alpha=.3)
  cb=fig.colorbar(hb,ax=ax,fraction=.045,pad=.025);cb.set_label("Cell density (log count)",fontsize=5.8);cb.ax.tick_params(labelsize=5.2)
  ax=axes[r,1];depth=-obs;max_depth=float(np.percentile(depth,98))
  bins=np.linspace(max(0,float(depth.min())),max_depth,10);keep=depth<=max_depth
  depth_view=depth[keep];res_view=res[keep];centres=[];med=[];q25=[];q75=[]
  for a,b in zip(bins[:-1],bins[1:]):
   m=(depth_view>=a)&(depth_view<(b if b<bins[-1] else b+1e-9))
   if int(m.sum())>=8:
    centres.append(float(np.median(depth_view[m])));med.append(float(np.median(res_view[m])))
    q25.append(float(np.percentile(res_view[m],25)));q75.append(float(np.percentile(res_view[m],75)))
  ax.scatter(depth_view,res_view,s=1.6,c="#64748B",alpha=.25,linewidths=0,rasterized=True)
  if centres:
   ax.fill_between(centres,q25,q75,color=site["color"],alpha=.19,label="25th–75th percentile")
   ax.plot(centres,med,color=site["color"],marker="o",markersize=2.7,lw=1.3,label="Median residual")
  ax.axhline(0,color="#172033",ls="--",lw=.8)
  ax.set(xlabel="Survey depth below local zero (m)",ylabel="Prediction − survey z (m)")
  panel_tag(ax,chr(98+r*2));ax.legend(loc="best",fontsize=5.4,frameon=True,facecolor="white",framealpha=.9);ax.grid(alpha=.3)
 save(fig,"Figure_04_validation_trends")
def fig05_results_maps():
 """Two-dimensional site-local bathymetry maps for labels and held-out predictions."""
 fig=plt.figure(figsize=(7.05,5.65),facecolor="white")
 gs=fig.add_gridspec(2,3,left=.055,right=.94,top=.985,bottom=.055,
                    width_ratios=[1,1,.045],wspace=.16,hspace=.10)
 for r,site in enumerate(SITES):
  pack,rgb,extent,_=context(site,pad=22)
  data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin());d=data[data.model==model]
  x=d.x_utm.to_numpy(float)/1000;y=d.y_utm.to_numpy(float)/1000
  observed=-d.reference_m.to_numpy(float);predicted=-d.prediction_m.to_numpy(float)
  vmax=max(2.,float(np.ceil(max(np.percentile(observed,98),np.percentile(predicted,98))/2)*2))
  last=None
  for c,values in enumerate((observed,predicted)):
   ax=fig.add_subplot(gs[r,c]);ax.imshow(rgb,extent=extent,origin="upper",alpha=.13,interpolation="nearest")
   last,_=add_depth_map(ax,x,y,values,site,extent,vmax)
   panel_tag(ax,chr(97+r*2+c));north(ax);scalebar(ax,extent,length=.5)
  cax=fig.add_subplot(gs[r,2]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap=DEPTH_CMAP),cax=cax)
  cb.set_label("Depth below local zero (m)",fontsize=6.4);cb.ax.tick_params(labelsize=5.7,pad=1)
 save(fig,"Figure_05_results_maps")
def fig06_results_3d():
 """Separate, enlarged XYZ bathymetry views for each site."""
 for site_index,site in enumerate(SITES):
  data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin());d=data[data.model==model]
  x=d.x_utm.to_numpy(float)/1000;y=d.y_utm.to_numpy(float)/1000
  observed=d.reference_m.to_numpy(float);predicted=d.prediction_m.to_numpy(float)
  observed_depth=-observed;predicted_depth=-predicted
  vmax=max(2.,float(np.ceil(max(np.percentile(observed_depth,98),np.percentile(predicted_depth,98))/2)*2))
  fig=plt.figure(figsize=(10.2,5.45),facecolor="white")
  gs=fig.add_gridspec(2,2,left=.035,right=.93,top=.97,bottom=.115,
                     height_ratios=[1,.055],wspace=.13,hspace=.20)
  for c,values in enumerate((observed,predicted)):
   ax=fig.add_subplot(gs[0,c],projection="3d")
   results_surface_3d(ax,x,y,values,site,vmax)
   panel_tag(ax,chr(97+site_index*2+c),fontsize=11)
  cax=fig.add_subplot(gs[1,:])
  cb=fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap=DEPTH_CMAP),
                  cax=cax,orientation="horizontal")
  cb.set_label("Depth below local zero (m; positive down)",fontsize=8.5,labelpad=2)
  cb.ax.tick_params(labelsize=7.5,pad=1)
  site_id=("anza_agadir","ras_el_ma")[site_index]
  save(fig,f"Figure_{6+site_index:02d}_results_3d_{site_id}")

def fig_s2_residual_maps():
 fig,axes=plt.subplots(1,2,figsize=(6.8,3.9),constrained_layout=True)
 for i,(ax,site) in enumerate(zip(axes,SITES)):
  pack,rgb,extent,_=context(site,pad=22);data=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  rm=data.groupby("model").residual_m.apply(lambda v:float(np.sqrt(np.mean(np.asarray(v,float)**2))))
  model=str(rm.idxmin());d=data[data.model==model];lim=max(.5,float(np.percentile(np.abs(d.residual_m),98)))
  ax.imshow(rgb,extent=extent,origin="upper",alpha=.2,interpolation="nearest")
  sc=ax.scatter(d.x_utm/1000,d.y_utm/1000,c=d.residual_m,s=MAP_DOT_SIZE,cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),linewidths=0,rasterized=True,alpha=.9)
  map_axes(ax,site,extent);north(ax);scalebar(ax,extent);panel_tag(ax,chr(97+i))
  cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.025);cb.set_label("Prediction − survey (m)",fontsize=6.2);cb.ax.tick_params(labelsize=5.7)
 save(fig,"Figure_S2_residual_maps")
def fig_s1_disagreement():
 fig,axes=plt.subplots(1,2,figsize=(6.8,4.0),constrained_layout=True)
 for i,(ax,site) in enumerate(zip(axes,SITES)):
  d=pd.read_csv(site["results"]/"out_of_fold_predictions.csv")
  wide=d.pivot_table(index=["pixel_id","x_utm","y_utm"],columns="model",values="prediction_m",aggfunc="first").reindex(columns=MODELS)
  spread=wide.std(axis=1,ddof=0).to_numpy(float);xy=wide.index.to_frame(index=False)
  sc=ax.scatter(xy.x_utm/1000,xy.y_utm/1000,c=spread,s=MAP_DOT_SIZE,cmap="cividis",
                vmin=0,vmax=max(.8,float(np.percentile(spread,98))),linewidths=0,rasterized=True)
  ax.set_xlabel("UTM easting (km)",fontsize=7);ax.set_ylabel(f"UTM northing (km; {site['utm']})",fontsize=7)
  ax.set_aspect("equal",adjustable="box");ax.tick_params(labelsize=6.5);ax.grid(alpha=.35);north(ax);panel_tag(ax,chr(97+i))
  cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.025);cb.set_label("Across-model SD (m)",fontsize=7);cb.ax.tick_params(labelsize=6.5)
 save(fig,"Figure_S1_model_disagreement")
def main():
 style(); fig01_data_overview(); fig02_cartographic_method(); fig04_spatial_validation(); fig05_errors(); fig05_results_maps(); fig06_results_3d(); fig_s1_disagreement(); fig_s2_residual_maps()
 print("Saved seven main cartographic/results figures and two supplementary maps to",OUT)
if __name__=="__main__":
 if len(sys.argv)>1 and sys.argv[1]=="--figure2-only":
  style();fig02_cartographic_method()
 else: main()

