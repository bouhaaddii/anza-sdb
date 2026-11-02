"""Generate model diagnostics and summary figures, including spatial maps.

All coordinates are read from delivered GeoTIFFs. Prediction maps inherit the
target grid; this does not repair its unverified alignment with the RGB grid.
No model inference, training, prediction interpolation or data filtering occurs.
"""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import SDB_Visualisation_Publication as P
import SDB_Architectures_Publication as A

plt=P.plt
ZMAP=LinearSegmentedColormap.from_list('signed_elevation', ['#053061','#4393C3','#F7F7F7','#DFC27D','#8C510A'])

def extent(s):
    return tuple(v/1000 for v in [s.bounds.left,s.bounds.right,s.bounds.bottom,s.bounds.top])

def centre_xy(transform,rows,cols):
    """Pixel centre coordinates in UTM kilometres, with an explicit half-cell offset."""
    x,y=rasterio.transform.xy(transform,np.asarray(rows),np.asarray(cols),offset='center')
    return np.asarray(x)/1000,np.asarray(y)/1000

def decorations(ax,ex,scale=500,north=True):
    x0,x1,y0,y1=ex;dx=x1-x0;dy=y1-y0
    x=x0+.07*dx;y=y0+.08*dy
    ax.plot([x,x+scale/1000],[y,y],color='#111111',lw=2.5,path_effects=[pe.Stroke(linewidth=4,foreground='white'),pe.Normal()])
    ax.text(x+scale/2000,y+.025*dy,f'{scale:g} m',ha='center',va='bottom',fontsize=9,
            fontweight='bold',bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
    if north:
        ax.annotate('Grid N',xy=(x1-.1*dx,y1-.07*dy),xytext=(x1-.1*dx,y0+.15*dy if dy/dx<.2 else y1-.24*dy),
            ha='center',va='center',fontsize=9,fontweight='bold',
            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1},
            arrowprops={'arrowstyle':'-|>','color':'#111111','lw':1.3})

def map_axes(ax,ex,title,xlabel=True,ylabel=True):
    ax.set(xlim=ex[:2],ylim=ex[2:],aspect='equal',title=title)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    for axis in [ax.xaxis,ax.yaxis]:axis.set_major_formatter(FormatStrFormatter('%.1f'))
    if xlabel:ax.set_xlabel('Easting (km)')
    if ylabel:ax.set_ylabel('Northing (km)')
    ax.tick_params(labelsize=10)

def study_map(rgb,rgbex,z,zex):
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.1),layout='constrained')
    ex=(min(rgbex[0],zex[0]),max(rgbex[1],zex[1]),min(rgbex[2],zex[2]),max(rgbex[3],zex[3]))
    axes[0].imshow(rgb.transpose(1,2,0),extent=rgbex,interpolation='nearest')
    lim=float(np.ceil(np.nanmax(abs(z))))
    im=axes[1].imshow(z,extent=zex,interpolation='nearest',cmap=ZMAP,norm=TwoSlopeNorm(0,vmin=-lim,vmax=lim))
    for ax,title in zip(axes,['(a) RGB image · 10 December 2021','(b) Supplied signed reference']):
        map_axes(ax,ex,title);decorations(ax,ex)
    fig.colorbar(im,ax=axes[1],shrink=.78,pad=.025,label='Reference z (nominal m)')
    return fig

def geographic_partition(coords,y,split,transform,ex):
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.0),layout='constrained')
    xy=centre_xy(transform,coords[:,0]+4,coords[:,1]+4)
    used=np.concatenate(list(split.values()));excluded=np.setdiff1d(np.arange(len(coords)),used)
    ax=axes[0];ax.scatter(xy[0][excluded],xy[1][excluded],s=2,color='#CDD3D8',marker='s',label=f'Excluded (n={len(excluded):,})',rasterized=True)
    for j,role in enumerate(P.ROLES):
        ids=split[role];ax.scatter(xy[0][ids],xy[1][ids],s=2,color=P.ROLE_COLORS[j],marker=P.MARKERS[j],label=f'{P.ROLE_LABELS[j]} (n={len(ids):,})',rasterized=True)
        x,q=P.ecdf(y[ids]);axes[1].step(x,q,where='post',color=P.ROLE_COLORS[j],ls=P.STYLES[j],label=P.ROLE_LABELS[j])
    map_axes(ax,ex,'(a) Spatial membership');decorations(ax,ex)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,fontsize=9,markerscale=4)
    axes[1].set(title='(b) Signed reference distributions',xlabel='Reference z (nominal m)',ylabel='Cumulative proportion',ylim=(0,1.02))
    axes[1].legend(loc='lower right')
    return fig

def prediction_maps(frames,transform,ex):
    combined=[pd.concat([frames[n,r] for r in P.ROLES]).sort_values('patch_id') for n in P.MODELS]
    f=combined[0];x,y=centre_xy(transform,f.row,f.col)
    values=[f.reference.to_numpy()]+[q.prediction.to_numpy() for q in combined]
    limit=float(np.ceil(np.max(abs(np.concatenate(values)))))
    norm=TwoSlopeNorm(0,vmin=-limit,vmax=limit)
    fig,axes=plt.subplots(2,3,figsize=(11.5,7.0),layout='constrained')
    test=frames[P.MODELS[0],'test'];tx,ty=centre_xy(transform,test.row,test.col)
    window=(tx.min()-.5*transform.a/1000,tx.max()+.5*transform.a/1000,
            ty.min()+.5*transform.e/1000,ty.max()-.5*transform.e/1000)
    for j,(ax,v,title) in enumerate(zip(axes.flat,values,['Reference at paired centres']+P.LABELS)):
        im=ax.scatter(x,y,c=v,s=2.2,marker='s',linewidths=0,cmap=ZMAP,norm=norm,rasterized=True)
        map_axes(ax,ex,f'({chr(97+j)}) {title}',xlabel=j>=3,ylabel=j%3==0)
        ax.add_patch(Rectangle((window[0],window[2]),window[1]-window[0],window[3]-window[2],fill=False,ec='#111111',lw=.8,ls='--'))
        if j==0:decorations(ax,ex)
    fig.colorbar(im,ax=list(axes.flat),fraction=.027,pad=.018,label='Signed elevation (nominal m)',shrink=.8)
    return fig,{'map_centres_per_panel':len(f),'map_colour_limits':[-limit,limit],'test_window_utm_km':window}

def test_residual_maps(frames,transform):
    f=frames[P.MODELS[0],'test'];x,y=centre_xy(transform,f.row,f.col)
    padx=abs(transform.a)/2000;pady=abs(transform.e)/2000
    ex=(x.min()-padx,x.max()+padx,y.min()-pady,y.max()+pady)
    errors=[frames[n,'test'].prediction.to_numpy()-frames[n,'test'].reference.to_numpy() for n in P.MODELS]
    lim=float(np.ceil(max(np.max(abs(e)) for e in errors)))
    fig,axes=plt.subplots(5,1,figsize=(10.5,8.7),layout='constrained')
    for j,(ax,e) in enumerate(zip(axes,errors)):
        im=ax.scatter(x,y,c=e,s=9,marker='s',linewidths=0,cmap='RdBu_r',norm=TwoSlopeNorm(0,vmin=-lim,vmax=lim),rasterized=True)
        map_axes(ax,ex,f'({chr(97+j)}) {P.LABELS[j]}',xlabel=j==4)
        ax.yaxis.set_major_locator(MaxNLocator(2));ax.tick_params(labelbottom=j==4)
        if j==0:decorations(ax,ex,scale=500,north=True)
    fig.colorbar(im,ax=list(axes),fraction=.035,pad=.025,label='Prediction − reference (nominal m)',shrink=.83)
    return fig,{'test_centres_per_map':len(f),'residual_colour_limits':[-lim,lim],
        'positive_residual_fraction':dict(zip(P.MODELS,[float((e>0).mean()) for e in errors]))}

def main():
    root=Path(__file__).parent;ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=root/'SDB_Resultats_Corriges')
    ap.add_argument('--data-root',type=Path,default=root/'data/anza_study')
    ap.add_argument('--out',type=Path,default=root/'SDB_Cartes_Anza')
    default_models=root/'src/original_models.py'
    ap.add_argument('--model-source',type=Path,default=default_models)
    a=ap.parse_args();data=a.data_root.resolve();results=a.results.resolve();out=a.out.resolve()
    if out==data or out==results or data in out.parents or results in out.parents:raise ValueError('Use a separate output folder.')
    out.mkdir(parents=True,exist_ok=True);P.style();inputs={}
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':8.2})
    architecture_audit=A.inspect_models(a.model_source)
    (out/'architecture_audit.json').write_text(json.dumps(architecture_audit,indent=2),encoding='utf-8')
    def track(p):
        key=p.relative_to(data).as_posix() if p.is_relative_to(data) else 'results/'+p.relative_to(results).as_posix()
        inputs[key]=P.sha(p);return p
    audit=json.loads(track(results/'reproduction_manifest.json').read_text())
    if not all(audit['scientific_functions_unchanged'].values()):raise ValueError('Scientific identity audit failed.')
    pack=track(data/'prepared/patches.npz');sp=track(data/'prepared/spatial_split.npz')
    if P.sha(pack)!=audit['source_hashes']['patches'] or P.sha(sp)!=audit['source_hashes']['split']:raise ValueError('Input hash mismatch.')
    with np.load(pack) as p:y=p['y'].reshape(-1);coords=p['coords']
    with np.load(sp) as p:split={k:p[k] for k in p.files}
    raster_meta={}
    def read_raster(name):
        with rasterio.open(track(data/name)) as s:
            if s.crs.to_epsg()!=32629 or s.transform.b or s.transform.d or s.transform.a<=0 or s.transform.e>=0:raise ValueError('Expected north-up EPSG:32629 raster.')
            raster_meta[name]={'crs':str(s.crs),'transform':list(s.transform),'shape':s.shape,'bounds_m':list(s.bounds)}
            return s.read(),extent(s),s.transform
    rgb,rgbex,_=read_raster('reference/raw/1.tif')
    rawz,zex,_=read_raster('reference/raw/2.tif')
    target,ex,transform=read_raster('reference/data/ANZA_crt.tif')
    frames={};table=pd.read_csv(track(results/'metrics_all_partitions.csv'))
    for n in P.MODELS:
        for role in P.ROLES:
            f=pd.read_csv(track(results/'pairs'/f'{n}_{role}.csv'));frames[n,role]=f
            if not np.array_equal(f.patch_id,split[role]):raise ValueError('Partition ID order mismatch.')
            if not np.array_equal(f[['row','col']].to_numpy(),coords[split[role]]+4):raise ValueError('Patch centre mismatch.')
            if not np.allclose(f.reference,target[0,f.row.to_numpy(),f.col.to_numpy()],atol=1e-5,rtol=0):raise ValueError('Target grid and reference mismatch.')
            m=P.metrics(f);rec=table[(table.model==n)&(table.role==role)].iloc[0]
            for k in ['rmse','mae','bias','r2']:
                if not np.isclose(m[k],rec[k],atol=1e-5,rtol=0):raise ValueError('Metric mismatch.')
    with np.load(track(data/'prepared/legacy_last_split.npz')) as p:ids=p['test']
    historical=[pd.DataFrame({'reference':y[ids],'prediction':np.load(track(data/f'reproduced/{n}_all_predictions.npy'))[ids]}) for n in P.MODELS]
    histories=[json.loads(track(data/f'experiments/spatial_seed20260908/{n}/history.json').read_text()) for n in P.MODELS]
    items=[]
    def save(label,fig,caption,function,layout='portrait',max_height=None):
        heights={'1':3.0,'2':6.0,'3':3.25,'4':2.9,'5':4.1,'6':3.65,'7':3.35,'8':3.7,'9':4.15,'S1':4.3,'S2':3.65,'S3':4.55}
        fig.set_size_inches(6.56,heights[label])
        if label not in ['2','4']:
            for text in fig.findobj(P.matplotlib.text.Text):text.set_fontsize(8.0)
            for ax in fig.axes:
                ax.title.set_fontsize(8.5);ax.xaxis.label.set_fontsize(8.2);ax.yaxis.label.set_fontsize(8.2)
                ax.tick_params(labelsize=7.3)
                for text in ax.texts:
                    if text.get_text().startswith('RMSE'):text.set_fontsize(7.2)
                for collection in ax.collections:
                    if isinstance(collection,P.matplotlib.collections.PathCollection) and collection.get_sizes().size:
                        collection.set_sizes(collection.get_sizes()*.45)
        if label=='7':
            ax=fig.axes[0];handles,labels=ax.get_legend_handles_labels()
            if ax.get_legend():ax.get_legend().remove()
            fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.045),ncol=3,fontsize=8,frameon=False)
        for ax in fig.axes:ax.minorticks_off()
        texts=[t.get_text() for t in fig.findobj(P.matplotlib.text.Text)]
        if any(re.search(r'\b(?:EN|original|SDB[-_ ]CNN)\b',s,re.I) for s in texts):raise ValueError('Outdated model or internal folder name in figure.')
        files={}
        for extn in ['png','pdf','svg']:
            p=out/f'Figure_{label}.{extn}';fig.savefig(p,dpi=400,bbox_inches='tight',pad_inches=.08);files[extn]={'file':p.name,'sha256':P.sha(p)}
        item={'label':label,'caption':caption,'function':function,'layout':'portrait','files':files,'visible_text':texts}
        item['max_width_inches']=6.56
        item['max_height_inches']=7.5
        items.append(item);plt.close(fig);print('Generated figure',label,flush=True)
    geo='Coordinates use WGS 84 / UTM zone 29N (EPSG:32629); axes are kilometres and scale bars are metres. Grid north is indicated. '
    save('1',study_map(rgb,rgbex,rawz[0],zex),'Fig. 1. Study image and signed reference surface for Anza. '+geo+'The RGB image is displayed at its stored 8-bit values without a contrast stretch; its acquisition date, 10 December 2021, is author-reported. Each raster retains its own recorded extent. The zero of the signed colour scale is not a verified shoreline or tidal datum.','study_map',max_height=3.8)
    save('2',A.architecture_figure(architecture_audit),'Fig. 2. Input preparation, common encoder and five model-specific heads. Convolutions use 3 × 3 kernels and ReLU. BN: batch normalisation; GAP: global average pooling; TConv: stride-2 transposed convolution with BN; dropout: 0.3. PI-CNN joins the feature in Equations (2)–(3) and uses MSE only. CNN-BiLSTM has one sequence step. FCNN evaluates its centre output. SeaBed-Net classes are reference intervals. All tensor shapes were checked against the implemented models.','architecture_figure')
    save('3',geographic_partition(coords,y,split,transform,ex),'Fig. 3. Spatial membership and signed-reference distributions. Gaps exclude shared input pixels between roles; counts are shown in the legend. The regions contain different elevation mixtures. EPSG:32629; axes in km, scale bar in m; grid north indicated. Coordinates inherit the reference grid and do not establish image registration.','geographic_partition')
    common='All paired records are retained. Density panels share axis limits and a logarithmic count scale; dashed lines mark identity. Panel (f) shows exact absolute-error ECDFs. '
    save('4',A.evaluation_figure({role:len(frames[P.MODELS[0],role]) for role in P.ROLES}),'Fig. 4. Training, checkpoint selection and evaluation. Training updates the weights; validation controls stopping, learning-rate changes and checkpoint selection. Selected models use the same test centres. Test values determine neither weight updates nor checkpoint selection. Roles share no input pixels.','evaluation_figure')
    save('5',P.learning_figure(histories),'Fig. 5. Training and validation objectives in the common spatial experiment. Diamonds indicate minimum validation-loss epochs; panel (f) places selection within each recorded run. SeaBed-Net combines regression and classification losses. FCNN supervises 81 repeated centre labels. Objective magnitudes are not directly equivalent between these architectures.','learning_figure',layout='landscape')
    save('6',P.paired_figure([frames[n,'test'] for n in P.MODELS]),'Fig. 6. Common spatial test predictions (n = 1,898 per model). Density panels include all pairs and share axis limits and logarithmic count scales; dashed lines mark identity. Panel (f) gives exact absolute-error ECDFs. This is an internal test within one scene.','paired_figure')
    save('7',P.role_figure(table),'Fig. 7. Inference metrics at validation-selected checkpoints. Circles denote training (n = 7,884), squares validation (n = 1,898), and triangles test (n = 1,898). All models share memberships. These are single-run estimates; differences between roles also reflect their reference distributions.','role_figure')
    fig,map_summary=prediction_maps(frames,transform,ex)
    save('8',fig,'Fig. 8. Signed reference and five model predictions at 11,680 retained centres. Panels share colour limits. EPSG:32629; axes in km, scale bar in m; grid north indicated. Dashed rectangles locate the test strip in Fig. 9. Maps include training locations and are descriptive. No predictions are interpolated; registration to RGB remains unverified.','prediction_maps')
    fig,error_summary=test_residual_maps(frames,transform)
    save('9',fig,'Fig. 9. Signed residuals (prediction minus reference) at the 1,898 common test centres for each model. All panels use the same strip and symmetric colour limits containing every residual. Red indicates overprediction; blue indicates underprediction. EPSG:32629; axes in km, scale bar in m; grid north indicated. No interpolation or smoothing is applied.','test_residual_maps')
    save('S1',P.paired_figure(historical),'Fig. S1. Initial predictions (n = 1,329 per model). Density panels include all pairs with shared axes and log-count scales; dashed lines mark identity. Panel (f) gives exact absolute-error ECDFs. Overlap and incomplete training memberships limit independent interpretation.','paired_figure')
    save('S2',P.residual_figure(historical[1]),'Fig. S2. Initial PI-CNN residuals (n = 1,329), defined as prediction minus reference. Panel (c) marks inverse empirical quantiles. Boxes show medians and interquartile ranges, with 1.5-IQR whiskers and all remaining points. Interval counts cover the full signed range.','residual_figure')
    save('S3',P.paired_figure([frames[n,'val'] for n in P.MODELS]),'Fig. S3. Predictions on the spatial validation partition (n = 1,898 per model). '+common+'This partition selected the checkpoints and is not an independent test.','paired_figure')
    report={'script':Path(__file__).name,'script_sha256':P.sha(__file__),'plotting_companion_sha256':P.sha(P.__file__),'architecture_script_sha256':P.sha(A.__file__),'architecture_audit':architecture_audit,
        'source_evaluation_script_sha256':audit['source_hashes']['corrected_script'],'original_script_sha256':audit['source_hashes']['original_script'],
        'source_hashes':inputs,'raster_metadata':raster_meta,'figures':items,'metrics_verified':15,'model_display_names':dict(zip(P.MODELS,P.LABELS)),
        'geolocation_checks':'Every paired ID matches its split; centre row/col equals top-left +4; target raster at each centre matches its recorded reference.',
        'map_summary':map_summary,'error_summary':error_summary,
        'scope':'Inherited georeferencing only; no registration repair, vertical datum inference, data filtering, fitting or prediction interpolation.'}
    (out/'figure_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complete:',out)

if __name__=='__main__':main()
