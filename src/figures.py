"""Study figures computed directly from frozen inputs and paired outputs."""
from core import *
import re
FIGURE_ROOT=Path(os.environ.get('ANZA_FIGURE_ROOT',str(ROOT/'figures'))).resolve()
FIGURE_ROOT.mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm, LogNorm
from rasterio.warp import transform
COLORS=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']
def style():
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':350,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold','figure.facecolor':'white'})
def finish(fig,name,caption,sources):
    # Figure labels describe data and model outputs, never local folder names.
    for obj in fig.findobj(matplotlib.text.Text):
        text=obj.get_text()
        text=re.sub(r'\b(?:Original|original|EN)\s*', '', text)
        text=re.sub(r'^([a-f])  ', r'(\1) ', text)
        obj.set_text(text)
    caption=caption.replace('EN ', '').replace('original ', '').replace('Original-', '')
    for ext in ['png','pdf','svg']:
        fig.savefig(FIGURE_ROOT/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    manifests.append({'figure':name,'caption':caption,'sources':sources})
def grid_extent(s):return (s.bounds.left,s.bounds.right,s.bounds.bottom,s.bounds.top)
def mapaxes(ax,title):
    ax.set_title(title,loc='left');ax.ticklabel_format(useOffset=False,style='plain');ax.tick_params(axis='x',rotation=25)
    ax.set_xlabel('Easting (m)');ax.set_ylabel('Northing (m)')
def box(ax,x,y,w,h,text,color='#EDF3F7',fontsize=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec='#64748B',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1,color='#475569'))
def make_figures():
    global manifests
    manifests=[];style()
    pack=np.load(ROOT/'prepared/patches.npz');X,y,coords=pack['X'],pack['y'][:,0],pack['coords']
    ids=np.load(ROOT/'prepared/legacy_last_split.npz')['test']
    with rasterio.open(ROOT/'reference/raw/1.tif') as src:raw=src.read();ext=grid_extent(src)
    with rasterio.open(ROOT/'reference/data/ANZA_crt.tif') as src:z=src.read(1);ze=grid_extent(src);tr=src.transform
    fig,ax=plt.subplots(1,3,figsize=(10.8,3.7),layout='constrained')
    ax[0].imshow(raw.transpose(1,2,0),extent=ext);mapaxes(ax[0],'a  Supplied three-channel RGB image')
    im=ax[1].imshow(z,extent=ze,cmap='terrain',norm=TwoSlopeNorm(0,vmin=-23,vmax=23));mapaxes(ax[1],'b  Original EN reference surface')
    fig.colorbar(im,ax=ax[1],shrink=.72,label='Signed target z (nominal m)')
    for role,col in [('bathymetry',COLORS[0]),('topography',COLORS[3])]:
        q=pd.read_csv(ROOT/f'tables/survey_{role}.csv')
        ax[2].scatter(q.candidate_utm_easting,q.candidate_utm_northing,s=.25,c=col,label=f'{role.capitalize()} (n={len(q):,})',rasterized=True)
    ax[2].set_aspect('equal');mapaxes(ax[2],'c  Original point deliveries')
    ax[2].legend(loc='lower left',fontsize=7,markerscale=5)
    ax[2].text(.02,.98,'Candidate transform\nMerchich Sud → UTM 29N',transform=ax[2].transAxes,va='top',fontsize=7)
    finish(fig,'Fig01_inputs','Supplied RGB image, original signed reference raster and separate survey deliveries. Map coordinates use EPSG:32629. EPSG:26192 is consistent with the Lambert Zone 2 parameters in the GEOMAITRISE plan dated November 2022; the exact transformation to the EN reference raster is unrecorded. The author identifies the image as S2_20211210 (10 December 2021); full product metadata, field dates and vertical datum remain unconfirmed. Positive targets are retained as signed elevations, not water depths.',['reference/raw/1.tif','reference/data/ANZA_crt.tif','tables/survey_bathymetry.csv','tables/survey_topography.csv','audit/dwg_titleblock.txt'])
    fig=plt.figure(figsize=(7.1,5.1));ax=fig.add_axes([.02,.04,.96,.92]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.01,.72,.23,.2,'Study evidence\nRGB image\nReference surface\nCheckpoints\nSource hashes')
    box(ax,.34,.72,.29,.2,'Faithful preparation\n280 × 444 × 6\nderived channels\n13,286 patches\nof 9 × 9 × 6')
    box(ax,.74,.72,.24,.2,'Seven scientific\nfunctions\nFive original\narchitectures\nAST identity checked')
    arrow(ax,(.25,.82),(.33,.82));arrow(ax,(.64,.82),(.73,.82))
    box(ax,.04,.32,.4,.23,'Checkpoint reproduction\nApril 2026 weights\nCurrent test pairs (n = 1,329)\nHistorical metrics verified\nTraining membership\npartly unrecorded',fontsize=9)
    box(ax,.56,.32,.4,.23,'Spatial sensitivity experiment\nSame inputs, models,\nlabels and losses\nShared split; empty boundary strips\nFresh weights\nPaired test exports',color='#FFF5E3',fontsize=9)
    arrow(ax,(.48,.7),(.24,.57));arrow(ax,(.5,.7),(.76,.57))
    box(ax,.19,.02,.62,.16,'Traceable metrics and map outputs\nMetrics use all finite pairs\nMaps distinguish training coverage\nSource uncertainty remains visible',fontsize=9)
    arrow(ax,(.24,.3),(.42,.19));arrow(ax,(.76,.3),(.59,.19))
    ax.set_xlim(-.015,1.015)
    finish(fig,'Fig02_workflow','Rebuild workflow. The original model functions are retained. The spatial split and explicit provenance are separate evaluation corrections; they do not repair the RGB-derived channel content or the original raster alignment.',['src/core.py','src/original_models.py','pipeline.py','audit/preparation_checks.json'])
    fig,ax=plt.subplots(figsize=(7.1,5.9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.02,.42,.19,.23,'Shared input\n9 × 9 × 6\n\nConv 32 → pool\nConv 64 → pool\nConv 128',fontsize=10)
    texts=[('CNN','GAP 128 → Dense 64 → 32 → 1'),('PI-CNN','GAP 128 + log-ratio proxy → 64 → 32 → 1\nLoss = MSE only; no physics penalty'),('CNN-BiLSTM','GAP 128 → reshape (1,128) → BiLSTM 64 × 2\n→ Dense 64 → 32 → 1; sequence length = 1'),('FCNN','2 × 2 × 128 → transpose-conv 64 → 32\n→ pad to 9 × 9 → conv 32 → 1\nAll output cells supervised by repeated centre z'),('SeaBed-Net','GAP 128 → separate Dense 64 → 32 heads\nDepth 1 + depth-threshold class 4\nLoss = 0.7 MSE + 0.3 cross-entropy')]
    for i,(label,txt) in enumerate(texts):
        yy=.82-i*.19;box(ax,.33,yy,.64,.155,label+'\n'+txt,fontsize=9)
        arrow(ax,(.22,.53),(.32,yy+.077))
    finish(fig,'Fig03_architectures','Implemented EN architectures. Convolutions use 3 × 3 kernels and batch normalisation; the first two blocks pool by 2. Ordinary Dropout, not SpatialDropout, is used. GAP denotes global average pooling. The PI-CNN feature uses channel indices 1 and 2, which are interpolated RGB combinations. SeaBed-Net classes use signed target thresholds 2, 5 and 15 and have no sediment or habitat observations.',['src/original_models.py','tables/model_parameters.csv'])
    fig,ax=plt.subplots(1,3,figsize=(10.8,3.5),layout='constrained')
    for k,(file,title) in enumerate([('legacy_last','a  Last historical random split'),('spatial','b  New spatial sensitivity split')]):
        ss=np.load(ROOT/f'prepared/{file}_split.npz')
        for group,col in [('train','#9AA9B6'),('val','#E69F00'),('test','#0072B2')]:
            cc=coords[ss[group]]+4;ax[k].scatter(cc[:,1],cc[:,0],s=1.3,c=col,label=f'{group} ({len(cc):,})',rasterized=True)
        ax[k].set(xlim=(0,444),ylim=(280,0),aspect='equal',xlabel='EN grid column',ylabel='EN grid row');ax[k].set_title(title,loc='left');ax[k].legend(fontsize=7,loc='upper right',markerscale=3)
    for group,col in [('train','#9AA9B6'),('val','#E69F00'),('test','#0072B2')]:
        ss=np.load(ROOT/'prepared/spatial_split.npz');ax[2].hist(y[ss[group]],bins=np.linspace(-24,24,25),density=True,histtype='step',color=col,label=group)
    ax[2].set(xlabel='Signed reference z (nominal m)',ylabel='Density');ax[2].set_title('c  Spatial split target distributions',loc='left');ax[2].legend(fontsize=8)
    finish(fig,'Fig04_splits','Patch centres for the recovered historical split and the new spatial split. Every historical test patch overlaps at least one training patch in the last saved split. In the spatial experiment, complete 9 × 9 patches lie within row regions separated by at least nine unused rows; input pixels are disjoint between partitions. Spatial autocorrelation beyond those gaps can remain. Distributions reveal potential spatial covariate shift.',['prepared/legacy_last_split.npz','prepared/spatial_split.npz','prepared/patches.npz'])
    fig,ax=plt.subplots(2,3,figsize=(10.8,6.8),layout='constrained');preds={};hexbins=[]
    for i,(m,label) in enumerate(zip(MODELS,LABELS)):
        p=np.load(ROOT/f'reproduced/{m}_all_predictions.npy')[ids];preds[m]=p;a=ax.flat[i]
        hb=a.hexbin(y[ids],p,gridsize=35,mincnt=1,cmap='viridis',extent=(-24,24,-24,30),rasterized=True)
        hexbins.append(hb)
        a.plot([-24,30],[-24,30],'--',color='#555555',lw=.8);a.set(xlim=(-24,24),ylim=(-24,30),xlabel='Reference z (nominal m)',ylabel='Prediction (nominal m)')
        met=metrics(y[ids],p);a.set_title(f'{chr(97+i)}  {label}',loc='left');a.text(.04,.95,f'RMSE {met["rmse"]:.3f}\nMAE {met["mae"]:.3f}\nR² {met["r2"]:.3f}\nn = {met["n"]:,}',va='top',transform=a.transAxes,fontsize=8,bbox={'fc':'white','ec':'none','alpha':.85})
    ax.flat[5].axis('off');ax.flat[5].text(.03,.84,'Historical checkpoint reproduction\n\nAll five April checkpoints use the\nsame currently saved test vector.\n\nThe original training subsets for\nfour models were overwritten.\nThese panels do not establish an\nindependent architecture ranking.',va='top',fontsize=11,linespacing=1.5)
    shared=LogNorm(vmin=1,vmax=max(float(hb.get_array().max()) for hb in hexbins))
    for hb in hexbins:hb.set_norm(shared)
    fig.colorbar(hexbins[0],ax=ax.ravel().tolist(),shrink=.6,label='Pairs per hexagon')
    finish(fig,'Fig05_checkpoint_pairs','Paired checkpoint reproduction on all 1,329 retained test records, without clipping or range filtering. Hexagon colours share a logarithmic count scale. Saved predictions are reproduced within numerical tolerance. These diagnostics must not be described as independent test performance because earlier training membership was overwritten and patches overlap spatially.',['tables/checkpoint_reproduction.csv','reproduced/*_current_split_pairs.csv'])
    fig,ax=plt.subplots(1,3,figsize=(10.8,3.5),layout='constrained')
    for m,label,col in zip(MODELS,LABELS,COLORS):
        p=preds[m];err=p-y[ids];ab=np.sort(abs(err))
        ax[0].plot(ab,np.arange(1,len(ab)+1)/len(ab),c=col,label=label)
        df=pd.DataFrame(stratified(y[ids],p));ax[1].plot(np.arange(len(df)),df.rmse,'o-',c=col,label=label)
    ax[0].set(xlabel='Absolute error (nominal m)',ylabel='Empirical cumulative probability',xlim=(0,15),ylim=(0,1));ax[0].set_title('a  Absolute error distribution',loc='left');ax[0].legend(fontsize=7)
    ax[1].set_xticks(range(6),['<−20','−20 to −10','−10 to 0','0 to 5','5 to 15','≥15'],rotation=40,ha='right');ax[1].set(ylabel='RMSE (nominal m)',xlabel='Reference z interval');ax[1].set_title('b  Error by signed target interval',loc='left')
    df=pd.DataFrame(stratified(y[ids],preds['seabed_net']));ax[2].bar(range(len(df)),df.n,color='#607D8B');ax[2].set_xticks(range(6),['<−20','−20 to −10','−10 to 0','0 to 5','5 to 15','≥15'],rotation=40,ha='right');ax[2].set(ylabel='Paired records',xlabel='Reference z interval');ax[2].set_title('c  Counts in each interval',loc='left')
    for i,v in enumerate(df.n):ax[2].text(i,v+4,str(v),ha='center',fontsize=8)
    finish(fig,'Fig06_error_structure','Historical checkpoint error distributions and complete signed-target intervals, including all values above +5. The ECDF panel displays absolute errors up to 15; its probability denominator includes every paired record and larger errors are retained in all metrics. The intervals describe stored elevations; neither water depth nor optical saturation is inferred. These errors do not certify IHO survey orders.',['reproduced/*_stratified.csv','reproduced/*_current_split_pairs.csv'])
    fig,ax=plt.subplots(2,3,figsize=(10.8,6.3),layout='constrained')
    for i,(m,label) in enumerate(zip(MODELS,LABELS)):
        p=np.load(ROOT/f'reproduced/{m}_all_predictions.npy');cc=coords+4
        a=ax.flat[i];sc=a.scatter(cc[:,1],cc[:,0],c=p,cmap='terrain',norm=TwoSlopeNorm(0,vmin=-23,vmax=23),s=2.5,marker='s',rasterized=True)
        a.set(xlim=(0,444),ylim=(280,0),aspect='equal',xlabel='EN grid column',ylabel='EN grid row');a.set_title(f'{chr(97+i)}  {label}',loc='left')
    a=ax.flat[5];a.imshow(z,cmap='terrain',norm=TwoSlopeNorm(0,vmin=-23,vmax=23));a.set_title('f  Original reference raster',loc='left');a.set(xlabel='EN grid column',ylabel='EN grid row')
    fig.colorbar(sc,ax=ax.ravel().tolist(),shrink=.6,label='Signed output (nominal m)')
    finish(fig,'Fig07_full_scene_outputs','Original-model predictions at every retained patch centre and reference raster. Full-scene inference includes training locations and is descriptive, not a test map. Sparse predictions use stored coordinates; missing centres are not interpolated. Colours use a common range from -23 to +23; values outside it retain their original values in the exports. Index-space display avoids implying repaired geospatial alignment.',['reproduced/*_all_predictions.npy','reference/data/ANZA_crt.tif','prepared/patches.npz'])
    fig,ax=plt.subplots(1,2,figsize=(10.8,3.5),layout='constrained')
    for m,label,col in zip(MODELS,LABELS,COLORS):
        histpath=sorted((ROOT/'reference/results').glob(f'history_{m}_*20260415*.json'))[-1];h=json.loads(histpath.read_text())
        key='depth_mae' if m=='seabed_net' else 'mae'
        ax[0].plot(np.arange(1,len(h[key])+1),h[key],color=col,label=label)
        ax[1].plot(np.arange(1,len(h['val_'+key])+1),h['val_'+key],color=col,label=label)
    for i,a in enumerate(ax):a.set(xlabel='Epoch',ylabel='MAE (nominal m)');a.set_title(['a  Original training histories','b  Original validation histories'][i],loc='left');a.legend(fontsize=7)
    finish(fig,'Fig08_training_histories','Saved April 2026 training and validation MAE histories. Depth-head MAE is used for SeaBed-Net. Historical runs had separately generated training splits; FCNN training MAE averages its repeated-target output grid and therefore differs from centre-only test MAE.',['reference/results/history_*20260415*.json'])
    runs=[ROOT/f'experiments/spatial_seed20260908/{m}/metrics.json' for m in MODELS]
    if all(p.exists() for p in runs):
        df=pd.DataFrame([json.loads(p.read_text()) for p in runs])
        fig,ax=plt.subplots(1,2,figsize=(10.8,3.7),layout='constrained')
        xx=np.arange(5);ax[0].bar(xx-.16,df.rmse,.32,label='RMSE',color='#0072B2');ax[0].bar(xx+.16,df.mae,.32,label='MAE',color='#E69F00')
        ax[0].set_xticks(xx,LABELS,rotation=25,ha='right');ax[0].set_ylabel('Error (nominal m)');ax[0].set_title('a  Fresh spatial test results',loc='left');ax[0].legend()
        for m,label,col in zip(MODELS,LABELS,COLORS):
            h=json.loads((ROOT/f'experiments/spatial_seed20260908/{m}/history.json').read_text());key='val_depth_mae' if m=='seabed_net' else 'val_mae';ax[1].plot(np.arange(1,len(h[key])+1),h[key],c=col,label=label)
        ax[1].set(xlabel='Epoch',ylabel='Validation MAE (nominal m)');ax[1].set_title('b  Spatial validation histories',loc='left');ax[1].legend(fontsize=7)
        finish(fig,'Fig09_spatial_experiment','Fresh training on the fixed spatial split with unchanged EN model functions and inputs. All models share the same 1,898 test centres. The experiment uses one seed and one geometric partition. It measures internal sensitivity to spatial separation and does not resolve source, datum, land-mask or radiometric limitations.',['tables/spatial_experiment.csv','experiments/spatial_seed20260908/*/config.json','experiments/spatial_seed20260908/*/test_pairs.csv'])
    save_json(FIGURE_ROOT/'figure_manifest.json',manifests)
    print('Created',len(manifests),'figures in PNG, PDF and editable SVG.')
