"""Result figures from the preserved experiment and evaluated paired records.

Plot-only companion to SDB_Visualisation_Avancee_Corrigee.py. No fitting,
inference, resampling, filtering of observations, or alteration of model code.
Run with --results, --data-root and --out; all source and output hashes are saved.
"""
from pathlib import Path
import argparse, hashlib, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

MODELS = ['sdb_cnn', 'pi_cnn', 'cnn_bilstm', 'fcnn', 'seabed_net']
LABELS = ['CNN', 'PI-CNN', 'CNN-BiLSTM', 'FCNN', 'SeaBed-Net']
# The first stored ID is retained only to locate the archived CNN checkpoints.
ROLES = ['train', 'val', 'test']
ROLE_LABELS = ['Training', 'Validation', 'Test']
ROLE_COLORS = ['#0072B2', '#E69F00', '#AA4499']
MODEL_COLORS = ['#0072B2', '#D89000', '#56B4E9', '#AA4499', '#333333']
MARKERS = ['o', 's', '^']
STYLES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1))]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(frame):
    a = frame[['reference', 'prediction']].to_numpy(dtype=float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('Paired observations must be nonempty and finite.')
    e = a[:, 1] - a[:, 0]
    den = np.sum((a[:, 0] - a[:, 0].mean()) ** 2)
    return dict(n=len(a), rmse=float(np.sqrt(np.mean(e**2))),
                mae=float(np.mean(abs(e))), bias=float(e.mean()),
                r2=float(1-np.sum(e**2)/den) if den else None)

def ecdf(values):
    a = np.sort(np.asarray(values, dtype=float))
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('ECDF requires finite observations.')
    return a, np.arange(1, len(a)+1)/len(a)

def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
        'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#5C6570', 'axes.linewidth': .7,
        'axes.grid': False, 'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'legend.frameon': False, 'lines.linewidth': 1.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})

def paired_figure(frames):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), layout='constrained')
    values = np.concatenate([f[['reference', 'prediction']].to_numpy().ravel() for f in frames])
    lo, hi = values.min(), values.max(); pad = (hi-lo)*.025
    lim = (lo-pad, hi+pad); hexes = []
    for j, (ax, frame) in enumerate(zip(axes.flat, frames)):
        h = ax.hexbin(frame.reference, frame.prediction, gridsize=34,
            extent=(*lim, *lim), mincnt=1, cmap='cividis', linewidths=0,
            norm=LogNorm(vmin=1), rasterized=False)
        hexes.append(h)
        ax.plot(lim, lim, color='#111111', linestyle='--', linewidth=1)
        ax.set(xlim=lim, ylim=lim, aspect='equal', title=f'({chr(97+j)}) {LABELS[j]}')
        if j % 3 == 0: ax.set_ylabel('Predicted z (nominal m)')
        if j >= 3: ax.set_xlabel('Reference z (nominal m)')
        m = metrics(frame)
        ax.text(.04, .96, f"RMSE {m['rmse']:.3f}\nMAE {m['mae']:.3f}\nR² {m['r2']:.3f}",
            transform=ax.transAxes, va='top', fontsize=10, fontweight='bold',
            bbox={'facecolor': 'white', 'alpha': .95, 'edgecolor': 'none', 'pad': 3})
    vmax = max(float(h.get_array().max()) for h in hexes)
    for h in hexes: h.set_clim(1, max(2, vmax))
    fig.colorbar(hexes[0], ax=list(axes.flat[:5]), fraction=.024, pad=.015,
                 label='Records per hexagon (log scale)', shrink=.75)
    ax = axes.flat[-1]
    for j, frame in enumerate(frames):
        x, y = ecdf(abs(frame.prediction-frame.reference))
        ax.step(np.r_[0, x], np.r_[0, y], where='post', color=MODEL_COLORS[j],
                linestyle=STYLES[j], label=LABELS[j])
    ax.set(title='(f) Absolute error distribution', xlabel='Absolute error (nominal m)',
           ylabel='Cumulative proportion', ylim=(0, 1.02), xlim=(0, None))
    ax.legend(loc='lower right', fontsize=9)
    return fig

def residual_figure(frame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    z = frame.reference.to_numpy(); e = frame.prediction.to_numpy()-z
    ax = axes[0, 0]
    h = ax.hexbin(z, e, gridsize=38, mincnt=1, norm=LogNorm(), cmap='cividis', linewidths=0)
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.set(title='(a) Signed residuals', xlabel='Reference z (nominal m)', ylabel='Prediction − reference (nominal m)')
    fig.colorbar(h, ax=ax, label='Records (log scale)', shrink=.8)
    ax = axes[0, 1]; ax.hist(e, bins=35, color='#0072B2', edgecolor='white', linewidth=.6)
    ax.axvline(0, color='black', ls='--', lw=1)
    ax.axvline(e.mean(), color='#AA4499', lw=1.8, label=f'Bias {e.mean():.3f}')
    ax.set(title='(b) Residual frequency', xlabel='Prediction − reference (nominal m)', ylabel='Records'); ax.legend()
    ax = axes[1, 0]; x, y = ecdf(abs(e))
    ax.step(np.r_[0, x], np.r_[0, y], where='post', color='#0072B2')
    for q, marker in zip([.5, .75, .9], MARKERS):
        v = x[int(np.ceil(q*len(x)))-1]
        ax.plot(v, q, marker=marker, color='#AA4499', ms=6, label=f'{int(q*100)}% ≤ {v:.2f}')
    ax.set(title='(c) Absolute error distribution', xlabel='Absolute error (nominal m)',
           ylabel='Cumulative proportion', ylim=(0, 1.02), xlim=(0, x.max())); ax.legend(loc='lower right')
    ax = axes[1, 1]
    edges = [-np.inf, -20, -10, 0, 5, 15, np.inf]
    groups = [e[(z>=a)&(z<b)] for a,b in zip(edges[:-1],edges[1:])]
    positions = [i+1 for i, g in enumerate(groups) if len(g)]
    bp = ax.boxplot([g for g in groups if len(g)], positions=positions, patch_artist=True,
        widths=.55, medianprops={'color': '#111111', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#0072B2', 'markeredgewidth': 0})
    for b in bp['boxes']: b.set(facecolor='#CDE7F3', edgecolor='#0072B2', linewidth=.8)
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.set(xticks=range(1,7), xticklabels=[f'{s}\n(n={len(g)})' for s,g in zip(['<−20','−20–<−10','−10–<0','0–<5','5–<15','≥15'],groups)],
        title='(d) Residuals by reference interval', xlabel='Reference z interval (nominal m)', ylabel='Residual (nominal m)')
    ax.tick_params(axis='x', labelsize=8)
    assert sum(map(len, groups)) == len(frame)
    return fig

def learning_figure(histories):
    fig, axes = plt.subplots(2,3,figsize=(12,7.1),layout='constrained')
    for j, (ax,h) in enumerate(zip(axes.flat,histories)):
        x = np.arange(1,len(h['loss'])+1); best = int(np.argmin(h['val_loss']))
        ax.plot(x,h['loss'],color=ROLE_COLORS[0],label='Training')
        ax.plot(x,h['val_loss'],color=ROLE_COLORS[1],ls='--',label='Validation')
        ax.plot(best+1,h['val_loss'][best],marker='D',ms=6,color='#222222',label=f'Selected epoch {best+1}')
        ax.set(title=f'({chr(97+j)}) {LABELS[j]}',xlabel='Epoch',
               ylabel='Weighted objective' if j==4 else 'MSE (nominal m²)',xlim=(1,len(x)))
        ax.legend(fontsize=9)
    ax=axes.flat[-1]
    for j,h in enumerate(histories):
        best=int(np.argmin(h['val_loss']))+1
        ax.plot([1,len(h['loss'])],[j,j],color='#D5DBE0',lw=4,solid_capstyle='round')
        ax.plot(best,j,marker='D',color=MODEL_COLORS[j],ms=6)
    ax.set(title='(f) Checkpoint selection',yticks=range(5),yticklabels=LABELS,xlabel='Epoch',xlim=(1,max(len(h['loss']) for h in histories)))
    ax.invert_yaxis()
    return fig

def role_figure(table):
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for i,(ax,key,label) in enumerate(zip(axes.flat,['rmse','mae','bias','r2'],['RMSE (nominal m)','MAE (nominal m)','Bias (nominal m)','R²'])):
        for j,role in enumerate(ROLES):
            values=table[table.role==role].set_index('model').reindex(MODELS)[key].to_numpy()
            ax.plot(values,np.arange(5)+(j-1)*.19,ls='none',marker=MARKERS[j],ms=6,
                color=ROLE_COLORS[j],label=ROLE_LABELS[j])
        ax.set(title=f'({chr(97+i)}) {label}',yticks=range(5),yticklabels=LABELS)
        ax.invert_yaxis();ax.axvline(0,color='#606770',lw=.6)
        if key in ['rmse','mae']:ax.set_xlim(left=0)
        if key=='r2':ax.set_xlim(min(0, table.r2.min()-.03),1)
    axes[0,0].legend(loc='lower right',fontsize=9)
    return fig

def partition_figure(coords,y,split):
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    used=np.concatenate(list(split.values()));excluded=np.setdiff1d(np.arange(len(coords)),used)
    ax=axes[0];ax.scatter(coords[excluded,1]+4,coords[excluded,0]+4,s=4,color='#CCD1D5',marker='s',label=f'Excluded (n={len(excluded):,})')
    for j,role in enumerate(ROLES):
        ids=split[role];ax.scatter(coords[ids,1]+4,coords[ids,0]+4,s=4,color=ROLE_COLORS[j],marker=MARKERS[j],label=f'{ROLE_LABELS[j]} (n={len(ids):,})')
        x,p=ecdf(y[ids]);axes[1].step(x,p,where='post',color=ROLE_COLORS[j],ls=STYLES[j],label=ROLE_LABELS[j])
    ax.set(title='(a) Patch centres',xlabel='Prepared-grid column',ylabel='Prepared-grid row',aspect='equal');ax.invert_yaxis()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=9,markerscale=3)
    axes[1].set(title='(b) Reference distributions',xlabel='Reference z (nominal m)',ylabel='Cumulative proportion',ylim=(0,1.02));axes[1].legend(loc='lower right')
    return fig

def main():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=root/'SDB_Resultats_Corriges')
    ap.add_argument('--data-root',type=Path,default=root/'data/anza_study')
    ap.add_argument('--out',type=Path,default=root/'SDB_Resultats_Figures')
    a=ap.parse_args();out=a.out.resolve();data=a.data_root.resolve();results=a.results.resolve()
    if out in [data,results] or data in out.parents or results in out.parents:
        raise ValueError('Use a separate results output folder.')
    out.mkdir(parents=True,exist_ok=True);style();inputs={}
    def track(p): inputs[str(p.relative_to(data) if p.is_relative_to(data) else 'results/'+p.relative_to(results).as_posix())]=sha(p);return p
    audit=json.loads(track(results/'reproduction_manifest.json').read_text())
    if not all(audit['scientific_functions_unchanged'].values()):raise ValueError('Model identity audit failed.')
    frames={}
    for name in MODELS:
        for role in ROLES:
            frames[name,role]=pd.read_csv(track(results/'pairs'/f'{name}_{role}.csv'))
    table=pd.read_csv(track(results/'metrics_all_partitions.csv'))
    for (name,role),frame in frames.items():
        m=metrics(frame);record=table[(table.model==name)&(table.role==role)].iloc[0]
        for key in ['rmse','mae','bias','r2']:
            if not np.isclose(m[key],record[key],rtol=0,atol=1e-5):raise ValueError(f'Metric mismatch {name} {role} {key}')
    pack=track(data/'prepared/patches.npz')
    if sha(pack)!=audit['source_hashes']['patches']:raise ValueError('Patch input hash mismatch.')
    with np.load(pack) as p: y=p['y'].reshape(-1);coords=p['coords']
    sp=track(data/'prepared/spatial_split.npz')
    if sha(sp)!=audit['source_hashes']['split']:raise ValueError('Split input hash mismatch.')
    with np.load(sp) as p: split={k:p[k] for k in p.files}
    with np.load(track(data/'prepared/legacy_last_split.npz')) as p: ids=p['test']
    historical=[pd.DataFrame({'reference':y[ids],'prediction':np.load(track(data/f'reproduced/{name}_all_predictions.npy'))[ids]}) for name in MODELS]
    histories=[json.loads(track(data/f'experiments/spatial_seed20260908/{name}/history.json').read_text()) for name in MODELS]
    items=[]
    def save(label,fig,caption,function):
        for ax in fig.axes: ax.minorticks_off()
        texts=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        if any(re.search(r'\b(?:EN|original)\b',s,re.I) for s in texts):raise ValueError('Internal folder label in figure.')
        files={}
        for ext in ['png','pdf','svg']:
            p=out/f'Figure_{label}.{ext}';fig.savefig(p,dpi=400,bbox_inches='tight',pad_inches=.08);files[ext]={'file':p.name,'sha256':sha(p)}
        items.append({'label':label,'caption':caption,'function':function,'files':files,'visible_text':texts})
        plt.close(fig);print('Generated figure',label,flush=True)
    common='Hexagons show every paired record on shared axes and a common logarithmic count scale; the dashed line denotes identity. Panel (f) shows exact empirical cumulative distributions of absolute error. No observations are removed.'
    save('1',paired_figure(historical),'Fig. 1. Historical checkpoint reproduction for five architectures (n = 1,329 each). '+common+' These overlapping historical samples do not provide independent validation.','paired_figure')
    save('2',residual_figure(historical[1]),'Fig. 2. Historical PI-CNN residual diagnostics (n = 1,329). Residuals equal prediction minus reference. Panel (c) marks inverse empirical quantiles. Boxes in (d) show the median and interquartile range, whiskers extend to 1.5 interquartile ranges, and points show all remaining observations. Interval counts cover the full signed target range.','residual_figure')
    save('3',learning_figure(histories),'Fig. 3. Recorded training and validation objectives for the common spatial experiment. Diamonds identify the minimum validation-loss epoch. Panel (f) compares that epoch with the full recorded run. SeaBed-Net combines regression and classification losses; FCNN supervises 81 repeated centre labels. Objective magnitudes across these architectures are not directly equivalent.','learning_figure')
    save('4',paired_figure([frames[n,'test'] for n in MODELS]),'Fig. 4. Spatial test predictions at 1,898 common centres. '+common+' The test patches share no input pixels with training or validation patches. This is one internal holdout from a single scene.','paired_figure')
    save('5',role_figure(table),'Fig. 5. Paired inference metrics at the validation-selected checkpoints. Circles, squares and triangles denote training (n = 7,884), validation (n = 1,898) and test (n = 1,898). All five models use identical memberships. Points are single-run estimates with no confidence intervals; differences between roles also reflect their reference distributions.','role_figure')
    save('S1',paired_figure([frames[n,'val'] for n in MODELS]),'Fig. S1. Spatial validation predictions at 1,898 common centres. '+common+' This partition selected the checkpoints and is not an independent test.','paired_figure')
    save('S2',partition_figure(coords,y,split),'Fig. S2. Common spatial membership and signed reference distributions. Coordinates denote centres in the prepared array, not a georeferenced bathymetric map. The partition includes 7,884 training, 1,898 validation, 1,898 test and 1,606 excluded patches. No patch footprints overlap between the three retained roles.','partition_figure')
    report={'script':Path(__file__).name,'script_sha256':sha(__file__),'source_evaluation_script_sha256':audit['source_hashes']['corrected_script'],
        'original_script_sha256':audit['source_hashes']['original_script'],'source_hashes':inputs,'figures':items,
        'metric_tolerance':1e-5,'metrics_verified':15,'scope':'Plot-only; all data unchanged. Nominal metres; vertical units and datum unverified.'}
    (out/'figure_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complete:',out)

if __name__=='__main__': main()
