"""Generate analysis plots by calling the supplied original plotting functions.

No model training or replacement plotting implementation is performed. This
adapter selects recorded paired outputs and adjusts only presentation settings.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();data=args.data_root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).parent/'original_scripts/SDB_Publication_Figures.py'
    spec=importlib.util.spec_from_file_location('supplied_figures',source);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    asts={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse(source.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}
    # Retain all finite paired records, including positive signed targets.
    # Bin and axis extents are presentation parameters, not water-depth classes.
    original.DEPTH_INTERVALS=[(-30,-20),(-20,-10),(-10,0),(0,5),(5,15),(15,30)]
    original.INTERVAL_LABELS=['−30 to −20','−20 to −10','−10 to 0','0 to 5','5 to 15','15 to 30']
    original.INTERVAL_COLORS+=['#6A1B9A','#795548']
    manifest=[];active={}
    def save(fig,path,dpi=300):
        label=active['label']
        if fig._suptitle is not None:fig._suptitle.set_text('')
        for ax in fig.axes:
            title=ax.get_title()
            if len(fig.axes)==1:ax.set_title('')
            ax.grid(False);ax.minorticks_off()
            if active['function']=='fig_scatter_intervals':
                ax.set_xlim(-25,30);ax.set_ylim(-25,30)
                # Extend the existing identity and OLS lines to the displayed range.
                for line in ax.lines:
                    x=np.asarray(line.get_xdata());y=np.asarray(line.get_ydata())
                    if len(x)==2 and x[1]!=x[0]:
                        slope=(y[1]-y[0])/(x[1]-x[0]);intercept=y[0]-slope*x[0]
                        xx=np.array([-25,30]);line.set_data(xx,slope*xx+intercept)
        for text in fig.findobj(matplotlib.text.Text):
            t=text.get_text();t=re.sub(r'\b(?:EN|Original|original)\s*','',t)
            t=t.replace('Reference depth (m)','Reference z (nominal m)').replace('Predicted depth (m)','Prediction (nominal m)')
            t=t.replace('depth interval','signed target interval').replace('Error vs. depth','Error vs. signed target')
            t=t.replace('Loss (MSE, m²)','Training objective').replace('Training loss (MSE)','Training objective').replace('Validation loss (MSE)','Validation objective')
            t=t.replace('(m)','(nominal m)')
            text.set_text(t)
        if active['function']=='fig_error_analysis':
            fig.set_size_inches(13,9)
            for ax in fig.axes:
                if ax.get_title()=='Absolute error by signed target interval':
                    ax.tick_params(axis='x',labelsize=8)
        for ext in ['png','pdf','svg']:fig.savefig(out/f'Figure_{label}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        entry=dict(active);entry['source_script']='original_scripts/SDB_Publication_Figures.py';entry['script_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();entry['function_ast_sha256']=asts[active['function']]
        entry['presentation_adjustments']='Folder labels removed; journal typography and legends; signed-target units; intervals and axes include all paired records. Figure title moved to caption. Training objectives labelled without assuming a common MSE loss.'
        manifest.append(entry);print('Generated Figure',label,'using',active['function'])
    original._save=save
    models=['sdb_cnn','pi_cnn','cnn_bilstm','fcnn','seabed_net'];labels=['CNN','PI-CNN','CNN-BiLSTM','FCNN','SeaBed-Net']
    def frame(reference,prediction):
        y=np.asarray(reference).reshape(-1);p=np.asarray(prediction).reshape(-1)
        assert y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all()
        df=pd.DataFrame({'test':y,'pred':p});df['error']=p-y;df['abs_error']=abs(p-y);return df
    pack=np.load(data/'prepared/patches.npz');ids=np.load(data/'prepared/legacy_last_split.npz')['test'];y=pack['y'].reshape(-1)[ids]
    hist={}
    for i,(model,label) in enumerate(zip(models,labels),1):
        df=frame(y,np.load(data/f'reproduced/{model}_all_predictions.npy')[ids]);hist[model]=df
        active={'label':str(i),'function':'fig_scatter_intervals','model':model,'protocol':'historical','n':len(df),'caption':f'Fig. {i}. {label} historical paired predictions. All 1,329 records are retained, including positive signed targets. Colours distinguish signed-target intervals; the dashed line indicates identity and the solid line is the ordinary least-squares fit. Statistics describe checkpoint reproduction and do not establish independent validation.','data_sources':['prepared/patches.npz','prepared/legacy_last_split.npz',f'reproduced/{model}_all_predictions.npy']}
        original.fig_scatter_intervals(df,label,str(out),350)
    active={'label':'6','function':'fig_error_analysis','model':'pi_cnn','protocol':'historical','n':1329,'caption':'Fig. 6. PI-CNN historical residual diagnostics. The supplied plotting function displays error frequencies, interval boxplots, residuals against signed targets, and a cumulative plotting-position curve. The cumulative curve uses linearly spaced positions from 0 to 100%; percentile lines are calculated from all absolute errors. All 1,329 records contribute. PI-CNN is shown because it has the smallest historical RMSE; this is not evidence of a stable architecture ranking.','data_sources':['prepared/patches.npz','prepared/legacy_last_split.npz','reproduced/pi_cnn_all_predictions.npy']}
    original.fig_error_analysis(hist['pi_cnn'],'PI-CNN',str(out),350)
    active={'label':'7','function':'fig_training_history','protocol':'historical','caption':'Fig. 7. Archived training and validation objectives. The supplied plotting function reads the saved April 2026 histories. SeaBed-Net uses a weighted regression and classification objective; the other models use mean squared error. FCNN averages repeated centre-label losses over its output grid. Curve magnitudes are therefore not directly comparable across every model, and historical training memberships differed.','data_sources':['reference/results/history_*20260415*.json']}
    original.fig_training_history(str(data/'reference/results'),'ANZA',str(out),350)
    for label,model,name in [('S1','pi_cnn','PI-CNN'),('S2','cnn_bilstm','CNN-BiLSTM')]:
        pairs=pd.read_csv(data/f'experiments/spatial_seed20260908/{model}/test_pairs.csv');df=frame(pairs.reference,pairs.prediction)
        active={'label':label,'function':'fig_scatter_intervals','model':model,'protocol':'spatial','n':len(df),'caption':f'Fig. {label}. {name} predictions on the common spatial holdout. The same supplied scatter function is applied to all 1,898 paired test centres. Colours denote signed-target intervals. This single-scene, single-seed experiment measures internal spatial sensitivity; it does not establish external bathymetric validation.','data_sources':[f'experiments/spatial_seed20260908/{model}/test_pairs.csv']}
        original.fig_scatter_intervals(df,name,str(out),350)
    (out/'original_figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
