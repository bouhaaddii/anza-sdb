"""Internal scientific pipeline. Use run.py to select a separate work directory."""
import argparse, json, os, sys, time
from pathlib import Path
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
CODE_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE_ROOT/'src'))
from core import *
for folder in ['prepared','tables','audit','figures','experiments','reproduced']:
    (ROOT/folder).mkdir(parents=True,exist_ok=True)

def reproduce():
    import tensorflow as tf
    from model_api import create_model
    pack=np.load(ROOT/'prepared/patches.npz');X,y,coords=pack['X'],pack['y'],pack['coords']
    split=np.load(ROOT/'prepared/legacy_last_split.npz');idx=split['test']
    dest=ROOT/'reproduced';dest.mkdir(exist_ok=True)
    reports=[];params=[]
    for name in MODELS:
        tf.keras.backend.clear_session()
        model,*_=create_model(name,input_size=(9,9,6))
        weights=sorted((ROOT/'reference/ckpts').glob(f'{name}_*20260415*.h5'))[-1]
        model.load_weights(weights)
        pred=model.predict(X,batch_size=512,verbose=0)
        if isinstance(pred,list):
            np.save(dest/f'{name}_class_probabilities.npy',pred[1]);pred=pred[0]
        if name=='fcnn':pred=pred[:,4,4,0]
        pred=pred.reshape(-1)
        np.save(dest/f'{name}_all_predictions.npy',pred)
        old=np.load(ROOT/f'reference/data/depth_pred_{weights.stem}.npy').reshape(-1)
        current=pred[idx]
        rec={'model':name,'checkpoint':weights.name,'checkpoint_sha256':sha(weights),
             'scope':'Current saved split; training membership of earlier models unverified',
             'saved_prediction_max_abs_difference':float(np.max(abs(old-current))),
             'saved_prediction_matches_current_split':bool(np.allclose(old,current,atol=1e-4,rtol=1e-5)),
             **metrics(y[idx],current)}
        oldrec=json.loads((ROOT/f'reference/results/results_{weights.stem}.json').read_text())
        rec['historical_logged_rmse']=oldrec['metrics']['RMSE'];rec['historical_logged_n']=oldrec['samples']
        reports.append(rec)
        params.append({'model':name,'total_parameters':model.count_params(),'trainable_parameters':sum(int(np.prod(w.shape)) for w in model.trainable_weights),'output_shape':str(model.output_shape)})
        df=pd.DataFrame({'patch_id':idx,'row':coords[idx,0]+4,'col':coords[idx,1]+4,'reference':y[idx,0],'prediction':current})
        df.to_csv(dest/f'{name}_current_split_pairs.csv',index=False)
        pd.DataFrame(stratified(y[idx,0],current)).to_csv(dest/f'{name}_stratified.csv',index=False)
        with rasterio.open(ROOT/'reference/data/ANZA_crt.tif') as s:profile=s.profile.copy();shape=s.shape
        sparse=np.full(shape,-9999,dtype='float32');sparse[coords[:,0]+4,coords[:,1]+4]=pred
        profile.update(dtype='float32',count=1,nodata=-9999,compress='lzw')
        with rasterio.open(dest/f'{name}_patch_centres.tif','w',**profile) as dst:
            dst.write(sparse,1);dst.update_tags(scope='All EN patch centres; includes training locations; original alignment limitations retained')
        print(name,rec,flush=True)
    pd.DataFrame(reports).to_csv(ROOT/'tables/checkpoint_reproduction.csv',index=False)
    pd.DataFrame(params).to_csv(ROOT/'tables/model_parameters.csv',index=False)
    save_json(ROOT/'audit/checkpoint_reproduction.json',reports)

def train(args):
    import tensorflow as tf
    from model_api import create_model,prepare_seabed_labels,archive_model_id
    pack=np.load(ROOT/'prepared/patches.npz');X=pack['X'];y=pack['y']
    split=np.load(ROOT/f'prepared/{args.protocol}_split.npz')
    dest=ROOT/'experiments'/f'{args.protocol}_seed{args.seed}';dest.mkdir(parents=True,exist_ok=True)
    selected=MODELS if args.model=='all' else [archive_model_id(args.model)]
    for name in selected:
        run=dest/name;run.mkdir(exist_ok=True)
        if (run/'metrics.json').exists() and not args.overwrite:
            print('Completed run retained:',name,flush=True);continue
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(args.seed)
        tf.config.experimental.enable_op_determinism()
        model,loss,loss_weights,met=create_model(name,input_size=(9,9,6))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=loss,loss_weights=loss_weights,metrics=met)
        def target(ids):
            v=y[ids]
            if name=='seabed_net':return {'depth':v,'seabed_class':prepare_seabed_labels(v)}
            if name=='fcnn':return np.broadcast_to(v[:,None,None,:],(len(v),9,9,1)).copy()
            return v
        checkpoint=run/'best.weights.h5'
        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                if epoch%10==0:print(name,'epoch',epoch+1,{k:round(float(v),4) for k,v in logs.items()},flush=True)
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint),save_weights_only=True,save_best_only=True,monitor='val_loss'),
                   tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=50,restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=10,min_lr=1e-7),Progress()]
        config={'model':name,'protocol':args.protocol,'seed':args.seed,'epochs_max':args.epochs,'batch_size':512,'learning_rate':.0001,'dropout':.3,'patience':50,'loss_weights':loss_weights,'input_sha256':sha(ROOT/'prepared/patches.npz'),'split_sha256':sha(ROOT/f'prepared/{args.protocol}_split.npz'),'scientific_functions_sha256':sha(CODE_ROOT/'src/original_models.py'),'counts':{k:len(split[k]) for k in split.files},'tensorflow':tf.__version__,'numpy':np.__version__,'devices':[str(d) for d in tf.config.list_physical_devices()],'scope':'New validation protocol using unchanged EN inputs, architectures, losses and labels. Internal sensitivity analysis; source metadata unresolved.'}
        save_json(run/'config.json',config)
        start=time.time()
        hist=model.fit(X[split['train']],target(split['train']),validation_data=(X[split['val']],target(split['val'])),epochs=args.epochs,batch_size=512,callbacks=callbacks,verbose=0)
        model.load_weights(checkpoint)
        pred=model.predict(X[split['test']],batch_size=512,verbose=0)
        if isinstance(pred,list):np.save(run/'class_probabilities.npy',pred[1]);pred=pred[0]
        if name=='fcnn':pred=pred[:,4,4,0]
        pred=pred.reshape(-1);yt=y[split['test'],0]
        pd.DataFrame({'patch_id':split['test'],'reference':yt,'prediction':pred}).to_csv(run/'test_pairs.csv',index=False)
        save_json(run/'history.json',{k:[float(x) for x in v] for k,v in hist.history.items()})
        result={'model':name,'protocol':args.protocol,'epochs_run':len(hist.history['loss']),'best_epoch':int(np.argmin(hist.history['val_loss'])+1),'runtime_seconds':time.time()-start,**metrics(yt,pred)}
        save_json(run/'metrics.json',result);print('FINISHED',result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');sub.add_parser('reproduce');sub.add_parser('figures')
    q=sub.add_parser('train');q.add_argument('--protocol',choices=['common_random','spatial'],default='spatial');q.add_argument('--seed',type=int,default=20260908);q.add_argument('--epochs',type=int,default=300);q.add_argument('--model',choices=MODELS+['cnn','all'],default='all');q.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(),indent=2))
    elif args.command=='reproduce':reproduce()
    elif args.command=='train':train(args)
    else:
        from figures import make_figures
        make_figures()
