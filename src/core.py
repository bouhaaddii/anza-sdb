"""Traceable data preparation, split reconstruction, and evaluation primitives."""
from pathlib import Path
import hashlib, json, os
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

ROOT=Path(os.environ['EN_DATA_ROOT']).resolve()
MODELS=['sdb_cnn','pi_cnn','cnn_bilstm','fcnn','seabed_net']
LABELS=['CNN','PI-CNN','CNN-BiLSTM','FCNN','SeaBed-Net']
def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metrics(y,p):
    y=np.asarray(y,dtype='float64').reshape(-1);p=np.asarray(p,dtype='float64').reshape(-1)
    if y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require paired finite arrays of equal length; no silent filtering.')
    e=p-y; ss=np.sum((y-y.mean())**2)
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),
            'medae':float(np.median(abs(e))),'bias':float(e.mean()),
            'r2':float(1-np.sum(e*e)/ss) if ss>0 else None,
            'p95_abs_error':float(np.quantile(abs(e),.95))}
def extract(rgb,z,window=9,stride=3):
    """Original acceptance rule, now with explicit retained top-left coordinates."""
    xs=[];ys=[];coords=[]
    for r in range(0,z.shape[0]-window+1,stride):
        for c in range(0,z.shape[1]-window+1,stride):
            target=z[r:r+window,c:c+window]
            if not np.isnan(target).any() and np.any(target!=0):
                xs.append(rgb[:,r:r+window,c:c+window].transpose(1,2,0))
                ys.append(z[r+window//2,c+window//2]);coords.append((r,c))
    return np.asarray(xs),np.asarray(ys).reshape(-1,1),np.asarray(coords,dtype='int32')
def prepare():
    """Reproduce actual EN raw preprocessing, explicitly retaining its limitations."""
    dest=ROOT/'prepared';dest.mkdir(exist_ok=True)
    with rasterio.open(ROOT/'reference/raw/1.tif') as s:raw=s.read()
    with rasterio.open(ROOT/'reference/raw/2.tif') as s:zraw=s.read()
    rgb=zoom(np.clip(raw.astype(np.float32)/10000,0,1),(6/raw.shape[0],280/raw.shape[1],444/raw.shape[2]),order=1)
    z=zoom(zraw.astype(np.float32),(1,280/zraw.shape[1],444/zraw.shape[2]),order=1)[0]
    with rasterio.open(ROOT/'reference/data/ANZA_rgb.tif') as s:refx=s.read()
    with rasterio.open(ROOT/'reference/data/ANZA_crt.tif') as s:refz=s.read(1)
    X,y,coords=extract(refx,refz)
    # Frozen prepared raster is authoritative; raw reproduction is checked, not assumed.
    check={'rgb_raw_rebuild_max_abs':float(np.max(abs(rgb-refx))),
           'target_raw_rebuild_max_abs':float(np.max(abs(z-refz))),
           'patches_exact':bool(np.array_equal(X,np.load(ROOT/'reference/data/rgbnss_ANZA.npy'))),
           'labels_exact':bool(np.array_equal(y,np.load(ROOT/'reference/data/depth_ANZA.npy'))),
           'n_patches':len(y),'window':9,'stride':3,'input_shape':list(X.shape),
           'target_range':[float(y.min()),float(y.max())],
           'scope':'Faithful EN inputs; RGB band interpolation, original grid mismatch, and signed target retained.'}
    assert check['patches_exact'] and check['labels_exact'],'Patch reconstruction failed'
    np.savez_compressed(dest/'patches.npz',X=X,y=y,coords=coords)
    # Reconstruct the last saved split from unique exact patch content; no depth-only guessing.
    def key(x,v):return hashlib.sha256(x.tobytes()+np.asarray(v).tobytes()).digest()
    lookup={}
    for i,x in enumerate(X):lookup.setdefault(key(x,y[i]),[]).append(i)
    idx={};ambiguous=0
    for name,suffix in [('train','trn'),('val','val'),('test','tst')]:
        old=np.load(ROOT/f'reference/data/rgbnss_{suffix}_ANZA.npy')
        old_y=np.load(ROOT/f'reference/data/depth_{suffix}_ANZA.npy')
        ids=[]
        for x,v in zip(old,old_y):
            hits=lookup[key(x,v)]
            if len(hits)!=1:ambiguous+=1;raise ValueError('Ambiguous patch identity')
            ids.append(hits[0])
        idx[name]=np.asarray(ids)
        assert np.array_equal(y[idx[name]],np.load(ROOT/f'reference/data/depth_{suffix}_ANZA.npy'))
    assert len(set(np.concatenate(list(idx.values()))))==len(X)
    np.savez(dest/'legacy_last_split.npz',**idx)
    d=cKDTree(coords[idx['train']]).query(coords[idx['test']],k=1,p=np.inf)[0]
    check.update({'legacy_test_n':len(idx['test']),'legacy_train_n':len(idx['train']),
                  'legacy_val_n':len(idx['val']),'test_patches_overlapping_train':int(np.sum(d<9)),
                  'test_overlap_fraction':float(np.mean(d<9)),
                  'test_positive_target_n':int((y[idx['test']]>=0).sum()),
                  'legacy_plot_filter_n':int(((y[idx['test']]>=-22)&(y[idx['test']]<22)).sum())})
    for seed in [20260908]:
        ids=np.random.default_rng(seed).permutation(len(X));i=int(.8*len(X));j=int(.9*len(X))
        np.savez(dest/'common_random_split.npz',train=ids[:i],val=ids[i:j],test=ids[j:])
    # Frozen geometric split independent of target values. A full 9-pixel empty strip
    # plus whole-patch containment at each boundary prevents shared input pixels.
    # Held-out row strips are an internal sensitivity experiment, not external validation.
    h=refz.shape[0];b1=int(.6*h);b2=int(.8*h);gap=9;r=coords[:,0]
    spatial={'train':np.flatnonzero(r+9<=b1),'val':np.flatnonzero((r>=b1+gap)&(r+9<=b2)),
             'test':np.flatnonzero(r>=b2+gap)}
    np.savez(dest/'spatial_split.npz',**spatial)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert np.min(cKDTree(coords[spatial[a]]).query(coords[spatial[b]],p=np.inf)[0])>=9
    check['spatial_counts']={k:len(v) for k,v in spatial.items()}
    check['spatial_row_boundaries']=[b1,b2];check['spatial_gap_pixels']=gap
    save_json(ROOT/'audit/preparation_checks.json',check)
    return check
def stratified(y,p):
    bins=[-np.inf,-20,-10,0,5,15,np.inf]
    names=['below -20','-20 to -10','-10 to 0','0 to 5','5 to 15','15 and above']
    rows=[]
    for lo,hi,name in zip(bins[:-1],bins[1:],names):
        m=(y>=lo)&(y<hi)
        if m.any(): rows.append({'interval':name,**metrics(y[m],p[m])})
    return rows
