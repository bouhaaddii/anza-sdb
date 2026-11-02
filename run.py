"""Portable entry point for model code and staged study inputs and outputs."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def source_identity():
    expected=json.loads((CODE/'metadata/scientific_functions.json').read_text(encoding='utf-8'))
    source=CODE/'src/original_models.py'
    current={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
        for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    if sha(source)!=expected['source_file_sha256'] or current!=expected['functions']:
        raise ValueError('Original scientific source identity changed')
    return len(current)

def load_core(root):
    os.environ['EN_DATA_ROOT']=str(root)
    sys.path.insert(0,str(CODE/'src'))
    import core
    return core

def checked_path(root, relative):
    target=(root/relative).resolve()
    if not target.is_relative_to(root):raise ValueError('Manifest path escapes data root')
    return target

def verify(root, inputs_only=False):
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    core=load_core(root)
    manifest=json.loads((root/'file_manifest.json').read_text(encoding='utf-8'))
    reports=[]
    def check(name, condition):
        if not condition:raise ValueError(name)
        reports.append(name)
    selected=[q for q in manifest['files'] if not inputs_only or q['path'].startswith('reference/')]
    check('Bundle hashes',all(sha(checked_path(root,q['path']))==q['sha256'] for q in selected))
    check('Seven original scientific functions',source_identity()==7)
    pack=np.load(root/'prepared/patches.npz');coords=pack['coords'];z=pack['y'][:,0]
    split=np.load(root/'prepared/spatial_split.npz');legacy=np.load(root/'prepared/legacy_last_split.npz')
    check('Unique patch coordinates',len(np.unique(coords,axis=0))==len(z)==13286)
    check('Full historical membership',np.array_equal(np.sort(np.concatenate([legacy[k] for k in legacy.files])),np.arange(len(z))))
    check('Spatial counts',[len(split[k]) for k in ['train','val','test']]==[7884,1898,1898])
    for a,b in [('train','val'),('train','test'),('val','test')]:
        check('No shared input pixels '+a+' '+b,cKDTree(coords[split[a]]).query(coords[split[b]],p=np.inf)[0].min()>=9)
    for model in core.MODELS:
        for protocol in ['historical','spatial']:
            if protocol=='historical':
                file=root/f'reproduced/{model}_current_split_pairs.csv';ids=legacy['test']
                summary=next(q for q in json.loads((root/'audit/checkpoint_reproduction.json').read_text()) if q['model']==model)
            else:
                file=root/f'experiments/spatial_seed20260908/{model}/test_pairs.csv';ids=split['test']
                summary=json.loads(file.with_name('metrics.json').read_text())
                config=json.loads(file.with_name('config.json').read_text())
                check(model+' original code hash',config['scientific_functions_sha256']==sha(CODE/'src/original_models.py'))
            pairs=pd.read_csv(file)
            check(model+' '+protocol+' pair identity',np.array_equal(pairs.patch_id,ids) and np.allclose(pairs.reference,z[ids],atol=2e-6,rtol=0))
            values=core.metrics(pairs.reference,pairs.prediction)
            check(model+' '+protocol+' full metrics',all(abs(values[k]-summary[k])<2e-6 for k in ['rmse','mae','medae','bias','r2','p95_abs_error']))
    return {'status':'passed','checks':reports,'passed':len(reports),'hashed_files':len(selected),'hash_scope':'reference inputs' if inputs_only else 'complete immutable bundle'}

def demo(output):
    import numpy as np
    import pandas as pd
    if output.exists() and any(output.iterdir()):raise ValueError('Choose an empty demonstration output directory')
    output.mkdir(parents=True,exist_ok=True)
    core=load_core(output)
    r,c=np.mgrid[:45,:36]
    # Analytic, non-geographic surfaces. No study pixels or labels are used.
    z=(0.3*r-0.2*c+np.sin(c/5)).astype('float32')
    rgb=np.stack([.01+.0001*r+.00005*c+j*.001 for j in range(6)]).astype('float32')
    X,y,coords=core.extract(rgb,z)
    pred=y[:,0]+0.25*np.sin(np.arange(len(y)))
    np.savez_compressed(output/'SYNTHETIC_ONLY.npz',X=X,y=y,coords=coords)
    pd.DataFrame({'patch_id':np.arange(len(y)),'reference':y[:,0],'prediction':pred}).to_csv(output/'SYNTHETIC_ONLY_pairs.csv',index=False)
    result={'scope':'SYNTHETIC SOFTWARE DEMONSTRATION ONLY. Analytic predictions, no trained model, no Anza data, no scientific benchmark.',**core.metrics(y[:,0],pred)}
    (output/'SYNTHETIC_ONLY_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def workspace(data, work):
    if work==data or work.is_relative_to(data):raise ValueError('Work directory must be outside the staged data root')
    stamp=work/'.en-workspace.json'
    source_hash=sha(data/'file_manifest.json')
    if work.exists() and any(work.iterdir()):
        if not stamp.exists() or json.loads(stamp.read_text())['source_manifest_sha256']!=source_hash:
            raise ValueError('Existing work directory does not match this data bundle; choose an empty directory')
    else:
        shutil.copytree(data,work,dirs_exist_ok=True)
        stamp.write_text(json.dumps({'source_manifest_sha256':source_hash})+'\n')
    return work

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check-code',help='Verify original scientific function identity without loading data or TensorFlow')
    q=sub.add_parser('demo',help='Generate synthetic diagnostics; these are not study results');q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('verify',help='Check real bundle and recompute all paired metrics');q.add_argument('--data-root',type=Path,required=True);q.add_argument('--inputs-only',action='store_true');q.add_argument('--report',type=Path)
    for name in ['prepare','reproduce','figures','train']:
        q=sub.add_parser(name);q.add_argument('--data-root',type=Path,required=True);q.add_argument('--work-root',type=Path,required=True)
        if name=='train':
            q.add_argument('--model',choices=['all','cnn','sdb_cnn','pi_cnn','cnn_bilstm','fcnn','seabed_net'],default='all',help='Use cnn for CNN; the old stored identifier remains accepted for compatibility.')
            q.add_argument('--seed',type=int,default=20260908);q.add_argument('--epochs',type=int,default=300)
            q.add_argument('--overwrite',action='store_true',help='Explicitly replace a completed model run in work-root only')
    args=p.parse_args()
    if args.command=='check-code':result={'unchanged_scientific_functions':source_identity()}
    elif args.command=='demo':result=demo(args.output.resolve())
    elif args.command=='verify':
        result=verify(args.data_root.resolve(),args.inputs_only)
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    else:
        data=args.data_root.resolve();work=workspace(data,args.work_root.resolve())
        env=os.environ.copy();env['EN_DATA_ROOT']=str(work)
        command=[sys.executable,str(CODE/'pipeline.py'),args.command]
        if args.command=='train':
            command+=['--protocol','spatial','--model',args.model,'--seed',str(args.seed),'--epochs',str(args.epochs)]
            if args.overwrite:command.append('--overwrite')
        completed=subprocess.run(command,env=env,check=False)
        return completed.returncode
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError) as error:
        print('ERROR: '+str(error),file=sys.stderr);sys.exit(2)
