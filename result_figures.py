"""Select result figures from outputs of the corrected model diagnostics."""
from pathlib import Path
import argparse,json,shutil,hashlib

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((a.results/'reproduction_manifest.json').read_text(encoding='utf-8'))
    if not all(audit['scientific_functions_unchanged'].values()):raise ValueError('Source identity was not verified.')
    items=[]
    def add(label,stem,function,caption):
        item={'label':label,'function':function,'source_script':'SDB_Visualisation_Avancee_Corrigee.py','original_source':'SDB_Visualisation_Avancee_EN.py','source_sha256':audit['source_hashes']['corrected_script'],'original_sha256':audit['source_hashes']['original_script'],'result_stem':stem,'caption':caption,'files':{}}
        for ext in ['png','pdf','svg']:
            source=a.results/'figures'/f'{stem}.{ext}';dest=out/f'Figure_{label}.{ext}'
            shutil.copy2(source,dest)
            item['files'][ext]={'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'exact_output_copy':True}
        items.append(item)
    models=['sdb_cnn','pi_cnn','cnn_bilstm','fcnn','seabed_net']
    names=['CNN','PI-CNN','CNN-BiLSTM','FCNN','SeaBed-Net']
    for i,(model,name) in enumerate(zip(models,names),1):
        add(str(i),f'historical/{model}/scatter_intervals_{model}','create_advanced_visualizations',f'Fig. {i}. {name} historical paired predictions. All 1,329 records are retained, including positive signed targets. Colours distinguish signed-target intervals; the dashed line indicates identity and the solid line is the ordinary least-squares fit. Statistics describe checkpoint reproduction and do not establish independent validation. The figure is an output of the corrected primary visualization script, with unchanged model definitions.')
    add('6','historical/pi_cnn/error_analysis_pi_cnn','create_advanced_visualizations','Fig. 6. PI-CNN historical residual diagnostics. Panels show (a) error frequencies, (b) interval boxplots, (c) residuals against signed targets, and (d) the empirical cumulative distribution of absolute errors. Quantile lines use all 1,329 records. PI-CNN illustrates the lowest historical RMSE; this descriptive choice does not establish a stable ranking. The primary-script diagnostics retain all signed intervals and use corrected cumulative probabilities.')
    add('7','overview/training_validation_objectives','training_curves','Fig. 7. Recorded objectives for the common spatial experiment. Blue and orange curves are training and validation losses. Dotted lines mark the minimum validation-loss epochs used to select weights. SeaBed-Net combines regression and classification losses; FCNN supervises a repeated centre label over its grid. Loss magnitudes are not directly comparable across every architecture. All runs use the same spatial partition.')
    add('S1','overview/paired_val_all_models','paired_panels','Fig. S1. Validation predictions at the selected checkpoints. All five models use the same 1,898 validation centres. Colours denote signed-target intervals and the dashed line indicates identity. The validation set was used for checkpoint selection; these results are diagnostic and do not provide an independent test.')
    add('S2','overview/paired_test_all_models','paired_panels','Fig. S2. Test predictions on the common spatial holdout. All five models use the same 1,898 test centres and checkpoints selected solely by validation loss. Colours denote signed-target intervals. This single-scene, single-seed experiment measures internal spatial sensitivity, without external bathymetric validation.')
    add('S3','overview/comparison_training_validation_test','role_comparison','Fig. S3. Paired metrics for training, validation and test at the selected checkpoints. Metrics use 7,884, 1,898 and 1,898 centres, respectively. Every role is evaluated in inference mode. The roles occupy different locations and target distributions, so differences are not a controlled estimate of an overfitting effect. Full R² and bias ranges are displayed.')
    add('S4','overview/spatial_partition','split_figure','Fig. S4. Spatial distribution of retained patch centres. Colours identify training, validation and test centres; grey marks 1,606 excluded centres. Coordinates are indices of the prepared grid. Whole 9 × 9 patch footprints from different roles do not share input pixels; this check does not eliminate longer-range spatial dependence.')
    (out/'original_figure_manifest.json').write_text(json.dumps(items,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Selected',len(items),'unaltered plotting outputs for analysis.')

if __name__=='__main__':main()
