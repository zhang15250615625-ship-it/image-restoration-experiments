"""CPU-only audit from saved floating outputs; requires no torch/checkpoints."""
from pathlib import Path
import argparse,ast,csv,hashlib,json
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def resolve(value):
    path=(ROOT/value.replace('\\','/')).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Result path leaves experiment directory')
    return path
def image(path):return np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results');args=parser.parse_args()
    directory=resolve(args.results);summary=read(directory/'summary.json');rows=summary['rows'];cfg=summary['config']
    assert cfg['complete'] is True
    assert len(rows)==len(cfg['images'])*len(cfg['tasks'])*(1+len(cfg['seeds']))
    manifest=read(ROOT/'data/manifest.json');max_psnr=max_ssim=0.;verified_pngs=0
    for record in manifest['records']:
        for key in ['gt','lq']:
            assert hashlib.sha256(resolve(record[key]).read_bytes()).hexdigest()==record[key+'_sha256']
    csvrows=list(csv.DictReader((directory/'metrics.csv').open(encoding='utf-8-sig',newline='')))
    assert len(csvrows)==len(rows)
    for row,csvrow in zip(rows,csvrows):
        for key,value in row.items():
            serialized=csvrow[key]
            if value is None:assert serialized==''
            elif isinstance(value,list):assert ast.literal_eval(serialized)==value
            elif isinstance(value,(int,float)):assert abs(float(serialized)-value)<1e-10
            else:assert serialized==value
        gt=image(ROOT/'data/gt'/f"{row['image']}.png")
        pred=np.load(resolve(row['array_path'])) if row['method']=='pixrestore_s' else image(resolve(row['image_path']))
        assert pred.shape==gt.shape==(512,512,3) and np.isfinite(pred).all()
        clipped=np.clip(pred,0,1)
        psnr=float(peak_signal_noise_ratio(gt,clipped,data_range=1.0))
        ssim=float(structural_similarity(gt,clipped,channel_axis=2,data_range=1.0))
        max_psnr=max(max_psnr,abs(psnr-row['psnr_db']));max_ssim=max(max_ssim,abs(ssim-row['ssim']))
        assert abs(psnr-row['psnr_db'])<1e-6 and abs(ssim-row['ssim'])<1e-6
        assert abs(float(np.mean((pred<0)|(pred>1)))-row['clipped_fraction'])<1e-10
        if row['method']=='pixrestore_s':
            assert row['nfe']==1 and row['model_seed']==row['base_seed']+cfg['images'].index(row['image'])
            assert len(row['seconds_repeats'])==cfg['repeats']
            assert abs(float(np.median(row['seconds_repeats']))-row['seconds_median'])<1e-10
            assert np.array_equal(np.rint(clipped*255).astype(np.uint8),np.asarray(Image.open(resolve(row['image_path']))))
            verified_pngs+=1
    for scope in ['aggregate','primary_seed_aggregate']:
        for task in cfg['tasks']:
            for method in ['input','pixrestore_s']:
                group=[r for r in rows if r['task']==task and r['method']==method and (scope=='aggregate' or method=='input' or r['base_seed']==cfg['seeds'][0])]
                stored=summary[scope][task][method];assert len(group)==stored['n']
                for metric in ['psnr_db','ssim']:assert abs(float(np.mean([r[metric] for r in group]))-stored[metric])<1e-10
                if method=='pixrestore_s':
                    assert abs(float(np.mean([r['seconds_median'] for r in group]))-stored['seconds_median_mean'])<1e-10
                    assert max(r['peak_allocated_mib'] for r in group)==stored['peak_allocated_mib_max']
            for metric in ['psnr','ssim']:
                key='psnr_db' if metric=='psnr' else metric
                means=[np.mean([r[key] for r in rows if r['task']==task and r['method']=='pixrestore_s' and r['base_seed']==seed]) for seed in cfg['seeds']]
                assert abs(float(np.mean(means))-summary['seed_stability'][task][metric+'_mean'])<1e-10
                if len(means)>1:assert abs(float(np.std(means,ddof=1))-summary['seed_stability'][task][metric+'_std'])<1e-10
    routes=read(directory/'router_weights.json')
    assert len(routes['rows'])==verified_pngs
    for row in routes['rows']:
        values=np.asarray(row['p_l']);assert values.shape==(6,) and np.isfinite(values).all() and (values>=0).all()
        assert abs(float(values.sum())-1)<.01 # BF16 casting introduces small rounding differences.
    report={'status':'PASS','rows_verified':len(rows),'float_model_outputs_verified':verified_pngs,'max_psnr_abs_error':max_psnr,
            'max_ssim_abs_error':max_ssim,'checks':['input SHA256','CSV versus JSON','float-output metrics','saved PNG quantization','seed mapping','NFE','repeat timing medians','aggregate means','seed sample standard deviations','router shape/normalization'],
            'limitation':'Recomputes archived evidence on CPU; does not independently rerun the GPU model or validate a population-level paper claim.'}
    (directory/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
