"""CPU-only independent audit of recorded DDRM results; does not run the model."""
from pathlib import Path
import csv, json
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
payload=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
with (OUT/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
    csvrows=list(csv.DictReader(f))
rows=payload['rows']
assert len(rows)==len(csvrows)==16, ('main rows incomplete',len(rows))
assert {r['image'] for r in rows}=={'astronaut','coffee','chelsea','rocket'}
assert {r['method'] for r in rows}=={'nearest','bicubic','ddrm_20','ddrm_100'}

def independent_ssim(a,b):
    # scikit-image defaults: 7x7 uniform window, K1=.01/K2=.03, sample covariance.
    result=[]
    for channel in range(3):
        x,y=a[:,:,channel],b[:,:,channel]
        ux,uy=uniform_filter(x,7),uniform_filter(y,7)
        vx=(uniform_filter(x*x,7)-ux*ux)*(49/48)
        vy=(uniform_filter(y*y,7)-uy*uy)*(49/48)
        vxy=(uniform_filter(x*y,7)-ux*uy)*(49/48)
        smap=((2*ux*uy+.01**2)*(2*vxy+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
        result.append(smap[3:-3,3:-3].mean(dtype=np.float64))
    return float(np.mean(result))

checks=[]
for r,c in zip(rows,csvrows):
    assert set(r)==set(c)
    for k,v in r.items():
        assert (str(v)==c[k] if isinstance(v,str) else float(v)==float(c[k])), (k,v,c[k])
    gt=np.asarray(Image.open(OUT/'images'/f"{r['image']}_original.png"),dtype=np.float32)/255
    predraw=np.load(OUT/'arrays'/f"{r['image']}_{r['method']}.npy")
    pred=np.clip(predraw,0,1)
    assert pred.shape==gt.shape==(256,256,3)
    mse=np.mean((gt-pred)**2,dtype=np.float64)
    psnr=float(10*np.log10(1/mse))
    ssim=independent_ssim(gt,pred)
    assert abs(psnr-r['psnr_db'])<1e-6,(r['image'],r['method'],'psnr',psnr,r['psnr_db'])
    assert abs(ssim-r['ssim'])<1e-6,(r['image'],r['method'],'ssim',ssim,r['ssim'])
    lr=gt.reshape(64,4,64,4,3).mean(axis=(1,3))
    for field,source in [('measurement_rmse',pred),('measurement_rmse_before_clip',predraw)]:
        measured=source.reshape(64,4,64,4,3).mean(axis=(1,3))
        error=float(np.sqrt(np.mean((lr-measured)**2)))
        # Prepared gt through avg_pool may differ by float rounding from NumPy averaging.
        assert abs(error-r[field])<1e-6,(field,error,r[field])
    png=np.asarray(Image.open(OUT/'images'/f"{r['image']}_{r['method']}.png"))
    assert np.array_equal(png,np.rint(pred*255).astype(np.uint8))
    assert r['seed']==20261116+['astronaut','coffee','chelsea','rocket'].index(r['image'])
    assert r['nfe']==({'ddrm_20':20,'ddrm_100':100}.get(r['method'],0))
    checks.append({'image':r['image'],'method':r['method'],'psnr_delta':psnr-r['psnr_db'],'ssim_delta':ssim-r['ssim']})

for method,agg in payload['aggregate'].items():
    selected=[r for r in rows if r['method']==method]
    assert agg['n']==len(selected)==4
    for key,value in agg.items():
        if key=='n': continue
        recomputed=float(np.mean([r[key] for r in selected]))
        assert abs(recomputed-value)<1e-10,(method,key,recomputed,value)

report={'status':'PASS','rows':len(rows),'checks':checks,'audited':'CSV vs summary exact; independently recalculated RGB PSNR, SSIM and measurement RMSE from float arrays; PNG rounding; main seed/NFE mapping; aggregate arithmetic means','limitation':'No additional GPU run. Timing is recorded single-run latency per image and is not validated by rerunning.'}
# Verification is read-only: archived result files remain unchanged.
print('PASS: 16 rows; CSV, summary, independent metrics, PNGs, seeds and NFE verified.')
print(json.dumps(report,indent=2))
