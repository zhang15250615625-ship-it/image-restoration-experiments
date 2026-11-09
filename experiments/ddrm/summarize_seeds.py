"""Summarize all three fixed seed runs, without selecting best samples."""
from pathlib import Path
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parent
sources = [ROOT / folder / 'summary.json' for folder in ['results','results_seed2','results_seed3']]
payloads = [json.loads(path.read_text(encoding='utf-8')) for path in sources]
rows = []
for run, payload in enumerate(payloads):
    for row in payload['rows']:
        if row['method'] == 'ddrm_20':
            rows.append({'run': run+1, 'base_seed': payload['config']['seed'], **row})

summary = {'method': 'ddrm_20', 'scope': '4 fixed public images, 3 independent seeds each; no selected best samples',
           'standard_deviation': 'sample standard deviation (ddof=1); not a confidence interval',
           'rows': rows, 'per_image': {}, 'per_run': []}
for name in payloads[0]['config']['images']:
    selected = [r for r in rows if r['image'] == name]
    summary['per_image'][name] = {'n': len(selected)}
    for metric in ['psnr_db', 'ssim', 'measurement_rmse', 'seconds']:
        values = np.asarray([r[metric] for r in selected])
        summary['per_image'][name][metric] = {'mean': float(values.mean()), 'std': float(values.std(ddof=1)),
                                            'min': float(values.min()), 'max': float(values.max())}
for run in range(1, 4):
    selected = [r for r in rows if r['run'] == run]
    summary['per_run'].append({'run': run, 'base_seed': selected[0]['base_seed'], **{
        metric: float(np.mean([r[metric] for r in selected])) for metric in ['psnr_db','ssim','seconds']}})
summary['across_run_means'] = {metric: {'mean': float(np.mean([r[metric] for r in summary['per_run']])),
    'std': float(np.std([r[metric] for r in summary['per_run']], ddof=1))} for metric in ['psnr_db','ssim','seconds']}
(ROOT / 'results/seed_stability.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
with (ROOT / 'results/seed_metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary['across_run_means'], indent=2))
