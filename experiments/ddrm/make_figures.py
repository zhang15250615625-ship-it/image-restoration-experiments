"""Build figures only from saved predictions and measured results."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
COLORS = {'nearest': '#94a3b8', 'bicubic': '#6b7280', 'ddrm_20': '#167d9a', 'ddrm_100': '#dca548'}
LABELS = {'nearest': 'Nearest', 'bicubic': 'Bicubic', 'ddrm_20': 'DDRM 20', 'ddrm_100': 'DDRM 100'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    result = ROOT / args.results
    payload = json.loads((result / 'summary.json').read_text(encoding='utf-8'))
    rows = payload['rows']
    aggregate = payload['aggregate']
    figures = result / 'figures'
    figures.mkdir(exist_ok=True)
    names = payload['config']['images']
    methods = ['original', 'nearest', 'bicubic', 'ddrm_20', 'ddrm_100']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(names), len(methods), figsize=(15, 3.1 * len(names)), layout='constrained')
    for row_index, name in enumerate(names):
        for col_index, method in enumerate(methods):
            ax = axes[row_index, col_index]
            ax.imshow(Image.open(result / 'images' / f'{name}_{method}.png'))
            ax.axis('off')
            if row_index == 0:
                ax.set_title('Ground truth' if method == 'original' else LABELS[method], fontsize=16)
            if col_index == 0:
                ax.text(-0.06, 0.5, name.capitalize(), rotation=90, transform=ax.transAxes, va='center', ha='right', fontsize=14)
            if method != 'original':
                matching = next(r for r in rows if r['image'] == name and r['method'] == method)
                ax.text(.5, -.04, f"{matching['psnr_db']:.2f} dB / {matching['ssim']:.3f}", transform=ax.transAxes, ha='center', fontsize=12)
    fig.savefig(figures / 'comparison_all.png', dpi=180, facecolor='white')
    plt.close(fig)
    # Plain grids intended for presentation labels supplied as editable PPT text.
    for name in names:
        tiles = [Image.open(result / 'images' / f'{name}_{m}.png').convert('RGB') for m in ['original','bicubic','ddrm_20','ddrm_100']]
        canvas = Image.new('RGB', (4 * 256, 256), 'white')
        for i, tile in enumerate(tiles):
            canvas.paste(tile, (i * 256, 0))
        canvas.save(figures / f'{name}_four_columns.png')
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), layout='constrained')
    chosen = ['nearest', 'bicubic', 'ddrm_20', 'ddrm_100']
    for ax, metric, title in zip(axes[:2], ['psnr_db', 'ssim'], ['RGB PSNR (dB)', 'RGB SSIM']):
        ax.bar(np.arange(4), [aggregate[m][metric] for m in chosen], color=[COLORS[m] for m in chosen])
        ax.set_xticks(np.arange(4), [LABELS[m] for m in chosen], rotation=20)
        ax.set_title(title)
        for i, m in enumerate(chosen):
            value = aggregate[m][metric]
            ax.text(i, value, f'{value:.2f}' if metric == 'psnr_db' else f'{value:.3f}', ha='center', va='bottom')
        ax.set_ylim(0, max(aggregate[m][metric] for m in chosen) * 1.15)
    methods_ddrm = ['ddrm_20', 'ddrm_100']
    axes[2].bar([0,1], [aggregate[m]['seconds'] for m in methods_ddrm], color=[COLORS[m] for m in methods_ddrm])
    axes[2].set_xticks([0,1], ['20 steps','100 steps'])
    axes[2].set_title('Measured seconds / image')
    for i,m in enumerate(methods_ddrm):
        axes[2].text(i,aggregate[m]['seconds'],f"{aggregate[m]['seconds']:.1f}s",ha='center',va='bottom')
    axes[2].set_ylim(0, max(aggregate[m]['seconds'] for m in methods_ddrm)*1.15)
    fig.savefig(figures / 'metric_and_runtime.png', dpi=180, facecolor='white')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8,4),layout='constrained')
    for name in names:
        selected = [next(r for r in rows if r['image']==name and r['method']==m) for m in methods_ddrm]
        ax.plot([r['nfe'] for r in selected],[r['psnr_db'] for r in selected],marker='o',label=name.capitalize())
    ax.set_xlabel('Number of function evaluations (NFE)')
    ax.set_ylabel('RGB PSNR (dB)')
    ax.set_xticks([20,100])
    ax.legend(frameon=False,ncols=2)
    ax.grid(axis='y',alpha=.2)
    fig.savefig(figures / 'steps_per_image.png',dpi=180,facecolor='white')
    plt.close(fig)
    print(f'Figures written to {figures}')

if __name__ == '__main__':
    main()
