"""Generate figures from saved results only; never runs a model."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LABELS={'sr4':'4x bicubic SR','gaussian25':'Gaussian noise (25/255)','jpeg10':'JPEG quality 10 (extra task)'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results');args=parser.parse_args()
    out=ROOT/args.results;figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    rows=summary['rows'];seed=summary['config']['seeds'][0];names=summary['config']['images'];tasks=summary['config']['tasks']
    def row(task,name,method):return next(r for r in rows if r['task']==task and r['image']==name and r['method']==method and (method=='input' or r['base_seed']==seed))
    def load(task,name,method):
        path=ROOT/'data/gt'/f'{name}.png' if method=='gt' else ROOT/row(task,name,method)['image_path'].replace('\\','/')
        return Image.open(path).convert('RGB')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    for task in tasks:
        fig,axes=plt.subplots(4,3,figsize=(11,14.1),layout='constrained')
        for i,name in enumerate(names):
            pictures=[load(task,name,m) for m in ['gt','input','pixrestore_s']]
            strip=Image.new('RGB',(512*3+16,512),'white')
            for j,picture in enumerate(pictures):
                strip.paste(picture,(j*520,0));axes[i,j].imshow(picture);axes[i,j].axis('off')
                title=f'{name} | reference' if j==0 else f"{['','Input','PixRestore-S'][j]} | {row(task,name,['gt','input','pixrestore_s'][j])['psnr_db']:.2f} dB / {row(task,name,['gt','input','pixrestore_s'][j])['ssim']:.3f}"
                axes[i,j].set_title(title,fontsize=11,pad=6)
            strip.save(figures/f'strip_{task}_{name}.png')
        fig.suptitle(f'{LABELS[task]} | fixed seed {seed} | PSNR / SSIM',fontsize=16)
        fig.savefig(figures/f'comparison_{task}.png',dpi=150);plt.close(fig)
    x=np.arange(len(tasks));width=.34
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for ax,metric,label in zip(axes,['psnr_db','ssim'],['PSNR (dB)','SSIM']):
        observed=[summary['aggregate'][t]['input'][metric] for t in tasks]
        restored=[summary['aggregate'][t]['pixrestore_s'][metric] for t in tasks]
        for offset,values,color,name in [(-width/2,observed,'#9BA9B9','Input'),(width/2,restored,'#007A78','PixRestore-S')]:
            bars=ax.bar(x+offset,values,width,color=color,label=name)
            ax.bar_label(bars,fmt='%.2f' if metric=='psnr_db' else '%.3f',padding=3,fontsize=10)
        ax.set_xticks(x,['4x SR','Gaussian 25','JPEG 10*']);ax.set_ylabel(label);ax.set_ylim(0,40 if metric=='psnr_db' else 1.08)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,ncol=2,loc='upper left')
    fig.suptitle('4 public images; PixRestore means over 3 seeds; *extra generalization task',fontsize=12)
    fig.savefig(figures/'quality_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.2),layout='constrained')
    changes=[summary['aggregate'][t]['pixrestore_s']['psnr_db']-summary['aggregate'][t]['input']['psnr_db'] for t in tasks]
    bars=ax.bar(x,changes,color=['#BB5349' if d<0 else '#007A78' for d in changes],width=.55)
    ax.bar_label(bars,labels=[f'{d:+.2f}' for d in changes],padding=4);ax.axhline(0,color='#344054',linewidth=.8)
    ax.set_xticks(x,['4x SR','Gaussian 25','JPEG 10*']);ax.set_ylabel('PSNR change vs. input (dB)');ax.set_ylim(-3,15)
    ax.set_title('A mixed result: retain the SR fidelity failure');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(figures/'psnr_gain.png',dpi=180);plt.close(fig)
    routes=json.loads((out/'router_weights.json').read_text(encoding='utf-8'));means=[];route_summary={}
    for task in tasks:
        selected=[r for r in routes['rows'] if r['task']==task and r['base_seed']==seed]
        values=np.array([r['p_l'] for r in selected]);means.append(values.mean(0))
        route_summary[task]={'n_distinct_inputs':len(selected),'layer_indices':routes['layer_indices'],'mean':values.mean(0).tolist(),'std_across_images':values.std(0,ddof=1).tolist()}
    (figures/'router_task_means.json').write_text(json.dumps(route_summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    im=ax.imshow(means,cmap='YlGnBu',vmin=0,vmax=max(max(v) for v in means)*1.15)
    ax.set_xticks(range(6),routes['layer_indices']);ax.set_xlabel('DINO block index (zero-based)')
    ax.set_yticks(range(3),['4x SR','Gaussian 25','JPEG 10*'])
    for i in range(3):
        for j in range(6):ax.text(j,i,f'{means[i][j]:.3f}',ha='center',va='center',color='white' if means[i][j]>.22 else 'black',fontsize=11)
    ax.set_title('Adaptive router weights: descriptive means over 4 inputs')
    fig.colorbar(im,ax=ax,label='Mean layer weight');fig.savefig(figures/'router_heatmap.png',dpi=180);plt.close(fig)
    print(f'Created figures in {figures}')
if __name__=='__main__':main()
