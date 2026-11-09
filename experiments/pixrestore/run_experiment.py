"""PixRestore-S: real one-step pretrained inference on fixed public images."""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,hashlib,importlib.metadata,json,platform,random,subprocess,sys,time
from datetime import datetime,timezone,timedelta
import numpy as np
from PIL import Image
import torch
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from runtime import ROOT,import_official,source_directory

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''): h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_array(tensor):
    return tensor.detach().float().cpu()[0].permute(1,2,0).numpy()

def metrics(gt,pred):
    clipped=np.clip(pred,0,1)
    return {'psnr_db':float(peak_signal_noise_ratio(gt,clipped,data_range=1.0)),
            'ssim':float(structural_similarity(gt,clipped,channel_axis=2,data_range=1.0)),
            'clipped_fraction':float(np.mean((pred<0)|(pred>1)))}

def save_image(path,pred):
    Image.fromarray(np.rint(np.clip(pred,0,1)*255).astype(np.uint8)).save(path)

def summarize(rows,config,output,router_rows):
    aggregate={}; primary={}
    for task in config['tasks']:
        aggregate[task]={};primary[task]={}
        for method in ['input','pixrestore_s']:
            selected=[r for r in rows if r['task']==task and r['method']==method]
            if not selected: continue
            def reduce(group):
                item={'n':len(group),'psnr_db':float(np.mean([r['psnr_db'] for r in group])),
                      'ssim':float(np.mean([r['ssim'] for r in group]))}
                if method=='pixrestore_s':
                    item.update({'seconds_median_mean':float(np.mean([r['seconds_median'] for r in group])),
                                 'peak_allocated_mib_max':max(r['peak_allocated_mib'] for r in group),
                                 'nfe':1})
                return item
            aggregate[task][method]=reduce(selected)
            paired=[r for r in selected if method=='input' or r['base_seed']==config['seeds'][0]]
            if paired: primary[task][method]=reduce(paired)
    stability={}
    for task in config['tasks']:
        per_seed=[]
        for seed in config['seeds']:
            group=[r for r in rows if r['task']==task and r['method']=='pixrestore_s' and r['base_seed']==seed]
            if group:
                per_seed.append({'base_seed':seed,'n':len(group),'psnr_db':float(np.mean([r['psnr_db'] for r in group])),
                                 'ssim':float(np.mean([r['ssim'] for r in group]))})
        if per_seed:
            stability[task]={'per_seed':per_seed,'statistic':'sample std across 4-image mean values; not a confidence interval',
                             'psnr_mean':float(np.mean([r['psnr_db'] for r in per_seed])),
                             'psnr_std':float(np.std([r['psnr_db'] for r in per_seed],ddof=1)) if len(per_seed)>1 else None,
                             'ssim_mean':float(np.mean([r['ssim'] for r in per_seed])),
                             'ssim_std':float(np.std([r['ssim'] for r in per_seed],ddof=1)) if len(per_seed)>1 else None}
    payload={'config':config,'aggregate':aggregate,'primary_seed_aggregate':primary,'seed_stability':stability,'rows':rows}
    (output/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'router_weights.json').write_text(json.dumps({'layer_indices':config['model_config']['encoder_layers'],
        'index_convention':'zero-based DINO block indices as used by official extract_layers',
        'source':'read-only hook on dino_prior_content_head; official softmax(float), bf16 cast, clamp_min(1e-8)',
        'rows':router_rows},indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[20261116,20261216,20261316])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',default=None,help='New relative directory; existing directories are never overwritten.')
    args=parser.parse_args()
    if args.repeats<1 or not args.seeds:parser.error('At least one seed and one timing repeat are required.')
    output=(ROOT/(args.output or datetime.now().strftime('runs/run_%Y%m%d_%H%M%S'))).resolve()
    if not output.is_relative_to(ROOT.resolve()):parser.error('--output must be inside this experiment directory.')
    if output.exists():parser.error(f'Output already exists: {output.name}. Choose a new --output directory.')
    if not torch.cuda.is_available():parser.error('This measured inference protocol requires a CUDA GPU.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    official=import_official()
    from accelerate import Accelerator
    from pixrestore.vision import load_dinov2,extract_layers
    for folder in ['images','arrays','figures']:(output/folder).mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    weight=ROOT/'weights/pixrestore-s/clean_weights/ema_model.safetensors'
    expected='13c10ca30fa865977de393457e731db9f2469159dbd69c85fb3c79b3edf06895'
    weight_hash=digest(weight)
    assert weight_hash==expected,'PixRestore checkpoint hash differs from published HF LFS SHA256'
    assert digest(ROOT/'weights/pixrestore-s/config.json')=='978deb1b0aaa544c12355ffb8ffffee1fbf012493de876736aba2c32185d18f4','Model configuration differs from the audited snapshot'
    assert digest(ROOT/'weights/dinov2_vits14_pretrain.pth')=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9','DINO checkpoint differs from the audited snapshot'
    model_config=official.read_config(ROOT/'weights/pixrestore-s/config.json')
    model_config['dinov2_repository']=str(source_directory('dinov2'))
    model_config['dinov2_checkpoint']=str(ROOT/'weights/dinov2_vits14_pretrain.pth')
    model_config['cfg_scale']=1.0
    state=official.read_state_dict(weight)
    model_config=official.apply_checkpoint_architecture_flags(model_config,state)
    cfg=SimpleNamespace(**model_config)
    accelerator=Accelerator(mixed_precision=cfg.mixed_precision)
    model=official.build_model(cfg)
    official.load_weights(model,weight,state=state)  # strict=True in upstream
    model.to(accelerator.device).eval().requires_grad_(False)
    encoder=load_dinov2(cfg.encoder_type,accelerator.device,repository=cfg.dinov2_repository,checkpoint=cfg.dinov2_checkpoint)
    flow=official.build_flow(cfg,accelerator)
    del state
    source_revisions={name:subprocess.check_output(['git','-C',str(source_directory(name)),'rev-parse','HEAD'],text=True).strip() for name in ['PixRestore','dinov2']}
    public_model_config={**model_config,'dinov2_repository':'external/dinov2','dinov2_checkpoint':'weights/dinov2_vits14_pretrain.pth'}
    config={'paper':'PixRestore: Unified Image Restoration via Pixel Diffusion Transformer',
            'paper_url':'https://arxiv.org/abs/2608.16793','first_submission':'2026-08-17',
            'executed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'scope':'Small-sample pretrained one-step inference; not retraining or full paper benchmark. JPEG is an extra generalization check outside the eight paper benchmark tasks.',
            'tasks':list(manifest['tasks']),'task_protocol':manifest['tasks'],'images':['astronaut','coffee','chelsea','rocket'],
            'resolution':512,'seeds':args.seeds,'seed_rule':'model seed = base seed + image index; reset before each timed repeat; independent fixed observation seed in data manifest',
            'nfe':1,'repeats':args.repeats,'precision':'official BF16 autocast; FP32 parameter storage',
            'metrics':'RGB full512x512 no border crop, clipped float[0,1]; skimage default SSIM7x7 uniform/sample covariance; PNG round-to-nearest saved after floating metrics',
            'timing':'one untimed global warmup; median of repeated CUDA-synchronized full restoration calls including DINO extraction and one DiT evaluation; excludes model loading, preprocessing, metrics, disk IO; no compiler',
            'source_revisions':source_revisions,'model_config':public_model_config,'checkpoint_sha256':weight_hash,
            'dino_checkpoint_sha256':digest(Path(cfg.dinov2_checkpoint)),
            'parameter_count_pixrestore':sum(p.numel() for p in model.parameters()),
            'parameter_count_dinov2':sum(p.numel() for p in encoder.parameters()),
            'hardware':torch.cuda.get_device_name(),'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,
            'python':sys.version,'platform':platform.platform(),'torch_cuda':torch.version.cuda,
            'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow','scikit-image','accelerate','timm','einops','safetensors','xformers']},
            'adaptations':['Windows/PyTorch2.1 cannot execute torch.compile; optional compilation decorators execute eagerly via outer adapter.',
                           'One GPU/batch1 instead of example 8-GPU accelerate launch.',
                           'Official inference build_model/load_weights(strict)/build_flow/load_dinov2/extract_layers/sample_multistep_fm used; sampler and model files unmodified.',
                           'Local DINO repository/checkpoint paths replace author absolute paths.',
                           'Synthetic public images, measurement freezing, metrics/timing/seed/routing logging added outside algorithm.']}
    (output/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    captured=[]
    def hook(module,inputs,value):captured.append(value.detach())
    hook_handle=model.dino_prior_content_head.register_forward_hook(hook)
    def infer(lq,seed):
        seed_all(seed)
        captured.clear()
        with torch.no_grad(),accelerator.autocast():
            features=extract_layers(encoder,lq,cfg.encoder_layers,cfg.encoder_input_size)
            restored=flow.sample_multistep_fm(model,lq,venc_fea=features,n_steps=1,schedule='linear')
        return restored
    rows=[];router_rows=[];warmed=False
    for item in manifest['records']:
        image=item['image'];task=item['task'];index=config['images'].index(image)
        gt=np.asarray(Image.open(ROOT/item['gt']),dtype=np.float32)/255
        observed=np.asarray(Image.open(ROOT/item['lq']),dtype=np.float32)/255
        lq=torch.from_numpy(observed).permute(2,0,1)[None].contiguous().to(accelerator.device)*2-1
        base={'task':task,'image':image,'method':'input','base_seed':None,'model_seed':None,
              'observation_seed':item['observation_seed'],'nfe':0,**metrics(gt,observed),
              'seconds_median':None,'seconds_repeats':[],'peak_allocated_mib':None,'peak_reserved_mib':None,
              'repeat_max_abs_difference':None,'array_path':None,'image_path':item['lq']}
        rows.append(base)
        save_image(output/'images'/f'{image}_gt.png',gt)
        save_image(output/'images'/f'{image}_{task}_input.png',observed)
        if not warmed:
            infer(lq,args.seeds[0]+index)
            torch.cuda.synchronize();warmed=True
        for base_seed in args.seeds:
            seed=base_seed+index;times=[];peaks=[];reserved=[];first=None;difference=0.0;route=None
            for repeat in range(args.repeats):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                restored=infer(lq,seed)
                torch.cuda.synchronize();times.append(time.perf_counter()-started)
                peaks.append(torch.cuda.max_memory_allocated()/1024**2)
                reserved.append(torch.cuda.max_memory_reserved()/1024**2)
                candidate=to_array((restored+1)/2)
                assert np.isfinite(candidate).all()
                if first is None:
                    first=candidate.copy()
                    assert len(captured)==1,f'Expected one router query, saw {len(captured)}'
                    route=torch.softmax(captured[0].float(),dim=-1).to(captured[0].dtype).clamp_min(1e-8).float().cpu().tolist()[0]
                else:difference=max(difference,float(np.abs(candidate-first).max()))
            image_path=output/'images'/f'{image}_{task}_pixrestore_s_seed{base_seed}.png'
            array_path=output/'arrays'/f'{image}_{task}_pixrestore_s_seed{base_seed}.npy'
            save_image(image_path,first);np.save(array_path,first)
            row={**base,'method':'pixrestore_s','base_seed':base_seed,'model_seed':seed,'nfe':1,
                 **metrics(gt,first),'seconds_median':float(np.median(times)),'seconds_repeats':times,
                 'peak_allocated_mib':max(peaks),'peak_reserved_mib':max(reserved),
                 'repeat_max_abs_difference':difference,'array_path':array_path.relative_to(ROOT).as_posix(),
                 'image_path':image_path.relative_to(ROOT).as_posix()}
            rows.append(row);router_rows.append({'task':task,'image':image,'base_seed':base_seed,'model_seed':seed,'p_l':route})
            summarize(rows,config,output,router_rows)
            print(json.dumps(row),flush=True)
    hook_handle.remove()
    config['complete']=True
    summarize(rows,config,output,router_rows)
    (output/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('FINISHED '+str(output/'summary.json'),flush=True)

if __name__=='__main__':main()
