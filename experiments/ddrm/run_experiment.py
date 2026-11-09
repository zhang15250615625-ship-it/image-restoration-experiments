"""Small-sample DDRM pretrained-inference reproduction; upstream code is unmodified."""
from pathlib import Path
import argparse, csv, gc, hashlib, importlib.metadata, json, platform, random, subprocess, sys, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'official_ddrm'))
from functions.denoising import efficient_generalized_steps
from functions.svd_replacement import SuperResolution
from guided_diffusion.script_util import create_model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def as_array(tensor):
    return tensor.detach().cpu()[0].permute(1, 2, 0).numpy().astype(np.float32)

def save_image(path, array):
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)

def evaluate(gt, prediction, lowres):
    # All scores are RGB, range [0,1], full 256x256, no border crop, no Y conversion.
    pred = np.clip(prediction, 0, 1)
    measured = pred.reshape(64, 4, 64, 4, 3).mean(axis=(1, 3))
    measured_raw = prediction.reshape(64, 4, 64, 4, 3).mean(axis=(1, 3))
    return {
        'psnr_db': float(peak_signal_noise_ratio(gt, pred, data_range=1.0)),
        'ssim': float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0)),
        'measurement_rmse': float(np.sqrt(np.mean((measured - lowres) ** 2))),
        'measurement_rmse_before_clip': float(np.sqrt(np.mean((measured_raw - lowres) ** 2))),
        'clipped_fraction': float(np.mean((prediction < 0) | (prediction > 1))),
    }

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as inp:
        for block in iter(lambda: inp.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def dump_results(rows, output, config):
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for method in dict.fromkeys(row['method'] for row in rows):
        selected = [r for r in rows if r['method'] == method]
        aggregate[method] = {'n': len(selected)}
        for key in ['psnr_db', 'ssim', 'measurement_rmse', 'measurement_rmse_before_clip', 'seconds', 'peak_allocated_mib', 'peak_reserved_mib', 'nfe']:
            aggregate[method][key] = float(np.mean([r[key] for r in selected]))
    (output / 'summary.json').write_text(json.dumps({'config': config, 'aggregate': aggregate, 'rows': rows}, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', default=['astronaut', 'coffee', 'chelsea', 'rocket'])
    parser.add_argument('--steps', nargs='+', type=int, default=[20, 100])
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--output', default='results')
    parser.add_argument('--checkpoint', default='weights/256x256_diffusion_uncond.pt')
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'A CUDA GPU is required by the unchanged official sampler.'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    (output / 'arrays').mkdir(exist_ok=True)
    model_config = yaml.safe_load((ROOT / 'official_ddrm/configs/imagenet_256.yml').read_text())
    commit = subprocess.check_output(['git', '-C', str(ROOT / 'official_ddrm'), 'rev-parse', 'HEAD'], text=True).strip()
    config = {
        'scope': '4-image pretrained inference study; not the paper ImageNet benchmark; no training',
        'paper': 'Denoising Diffusion Restoration Models (NeurIPS 2022)',
        'official_commit': commit,
        'checkpoint': args.checkpoint, 'checkpoint_sha256': sha256(ROOT / args.checkpoint),
        'operator': 'official SuperResolution(channels=3,img_dim=256,ratio=4): nonoverlapping 4x4 average pooling',
        'noise_std_01': 0.0, 'sigma_0_model_space': 0.0,
        'etaA': 0.85, 'etaC': 0.85, 'etaB': 1.0,
        'schedule': 'official range(0,1000,1000//steps); 20 steps starts at t=950; 100 starts at t=990',
        'steps': args.steps, 'images': args.images, 'seed': args.seed, 'seed_rule': 'base seed plus image index; reset for each step count',
        'precision': 'official convert_to_fp16() for UNet torso; fp32 diffusion/operator arithmetic',
        'batch_size': 1, 'model_configuration': model_config,
        'metrics': 'PSNR/SSIM on float32 clipped RGB [0,1], full 256x256, no border crop; SSIM skimage default 7x7/uniform/sample covariance',
        'timing': 'single warm forward before runs; torch.cuda.synchronize before/after sampler, includes official CPU trajectory copies; excludes model loading, preprocessing, metric calculation, image saves',
        'baseline_timing': 'one CPU interpolation call per image, no repeated timing; not a robust benchmark',
        'determinism': 'seeded Python/NumPy/PyTorch; cuDNN deterministic=True benchmark=False; TF32=False; no cross-version bitwise guarantee',
        'hardware': torch.cuda.get_device_name(), 'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 1024**2,
        'python': sys.version, 'platform': platform.platform(), 'torch_cuda': torch.version.cuda,
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy', 'Pillow', 'scikit-image', 'scipy', 'PyYAML', 'tqdm', 'matplotlib']},
        'adaptations': ['Standalone wrapper replaces official CLI/data loader.', 'Batch size 1 for 6GB GPU.', 'Loads checkpoint on CPU before CUDA to avoid transient duplicate GPU weights.', 'Official efficient_generalized_steps, SuperResolution, create_model and model YAML used without edits.', 'Uses center-cropped public scikit-image samples rather than ImageNet.', 'Adds fair baselines, measurement-consistency metrics, synchronized timing, memory measurement, output manifests.'],
    }
    (output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Loading checkpoint on CPU...', flush=True)
    seed_all(args.seed)
    model = create_model(**model_config['model'])
    model.convert_to_fp16()
    state = torch.load(ROOT / args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    model.to('cuda').eval()
    beta = torch.from_numpy(np.linspace(0.0001, 0.02, 1000, dtype=np.float64)).float().cuda()
    operator = SuperResolution(3, 256, 4, 'cuda')
    with torch.no_grad():
        warm = model(torch.zeros(1, 3, 256, 256, device='cuda'), torch.ones(1, device='cuda') * 500)
        del warm
    torch.cuda.synchronize()
    rows = []
    for index, name in enumerate(args.images):
        array = np.asarray(Image.open(ROOT / 'data' / f'{name}.png').convert('RGB'), dtype=np.float32) / 255
        gt_cpu = torch.from_numpy(array).permute(2, 0, 1)[None].contiguous()
        gt = gt_cpu.cuda()
        low_cpu = F.avg_pool2d(gt_cpu, 4)
        low = as_array(low_cpu)
        y = operator.H(gt * 2 - 1)
        expected = F.avg_pool2d(gt * 2 - 1, 4).reshape(1, -1)
        residual = float((y - expected).abs().max())
        assert residual < 2e-6, residual
        print(f'{name}: operator max error={residual:.3g}', flush=True)
        save_image(output / 'images' / f'{name}_original.png', array)
        save_image(output / 'images' / f'{name}_observation_64.png', low)
        for method in ['nearest', 'bicubic']:
            start = time.perf_counter()
            kwargs = {'align_corners': False} if method == 'bicubic' else {}
            restored = F.interpolate(low_cpu, size=(256, 256), mode=method, **kwargs)
            elapsed = time.perf_counter() - start
            prediction = as_array(restored)
            row = {'image': name, 'method': method, 'seed': args.seed + index, 'nfe': 0, 'seconds': elapsed,
                'peak_allocated_mib': 0.0, 'peak_reserved_mib': 0.0, 'operator_max_abs_error': residual, **evaluate(array, prediction, low)}
            rows.append(row)
            save_image(output / 'images' / f'{name}_{method}.png', prediction)
            np.save(output / 'arrays' / f'{name}_{method}.npy', prediction)
        for steps in args.steps:
            assert 1000 % steps == 0
            seed_all(args.seed + index)
            x = torch.randn_like(gt)
            sequence = list(range(0, 1000, 1000 // steps))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            trajectories, predictions = efficient_generalized_steps(x, sequence, model, beta, operator, y,
                sigma_0=0.0, etaB=1.0, etaA=0.85, etaC=0.85)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / 1024**2
            reserved = torch.cuda.max_memory_reserved() / 1024**2
            prediction = as_array((trajectories[-1] + 1) / 2)
            assert np.isfinite(prediction).all()
            row = {'image': name, 'method': f'ddrm_{steps}', 'seed': args.seed + index, 'nfe': len(sequence),
                'seconds': elapsed, 'peak_allocated_mib': peak, 'peak_reserved_mib': reserved,
                'operator_max_abs_error': residual, **evaluate(array, prediction, low)}
            rows.append(row)
            save_image(output / 'images' / f'{name}_ddrm_{steps}.png', prediction)
            np.save(output / 'arrays' / f'{name}_ddrm_{steps}.npy', prediction)
            dump_results(rows, output, config)
            print(json.dumps(row), flush=True)
            del trajectories, predictions, x
            gc.collect()
        del gt, y
    dump_results(rows, output, config)
    print('FINISHED ' + str(output / 'summary.json'), flush=True)

if __name__ == '__main__':
    main()
