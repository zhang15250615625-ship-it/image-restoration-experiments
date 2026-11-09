"""Build fixed 512px references and three synthetic degradations from public originals."""
from pathlib import Path
from io import BytesIO
import hashlib, inspect, json
import numpy as np
from PIL import Image
from skimage import data

ROOT = Path(__file__).resolve().parent
NAMES = ['astronaut', 'coffee', 'chelsea', 'rocket']
TASKS = ['sr4', 'gaussian25', 'jpeg10']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    (ROOT/'data/gt').mkdir(parents=True,exist_ok=True)
    for task in TASKS:
        (ROOT/'data/lq'/task).mkdir(parents=True,exist_ok=True)
    records=[]
    for index,name in enumerate(NAMES):
        raw=getattr(data,name)()
        source=Image.fromarray(raw).convert('RGB')
        width,height=source.size
        side=min(width,height)
        box=((width-side)//2,(height-side)//2,(width-side)//2+side,(height-side)//2+side)
        gt=source.crop(box).resize((512,512),Image.Resampling.BICUBIC)
        gt_path=ROOT/'data/gt'/f'{name}.png'
        gt.save(gt_path)
        # SR: true 512 -> 128 bicubic downsampling, then 128 -> 512 bicubic input.
        observation=gt.resize((128,128),Image.Resampling.BICUBIC)
        observation.save(ROOT/'data/lq/sr4'/f'{name}_observation128.png')
        sr4=observation.resize((512,512),Image.Resampling.BICUBIC)
        # A fixed observation is shared across all model seeds. Clipping and 8-bit
        # quantization intentionally match the PNG input pathway of the public CLI.
        seed=20261001+index
        rng=np.random.default_rng(seed)
        noisy=np.asarray(gt,dtype=np.float32)/255+rng.normal(0,25/255,(512,512,3))
        gaussian25=Image.fromarray(np.rint(np.clip(noisy,0,1)*255).astype(np.uint8))
        buffer=BytesIO()
        gt.save(buffer,format='JPEG',quality=10,subsampling=2,optimize=False)
        buffer.seek(0)
        jpeg10=Image.open(buffer).convert('RGB')
        outputs={'sr4':sr4,'gaussian25':gaussian25,'jpeg10':jpeg10}
        for task,image in outputs.items():
            path=ROOT/'data/lq'/task/f'{name}.png'
            image.save(path)
            records.append({'image':name,'task':task,'gt':gt_path.relative_to(ROOT).as_posix(),
                            'lq':path.relative_to(ROOT).as_posix(),'gt_sha256':digest(gt_path),
                            'lq_sha256':digest(path),'observation_seed':seed if task=='gaussian25' else None})
    sources=[]
    for name in NAMES:
        raw=getattr(data,name)()
        h,w=raw.shape[:2]; side=min(h,w)
        sources.append({'image':name,'loader':f'skimage.data.{name}', 'original_shape':list(raw.shape),
                        'source_array_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                        'crop_box':[(w-side)//2,(h-side)//2,(w-side)//2+side,(h-side)//2+side],
                        'resize':'Center square from original skimage array, Pillow bicubic to 512x512, uint8 RGB PNG; never upsamples previous 256px experiment inputs.',
                        'source_url':f'https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.{name}',
                        'provenance_and_license':inspect.getdoc(getattr(data,name))})
    config={'resolution':512,'tasks':{
        'sr4':'Pillow bicubic 512->128; bicubic 128->512 input. No additional noise.',
        'gaussian25':'Independent RGB Gaussian noise std25/255 using NumPy default_rng, clip[0,1], round to uint8 PNG. Same fixed observed image for every model seed.',
        'jpeg10':'JPEG quality10, chroma subsampling2 (4:2:0), optimize=False, decoded to RGB PNG. Extra generalization check outside the paper eight benchmark tasks.'},
        'sources':sources,'records':records}
    (ROOT/'data/manifest.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Prepared {len(records)} fixed 512x512 observations from original source arrays.',flush=True)

if __name__=='__main__':
    main()
