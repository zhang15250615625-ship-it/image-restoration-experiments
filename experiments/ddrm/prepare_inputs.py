"""Freeze four redistributable scikit-image examples as 256x256 RGB inputs."""
from pathlib import Path
from PIL import Image
from skimage import data
import hashlib, inspect, json

ROOT = Path(__file__).resolve().parent

def main():
    target = ROOT / 'data'
    target.mkdir(exist_ok=True)
    records = []
    for name in ['astronaut', 'coffee', 'chelsea', 'rocket']:
        source = getattr(data, name)()
        image = Image.fromarray(source).convert('RGB')
        width, height = image.size
        side = min(width, height)
        box = ((width-side)//2, (height-side)//2, (width-side)//2+side, (height-side)//2+side)
        prepared = image.crop(box).resize((256, 256), Image.Resampling.BICUBIC)
        dest = target / f'{name}.png'
        prepared.save(dest)
        records.append({'name': name, 'loader': f'skimage.data.{name}',
            'source_array_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
            'original_shape': list(source.shape), 'center_crop_box': box,
            'resize': 'Pillow bicubic, 256x256; stored uint8 RGB PNG',
            'prepared_file_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'documentation_url': f'https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.{name}',
            'provenance_and_license': inspect.getdoc(getattr(data, name))})
    (target / 'sources.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {len(records)} images in {target}')

if __name__ == '__main__':
    main()
