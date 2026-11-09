"""Download official files and verify SHA256 before making them available."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent
FILES=[
    ('pixrestore-s/clean_weights/ema_model.safetensors','https://huggingface.co/VCLab-PolyU/PixRestore/resolve/main/pixrestore-s/clean_weights/ema_model.safetensors',128709340,'13c10ca30fa865977de393457e731db9f2469159dbd69c85fb3c79b3edf06895','HF published LFS SHA256'),
    ('pixrestore-s/config.json','https://huggingface.co/VCLab-PolyU/PixRestore/resolve/main/pixrestore-s/config.json',2281,'978deb1b0aaa544c12355ffb8ffffee1fbf012493de876736aba2c32185d18f4','SHA256 of official file downloaded for this experiment'),
    ('dinov2_vits14_pretrain.pth','https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',88283115,'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9','SHA256 of official file downloaded for this experiment'),
]
def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def verify(path,size,expected):return path.is_file() and path.stat().st_size==size and sha256(path)==expected
def main():
    manifest=[]
    for relative,url,size,expected,provenance in FILES:
        dest=ROOT/'weights'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if not verify(dest,size,expected):raise RuntimeError(f'Existing {relative} fails verification; it was not overwritten.')
        else:
            temporary=dest.with_suffix(dest.suffix+'.partial')
            request=urllib.request.Request(url,headers={'User-Agent':'PixRestore-Reproduction/1.0'})
            print(f'Downloading {relative} ({size:,} bytes)',flush=True)
            with urllib.request.urlopen(request,timeout=180) as response,temporary.open('wb') as handle:
                while block:=response.read(1024*1024):handle.write(block)
            if not verify(temporary,size,expected):raise RuntimeError(f'SHA256 or size mismatch: {relative}; incomplete file retained for inspection.')
            temporary.replace(dest)
        print(f'Verified {relative}: {expected}',flush=True)
        manifest.append({'path':'weights/'+relative,'url':url,'bytes':size,'sha256':expected,'hash_provenance':provenance})
    (ROOT/'weights/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
