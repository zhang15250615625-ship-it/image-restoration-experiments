"""Resume the official public checkpoint with validated HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib, json, time, urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
SIZE = 2211383297
EXPECTED_SHA256 = 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1'
CHUNK = 8 * 1024 * 1024
DEST = ROOT / 'weights' / '256x256_diffusion_uncond.pt'
PARTS = ROOT / 'weights' / 'download_parts'

def fetch(i):
    start, end = i * CHUNK, min((i + 1) * CHUNK, SIZE) - 1
    dest = PARTS / f'{i:04d}.part'
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return i, dest.stat().st_size, 'cached'
    for attempt in range(10):
        try:
            request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=30) as response:
                assert response.status == 206, response.status
                assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'), response.headers
                payload = response.read()
            assert len(payload) == end - start + 1
            dest.write_bytes(payload)
            return i, len(payload), 'downloaded'
        except Exception as error:
            print(f'chunk {i} attempt {attempt+1} failed: {type(error).__name__}: {error}', flush=True)
            if attempt == 9:
                raise
            time.sleep(2 * (attempt + 1))

def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not DEST.exists() or DEST.stat().st_size != SIZE:
        count = (SIZE + CHUNK - 1) // CHUNK
        done = 0
        failures = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(fetch, i) for i in range(count)]):
                try:
                    _, length, status = future.result()
                except Exception as error:
                    failures.append(repr(error))
                    print(f'Chunk exhausted retries: {error}', flush=True)
                    continue
                done += length
                print(f'{done / SIZE:.1%} {done / 1e6:.1f} MB {done / 1e6 / (time.perf_counter()-start):.2f} MB/s {status}', flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} chunks failed; rerun to resume. First error: {failures[0]}')
        with DEST.open('wb') as out:
            for i in range(count):
                out.write((PARTS / f'{i:04d}.part').read_bytes())
    digest = hashlib.file_digest(DEST.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else sha256(DEST)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f'Checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {digest}')
    manifest = {'url': URL, 'bytes': DEST.stat().st_size, 'sha256': digest}
    (ROOT / 'weights' / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)

def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

if __name__ == '__main__':
    main()
