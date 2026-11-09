"""Small runtime adapter; upstream model and sampler files remain unchanged."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parent

def eager_compile(function=None,*args,**kwargs):
    """Disable an optional compiler unavailable on Windows/PyTorch2.1."""
    return function if function is not None else lambda f:f

def source_directory(name):
    candidates=[ROOT/'external'/name, ROOT.parent/'work/candidates'/name]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f'Clone {name} via setup_sources.py first.')

def import_official():
    # This is an execution-mode change only. Decorated Python functions execute
    # their original bodies, with no changes to checkpoint, equations or routing.
    torch.compile=eager_compile
    sys.path.insert(0,str(source_directory('PixRestore')))
    import inference
    return inference
